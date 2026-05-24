#!/usr/bin/env python3
"""DRA-GRPO baseline using this repo's exact GRPO data/prompt/training stack.

This script mirrors Answer_only_GRPO.py and changes only the reward construction:

1. Compute the same final-answer correctness reward as Answer_only_GRPO.py.
2. Embed completions within each rollout group.
3. Apply the DRA diversity weight from the official DRA-GRPO implementation:
   reward_i <- reward_i * 1 / sum_j cosine(emb_i, emb_j).

The resulting scalar rewards are then passed to the same TRL GRPOTrainer.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    DEFAULT_DATASET_NAME,
    DEFAULT_EVAL_METADATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_METADATA_PATH,
    add_bool_arg,
    add_vllm_args,
    align_gold_answers,
    align_values,
    answers_match,
    attach_percent_progress_callback,
    build_grpo_training_components,
    build_peft_config,
    build_training_config_payload,
    create_grpo_config,
    create_grpo_trainer,
    default_bf16_enabled,
    ensure_output_dir,
    extract_final_answer_from_output,
    extract_text_from_completion,
    import_torch,
    load_tokenizer,
    load_training_data,
    maybe_warn_on_smoke_metrics,
    parse_completion_sections,
    run_parser_self_test,
    run_smoke_test,
    set_seed,
    write_json,
)


DEFAULT_OUTPUT_DIR = "outputs/grpo_dra_qwen3b_bigmath"
DEFAULT_DRA_DEBUG_JSONL = "dra_reward_debug.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DRA-GRPO baseline training with this repo's fixed metadata and verifier."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--min_solve_rate", type=float, default=0.2)
    parser.add_argument("--max_solve_rate", type=float, default=0.8)
    parser.add_argument("--train_size", type=int, default=None, help="Number of train rows to use. Defaults to all rows.")
    parser.add_argument("--eval_size", type=int, default=None, help="Number of eval rows to use. Defaults to all rows.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    add_vllm_args(parser)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--loss_type",
        choices=[
            "grpo",
            "dapo",
            "bnpo",
            "dr_grpo",
            "cispo",
            "sapo",
            "vespo",
            "luspo",
        ],
        default=None,
        help="Optional TRL GRPO loss formulation override. If omitted, the installed TRL default is used.",
    )
    parser.add_argument("--scale_rewards", choices=["group", "batch", "none"], default=None)
    parser.add_argument("--importance_sampling_level", choices=["token", "sequence"], default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--progress_interval_percent",
        type=int,
        default=10,
        help="Print compact training progress updates every N percent of max_steps.",
    )

    add_bool_arg(parser, "use_lora", True, "Enable LoRA adapters.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    add_bool_arg(parser, "bf16", True, "Use bf16 when supported.")
    add_bool_arg(parser, "gradient_checkpointing", True, "Enable gradient checkpointing.")

    add_bool_arg(parser, "smoke_test_only", False, "Run prompt-format smoke test only.")
    parser.add_argument("--smoke_test_examples", type=int, default=8)
    parser.add_argument("--smoke_test_generations", type=int, default=4)
    add_bool_arg(parser, "run_pretrain_smoke", False, "Run smoke test before GRPO training.")
    parser.add_argument("--min_smoke_success_rate", type=float, default=0.8)
    add_bool_arg(parser, "parser_self_test", False, "Run parser self-test cases and exit.")

    parser.add_argument("--format_bonus", type=float, default=0.0)
    add_bool_arg(
        parser,
        "use_fixed_metadata",
        False,
        "Load fixed selected_train/selected_eval metadata for a fair comparison.",
    )
    parser.add_argument("--train_metadata_path", default=DEFAULT_TRAIN_METADATA_PATH)
    parser.add_argument("--eval_metadata_path", default=DEFAULT_EVAL_METADATA_PATH)

    parser.add_argument(
        "--dra_embedding_backend",
        choices=["jina", "sentence_transformers", "hf_mean_pool"],
        default="jina",
        help="Embedding backend used for DRA completion diversity.",
    )
    parser.add_argument("--dra_embedding_model", default="jinaai/jina-embeddings-v2-small-en")
    parser.add_argument("--dra_embedding_max_length", type=int, default=3584)
    parser.add_argument("--dra_eps", type=float, default=1e-6)
    parser.add_argument("--dra_debug_jsonl", default=None)

    return parser.parse_args()


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


class CompletionEmbedder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model: Any = None
        self.tokenizer: Any = None

    def _load(self) -> None:
        if self.model is not None:
            return

        torch = import_torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.args.dra_embedding_backend == "jina":
            # Some newer minimal Transformers builds no longer expose transformers.onnx,
            # while Jina's remote config imports OnnxConfig even for normal PyTorch loading.
            if "transformers.onnx" not in sys.modules:
                onnx_stub = types.ModuleType("transformers.onnx")

                class OnnxConfig:  # pragma: no cover - only used as an import shim
                    pass

                onnx_stub.OnnxConfig = OnnxConfig
                sys.modules["transformers.onnx"] = onnx_stub
            import transformers.pytorch_utils as pytorch_utils

            if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

                def find_pruneable_heads_and_indices(
                    heads,
                    n_heads,
                    head_size,
                    already_pruned_heads,
                ):
                    mask = torch.ones(n_heads, head_size)
                    heads = set(heads) - set(already_pruned_heads)
                    for head in heads:
                        head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                        mask[head] = 0
                    mask = mask.view(-1).contiguous().eq(1)
                    index = torch.arange(len(mask))[mask].long()
                    return heads, index

                pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
            from transformers import AutoConfig, AutoModel

            config = AutoConfig.from_pretrained(
                self.args.dra_embedding_model,
                trust_remote_code=True,
            )
            for attr_name, default_value in {
                "is_decoder": False,
                "add_cross_attention": False,
                "chunk_size_feed_forward": 0,
            }.items():
                if not hasattr(config, attr_name):
                    setattr(config, attr_name, default_value)

            with torch.device("cpu"):
                self.model = AutoModel.from_pretrained(
                    self.args.dra_embedding_model,
                    config=config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                )
            self.model.to(device)
            self.model.eval()
        elif self.args.dra_embedding_backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.args.dra_embedding_model,
                trust_remote_code=True,
                device=device,
            )
        elif self.args.dra_embedding_backend == "hf_mean_pool":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.dra_embedding_model,
                use_fast=True,
            )
            with torch.device("cpu"):
                self.model = AutoModel.from_pretrained(
                    self.args.dra_embedding_model,
                    trust_remote_code=False,
                    low_cpu_mem_usage=False,
                )
            self.model.to(device)
            self.model.eval()
        else:  # pragma: no cover - argparse enforces choices
            raise RuntimeError(f"Unsupported DRA embedding backend: {self.args.dra_embedding_backend}")

    def encode(self, texts: list[str]):
        self._load()
        torch = import_torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            if self.args.dra_embedding_backend == "jina":
                embeddings = self.model.encode(
                    texts,
                    max_length=self.args.dra_embedding_max_length,
                    device=device,
                )
            elif self.args.dra_embedding_backend == "sentence_transformers":
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            elif self.args.dra_embedding_backend == "hf_mean_pool":
                max_length = int(self.args.dra_embedding_max_length)
                for candidate in (
                    getattr(self.tokenizer, "model_max_length", None),
                    getattr(self.model.config, "max_position_embeddings", None),
                ):
                    if isinstance(candidate, int) and 0 < candidate < 1_000_000:
                        max_length = min(max_length, candidate)
                batch = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = self.model(**batch)
                token_embeddings = outputs.last_hidden_state.float()
                attention_mask = batch["attention_mask"].unsqueeze(-1).float()
                embeddings = (token_embeddings * attention_mask).sum(dim=1)
                embeddings = embeddings / attention_mask.sum(dim=1).clamp(min=1e-6)
            else:  # pragma: no cover - argparse enforces choices
                raise RuntimeError(f"Unsupported DRA embedding backend: {self.args.dra_embedding_backend}")
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class DRADiversityReward:
    def __init__(self, args: argparse.Namespace, output_dir: Path) -> None:
        self.args = args
        self.embedder = CompletionEmbedder(args)
        self.debug_path = Path(args.dra_debug_jsonl) if args.dra_debug_jsonl else output_dir / DEFAULT_DRA_DEBUG_JSONL
        self.reward_call_index = 0
        self.__name__ = "dra_diversity_reward"
        self.embedder._load()

    def _base_rewards(
        self,
        completion_texts: list[str],
        gold_answers: list[str],
        problem_texts: list[str],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        rewards: list[float] = []
        debug_rows: list[dict[str, Any]] = []
        for completion_text, gold_answer, problem_text in zip(completion_texts, gold_answers, problem_texts):
            predicted_answer = extract_final_answer_from_output(completion_text)
            correct = answers_match(predicted_answer, gold_answer, problem_text=problem_text)
            parsed_sections = parse_completion_sections(completion_text)
            reward = 1.0 if correct else 0.0
            if self.args.format_bonus > 0 and correct and parsed_sections["exact_format_success"]:
                reward += self.args.format_bonus
            rewards.append(float(reward))
            debug_rows.append(
                {
                    "problem": problem_text,
                    "gold_answer": gold_answer,
                    "raw_completion": completion_text,
                    "parsed_final_answer": predicted_answer,
                    "answer_correctness": float(correct),
                    "base_reward": float(reward),
                    "exact_format_success": bool(parsed_sections["exact_format_success"]),
                    "final_answer_section_present": bool(parsed_sections["final_answer_section_present"]),
                    "suspicious_final_answer": bool(parsed_sections["suspicious_final_answer"]),
                }
            )
        return rewards, debug_rows

    def _diversity_weights(self, completion_texts: list[str]) -> list[float]:
        torch = import_torch()
        group_size = int(self.args.num_generations)
        if group_size <= 1 or not completion_texts:
            return [1.0] * len(completion_texts)

        embeddings = self.embedder.encode(completion_texts)
        weights = torch.ones(len(completion_texts), dtype=torch.float32, device=embeddings.device)
        full_groups = len(completion_texts) // group_size
        if full_groups <= 0:
            return weights.detach().cpu().tolist()

        grouped = embeddings[: full_groups * group_size].view(full_groups, group_size, -1)
        similarity = torch.bmm(grouped, grouped.transpose(1, 2))
        row_sum = similarity.sum(dim=-1)
        grouped_weights = 1.0 / (row_sum + float(self.args.dra_eps))
        weights[: full_groups * group_size] = grouped_weights.reshape(-1)
        return weights.detach().cpu().tolist()

    def __call__(self, completions, answer=None, problem=None, **kwargs):
        self.reward_call_index += 1

        gold_answers = answer if answer is not None else kwargs.get("answer")
        if gold_answers is None:
            gold_answers = kwargs.get("solution")
        problem_texts = problem if problem is not None else kwargs.get("problem")

        completion_texts = [extract_text_from_completion(completion) for completion in completions]
        aligned_golds = align_gold_answers(gold_answers, len(completion_texts))
        aligned_problems = align_values(problem_texts, len(completion_texts))

        base_rewards, debug_rows = self._base_rewards(completion_texts, aligned_golds, aligned_problems)
        diversity_weights = self._diversity_weights(completion_texts)
        adjusted_rewards = [
            float(base_reward) * float(weight)
            for base_reward, weight in zip(base_rewards, diversity_weights)
        ]

        for index, (debug_row, diversity_weight, adjusted_reward) in enumerate(
            zip(debug_rows, diversity_weights, adjusted_rewards)
        ):
            group_index = index // int(self.args.num_generations)
            rollout_id = index % int(self.args.num_generations)
            debug_row.update(
                {
                    "global_reward_call_index": self.reward_call_index,
                    "completion_index_within_call": index,
                    "group_index_within_call": group_index,
                    "group_rollout_id": rollout_id,
                    "dra_diversity_weight": float(diversity_weight),
                    "dra_reward": float(adjusted_reward),
                    "num_generations": int(self.args.num_generations),
                    "dra_embedding_backend": self.args.dra_embedding_backend,
                    "dra_embedding_model": self.args.dra_embedding_model,
                }
            )
            append_jsonl_row(self.debug_path, debug_row)

        return adjusted_rewards


def main() -> None:
    args = parse_args()
    args.bf16 = bool(args.bf16 and default_bf16_enabled())

    if args.parser_self_test:
        success = run_parser_self_test()
        if not success:
            raise SystemExit(1)
        return

    if args.train_size is not None and args.train_size <= 0:
        raise SystemExit("--train_size must be positive.")
    if args.eval_size is not None and args.eval_size < 0:
        raise SystemExit("--eval_size must be non-negative.")
    if args.smoke_test_examples <= 0:
        raise SystemExit("--smoke_test_examples must be positive.")
    if args.smoke_test_generations <= 0:
        raise SystemExit("--smoke_test_generations must be positive.")
    if args.num_generations <= 1:
        raise SystemExit("--num_generations must be greater than 1 for DRA-GRPO.")
    if args.min_solve_rate > args.max_solve_rate:
        raise SystemExit("--min_solve_rate must be <= --max_solve_rate.")

    output_dir = ensure_output_dir(args.output_dir)
    set_seed(args.seed)

    tokenizer = load_tokenizer(args.model_name)
    train_dataset, eval_dataset, smoke_rows, dataset_stats, metadata_context = load_training_data(
        args,
        tokenizer,
        output_dir,
    )

    if args.smoke_test_only or args.run_pretrain_smoke:
        smoke_metrics = run_smoke_test(
            args=args,
            tokenizer=tokenizer,
            smoke_rows=smoke_rows,
            output_dir=output_dir,
        )
        maybe_warn_on_smoke_metrics(smoke_metrics, args.min_smoke_success_rate)
        if args.smoke_test_only:
            print(f"[DONE] smoke test artifacts saved to {output_dir}")
            return

    GRPOConfig, GRPOTrainer = build_grpo_training_components(args)
    peft_config = build_peft_config(args)
    training_args, dropped_grpo_config_kwargs = create_grpo_config(args, GRPOConfig)
    reward_fn = DRADiversityReward(args=args, output_dir=output_dir)
    trainer, dropped_trainer_kwargs = create_grpo_trainer(
        args=args,
        GRPOTrainer=GRPOTrainer,
        training_args=training_args,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    attach_percent_progress_callback(
        trainer,
        progress_interval_percent=args.progress_interval_percent,
        expected_total_steps=args.max_steps,
    )

    training_config = build_training_config_payload(
        args=args,
        dataset_stats=dataset_stats,
        dropped_grpo_config_kwargs=dropped_grpo_config_kwargs,
        dropped_trainer_kwargs=dropped_trainer_kwargs,
    )
    training_config.update(
        {
            "reward_type": "dra_grpo_answer_correctness_diversity_adjusted",
            "dra_base_reward": "answer_only_correctness",
            "dra_adjustment": "reward_i *= 1 / sum_j cosine(embedding_i, embedding_j)",
            "dra_embedding_backend": args.dra_embedding_backend,
            "dra_embedding_model": args.dra_embedding_model,
            "dra_embedding_max_length": args.dra_embedding_max_length,
            "dra_eps": args.dra_eps,
            "dra_debug_jsonl": str(reward_fn.debug_path),
            "use_fixed_metadata": metadata_context["used_fixed_metadata"],
            "use_fixed_metadata_requested": args.use_fixed_metadata,
            "fixed_train_metadata_path": metadata_context["fixed_train_metadata_path"],
            "fixed_eval_metadata_path": metadata_context["fixed_eval_metadata_path"],
            "fixed_train_metadata_sha256": metadata_context["fixed_train_metadata_sha256"],
            "fixed_eval_metadata_sha256": metadata_context["fixed_eval_metadata_sha256"],
        }
    )
    write_json(output_dir / "training_config.json", training_config)
    print("[INFO] final resolved configuration:")
    print(json.dumps(training_config, ensure_ascii=False, indent=2, sort_keys=True))

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] saved to {output_dir}")


if __name__ == "__main__":
    main()
