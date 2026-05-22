#!/usr/bin/env python3
"""Experiment A: Answer-only GRPO / RLVR baseline on Big-Math-RL-Verified."""

from __future__ import annotations

import argparse
import hashlib
import math
import inspect
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATASET_NAME = "SynthLabsAI/Big-Math-RL-Verified"
DEFAULT_OUTPUT_DIR = "outputs/grpo_answer_only_qwen3b_bigmath"
DEFAULT_TRAIN_METADATA_PATH = (
    "outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_train_metadata.jsonl"
)
DEFAULT_EVAL_METADATA_PATH = (
    "outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_eval_metadata.jsonl"
)
NUMERIC_TOLERANCE = 1e-6
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉₋", "0123456789-")
ORDER_INSENSITIVE_HINTS = (
    "find all",
    "all solutions",
    "solutions",
    "roots",
    "possible values",
    "values of",
    "integers",
    "numbers",
    "real values",
    "what are the",
)
SUSPICIOUS_FINAL_ANSWER_LITERALS = {"", "\\[", "\\]", "\\(", "\\)", "[", "]", "(", ")"}

SYSTEM_PROMPT = """You are a careful mathematical reasoning assistant.
Solve the problem independently.
You must follow the required output format exactly.
Do not skip the final answer section."""


@dataclass
class AnswerNormalization:
    raw: str
    normalized: str
    stripped_units: bool
    had_pi: bool
    had_fraction: bool
    had_base_notation: bool
    comma_items: Optional[list[str]]
    numeric_value: Optional[float]


def build_user_prompt(problem: str) -> str:
    return f"""Solve the given problem independently.

First, write a concise strategy that you believe is appropriate for solving the problem.
Then, solve the problem by following that strategy.
Finally, provide the final answer.

Do not force an unusual strategy.
Do not choose a strategy from a predefined list.
Use whatever strategy naturally fits the problem.

You MUST include all three exact section headers:
[Strategy]
[Reasoning]
[Final Answer]

The final answer must be written under the exact header [Final Answer].
Do not omit [Final Answer].
Do not end the response inside [Reasoning].
Keep the reasoning concise enough to always include [Final Answer].
Prefer \\boxed{{...}} when appropriate.
Do not put extra explanation after the final answer.

Return your response in the following format:

[Strategy]
...

[Reasoning]
...

[Final Answer]
...

Problem:
{problem}
"""


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dashed_name = name.replace("_", "-")
    positive_options = [f"--{name}"]
    if dashed_name != name:
        positive_options.append(f"--{dashed_name}")

    negative_options = [f"--no-{name}", f"--no_{name}"]
    if dashed_name != name:
        negative_options.append(f"--no-{dashed_name}")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        *positive_options,
        dest=name,
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        *negative_options,
        dest=name,
        action="store_false",
        help=f"Disable: {help_text}",
    )
    parser.set_defaults(**{name: default})


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    add_bool_arg(
        parser,
        "use_vllm",
        False,
        "Use vLLM for GRPO rollout generation when supported by the installed TRL version.",
    )
    parser.add_argument(
        "--vllm_mode",
        choices=("colocate", "server"),
        default="colocate",
        help="TRL vLLM execution mode. colocate keeps the vLLM engine in the trainer process.",
    )
    parser.add_argument(
        "--vllm_model_impl",
        default="vllm",
        help="TRL vLLM model implementation.",
    )
    add_bool_arg(
        parser,
        "vllm_enable_sleep_mode",
        False,
        "Enable TRL/vLLM sleep mode to release cache memory between rollout phases when available.",
    )
    parser.add_argument("--vllm_server_base_url", default=None)
    parser.add_argument("--vllm_server_host", default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_server_timeout", type=float, default=240.0)
    parser.add_argument("--vllm_group_port", type=int, default=51216)
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="Fraction of GPU memory vLLM may reserve in colocate/server mode.",
    )
    parser.add_argument(
        "--vllm_max_model_length",
        type=int,
        default=0,
        help="vLLM max model length. 0 lets TRL/vLLM infer it from the model/config.",
    )
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer-only GRPO / RLVR baseline training for Big-Math-RL-Verified."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--min_solve_rate", type=float, default=0.2)
    parser.add_argument("--max_solve_rate", type=float, default=0.8)
    parser.add_argument("--train_size", type=int, default=None, help="Number of train rows to use. Defaults to all rows.")
    parser.add_argument("--eval_size", type=int, default=None, help="Number of eval rows to use. Defaults to all rows.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_generations", type=int, default=4)
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

    return parser.parse_args()


def ensure_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("PyTorch is required to run this script.") from exc
    return torch


def default_bf16_enabled() -> bool:
    try:
        torch = import_torch()
    except RuntimeError:
        return False
    if not torch.cuda.is_available():
        return False
    is_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_supported):
        return bool(is_supported())
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        torch = import_torch()
    except RuntimeError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_solve_rate(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric


def difficulty_bucket_from_solve_rate(solve_rate: Optional[float]) -> str:
    if solve_rate is None:
        return "unknown"
    if solve_rate >= 0.7:
        return "easy"
    if solve_rate >= 0.2:
        return "medium"
    if solve_rate >= 0.02:
        return "hard_but_learnable"
    return "too_hard"


def filter_nonempty_problem_answer(row: dict[str, Any]) -> bool:
    problem = str(row.get("problem", "") or "").strip()
    answer = str(row.get("answer", "") or "").strip()
    return bool(problem and answer)


def filter_by_solve_rate(row: dict[str, Any], min_solve_rate: float, max_solve_rate: float) -> bool:
    solve_rate = parse_solve_rate(row.get("llama8b_solve_rate"))
    if solve_rate is None:
        return False
    return min_solve_rate <= solve_rate <= max_solve_rate


def metadata_row_from_example(row: dict[str, Any]) -> dict[str, Any]:
    solve_rate = parse_solve_rate(row.get("llama8b_solve_rate"))
    return {
        "problem": str(row.get("problem", "")),
        "answer": str(row.get("answer", "")),
        "source": str(row.get("source", "") or ""),
        "domain": str(row.get("domain", "") or ""),
        "llama8b_solve_rate": solve_rate,
        "difficulty_bucket": difficulty_bucket_from_solve_rate(solve_rate),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            rows.append(dict(row))
    return rows


def normalize_metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    solve_rate = parse_solve_rate(row.get("llama8b_solve_rate"))
    difficulty_bucket = str(row.get("difficulty_bucket", "") or "").strip()
    if not difficulty_bucket:
        difficulty_bucket = difficulty_bucket_from_solve_rate(solve_rate)
    return {
        "problem": str(row.get("problem", "") or ""),
        "answer": str(row.get("answer", "") or ""),
        "source": str(row.get("source", "") or ""),
        "domain": str(row.get("domain", "") or ""),
        "llama8b_solve_rate": solve_rate,
        "difficulty_bucket": difficulty_bucket,
    }


def build_training_row(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    training_row = dict(row)
    training_row["prompt"] = render_prompt(row["problem"], tokenizer)
    return training_row


def render_prompt_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(problem)},
    ]


def render_prompt(problem: str, tokenizer: Any) -> str:
    messages = render_prompt_messages(problem)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"User:\n{build_user_prompt(problem)}\n\n"
        "Assistant:\n"
    )


def print_counter(label: str, counts: Counter[str]) -> None:
    printable = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    print(f"[INFO] {label}: {printable}")


def summarize_selected_train(train_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    solve_rates = [row["llama8b_solve_rate"] for row in train_metadata if row["llama8b_solve_rate"] is not None]
    solve_rate_summary = {
        "min": min(solve_rates) if solve_rates else None,
        "max": max(solve_rates) if solve_rates else None,
        "mean": statistics.fmean(solve_rates) if solve_rates else None,
    }
    source_distribution = Counter(row["source"] or "unknown" for row in train_metadata)
    difficulty_distribution = Counter(row["difficulty_bucket"] for row in train_metadata)
    return {
        "solve_rate_summary": solve_rate_summary,
        "source_distribution": dict(source_distribution),
        "difficulty_distribution": dict(difficulty_distribution),
    }


def load_dataset_rows(args: argparse.Namespace, tokenizer: Any, output_dir: Path):
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("datasets is required. Install it with `pip install datasets`.") from exc

    dataset = load_dataset(args.dataset_name, split="train")
    total_before = len(dataset)
    print(f"[INFO] dataset size before filtering: {total_before}")

    dataset = dataset.filter(filter_nonempty_problem_answer)
    after_problem_answer = len(dataset)
    print(f"[INFO] dataset size after problem/answer filtering: {after_problem_answer}")

    dataset = dataset.filter(
        lambda row: filter_by_solve_rate(row, args.min_solve_rate, args.max_solve_rate)
    )
    after_solve_rate = len(dataset)
    print(f"[INFO] dataset size after solve-rate filtering: {after_solve_rate}")

    if after_solve_rate == 0:
        raise RuntimeError("No Big-Math rows remain after filtering.")

    dataset = dataset.shuffle(seed=args.seed)
    train_size = after_solve_rate if args.train_size is None else args.train_size
    eval_size = 0 if args.eval_size is None else args.eval_size
    if train_size < 0 or eval_size < 0:
        raise SystemExit("--train_size and --eval_size must be non-negative.")

    total_needed = min(after_solve_rate, train_size + eval_size)
    if total_needed < train_size + eval_size:
        print(
            f"[WARN] Filtered dataset has only {after_solve_rate} rows, "
            f"smaller than requested train+eval size {train_size + eval_size}."
        )
    selected = dataset.select(range(total_needed))
    selected_rows = [metadata_row_from_example(selected[index]) for index in range(len(selected))]

    train_rows = selected_rows[: min(train_size, len(selected_rows))]
    eval_start = len(train_rows)
    eval_rows = selected_rows[eval_start : eval_start + eval_size]
    args.train_size = len(train_rows)
    args.eval_size = len(eval_rows)

    print(f"[INFO] train size: {len(train_rows)}")
    print(f"[INFO] eval size: {len(eval_rows)}")

    selected_train_path = output_dir / "selected_train_metadata.jsonl"
    selected_eval_path = output_dir / "selected_eval_metadata.jsonl"
    write_jsonl(selected_train_path, train_rows)
    write_jsonl(selected_eval_path, eval_rows)

    train_summary = summarize_selected_train(train_rows)
    eval_summary = summarize_selected_train(eval_rows) if eval_rows else {
        "solve_rate_summary": {"min": None, "max": None, "mean": None},
        "source_distribution": {},
        "difficulty_distribution": {},
    }
    solve_rate_summary = train_summary["solve_rate_summary"]
    print(
        "[INFO] selected train solve-rate min/max/mean: "
        f"{solve_rate_summary['min']} / {solve_rate_summary['max']} / {solve_rate_summary['mean']}"
    )
    print_counter("selected train source distribution", Counter(train_summary["source_distribution"]))
    print_counter(
        "selected train difficulty bucket distribution",
        Counter(train_summary["difficulty_distribution"]),
    )

    train_dataset = Dataset.from_list([build_training_row(row, tokenizer) for row in train_rows])
    eval_dataset = Dataset.from_list([build_training_row(row, tokenizer) for row in eval_rows]) if eval_rows else None

    smoke_count = min(len(dataset), max(args.smoke_test_examples, 1))
    smoke_pool = [metadata_row_from_example(dataset[index]) for index in range(smoke_count)]
    dataset_stats = {
        "dataset_mode": "fresh_sampled",
        "dataset_size_before_filtering": total_before,
        "dataset_size_after_problem_answer_filtering": after_problem_answer,
        "dataset_size_after_solve_rate_filtering": after_solve_rate,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "selected_train_solve_rate_summary": solve_rate_summary,
        "selected_train_source_distribution": train_summary["source_distribution"],
        "selected_train_difficulty_distribution": train_summary["difficulty_distribution"],
        "selected_eval_solve_rate_summary": eval_summary["solve_rate_summary"],
        "selected_eval_source_distribution": eval_summary["source_distribution"],
        "selected_eval_difficulty_distribution": eval_summary["difficulty_distribution"],
    }
    return train_dataset, eval_dataset, smoke_pool, dataset_stats


def load_fixed_metadata_datasets(
    args: argparse.Namespace,
    tokenizer: Any,
    output_dir: Path,
):
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("datasets is required. Install it with `pip install datasets`.") from exc

    train_path = Path(args.train_metadata_path)
    eval_path = Path(args.eval_metadata_path)

    full_train_rows = [normalize_metadata_row(row) for row in load_jsonl_rows(train_path)]
    full_eval_rows = [normalize_metadata_row(row) for row in load_jsonl_rows(eval_path)]
    if not full_train_rows:
        raise RuntimeError(f"Fixed train metadata is empty: {train_path}")
    train_size = len(full_train_rows) if args.train_size is None else args.train_size
    eval_size = len(full_eval_rows) if args.eval_size is None else args.eval_size
    if train_size < 0 or eval_size < 0:
        raise SystemExit("--train_size and --eval_size must be non-negative.")
    if eval_size > 0 and not full_eval_rows:
        raise RuntimeError(f"Fixed eval metadata is empty: {eval_path}")

    if train_size > len(full_train_rows):
        print(
            f"[WARN] Requested train_size={train_size} exceeds fixed metadata size={len(full_train_rows)}. "
            "Clipping to available rows."
        )
    if eval_size > len(full_eval_rows):
        print(
            f"[WARN] Requested eval_size={eval_size} exceeds fixed metadata size={len(full_eval_rows)}. "
            "Clipping to available rows."
        )

    train_rows = full_train_rows[: min(train_size, len(full_train_rows))]
    eval_rows = full_eval_rows[: min(eval_size, len(full_eval_rows))]
    args.train_size = len(train_rows)
    args.eval_size = len(eval_rows)

    write_jsonl(output_dir / "selected_train_metadata.jsonl", train_rows)
    write_jsonl(output_dir / "selected_eval_metadata.jsonl", eval_rows)

    train_summary = summarize_selected_train(train_rows)
    eval_summary = summarize_selected_train(eval_rows) if eval_rows else {
        "solve_rate_summary": {"min": None, "max": None, "mean": None},
        "source_distribution": {},
        "difficulty_distribution": {},
    }

    print(f"[INFO] selected fixed train metadata count: {len(train_rows)}")
    print(f"[INFO] selected fixed eval metadata count: {len(eval_rows)}")
    print(
        "[INFO] selected fixed train solve-rate min/max/mean: "
        f"{train_summary['solve_rate_summary']['min']} / "
        f"{train_summary['solve_rate_summary']['max']} / "
        f"{train_summary['solve_rate_summary']['mean']}"
    )
    print(
        "[INFO] selected fixed eval solve-rate min/max/mean: "
        f"{eval_summary['solve_rate_summary']['min']} / "
        f"{eval_summary['solve_rate_summary']['max']} / "
        f"{eval_summary['solve_rate_summary']['mean']}"
    )
    print_counter("selected fixed train source distribution", Counter(train_summary["source_distribution"]))
    print_counter(
        "selected fixed train difficulty bucket distribution",
        Counter(train_summary["difficulty_distribution"]),
    )
    print_counter("selected fixed eval source distribution", Counter(eval_summary["source_distribution"]))
    print_counter(
        "selected fixed eval difficulty bucket distribution",
        Counter(eval_summary["difficulty_distribution"]),
    )

    train_dataset = Dataset.from_list([build_training_row(row, tokenizer) for row in train_rows])
    eval_dataset = Dataset.from_list([build_training_row(row, tokenizer) for row in eval_rows]) if eval_rows else None
    smoke_rows = train_rows[: min(len(train_rows), max(args.smoke_test_examples, 1))]
    dataset_stats = {
        "dataset_mode": "fixed_metadata",
        "dataset_size_before_filtering": len(full_train_rows) + len(full_eval_rows),
        "dataset_size_after_problem_answer_filtering": len(full_train_rows) + len(full_eval_rows),
        "dataset_size_after_solve_rate_filtering": len(full_train_rows) + len(full_eval_rows),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "selected_train_solve_rate_summary": train_summary["solve_rate_summary"],
        "selected_train_source_distribution": train_summary["source_distribution"],
        "selected_train_difficulty_distribution": train_summary["difficulty_distribution"],
        "selected_eval_solve_rate_summary": eval_summary["solve_rate_summary"],
        "selected_eval_source_distribution": eval_summary["source_distribution"],
        "selected_eval_difficulty_distribution": eval_summary["difficulty_distribution"],
        "fixed_train_metadata_full_rows": len(full_train_rows),
        "fixed_eval_metadata_full_rows": len(full_eval_rows),
    }
    metadata_context = {
        "used_fixed_metadata": True,
        "fixed_train_metadata_path": str(train_path),
        "fixed_eval_metadata_path": str(eval_path),
        "fixed_train_metadata_sha256": sha256_file(train_path),
        "fixed_eval_metadata_sha256": sha256_file(eval_path),
    }
    return train_dataset, eval_dataset, smoke_rows, dataset_stats, metadata_context


def load_training_data(
    args: argparse.Namespace,
    tokenizer: Any,
    output_dir: Path,
):
    train_path = Path(args.train_metadata_path)
    eval_path = Path(args.eval_metadata_path)
    if args.use_fixed_metadata and train_path.exists() and eval_path.exists():
        return load_fixed_metadata_datasets(args, tokenizer, output_dir)

    if args.use_fixed_metadata:
        print(
            "[WARN] Fixed metadata not found. Falling back to fresh Big-Math sampling. "
            "This is not a perfectly fair comparison."
        )
    train_dataset, eval_dataset, smoke_rows, dataset_stats = load_dataset_rows(args, tokenizer, output_dir)
    metadata_context = {
        "used_fixed_metadata": False,
        "fixed_train_metadata_path": str(train_path),
        "fixed_eval_metadata_path": str(eval_path),
        "fixed_train_metadata_sha256": sha256_file(train_path) if train_path.exists() else None,
        "fixed_eval_metadata_sha256": sha256_file(eval_path) if eval_path.exists() else None,
    }
    return train_dataset, eval_dataset, smoke_rows, dataset_stats, metadata_context


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "transformers is required. Install it with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_smoke(model_name: str, bf16: bool):
    torch = import_torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "transformers is required. Install it with `pip install transformers`."
        ) from exc

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model


def extract_section(text: str, heading: str, following_headings: list[str]) -> str:
    if following_headings:
        next_pattern = "|".join(re.escape(item) for item in following_headings)
        pattern = re.compile(
            rf"\[{re.escape(heading)}\]\s*(.*?)(?=(?:\n\s*\[(?:{next_pattern})\])|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        pattern = re.compile(
            rf"\[{re.escape(heading)}\]\s*(.*)$",
            flags=re.IGNORECASE | re.DOTALL,
        )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def unwrap_boxed(text: str) -> str:
    cleaned = text.strip()
    prefix = "\\boxed{"
    while cleaned.startswith(prefix) and cleaned.endswith("}"):
        cleaned = cleaned[len(prefix) : -1].strip()
    return cleaned


def strip_outer_text_wrapper(text: str) -> str:
    match = re.fullmatch(r"\\text\{(.+)\}", text)
    return match.group(1) if match else text


def cleanup_extracted_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = re.sub(r"^(?:the answer is|answer is|is)\s+", "", answer, flags=re.IGNORECASE).strip()
    return answer.strip(" \t\r\n$`")


def strip_trailing_punctuation(text: str) -> str:
    return text.rstrip(" \t\r\n.,;:!?")


def strip_outer_braces(text: str) -> str:
    cleaned = text.strip()
    while cleaned.startswith("{") and cleaned.endswith("}"):
        depth = 0
        balanced = True
        for index, char in enumerate(cleaned):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0 and index != len(cleaned) - 1:
                balanced = False
                break
        if not balanced:
            break
        cleaned = cleaned[1:-1].strip()
    return cleaned


def replace_latex_fractions(text: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = pattern.sub(lambda match: f"({match.group(1)})/({match.group(2)})", cleaned)
    return cleaned


def normalize_base_notation_text(text: str) -> tuple[str, bool]:
    original = text
    cleaned = re.sub(
        r"([A-Za-z0-9+-]+)([₀₁₂₃₄₅₆₇₈₉]+)\b",
        lambda match: f"{match.group(1)}_{match.group(2).translate(SUBSCRIPT_TRANSLATION)}",
        text,
    )
    cleaned = cleaned.translate(SUBSCRIPT_TRANSLATION)
    cleaned = re.sub(r"\b([A-Za-z0-9+-]+)\s*_\s*(\d+)\b", r"\1_\2", cleaned)
    cleaned = re.sub(
        r"\b([A-Za-z0-9+-]+)\s+(?:in\s+)?base\s+(\d+)\b",
        r"\1_\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned, cleaned != original


def normalize_pi_text(text: str) -> tuple[str, bool]:
    original = text
    cleaned = text.replace("\\pi", "pi").replace("π", "pi")
    cleaned = re.sub(r"(?<=\d)\s*pi\b", "*pi", cleaned)
    cleaned = re.sub(r"(?<=\))\s*pi\b", "*pi", cleaned)
    cleaned = re.sub(r"\bpi(?=\()", "pi*", cleaned)
    cleaned = re.sub(r"(?<=\d)pi\b", "*pi", cleaned)
    cleaned = re.sub(r"\bpi(?=\d)", "pi*", cleaned)
    return cleaned, cleaned != original


def expression_looks_numeric_like(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    if re.fullmatch(r"[-+*/().0-9pi_,\s]+", candidate):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9+-]+_\d+", candidate))


def strip_trailing_units(text: str) -> tuple[str, bool]:
    cleaned = text.strip()
    match = re.fullmatch(r"(.+?)\s+([A-Za-z%]+(?:\s+[A-Za-z%]+)*)", cleaned)
    if match is None:
        return cleaned, False
    expr, unit_phrase = match.groups()
    if not expression_looks_numeric_like(expr):
        return cleaned, False
    if "base" in unit_phrase.lower():
        return cleaned, False
    return expr.strip(), True


def safe_eval_numeric_expression(text: str) -> Optional[float]:
    candidate = text.strip()
    if not candidate or "_" in candidate or "," in candidate:
        return None
    candidate = candidate.replace("^", "**")
    if not re.fullmatch(r"[0-9pi+\-*/(). ]+", candidate):
        return None
    try:
        value = eval(candidate, {"__builtins__": {}}, {"pi": math.pi})
    except Exception:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def split_comma_collection(text: str) -> Optional[list[str]]:
    if "," not in text:
        return None
    if any(char in text for char in "()[]"):
        return None
    items = [item.strip() for item in text.split(",")]
    if len(items) < 2 or any(not item for item in items):
        return None
    return items


def normalize_answer_details(value: Optional[str], allow_collection: bool = True) -> AnswerNormalization:
    raw = "" if value is None else str(value).strip()
    cleaned = cleanup_extracted_answer(raw)
    cleaned = cleaned.strip().strip("$").strip("`")
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace("\\!", "")
    cleaned = cleaned.replace("\\;", "")
    cleaned = cleaned.replace("\\:", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = cleaned.replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\dfrac", "\\frac")
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    cleaned = re.sub(r"\\text\{([^{}]+)\}", r" \1 ", cleaned)
    cleaned = re.sub(r"\\(?:mathrm|operatorname)\{([^{}]+)\}", r"\1", cleaned)
    cleaned = cleaned.replace("\\[", " ").replace("\\]", " ")
    cleaned = cleaned.replace("\\(", " ").replace("\\)", " ")
    cleaned = unwrap_boxed(cleaned)
    cleaned = strip_outer_text_wrapper(cleaned)
    cleaned = strip_outer_braces(cleaned)
    cleaned = strip_trailing_punctuation(cleaned)

    had_fraction = "\\frac" in cleaned or bool(re.fullmatch(r"[-+]?\d+\s*/\s*\d+", cleaned))
    cleaned = replace_latex_fractions(cleaned)

    cleaned, had_base_notation = normalize_base_notation_text(cleaned)
    cleaned, had_pi = normalize_pi_text(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned, stripped_units = strip_trailing_units(cleaned)
    cleaned = strip_trailing_punctuation(cleaned)
    cleaned = strip_outer_braces(cleaned)
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)

    comma_items: Optional[list[str]] = None
    if allow_collection:
        raw_items = split_comma_collection(cleaned)
        if raw_items is not None:
            comma_items = [
                normalize_answer_details(item, allow_collection=False).normalized for item in raw_items
            ]

    numeric_value = safe_eval_numeric_expression(cleaned)
    return AnswerNormalization(
        raw=raw,
        normalized=cleaned,
        stripped_units=stripped_units,
        had_pi=had_pi,
        had_fraction=had_fraction,
        had_base_notation=had_base_notation,
        comma_items=comma_items,
        numeric_value=numeric_value,
    )


def collection_key(items: list[str]) -> tuple[str, ...]:
    keys: list[str] = []
    for item in items:
        numeric = safe_eval_numeric_expression(item)
        if numeric is not None:
            keys.append(f"num:{numeric:.12g}")
        else:
            keys.append(f"str:{item}")
    return tuple(sorted(keys))


def should_treat_as_unordered_collection(
    problem_text: str,
    predicted: AnswerNormalization,
    gold: AnswerNormalization,
) -> bool:
    if predicted.comma_items is None or gold.comma_items is None:
        return False
    if any(char in predicted.raw + gold.raw for char in "()[]"):
        return False
    if any(hint in problem_text.lower() for hint in ORDER_INSENSITIVE_HINTS):
        return True
    return "{" in predicted.raw or "{" in gold.raw or "}" in predicted.raw or "}" in gold.raw


def verify_answer(predicted_answer: Optional[str], gold_answer: Optional[str], problem_text: str = "") -> dict[str, Any]:
    predicted = normalize_answer_details(predicted_answer)
    gold = normalize_answer_details(gold_answer)

    verification_method = "no_match"
    correct = False

    if predicted.normalized and predicted.normalized == gold.normalized:
        correct = True
        if predicted.stripped_units or gold.stripped_units:
            verification_method = "unit_stripped_match"
        elif predicted.had_pi or gold.had_pi:
            verification_method = "latex_pi_match"
        else:
            verification_method = "exact_string"
    elif should_treat_as_unordered_collection(problem_text, predicted, gold):
        if predicted.comma_items is not None and gold.comma_items is not None:
            if collection_key(predicted.comma_items) == collection_key(gold.comma_items):
                correct = True
                verification_method = "comma_set_match"
    elif predicted.numeric_value is not None and gold.numeric_value is not None:
        if abs(predicted.numeric_value - gold.numeric_value) <= NUMERIC_TOLERANCE:
            correct = True
            if predicted.had_pi or gold.had_pi:
                verification_method = "latex_pi_match"
            elif predicted.had_fraction or gold.had_fraction:
                verification_method = "fraction_decimal_match"
            else:
                verification_method = "numeric_match"

    return {
        "normalized_predicted_answer": predicted.normalized,
        "normalized_gold_answer": gold.normalized,
        "correct": correct,
        "verification_method": verification_method,
    }


def extract_last_boxed_content(text: str) -> str:
    if not text:
        return ""
    search_term = "\\boxed{"
    position = 0
    last_content = ""
    while True:
        start = text.find(search_term, position)
        if start == -1:
            break
        brace_start = start + len(search_term) - 1
        depth = 0
        end_index: Optional[int] = None
        for index in range(brace_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_index = index
                    break
        if end_index is not None:
            last_content = text[brace_start + 1 : end_index].strip()
        position = start + len(search_term)
    return last_content


def strip_math_delimiters(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.replace("\\[", "\n").replace("\\]", "\n")
    cleaned = cleaned.replace("\\(", "\n").replace("\\)", "\n")
    cleaned = cleaned.replace("$", "")
    return cleaned


def is_suspicious_final_answer(text: str) -> bool:
    cleaned = cleanup_extracted_answer(text)
    cleaned = strip_trailing_punctuation(cleaned)
    if cleaned in SUSPICIOUS_FINAL_ANSWER_LITERALS:
        return True
    stripped = re.sub(r"[\s\.,;:!?]+", "", cleaned)
    if stripped in SUSPICIOUS_FINAL_ANSWER_LITERALS:
        return True
    if not stripped:
        return True
    if re.fullmatch(r"[\\\[\]\(\)\{\}]+", stripped):
        return True
    return False


def first_meaningful_answer_line(text: str) -> str:
    cleaned = strip_math_delimiters(text)
    lines = [line.strip() for line in cleaned.splitlines()]
    for line in lines:
        candidate = cleanup_extracted_answer(line)
        if candidate and not is_suspicious_final_answer(candidate):
            return candidate
    return ""


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    final_answer_section = extract_section(text, "Final Answer", [])
    if final_answer_section:
        boxed_content = extract_last_boxed_content(final_answer_section)
        if boxed_content and not is_suspicious_final_answer(boxed_content):
            return cleanup_extracted_answer(boxed_content)

        meaningful_line = first_meaningful_answer_line(final_answer_section)
        if meaningful_line and not is_suspicious_final_answer(meaningful_line):
            return cleanup_extracted_answer(meaningful_line)

    boxed_content = extract_last_boxed_content(text)
    if boxed_content and not is_suspicious_final_answer(boxed_content):
        return cleanup_extracted_answer(boxed_content)

    patterns = [
        r"\[Final Answer\]\s*(.+)",
        r"FINAL_ANSWER:\s*(.+)",
        r"Final answer\s*[:：]\s*(.+)",
        r"final answer\s*[:：]\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            candidate = first_meaningful_answer_line(matches[-1])
            if candidate and not is_suspicious_final_answer(candidate):
                return cleanup_extracted_answer(candidate)

    tail_candidate = first_meaningful_answer_line("\n".join(text.splitlines()[-8:]))
    if tail_candidate and not is_suspicious_final_answer(tail_candidate):
        return cleanup_extracted_answer(tail_candidate)
    return ""


def parse_completion_sections(text: str) -> dict[str, Any]:
    strategy = extract_section(text, "Strategy", ["Reasoning", "Final Answer"])
    reasoning = extract_section(text, "Reasoning", ["Final Answer"])
    final_answer_section = extract_section(text, "Final Answer", [])

    parsed_final_answer = extract_final_answer(text)
    suspicious_final_answer = is_suspicious_final_answer(parsed_final_answer)
    if suspicious_final_answer:
        parsed_final_answer = ""

    return {
        "strategy_section_present": bool(strategy.strip()),
        "reasoning_section_present": bool(reasoning.strip()),
        "final_answer_section_present": bool(final_answer_section.strip()),
        "parsed_final_answer": parsed_final_answer,
        "suspicious_final_answer": suspicious_final_answer,
        "exact_format_success": bool(
            strategy.strip() and reasoning.strip() and final_answer_section.strip() and parsed_final_answer
        ),
    }


def extract_final_answer_from_output(text: str) -> str:
    return parse_completion_sections(text)["parsed_final_answer"]


def answers_match(predicted: str, gold: str, problem_text: str = "") -> bool:
    return bool(verify_answer(predicted, gold, problem_text=problem_text)["correct"])


def extract_text_from_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            content = completion["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(extract_text_from_completion(item) for item in content)
        for key in ("text", "generated_text", "completion"):
            if key in completion:
                return extract_text_from_completion(completion[key])
        return str(completion)
    if isinstance(completion, list):
        assistant_contents = [
            extract_text_from_completion(item)
            for item in completion
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if assistant_contents:
            return assistant_contents[-1]
        return "".join(extract_text_from_completion(item) for item in completion)
    return str(completion)


def align_gold_answers(answer: Any, target_len: int) -> list[str]:
    if isinstance(answer, list):
        golds = [str(item) for item in answer]
    elif isinstance(answer, tuple):
        golds = [str(item) for item in answer]
    elif answer is None:
        golds = []
    else:
        golds = [str(answer)]

    if not golds:
        return [""] * target_len
    if len(golds) == target_len:
        return golds
    if target_len % len(golds) == 0:
        repeat_factor = target_len // len(golds)
        return [gold for gold in golds for _ in range(repeat_factor)]
    return [golds[index % len(golds)] for index in range(target_len)]


def align_values(values: Any, target_len: int) -> list[str]:
    if isinstance(values, list):
        items = [str(item) for item in values]
    elif isinstance(values, tuple):
        items = [str(item) for item in values]
    elif values is None:
        items = []
    else:
        items = [str(values)]

    if not items:
        return [""] * target_len
    if len(items) == target_len:
        return items
    if target_len % len(items) == 0:
        repeat_factor = target_len // len(items)
        return [item for item in items for _ in range(repeat_factor)]
    return [items[index % len(items)] for index in range(target_len)]


def build_answer_only_reward_fn(format_bonus: float):
    def reward_fn(completions, answer=None, **kwargs):
        gold_answers = answer if answer is not None else kwargs.get("answer")
        if gold_answers is None:
            gold_answers = kwargs.get("solution")
        problem_texts = kwargs.get("problem")

        completion_texts = [extract_text_from_completion(completion) for completion in completions]
        aligned_golds = align_gold_answers(gold_answers, len(completion_texts))
        aligned_problems = align_values(problem_texts, len(completion_texts))
        rewards: list[float] = []

        for completion_text, gold_answer, problem_text in zip(completion_texts, aligned_golds, aligned_problems):
            predicted_answer = extract_final_answer_from_output(completion_text)
            correct = answers_match(predicted_answer, gold_answer, problem_text=problem_text)
            reward = 1.0 if correct else 0.0
            if format_bonus > 0 and correct and parse_completion_sections(completion_text)["exact_format_success"]:
                reward += format_bonus
            rewards.append(float(reward))
        return rewards

    return reward_fn


def generate_smoke_outputs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    torch = import_torch()
    inputs = tokenizer([prompt], return_tensors="pt")
    try:
        model_device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "num_return_sequences": num_generations,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = max(0.01, temperature)
        generate_kwargs["top_p"] = top_p
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    decoded: list[str] = []
    for output in outputs:
        decoded.append(tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip())
    return decoded


def run_smoke_test(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    smoke_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    model = load_model_for_smoke(args.model_name, args.bf16)
    smoke_examples = smoke_rows[: min(len(smoke_rows), args.smoke_test_examples)]
    if not smoke_examples:
        raise RuntimeError("Smoke test requested but no filtered Big-Math examples are available.")

    raw_output_rows: list[dict[str, Any]] = []
    metrics_counter = Counter()
    parsed_final_answers: list[str] = []
    failed_parse_examples: list[dict[str, Any]] = []

    for problem_index, row in enumerate(smoke_examples):
        prompt = render_prompt(row["problem"], tokenizer)
        completions = generate_smoke_outputs(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_generations=args.smoke_test_generations,
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for generation_index, completion_text in enumerate(completions):
            parsed = parse_completion_sections(completion_text)
            metrics_counter["total_generations"] += 1
            metrics_counter["strategy_section_present"] += int(parsed["strategy_section_present"])
            metrics_counter["reasoning_section_present"] += int(parsed["reasoning_section_present"])
            metrics_counter["final_answer_section_present"] += int(parsed["final_answer_section_present"])
            metrics_counter["nonempty_final_answer"] += int(bool(parsed["parsed_final_answer"]))
            metrics_counter["exact_format_success"] += int(parsed["exact_format_success"])
            metrics_counter["suspicious_final_answer"] += int(parsed["suspicious_final_answer"])
            if parsed["parsed_final_answer"]:
                parsed_final_answers.append(parsed["parsed_final_answer"])
            if parsed["suspicious_final_answer"] or not parsed["parsed_final_answer"]:
                if len(failed_parse_examples) < 20:
                    failed_parse_examples.append(
                        {
                            "problem": row["problem"],
                            "raw_output": completion_text,
                            "parsed_final_answer": parsed["parsed_final_answer"],
                            "strategy_section_present": parsed["strategy_section_present"],
                            "reasoning_section_present": parsed["reasoning_section_present"],
                            "final_answer_section_present": parsed["final_answer_section_present"],
                        }
                    )

            raw_output_rows.append(
                {
                    "problem_index": problem_index,
                    "generation_index": generation_index,
                    "problem": row["problem"],
                    "answer": row["answer"],
                    "source": row["source"],
                    "domain": row["domain"],
                    "llama8b_solve_rate": row["llama8b_solve_rate"],
                    "difficulty_bucket": row["difficulty_bucket"],
                    "raw_output": completion_text,
                    "strategy_section_present": parsed["strategy_section_present"],
                    "reasoning_section_present": parsed["reasoning_section_present"],
                    "final_answer_section_present": parsed["final_answer_section_present"],
                    "parsed_final_answer": parsed["parsed_final_answer"],
                    "suspicious_final_answer": parsed["suspicious_final_answer"],
                    "exact_format_success": parsed["exact_format_success"],
                }
            )

    total_generations = metrics_counter["total_generations"]
    smoke_metrics = {
        "num_problems": len(smoke_examples),
        "num_generations_per_problem": args.smoke_test_generations,
        "total_generations": total_generations,
        "strategy_section_present_rate": metrics_counter["strategy_section_present"] / total_generations,
        "reasoning_section_present_rate": metrics_counter["reasoning_section_present"] / total_generations,
        "final_answer_section_present_rate": metrics_counter["final_answer_section_present"] / total_generations,
        "nonempty_final_answer_rate": metrics_counter["nonempty_final_answer"] / total_generations,
        "exact_format_success_rate": metrics_counter["exact_format_success"] / total_generations,
        "suspicious_final_answer_count": metrics_counter["suspicious_final_answer"],
        "suspicious_final_answer_rate": metrics_counter["suspicious_final_answer"] / total_generations,
        "parsed_final_answer_examples": parsed_final_answers[:20],
        "sample_parsed_final_answers": parsed_final_answers[:20],
        "failed_parse_examples": failed_parse_examples,
    }
    write_jsonl(output_dir / "smoke_raw_outputs.jsonl", raw_output_rows)
    write_json(output_dir / "smoke_metrics.json", smoke_metrics)

    del model
    try:
        torch = import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError:
        pass

    print(f"[INFO] smoke metrics: {json.dumps(smoke_metrics, ensure_ascii=False, indent=2)}")
    return smoke_metrics


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    signature = inspect.signature(callable_obj)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_var_kwargs:
        return dict(kwargs), []
    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    dropped = sorted(key for key in kwargs if key not in filtered)
    return filtered, dropped


def build_peft_config(args: argparse.Namespace):
    if not args.use_lora:
        return None
    try:
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("peft is required for LoRA. Install it with `pip install peft`.") from exc

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_grpo_training_components(args: argparse.Namespace):
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("trl is required. Install it with `pip install trl`.") from exc
    return GRPOConfig, GRPOTrainer


def create_grpo_config(args: argparse.Namespace, GRPOConfig: Any) -> tuple[Any, list[str]]:
    config_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "log_completions": True,
        "report_to": "none",
        "seed": args.seed,
    }
    if getattr(args, "use_vllm", False):
        config_kwargs.update(
            {
                "use_vllm": True,
                "vllm_mode": args.vllm_mode,
                "vllm_model_impl": args.vllm_model_impl,
                "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
                "vllm_server_host": args.vllm_server_host,
                "vllm_server_port": args.vllm_server_port,
                "vllm_server_timeout": args.vllm_server_timeout,
                "vllm_group_port": args.vllm_group_port,
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            }
        )
        if args.vllm_server_base_url:
            config_kwargs["vllm_server_base_url"] = args.vllm_server_base_url
        if args.vllm_max_model_length and args.vllm_max_model_length > 0:
            config_kwargs["vllm_max_model_length"] = args.vllm_max_model_length
    filtered_kwargs, dropped = filter_supported_kwargs(GRPOConfig.__init__, config_kwargs)
    return GRPOConfig(**filtered_kwargs), dropped


def create_grpo_trainer(
    *,
    args: argparse.Namespace,
    GRPOTrainer: Any,
    training_args: Any,
    reward_fn: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    peft_config: Any,
) -> tuple[Any, list[str]]:
    trainer_kwargs: dict[str, Any] = {
        "model": args.model_name,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }

    trainer_signature = inspect.signature(GRPOTrainer.__init__)
    trainer_parameters = trainer_signature.parameters
    if "reward_funcs" in trainer_parameters:
        trainer_kwargs["reward_funcs"] = reward_fn
    elif "reward_func" in trainer_parameters:
        trainer_kwargs["reward_func"] = reward_fn
    else:  # pragma: no cover - depends on env
        raise RuntimeError("Installed GRPOTrainer does not accept reward_func(s).")

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    filtered_kwargs, dropped = filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs)
    return GRPOTrainer(**filtered_kwargs), dropped


def build_training_config_payload(
    args: argparse.Namespace,
    dataset_stats: dict[str, Any],
    dropped_grpo_config_kwargs: list[str],
    dropped_trainer_kwargs: list[str],
) -> dict[str, Any]:
    try:
        torch = import_torch()
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
    except RuntimeError:
        cuda_available = False
        device_count = 0

    return {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "output_dir": args.output_dir,
        "min_solve_rate": args.min_solve_rate,
        "max_solve_rate": args.max_solve_rate,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_vllm": getattr(args, "use_vllm", False),
        "vllm_mode": getattr(args, "vllm_mode", None),
        "vllm_model_impl": getattr(args, "vllm_model_impl", None),
        "vllm_enable_sleep_mode": getattr(args, "vllm_enable_sleep_mode", None),
        "vllm_server_base_url": getattr(args, "vllm_server_base_url", None),
        "vllm_server_host": getattr(args, "vllm_server_host", None),
        "vllm_server_port": getattr(args, "vllm_server_port", None),
        "vllm_server_timeout": getattr(args, "vllm_server_timeout", None),
        "vllm_group_port": getattr(args, "vllm_group_port", None),
        "vllm_gpu_memory_utilization": getattr(args, "vllm_gpu_memory_utilization", None),
        "vllm_max_model_length": getattr(args, "vllm_max_model_length", None),
        "vllm_tensor_parallel_size": getattr(args, "vllm_tensor_parallel_size", None),
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "progress_interval_percent": args.progress_interval_percent,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "smoke_test_only": args.smoke_test_only,
        "smoke_test_examples": args.smoke_test_examples,
        "smoke_test_generations": args.smoke_test_generations,
        "run_pretrain_smoke": args.run_pretrain_smoke,
        "min_smoke_success_rate": args.min_smoke_success_rate,
        "format_bonus": args.format_bonus,
        "report_to": "none",
        "cuda_available": cuda_available,
        "device_count": device_count,
        "dataset_stats": dataset_stats,
        "dropped_grpo_config_kwargs": dropped_grpo_config_kwargs,
        "dropped_trainer_kwargs": dropped_trainer_kwargs,
    }


def maybe_warn_on_smoke_metrics(smoke_metrics: dict[str, Any], min_success_rate: float) -> None:
    success_rate = float(smoke_metrics["exact_format_success_rate"])
    if success_rate < min_success_rate:
        print(
            "[WARN] Smoke exact format success rate is below threshold: "
            f"{success_rate:.4f} < {min_success_rate:.4f}"
        )


def attach_percent_progress_callback(trainer: Any, progress_interval_percent: int, expected_total_steps: int) -> None:
    interval = max(1, min(100, int(progress_interval_percent)))
    if not hasattr(trainer, "add_callback"):
        print("[WARN] Trainer does not support add_callback; progress callback is disabled.")
        return

    try:
        from transformers import TrainerCallback
    except ImportError:
        print("[WARN] transformers TrainerCallback is unavailable; progress callback is disabled.")
        return

    class PercentProgressCallback(TrainerCallback):
        def __init__(self, interval_percent: int, fallback_total_steps: int) -> None:
            self.interval_percent = interval_percent
            self.fallback_total_steps = fallback_total_steps
            self.milestones: list[tuple[int, int]] = []
            self.next_milestone_index = 0
            self.total_steps = 0
            self.finished_reported = False

        def _resolve_total_steps(self, state: Any) -> int:
            state_total = int(getattr(state, "max_steps", 0) or 0)
            return state_total if state_total > 0 else max(0, int(self.fallback_total_steps))

        def _build_milestones(self, total_steps: int) -> list[tuple[int, int]]:
            step_to_percent: dict[int, int] = {}
            target_percents = list(range(self.interval_percent, 101, self.interval_percent))
            if 100 not in target_percents:
                target_percents.append(100)
            for percent in target_percents:
                step = max(1, math.ceil(total_steps * percent / 100.0))
                step_to_percent[step] = max(percent, step_to_percent.get(step, 0))
            return sorted(step_to_percent.items())

        def on_train_begin(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True):
                return control
            self.total_steps = self._resolve_total_steps(state)
            if self.total_steps <= 0:
                print("[INFO] training progress callback enabled, but total step count is unknown.", flush=True)
                return control
            self.milestones = self._build_milestones(self.total_steps)
            checkpoint_preview = ", ".join(
                f"{percent}%={step}/{self.total_steps}" for step, percent in self.milestones
            )
            print(f"[INFO] training progress checkpoints: {checkpoint_preview}", flush=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True) or not self.milestones:
                return control
            current_step = int(getattr(state, "global_step", 0) or 0)
            while self.next_milestone_index < len(self.milestones):
                target_step, target_percent = self.milestones[self.next_milestone_index]
                if current_step < target_step:
                    break
                print(
                    f"[PROGRESS] {target_percent}% ({min(current_step, self.total_steps)}/{self.total_steps} steps)",
                    flush=True,
                )
                self.next_milestone_index += 1
                if target_percent >= 100:
                    self.finished_reported = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if (
                getattr(state, "is_world_process_zero", True)
                and self.total_steps > 0
                and not self.finished_reported
            ):
                final_step = int(getattr(state, "global_step", 0) or 0)
                print(
                    f"[PROGRESS] 100% ({min(final_step, self.total_steps)}/{self.total_steps} steps)",
                    flush=True,
                )
            return control

    trainer.add_callback(PercentProgressCallback(interval, expected_total_steps))


def run_parser_self_test() -> bool:
    test_cases = [
        {
            "name": "display_boxed_fraction",
            "text": "[Final Answer]\n\\[\n\\boxed{\\frac{\\sqrt{6}}{2}}\n\\]",
            "expected_substring": "\\frac{\\sqrt{6}}{2}",
        },
        {
            "name": "inline_boxed_fraction",
            "text": "[Final Answer]\n\\(\\boxed{\\frac{3}{5}}\\)",
            "expected_substring": "\\frac{3}{5}",
        },
        {
            "name": "percent_boxed",
            "text": "[Final Answer] \\boxed{40\\%}",
            "expected_substring": "40\\%",
        },
        {
            "name": "fallback_last_boxed",
            "text": "[Strategy]\nDirect.\n[Reasoning]\nCompute carefully.\nThus final quantity is \\boxed{17}.",
            "expected_substring": "17",
        },
        {
            "name": "plain_number",
            "text": "[Final Answer]\n3",
            "expected_substring": "3",
        },
    ]

    passed = 0
    for case in test_cases:
        parsed = extract_final_answer(case["text"])
        success = bool(parsed) and case["expected_substring"] in parsed
        status = "PASS" if success else "FAIL"
        print(
            f"[{status}] {case['name']} | parsed={parsed!r} | "
            f"expected_substring={case['expected_substring']!r}"
        )
        if success:
            passed += 1

    all_passed = passed == len(test_cases)
    print(f"[INFO] parser self-test: {passed}/{len(test_cases)} passed")
    return all_passed


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
    if args.num_generations <= 0:
        raise SystemExit("--num_generations must be positive.")
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
    reward_fn = build_answer_only_reward_fn(args.format_bonus)
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
            "reward_type": "answer_only_correctness",
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] saved to {output_dir}")


if __name__ == "__main__":
    main()

# Smoke test:
# CUDA_VISIBLE_DEVICES=3 python3 Answer_only_GRPO.py \
#   --smoke_test_only \
#   --model_name Qwen/Qwen2.5-3B-Instruct \
#   --train_size 64 \
#   --eval_size 16 \
#   --num_generations 4 \
#   --smoke_test_examples 8 \
#   --smoke_test_generations 4 \
#   --output_dir outputs/smoke_answer_only_qwen3b_bigmath
#
# Pilot training:
# CUDA_VISIBLE_DEVICES=3 nohup python3 Answer_only_GRPO.py \
#   --model_name Qwen/Qwen2.5-3B-Instruct \
#   --train_size 1000 \
#   --eval_size 100 \
#   --num_generations 4 \
#   --max_steps 200 \
#   --max_completion_length 1024 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 5e-6 \
#   --min_solve_rate 0.2 \
#   --max_solve_rate 0.8 \
#   --run_pretrain_smoke \
#   --output_dir outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200 \
#   > train_answer_only_qwen3b_n4.log 2>&1 &
