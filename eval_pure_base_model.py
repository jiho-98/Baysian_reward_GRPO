#!/usr/bin/env python3
"""Evaluate base models with raw problem-only inputs.

This is intentionally different from the prompted baseline eval. It does not
use a system prompt, a user instruction, or the tokenizer chat template. Each
input is exactly the metadata `problem` text, and answer extraction is
format-agnostic.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    DEFAULT_MODEL_NAME,
    cleanup_extracted_answer,
    extract_last_boxed_content,
    is_suspicious_final_answer,
    parse_completion_sections,
    verify_answer,
)


DATASET_METADATA_PATHS = {
    "gsm8k": "outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl",
    "math500": "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl",
    "aime26": "outputs/eval_benchmarks/aime26_metadata.jsonl",
    "minervamath": "outputs/eval_benchmarks/minervamath_metadata.jsonl",
    "olympiadbench": "outputs/eval_benchmarks/olympiadbench_metadata.jsonl",
}

DATASET_ALIASES = {
    "gsm": "gsm8k",
    "gsm8k": "gsm8k",
    "math": "math500",
    "math500": "math500",
    "math-500": "math500",
    "math_500": "math500",
    "aime": "aime26",
    "aime26": "aime26",
    "aime2026": "aime26",
    "minerva": "minervamath",
    "minervamath": "minervamath",
    "minerva_math": "minervamath",
    "olympiad": "olympiadbench",
    "olympiadbench": "olympiadbench",
    "olympiad_bench": "olympiadbench",
}

DATASET_DEFAULT_MAX_NEW_TOKENS = {
    "gsm8k": 1024,
    "math500": 1024,
    "aime26": 4096,
    "minervamath": 1024,
    "olympiadbench": 1024,
}

MODEL_ALIASES = {
    "qwen3_1p7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no_{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure base eval: raw problem text in, no prompt/instruction/chat template."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model_key",
        default="",
        choices=sorted(MODEL_ALIASES),
        help="Optional shortcut for Qwen3 model names.",
    )
    parser.add_argument(
        "--dataset_key",
        default="all",
        help="Dataset key: all, gsm8k, math500, aime26, minervamath, olympiadbench.",
    )
    parser.add_argument(
        "--eval_metadata_path",
        default="",
        help="Custom metadata path. If set, dataset_key is used only as an output label.",
    )
    parser.add_argument("--output_root", default="")
    parser.add_argument("--output_dir", default="", help="Only valid for a single dataset eval.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means evaluate all rows.")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=0,
        help="0 uses dataset defaults: AIME26=4096, all others=1024.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Resolve jobs and paths without loading the model.")
    add_bool_arg(parser, "do_sample", False, "Use stochastic sampling during evaluation.")
    add_bool_arg(parser, "bf16", True, "Use bf16 when CUDA supports it.")
    add_bool_arg(parser, "use_vllm", False, "Use vLLM instead of transformers.generate for inference.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument(
        "--vllm_dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "half", "float32"),
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "model"


def normalize_dataset_key(dataset_key: str) -> str:
    key = str(dataset_key or "").strip().lower()
    if key == "all":
        return "all"
    if key not in DATASET_ALIASES:
        raise SystemExit(
            f"Unknown dataset_key={dataset_key!r}. "
            f"Use one of: all, {', '.join(sorted(DATASET_METADATA_PATHS))}."
        )
    return DATASET_ALIASES[key]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def bucket_value(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "") or "unknown")


def compute_accuracy_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped_total: Counter[str] = Counter()
    grouped_correct: Counter[str] = Counter()
    for row in rows:
        value = bucket_value(row, key)
        grouped_total[value] += 1
        grouped_correct[value] += int(bool(row.get("correct")))
    return {
        group: (grouped_correct[group] / grouped_total[group]) if grouped_total[group] else 0.0
        for group in sorted(grouped_total)
    }


def compute_group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(bucket_value(row, key), []).append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for group in sorted(grouped):
        group_rows = grouped[group]
        total = len(group_rows)
        correct_count = sum(int(bool(item.get("correct"))) for item in group_rows)
        extraction_count = sum(int(bool(item.get("parsed_final_answer"))) for item in group_rows)
        metrics[group] = {
            "num_examples": total,
            "correct": correct_count,
            "accuracy": (correct_count / total) if total else 0.0,
            "answer_extraction_success_rate": (extraction_count / total) if total else 0.0,
        }
    return metrics


def get_first_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vllm_sampling_params(args: argparse.Namespace, max_new_tokens: int):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment. Install `vllm` before using --use_vllm.") from exc

    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=max(float(args.temperature), 1e-6) if args.do_sample else 0.0,
        top_p=float(args.top_p) if args.do_sample else 1.0,
    )


def build_vllm_engine(args: argparse.Namespace, model_name: str):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment. Install `vllm` before using --use_vllm.") from exc

    dtype = args.vllm_dtype
    if dtype == "auto" and args.bf16:
        dtype = "bfloat16"

    engine_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": dtype,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
    }
    if args.vllm_max_model_len > 0:
        engine_kwargs["max_model_len"] = args.vllm_max_model_len
    return LLM(**engine_kwargs)


def clean_answer_candidate(candidate: str) -> str:
    cleaned = cleanup_extracted_answer(str(candidate or "").strip())
    cleaned = re.sub(r"^[\\.:=\-\s]+", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("`")
    return cleaned


def first_good_candidate(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for candidate, source in candidates:
        cleaned = clean_answer_candidate(candidate)
        if cleaned and not is_suspicious_final_answer(cleaned):
            return cleaned, source
    return "", "none"


def extract_answer_from_pure_completion(text: str) -> tuple[str, str]:
    """Extract a final answer without assuming instruction-following format."""

    raw = str(text or "")
    if not raw.strip():
        return "", "empty_completion"

    parsed = parse_completion_sections(raw)
    if parsed.get("final_answer_section_present") and parsed.get("parsed_final_answer"):
        return str(parsed["parsed_final_answer"]), "final_answer_section"

    candidates: list[tuple[str, str]] = []

    gsm_matches = re.findall(r"####\s*(.+)", raw)
    if gsm_matches:
        candidates.append((gsm_matches[-1], "gsm8k_hash_answer"))

    boxed = extract_last_boxed_content(raw)
    if boxed:
        candidates.append((boxed, "boxed"))

    explicit_patterns = [
        r"(?im)(?:final\s+answer|answer|ans)\s*(?:is|=|:)\s*([^\n]+)",
        r"(?im)(?:therefore|thus|so),?\s+(?:the\s+)?answer\s+(?:is|=)\s*([^\n]+)",
        r"(?im)(?:we\s+get|we\s+obtain)\s+([^\n]+)",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, raw)
        if matches:
            candidates.append((matches[-1], "explicit_answer_phrase"))

    tail = "\n".join(raw.splitlines()[-12:])
    equals_matches = re.findall(r"=\s*([^=\n]+?)(?:[.\s]*$|\n)", tail)
    if equals_matches:
        candidates.append((equals_matches[-1], "last_equals_expression"))

    math_token_pattern = re.compile(
        r"("
        r"\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}"
        r"|[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?%?"
        r")"
    )
    math_tokens = math_token_pattern.findall(tail)
    if math_tokens:
        candidates.append((math_tokens[-1], "last_numeric_or_fraction_token"))

    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if nonempty_lines:
        candidates.append((nonempty_lines[-1], "last_nonempty_line"))

    return first_good_candidate(candidates)


def resolve_eval_jobs(args: argparse.Namespace, model_name: str) -> list[dict[str, Any]]:
    model_slug = slugify(model_name)
    output_root = Path(args.output_root or f"outputs/pure_base_eval/{model_slug}")

    if args.eval_metadata_path:
        dataset_key = normalize_dataset_key(args.dataset_key)
        if dataset_key == "all":
            dataset_key = "custom"
        output_dir = Path(args.output_dir) if args.output_dir else output_root / dataset_key
        return [
            {
                "dataset_key": dataset_key,
                "eval_metadata_path": Path(args.eval_metadata_path),
                "output_dir": output_dir,
            }
        ]

    dataset_key = normalize_dataset_key(args.dataset_key)
    keys = list(DATASET_METADATA_PATHS) if dataset_key == "all" else [dataset_key]
    if args.output_dir and len(keys) > 1:
        raise SystemExit("--output_dir can only be used with a single dataset.")

    jobs: list[dict[str, Any]] = []
    for key in keys:
        output_dir = Path(args.output_dir) if args.output_dir else output_root / key
        jobs.append(
            {
                "dataset_key": key,
                "eval_metadata_path": Path(DATASET_METADATA_PATHS[key]),
                "output_dir": output_dir,
            }
        )
    return jobs


def evaluate_dataset(
    *,
    model: Any | None,
    llm: Any | None,
    tokenizer: Any,
    device: Any | None,
    model_name: str,
    dataset_key: str,
    eval_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = load_jsonl(eval_path)
    if args.max_examples and args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise RuntimeError(f"No eval rows loaded from {eval_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    predictions_path.write_text("", encoding="utf-8")

    max_new_tokens = (
        DATASET_DEFAULT_MAX_NEW_TOKENS.get(dataset_key, 1024)
        if args.max_new_tokens <= 0
        else args.max_new_tokens
    )

    print(f"[INFO] dataset={dataset_key} rows={len(rows)} eval_path={eval_path}", flush=True)
    print(f"[INFO] output_dir={output_dir}", flush=True)
    print(f"[INFO] max_new_tokens={max_new_tokens}", flush=True)

    correct = 0
    extraction_success = 0
    generated_lengths: list[int] = []
    result_rows: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [str(row.get("problem", "") or "") for row in batch_rows]
        if args.use_vllm:
            if llm is None:
                raise RuntimeError("vLLM engine is not initialized.")
            outputs = llm.generate(prompts, build_vllm_sampling_params(args, max_new_tokens))
            decoded = [output.outputs[0].text for output in outputs]
            batch_generated_lengths = [len(output.outputs[0].token_ids) for output in outputs]
        else:
            if model is None or device is None:
                raise RuntimeError("Transformers model is not initialized.")
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": bool(args.do_sample),
            }
            if args.do_sample:
                generation_kwargs["temperature"] = max(float(args.temperature), 1e-6)
                generation_kwargs["top_p"] = float(args.top_p)

            import torch

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            continuation_ids = output_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
            batch_generated_lengths = [len(continuation_ids[local_idx]) for local_idx in range(len(batch_rows))]

        for local_idx, (row, prompt, completion) in enumerate(zip(batch_rows, prompts, decoded)):
            problem = str(row.get("problem", "") or "")
            gold = str(row.get("answer", "") or "")
            pred_answer, extraction_source = extract_answer_from_pure_completion(completion)
            verification = verify_answer(pred_answer, gold, problem_text=problem)
            is_correct = bool(verification.get("correct"))
            has_answer = bool(pred_answer)

            correct += int(is_correct)
            extraction_success += int(has_answer)
            generated_lengths.append(batch_generated_lengths[local_idx])

            out = {
                "index": start + local_idx,
                "dataset_key": dataset_key,
                "problem": problem,
                "input_prompt": prompt,
                "gold_answer": gold,
                "source": str(row.get("source", "") or ""),
                "domain": str(row.get("domain", "") or ""),
                "difficulty_bucket": str(row.get("difficulty_bucket", "") or ""),
                "llama8b_solve_rate": row.get("llama8b_solve_rate"),
                "completion": completion,
                "parsed_final_answer": pred_answer,
                "answer_extraction_source": extraction_source,
                "correct": is_correct,
                "verification": verification,
            }
            result_rows.append(out)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")

        done = min(start + len(batch_rows), len(rows))
        print(f"[PROGRESS] {dataset_key}: evaluated {done}/{len(rows)}", flush=True)

    accuracy = correct / len(rows)
    source_distribution = Counter(bucket_value(row, "source") for row in result_rows)
    domain_distribution = Counter(bucket_value(row, "domain") for row in result_rows)
    difficulty_distribution = Counter(bucket_value(row, "difficulty_bucket") for row in result_rows)
    solve_rates = [
        float(row["llama8b_solve_rate"])
        for row in result_rows
        if row.get("llama8b_solve_rate") is not None
    ]
    summary = {
        "model_name": model_name,
        "dataset_key": dataset_key,
        "eval_metadata_path": str(eval_path),
        "prompt_mode": "raw_problem_only",
        "uses_system_prompt": False,
        "uses_user_instruction": False,
        "uses_chat_template": False,
        "num_examples": len(rows),
        "correct": correct,
        "correct_count": correct,
        "accuracy": accuracy,
        "answer_extraction_success_rate": extraction_success / len(rows),
        "source_distribution": dict(source_distribution),
        "domain_distribution": dict(domain_distribution),
        "difficulty_distribution": dict(difficulty_distribution),
        "accuracy_by_source": compute_accuracy_breakdown(result_rows, "source"),
        "accuracy_by_domain": compute_accuracy_breakdown(result_rows, "domain"),
        "accuracy_by_difficulty_bucket": compute_accuracy_breakdown(result_rows, "difficulty_bucket"),
        "metrics_by_source": compute_group_metrics(result_rows, "source"),
        "metrics_by_domain": compute_group_metrics(result_rows, "domain"),
        "metrics_by_difficulty_bucket": compute_group_metrics(result_rows, "difficulty_bucket"),
        "solve_rate_min": min(solve_rates) if solve_rates else None,
        "solve_rate_max": max(solve_rates) if solve_rates else None,
        "solve_rate_mean": statistics.fmean(solve_rates) if solve_rates else None,
        "batch_size": args.batch_size,
        "inference_backend": "vllm" if args.use_vllm else "transformers",
        "use_vllm": bool(args.use_vllm),
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization if args.use_vllm else None,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size if args.use_vllm else None,
        "vllm_max_model_len": args.vllm_max_model_len if args.use_vllm else None,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "generated_length_mean": statistics.fmean(generated_lengths) if generated_lengths else None,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[DONE] pure-base eval summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive.")
    if args.max_new_tokens < 0:
        raise SystemExit("--max_new_tokens must be non-negative.")

    model_name = MODEL_ALIASES.get(args.model_key, args.model_name) if args.model_key else args.model_name
    random.seed(args.seed)

    jobs = resolve_eval_jobs(args, model_name)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_name": model_name,
                    "prompt_mode": "raw_problem_only",
                    "inference_backend": "vllm" if args.use_vllm else "transformers",
                    "jobs": [
                        {
                            "dataset_key": job["dataset_key"],
                            "eval_metadata_path": str(job["eval_metadata_path"]),
                            "eval_metadata_exists": job["eval_metadata_path"].exists(),
                            "output_dir": str(job["output_dir"]),
                            "max_new_tokens": (
                                DATASET_DEFAULT_MAX_NEW_TOKENS.get(job["dataset_key"], 1024)
                                if args.max_new_tokens <= 0
                                else args.max_new_tokens
                            ),
                        }
                        for job in jobs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[INFO] pure-base eval")
    print(f"[INFO] model={model_name}")
    print("[INFO] prompt_mode=raw_problem_only")
    print("[INFO] no system prompt, no user instruction, no chat template")
    print(f"[INFO] jobs={[job['dataset_key'] for job in jobs]}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = None
    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

    model = None
    device = None
    llm = None
    if args.use_vllm:
        llm = build_vllm_engine(args, model_name)
        print("[INFO] inference_backend=vllm")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        device = get_first_device(model)
        print("[INFO] inference_backend=transformers")

    summaries = []
    for job in jobs:
        summaries.append(
            evaluate_dataset(
                model=model,
                llm=llm,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                dataset_key=job["dataset_key"],
                eval_path=job["eval_metadata_path"],
                output_dir=job["output_dir"],
                args=args,
            )
        )

    if len(summaries) > 1:
        output_root = Path(args.output_root or f"outputs/pure_base_eval/{slugify(model_name)}")
        output_root.mkdir(parents=True, exist_ok=True)
        aggregate_path = output_root / "all_summary.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "prompt_mode": "raw_problem_only",
                    "datasets": summaries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[DONE] aggregate_summary={aggregate_path}")


if __name__ == "__main__":
    main()
