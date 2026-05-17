#!/usr/bin/env python3
"""Standalone evaluation for Answer-only GRPO / RLVR models."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from Answer_only_GRPO import (
    DEFAULT_MODEL_NAME,
    default_bf16_enabled,
    extract_section,
    generate_smoke_outputs,
    import_torch,
    load_tokenizer,
    parse_completion_sections,
    render_prompt,
    set_seed,
    verify_answer,
)


DEFAULT_EVAL_PATH = "outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_eval_metadata.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/eval_answer_only_grpo_compare"
DEFAULT_N4_ADAPTER_PATH = "outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/checkpoint-200"
DEFAULT_N8_ADAPTER_PATH = "outputs/grpo_answer_only_qwen3b_bigmath_n8_steps200/checkpoint-200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen2.5-3B-Instruct and Answer-only GRPO LoRA adapters."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--eval_path", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--n4_adapter_path", default=DEFAULT_N4_ADAPTER_PATH)
    parser.add_argument("--n8_adapter_path", default=DEFAULT_N8_ADAPTER_PATH)
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


def load_eval_rows(eval_path: str, limit: Optional[int]) -> list[dict[str, Any]]:
    path = Path(eval_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise RuntimeError(f"No eval rows found in {path}")
    return rows


def parse_solver_output(raw_output: str) -> dict[str, Any]:
    strategy = extract_section(raw_output, "Strategy", ["Reasoning", "Final Answer"])
    reasoning = extract_section(raw_output, "Reasoning", ["Final Answer"])
    parsed = parse_completion_sections(raw_output)
    return {
        "parsed_strategy": strategy.strip(),
        "parsed_reasoning": reasoning.strip(),
        "parsed_final_answer": parsed["parsed_final_answer"],
        "strategy_section_present": parsed["strategy_section_present"],
        "reasoning_section_present": parsed["reasoning_section_present"],
        "final_answer_section_present": parsed["final_answer_section_present"],
        "suspicious_final_answer": parsed["suspicious_final_answer"],
        "format_valid": parsed["exact_format_success"],
    }


def build_result_row(
    *,
    index: int,
    model_label: str,
    example: dict[str, Any],
    raw_output: str,
    parsed_output: dict[str, Any],
    verification: dict[str, Any],
    generation_error: Optional[str],
) -> dict[str, Any]:
    return {
        "index": index,
        "model_label": model_label,
        "problem": str(example.get("problem", "")),
        "gold_answer": str(example.get("answer", "")),
        "source": str(example.get("source", "") or ""),
        "domain": str(example.get("domain", "") or ""),
        "llama8b_solve_rate": example.get("llama8b_solve_rate"),
        "difficulty_bucket": str(example.get("difficulty_bucket", "") or ""),
        "raw_output": raw_output,
        "parsed_strategy": parsed_output["parsed_strategy"],
        "parsed_reasoning": parsed_output["parsed_reasoning"],
        "parsed_final_answer": parsed_output["parsed_final_answer"],
        "normalized_predicted_answer": verification.get("normalized_predicted_answer", ""),
        "normalized_gold_answer": verification.get("normalized_gold_answer", ""),
        "correct": bool(verification.get("correct", False)),
        "verification_method": verification.get("verification_method", "no_match"),
        "possible_miss_reasons": verification.get("possible_miss_reasons", []),
        "format_valid": bool(parsed_output["format_valid"]),
        "strategy_section_present": bool(parsed_output["strategy_section_present"]),
        "reasoning_section_present": bool(parsed_output["reasoning_section_present"]),
        "final_answer_section_present": bool(parsed_output["final_answer_section_present"]),
        "suspicious_final_answer": bool(parsed_output["suspicious_final_answer"]),
        "generation_error": generation_error,
    }


def compute_accuracy_breakdown(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    grouped_total: Counter[str] = Counter()
    grouped_correct: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key, "") or "unknown")
        grouped_total[value] += 1
        grouped_correct[value] += int(bool(row["correct"]))
    return {
        group: (grouped_correct[group] / grouped_total[group]) if grouped_total[group] else 0.0
        for group in sorted(grouped_total)
    }


def compute_metrics(
    *,
    model_label: str,
    model_name: str,
    adapter_path: Optional[str],
    eval_path: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    num_examples = len(results)
    correct_count = sum(int(row["correct"]) for row in results)
    accuracy = (correct_count / num_examples) if num_examples else 0.0
    format_success_count = sum(int(row["format_valid"]) for row in results)
    strategy_present_count = sum(int(row["strategy_section_present"]) for row in results)
    reasoning_present_count = sum(int(row["reasoning_section_present"]) for row in results)
    final_answer_present_count = sum(int(row["final_answer_section_present"]) for row in results)
    nonempty_final_answer_count = sum(int(bool(row["parsed_final_answer"])) for row in results)
    suspicious_final_answer_count = sum(int(row["suspicious_final_answer"]) for row in results)
    completion_lengths = [len(row["raw_output"]) for row in results]
    source_distribution = Counter(str(row["source"] or "unknown") for row in results)
    difficulty_distribution = Counter(str(row["difficulty_bucket"] or "unknown") for row in results)
    solve_rates = [
        float(row["llama8b_solve_rate"])
        for row in results
        if row.get("llama8b_solve_rate") is not None
    ]

    return {
        "model_label": model_label,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "eval_path": eval_path,
        "num_examples": num_examples,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "format_success_rate": (format_success_count / num_examples) if num_examples else 0.0,
        "strategy_section_present_rate": (strategy_present_count / num_examples) if num_examples else 0.0,
        "reasoning_section_present_rate": (reasoning_present_count / num_examples) if num_examples else 0.0,
        "final_answer_section_present_rate": (final_answer_present_count / num_examples) if num_examples else 0.0,
        "nonempty_final_answer_rate": (nonempty_final_answer_count / num_examples) if num_examples else 0.0,
        "suspicious_final_answer_rate": (suspicious_final_answer_count / num_examples) if num_examples else 0.0,
        "avg_completion_chars": statistics.fmean(completion_lengths) if completion_lengths else 0.0,
        "source_distribution": dict(source_distribution),
        "difficulty_distribution": dict(difficulty_distribution),
        "accuracy_by_difficulty_bucket": compute_accuracy_breakdown(results, "difficulty_bucket"),
        "accuracy_by_source": compute_accuracy_breakdown(results, "source"),
        "solve_rate_min": min(solve_rates) if solve_rates else None,
        "solve_rate_max": max(solve_rates) if solve_rates else None,
        "solve_rate_mean": statistics.fmean(solve_rates) if solve_rates else None,
    }


def cleanup_model_memory(model: Any) -> None:
    del model
    torch = import_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_for_eval(model_name: str, adapter_path: Optional[str], use_bf16: bool):
    torch = import_torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("transformers is required. Install it with `pip install transformers`.") from exc

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if use_bf16 else "auto",
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = base_model

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError("peft is required to load LoRA adapters. Install it with `pip install peft`.") from exc
        model = PeftModel.from_pretrained(base_model, adapter_path)

    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model


def evaluate_model(
    *,
    model_label: str,
    file_stem: str,
    model_name: str,
    adapter_path: Optional[str],
    eval_rows: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    output_dir: Path,
    use_bf16: bool,
) -> Optional[dict[str, Any]]:
    if adapter_path is not None and not Path(adapter_path).exists():
        print(f"[WARN] Adapter path does not exist, skipping {model_label}: {adapter_path}")
        return None

    print(f"[INFO] Loading model for {model_label}...")
    try:
        model = load_model_for_eval(model_name, adapter_path, use_bf16)
    except Exception as exc:
        if adapter_path is not None:
            print(f"[WARN] Failed to load {model_label}, skipping: {exc}")
            return None
        raise

    results: list[dict[str, Any]] = []
    for index, example in enumerate(eval_rows):
        raw_output = ""
        generation_error: Optional[str] = None
        parsed_output = {
            "parsed_strategy": "",
            "parsed_reasoning": "",
            "parsed_final_answer": "",
            "strategy_section_present": False,
            "reasoning_section_present": False,
            "final_answer_section_present": False,
            "suspicious_final_answer": True,
            "format_valid": False,
        }
        verification = {
            "normalized_predicted_answer": "",
            "normalized_gold_answer": "",
            "correct": False,
            "verification_method": "no_match",
            "possible_miss_reasons": [],
        }

        try:
            prompt = render_prompt(str(example["problem"]), tokenizer)
            outputs = generate_smoke_outputs(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                num_generations=1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            raw_output = outputs[0] if outputs else ""
            parsed_output = parse_solver_output(raw_output)
            verification = verify_answer(
                parsed_output["parsed_final_answer"],
                str(example["answer"]),
                problem_text=str(example["problem"]),
            )
        except Exception as exc:
            generation_error = str(exc)

        results.append(
            build_result_row(
                index=index,
                model_label=model_label,
                example=example,
                raw_output=raw_output,
                parsed_output=parsed_output,
                verification=verification,
                generation_error=generation_error,
            )
        )

        if (index + 1) % 10 == 0 or (index + 1) == len(eval_rows):
            print(f"[INFO] {model_label}: processed {index + 1}/{len(eval_rows)} examples")

    results_path = output_dir / f"{file_stem}_eval_results.jsonl"
    metrics_path = output_dir / f"{file_stem}_metrics.json"
    metrics = compute_metrics(
        model_label=model_label,
        model_name=model_name,
        adapter_path=adapter_path,
        eval_path=args.eval_path,
        results=results,
    )
    write_jsonl(results_path, results)
    write_json(metrics_path, metrics)
    cleanup_model_memory(model)
    return metrics


def maybe_add_model_specs(args: argparse.Namespace) -> list[tuple[str, str, Optional[str]]]:
    if args.base_only and args.skip_base:
        raise SystemExit("--base_only and --skip_base cannot both be set.")

    specs: list[tuple[str, str, Optional[str]]] = []
    if args.base_only:
        return [("base", "base", None)]

    if not args.skip_base:
        specs.append(("base", "base", None))
    specs.append(("answer_only_grpo_n4", "n4", args.n4_adapter_path))
    specs.append(("answer_only_grpo_n8", "n8", args.n8_adapter_path))
    return specs


def safe_delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def build_comparison_summary(
    *,
    eval_path: str,
    metrics_by_stem: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_accuracy = metrics_by_stem.get("base", {}).get("accuracy")
    n4_accuracy = metrics_by_stem.get("n4", {}).get("accuracy")
    n8_accuracy = metrics_by_stem.get("n8", {}).get("accuracy")
    base_format_success_rate = metrics_by_stem.get("base", {}).get("format_success_rate")
    n4_format_success_rate = metrics_by_stem.get("n4", {}).get("format_success_rate")
    n8_format_success_rate = metrics_by_stem.get("n8", {}).get("format_success_rate")

    evaluated = [
        (stem, metrics["accuracy"])
        for stem, metrics in metrics_by_stem.items()
        if metrics is not None
    ]
    which_model_best = None
    if evaluated:
        best_stem, _ = max(evaluated, key=lambda item: (item[1], item[0]))
        which_model_best = {
            "base": "base",
            "n4": "answer_only_grpo_n4",
            "n8": "answer_only_grpo_n8",
        }[best_stem]

    return {
        "eval_path": eval_path,
        "base_accuracy": base_accuracy,
        "n4_accuracy": n4_accuracy,
        "n8_accuracy": n8_accuracy,
        "delta_n4_minus_base": safe_delta(n4_accuracy, base_accuracy),
        "delta_n8_minus_base": safe_delta(n8_accuracy, base_accuracy),
        "delta_n8_minus_n4": safe_delta(n8_accuracy, n4_accuracy),
        "base_format_success_rate": base_format_success_rate,
        "n4_format_success_rate": n4_format_success_rate,
        "n8_format_success_rate": n8_format_success_rate,
        "which_model_best": which_model_best,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def print_compact_table(metrics_by_stem: dict[str, dict[str, Any]]) -> None:
    print("\nModel | Accuracy | Correct/Total | Format Success | Final Answer Present | Suspicious Rate")
    display_order = [("base", "base"), ("n4", "answer_only_grpo_n4"), ("n8", "answer_only_grpo_n8")]
    for stem, label in display_order:
        metrics = metrics_by_stem.get(stem)
        if metrics is None:
            continue
        accuracy = metrics["accuracy"]
        correct_count = metrics["correct_count"]
        total = metrics["num_examples"]
        format_success = metrics["format_success_rate"]
        final_answer_present = metrics["final_answer_section_present_rate"]
        suspicious_rate = metrics["suspicious_final_answer_rate"]
        print(
            f"{label:<20} {accuracy:.2f}  {correct_count}/{total}  "
            f"{format_success:.2f}  {final_answer_present:.2f}  {suspicious_rate:.2f}"
        )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    set_seed(args.seed)

    eval_rows = load_eval_rows(args.eval_path, args.limit)
    tokenizer = load_tokenizer(args.model_name)
    use_bf16 = default_bf16_enabled()

    model_specs = maybe_add_model_specs(args)
    metrics_by_stem: dict[str, dict[str, Any]] = {}

    for model_label, file_stem, adapter_path in model_specs:
        metrics = evaluate_model(
            model_label=model_label,
            file_stem=file_stem,
            model_name=args.model_name,
            adapter_path=adapter_path,
            eval_rows=eval_rows,
            tokenizer=tokenizer,
            args=args,
            output_dir=output_dir,
            use_bf16=use_bf16,
        )
        if metrics is not None:
            metrics_by_stem[file_stem] = metrics

    if not metrics_by_stem:
        raise RuntimeError("No models were evaluated.")

    comparison_summary = build_comparison_summary(
        eval_path=args.eval_path,
        metrics_by_stem=metrics_by_stem,
    )
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    print_compact_table(metrics_by_stem)


if __name__ == "__main__":
    main()

# Evaluate all three on the same 100 eval examples:
# CUDA_VISIBLE_DEVICES=3 python3 eval_answer_only_grpo.py \
#   --model_name Qwen/Qwen2.5-3B-Instruct \
#   --eval_path outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_eval_metadata.jsonl \
#   --n4_adapter_path outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/checkpoint-200 \
#   --n8_adapter_path outputs/grpo_answer_only_qwen3b_bigmath_n8_steps200/checkpoint-200 \
#   --output_dir outputs/eval_answer_only_grpo_compare_100
#
# Quick test on 10 examples:
# CUDA_VISIBLE_DEVICES=3 python3 eval_answer_only_grpo.py \
#   --limit 10 \
#   --output_dir outputs/eval_answer_only_grpo_quick10
