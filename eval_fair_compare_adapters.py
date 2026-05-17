#!/usr/bin/env python3
"""Evaluate base Qwen and multiple LoRA adapters on the same fixed Big-Math eval split."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen and multiple adapters on the same fixed Big-Math eval split."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Adapter spec in the form label:path. Repeat for multiple adapters.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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


def parse_adapter_specs(raw_specs: list[str]) -> list[tuple[str, str]]:
    if not raw_specs:
        raise SystemExit("At least one --adapter label:path is required.")

    specs: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for raw_spec in raw_specs:
        if ":" not in raw_spec:
            raise SystemExit(f"Invalid --adapter spec (expected label:path): {raw_spec}")
        label, path = raw_spec.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(f"Invalid --adapter spec (empty label or path): {raw_spec}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise SystemExit(
                f"Invalid adapter label {label!r}. Use only letters, numbers, underscore, dot, and hyphen."
            )
        if label == "base":
            raise SystemExit("Adapter label 'base' is reserved.")
        if label in seen_labels:
            raise SystemExit(f"Duplicate adapter label: {label}")
        seen_labels.add(label)
        specs.append((label, path))
    return specs


def load_eval_rows(eval_path: str, limit: Optional[int]) -> list[dict[str, Any]]:
    path = Path(eval_path)
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

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
        "format_valid": bool(parsed_output["format_valid"]),
        "strategy_section_present": bool(parsed_output["strategy_section_present"]),
        "reasoning_section_present": bool(parsed_output["reasoning_section_present"]),
        "final_answer_section_present": bool(parsed_output["final_answer_section_present"]),
        "suspicious_final_answer": bool(parsed_output["suspicious_final_answer"]),
        "generation_error": generation_error,
    }


def compute_accuracy_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
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
    eval_path: str,
    tokenizer: Any,
    max_new_tokens: int,
    output_dir: Path,
    use_bf16: bool,
) -> dict[str, Any]:
    if adapter_path is not None and not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter path does not exist for {model_label}: {adapter_path}")

    print(f"[INFO] Loading model for {model_label}...")
    model = load_model_for_eval(model_name, adapter_path, use_bf16)

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
        }

        try:
            prompt = render_prompt(str(example["problem"]), tokenizer)
            outputs = generate_smoke_outputs(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                num_generations=1,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
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

    metrics = compute_metrics(
        model_label=model_label,
        model_name=model_name,
        adapter_path=adapter_path,
        eval_path=eval_path,
        results=results,
    )
    write_jsonl(output_dir / f"{file_stem}_eval_results.jsonl", results)
    write_json(output_dir / f"{file_stem}_metrics.json", metrics)
    cleanup_model_memory(model)
    return metrics


def safe_delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def build_comparison_summary(
    *,
    eval_path: str,
    metrics_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_metrics = metrics_by_label.get("base", {})
    answer_metrics = metrics_by_label.get("answer_only_n8", {})
    bayes_metrics = metrics_by_label.get("bayesian_ah080_n8", {})
    labels_in_order = list(metrics_by_label)

    return {
        "eval_path": eval_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_examples": base_metrics.get("num_examples"),
        "labels_in_order": labels_in_order,
        "base_accuracy": base_metrics.get("accuracy"),
        "answer_only_n8_accuracy": answer_metrics.get("accuracy"),
        "bayesian_ah080_n8_accuracy": bayes_metrics.get("accuracy"),
        "delta_answer_only_minus_base": safe_delta(answer_metrics.get("accuracy"), base_metrics.get("accuracy")),
        "delta_bayesian_minus_base": safe_delta(bayes_metrics.get("accuracy"), base_metrics.get("accuracy")),
        "delta_bayesian_minus_answer_only": safe_delta(
            bayes_metrics.get("accuracy"),
            answer_metrics.get("accuracy"),
        ),
        "base_format_success_rate": base_metrics.get("format_success_rate"),
        "answer_only_n8_format_success_rate": answer_metrics.get("format_success_rate"),
        "bayesian_ah080_n8_format_success_rate": bayes_metrics.get("format_success_rate"),
        "base_final_answer_present_rate": base_metrics.get("final_answer_section_present_rate"),
        "answer_only_n8_final_answer_present_rate": answer_metrics.get("final_answer_section_present_rate"),
        "bayesian_ah080_n8_final_answer_present_rate": bayes_metrics.get("final_answer_section_present_rate"),
        "base_suspicious_rate": base_metrics.get("suspicious_final_answer_rate"),
        "answer_only_n8_suspicious_rate": answer_metrics.get("suspicious_final_answer_rate"),
        "bayesian_ah080_n8_suspicious_rate": bayes_metrics.get("suspicious_final_answer_rate"),
        "accuracy_by_difficulty_bucket": {
            label: metrics["accuracy_by_difficulty_bucket"] for label, metrics in metrics_by_label.items()
        },
        "accuracy_by_source": {
            label: metrics["accuracy_by_source"] for label, metrics in metrics_by_label.items()
        },
        "source_distribution": base_metrics.get("source_distribution"),
        "difficulty_distribution": base_metrics.get("difficulty_distribution"),
        "avg_completion_chars": {
            label: metrics["avg_completion_chars"] for label, metrics in metrics_by_label.items()
        },
        "per_model_metrics": metrics_by_label,
    }


def print_compact_table(model_order: list[str], metrics_by_label: dict[str, dict[str, Any]]) -> None:
    print("\nModel | Accuracy | Correct/Total | Format Success | Final Answer Present | Suspicious Rate")
    for label in model_order:
        metrics = metrics_by_label[label]
        print(
            f"{label:<20} | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['correct_count']}/{metrics['num_examples']} | "
            f"{metrics['format_success_rate']:.4f} | "
            f"{metrics['final_answer_section_present_rate']:.4f} | "
            f"{metrics['suspicious_final_answer_rate']:.4f}"
        )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    set_seed(args.seed)

    adapter_specs = parse_adapter_specs(args.adapter)
    eval_rows = load_eval_rows(args.eval_path, args.limit)
    tokenizer = load_tokenizer(args.model_name)
    use_bf16 = default_bf16_enabled()

    metrics_by_label: dict[str, dict[str, Any]] = {}
    model_order = ["base"] + [label for label, _ in adapter_specs]

    metrics_by_label["base"] = evaluate_model(
        model_label="base",
        file_stem="base",
        model_name=args.model_name,
        adapter_path=None,
        eval_rows=eval_rows,
        eval_path=args.eval_path,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        use_bf16=use_bf16,
    )

    for label, adapter_path in adapter_specs:
        metrics_by_label[label] = evaluate_model(
            model_label=label,
            file_stem=label,
            model_name=args.model_name,
            adapter_path=adapter_path,
            eval_rows=eval_rows,
            eval_path=args.eval_path,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            output_dir=output_dir,
            use_bf16=use_bf16,
        )

    comparison_summary = build_comparison_summary(
        eval_path=args.eval_path,
        metrics_by_label=metrics_by_label,
    )
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    print_compact_table(model_order, metrics_by_label)


if __name__ == "__main__":
    main()
