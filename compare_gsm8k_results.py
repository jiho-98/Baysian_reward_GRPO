#!/usr/bin/env python3
"""Aggregate GSM8K experiment summaries into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENTS_ROOT = "outputs/gsm8k_experiments"
DEFAULT_OUTPUT_JSON = "outputs/gsm8k_experiments/gsm8k_main_comparison.json"
DEFAULT_OUTPUT_MD = "outputs/gsm8k_experiments/gsm8k_main_comparison.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate GSM8K main-experiment summaries."
    )
    parser.add_argument("--experiments_root", default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_accuracy(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    value = summary.get("accuracy")
    if value is None:
        return None
    return float(value)


def format_correct_total(summary: dict[str, Any] | None) -> str | None:
    if summary is None:
        return None
    correct = summary.get("correct")
    if correct is None:
        correct = summary.get("correct_count")
    total = summary.get("num_examples")
    if correct is None or total is None:
        return None
    return f"{int(correct)}/{int(total)}"


def maybe_float_text(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def build_method_row(
    *,
    method: str,
    train_data: str,
    reward: str,
    valid_summary: dict[str, Any] | None,
    test_summary: dict[str, Any] | None,
    notes: str,
    valid_summary_path: Path,
    test_summary_path: Path,
) -> dict[str, Any]:
    return {
        "Method": method,
        "Train Data": train_data,
        "Reward": reward,
        "Valid Accuracy": format_accuracy(valid_summary),
        "Test Accuracy": format_accuracy(test_summary),
        "Correct/Test Total": format_correct_total(test_summary),
        "Notes": notes,
        "valid_summary_path": str(valid_summary_path),
        "test_summary_path": str(test_summary_path),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Method",
        "Train Data",
        "Reward",
        "Valid Accuracy",
        "Test Accuracy",
        "Correct/Test Total",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Method"]),
                    str(row["Train Data"]),
                    str(row["Reward"]),
                    maybe_float_text(row["Valid Accuracy"]),
                    maybe_float_text(row["Test Accuracy"]),
                    str(row["Correct/Test Total"] or "NA"),
                    str(row["Notes"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.experiments_root)

    base_valid = root / "base_qwen3b" / "valid" / "summary.json"
    base_test = root / "base_qwen3b" / "test" / "summary.json"

    sft_valid = root / "sft_qwen3b_train3000" / "valid" / "summary.json"
    sft_test = root / "sft_qwen3b_train3000" / "test" / "summary.json"

    answer_valid = root / "grpo_answer_only_qwen3b_train3000_n8_steps500" / "valid" / "summary.json"
    answer_test = root / "grpo_answer_only_qwen3b_train3000_n8_steps500" / "test" / "summary.json"

    bayes_root = root / "grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10"
    bayes_valid = bayes_root / "valid" / "summary.json"
    bayes_test = bayes_root / "test" / "summary.json"
    bayes_diag = bayes_root / "bayesian_reward_diagnostics" / "summary.json"

    bayes_diag_payload = load_json_if_exists(bayes_diag)
    bayes_notes = ""
    if bayes_diag_payload is None:
        bayes_notes = "diagnostics missing"
    else:
        diag = bayes_diag_payload.get("diagnostics", {})
        bayes_notes = (
            "lambda=1.0; "
            f"prior_parse={float(diag.get('prior_parse_rate', 0.0)):.3f}; "
            f"evidence_parse={float(diag.get('evidence_parse_rate', 0.0)):.3f}; "
            f"mass_correct={float(diag.get('mass_on_correct_mean', 0.0)):.3f}"
        )

    rows = [
        build_method_row(
            method="Base Qwen2.5-3B-Instruct",
            train_data="0",
            reward="none",
            valid_summary=load_json_if_exists(base_valid),
            test_summary=load_json_if_exists(base_test),
            notes="deterministic base eval",
            valid_summary_path=base_valid,
            test_summary_path=base_test,
        ),
        build_method_row(
            method="SFT on GSM8K-3K",
            train_data="GSM8K-3K",
            reward="supervised",
            valid_summary=load_json_if_exists(sft_valid),
            test_summary=load_json_if_exists(sft_test),
            notes="LoRA SFT with GSM8K reasoning targets",
            valid_summary_path=sft_valid,
            test_summary_path=sft_test,
        ),
        build_method_row(
            method="GRPO Answer-only on GSM8K-3K",
            train_data="GSM8K-3K",
            reward="answer correctness",
            valid_summary=load_json_if_exists(answer_valid),
            test_summary=load_json_if_exists(answer_test),
            notes="reward=1 iff final answer correct",
            valid_summary_path=answer_valid,
            test_summary_path=answer_test,
        ),
        build_method_row(
            method="GRPO Bayesian Prompted Analyzer on GSM8K-3K",
            train_data="GSM8K-3K",
            reward="full Bayesian posterior",
            valid_summary=load_json_if_exists(bayes_valid),
            test_summary=load_json_if_exists(bayes_test),
            notes=bayes_notes,
            valid_summary_path=bayes_valid,
            test_summary_path=bayes_test,
        ),
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments_root": str(root),
        "rows": rows,
        "bayesian_diagnostics_path": str(bayes_diag),
    }
    write_json(Path(args.output_json), payload)
    write_text(Path(args.output_md), markdown_table(rows))

    print("[INFO] GSM8K comparison summary")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
