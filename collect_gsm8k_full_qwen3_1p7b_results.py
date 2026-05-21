#!/usr/bin/env python3
"""Collect Qwen3-1.7B GSM8K full-train experiment results into JSON and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "outputs/gsm8k_full_qwen3_1p7b"
EXPECTED_MODEL_NAME = "Qwen/Qwen3-1.7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect GSM8K full-train Qwen3-1.7B comparison rows."
    )
    parser.add_argument("--root_dir", default=DEFAULT_ROOT)
    parser.add_argument("--metadata_dir", default="outputs/gsm8k_full_train_seed42")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_float(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def optional_int(payload: dict[str, Any] | None, key: str) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def optional_str(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def config_value(path: Path, key: str) -> Any:
    payload = load_json_if_exists(path)
    if payload is None:
        return None
    return payload.get(key)


def resolve_model_name(
    *,
    root_summary: dict[str, Any] | None,
    test_summary: dict[str, Any] | None,
    config_summary: dict[str, Any] | None,
) -> str | None:
    for payload, key in (
        (root_summary, "model_name"),
        (test_summary, "model_name"),
        (config_summary, "model_name"),
    ):
        value = optional_str(payload, key)
        if value is not None:
            return value
    return None


def resolve_config_dir(experiment_dir: Path) -> dict[str, Any] | None:
    for candidate in (
        experiment_dir / "training_config.json",
        experiment_dir / "launcher_config.json",
    ):
        payload = load_json_if_exists(candidate)
        if payload is not None:
            payload["_config_path"] = str(candidate)
            return payload
    return None


def collect_row(
    *,
    method: str,
    train_data: str,
    reward_type: str,
    analyzer_type: str,
    experiment_dir: Path,
    diagnostics_path: Path | None,
    metadata_summary: dict[str, Any] | None,
    notes: str,
) -> dict[str, Any]:
    root_summary = load_json_if_exists(experiment_dir / "summary.json")
    test_summary_path = experiment_dir / "test" / "summary.json"
    test_summary = load_json_if_exists(test_summary_path)
    config_summary = resolve_config_dir(experiment_dir)
    diagnostics_summary = load_json_if_exists(diagnostics_path)

    model_name = resolve_model_name(
        root_summary=root_summary,
        test_summary=test_summary,
        config_summary=config_summary,
    )

    max_steps = None
    num_generations = None
    if config_summary is not None:
        max_steps = config_summary.get("max_steps")
        num_generations = config_summary.get("num_generations")
    use_lora = None
    gradient_checkpointing = None
    training_mode = "base_eval" if reward_type == "none" else None

    train_size = None
    test_size = None
    valid_accuracy = None
    test_accuracy = optional_float(test_summary, "accuracy")
    correct_count = optional_int(test_summary, "correct")
    test_total = optional_int(test_summary, "num_examples")

    if root_summary is not None:
        train_size = root_summary.get("train_size")
        if root_summary.get("train_size_full") is not None:
            train_size = root_summary.get("train_size_full")
        test_size = root_summary.get("test_size")
        if root_summary.get("test_size_full") is not None:
            test_size = root_summary.get("test_size_full")
        valid_accuracy = root_summary.get("valid_accuracy")
        test_accuracy = root_summary.get("test_accuracy", test_accuracy)
        correct_count = root_summary.get("correct", correct_count)
        test_total = root_summary.get("test_total", test_total)
        notes = str(root_summary.get("notes", notes))
        reward_type = str(root_summary.get("reward_type", reward_type))
        analyzer_type = str(root_summary.get("analyzer_type", analyzer_type))
        train_data = str(root_summary.get("train_data", train_data))
        use_lora = root_summary.get("use_lora", use_lora)
        gradient_checkpointing = root_summary.get(
            "gradient_checkpointing",
            gradient_checkpointing,
        )
        training_mode = root_summary.get("training_mode", training_mode)

    if use_lora is None and config_summary is not None:
        use_lora = config_summary.get("use_lora")
    if gradient_checkpointing is None and config_summary is not None:
        gradient_checkpointing = config_summary.get("gradient_checkpointing")
    if training_mode is None:
        if use_lora is True:
            training_mode = "lora_grpo"
        elif use_lora is False:
            training_mode = "full_finetune_grpo"

    if train_size is None and metadata_summary is not None and reward_type != "none":
        train_size = metadata_summary.get("train_size_full") or metadata_summary.get("train_size")
    if test_size is None:
        if metadata_summary is not None:
            test_size = metadata_summary.get("test_size_full") or metadata_summary.get("test_size")
        elif test_total is not None:
            test_size = test_total

    correct_over_total = None
    if correct_count is not None and test_total is not None:
        correct_over_total = f"{int(correct_count)}/{int(test_total)}"

    row = {
        "Method": method,
        "Model": model_name,
        "Train Data": train_data,
        "Train Size": train_size if reward_type != "none" else 0,
        "Test Size": test_size,
        "Reward Type": reward_type,
        "Analyzer Type": analyzer_type,
        "Training Mode": training_mode,
        "Use LoRA": use_lora,
        "Gradient Checkpointing": gradient_checkpointing,
        "Max Steps": max_steps,
        "Num Generations": num_generations,
        "Valid Accuracy": valid_accuracy,
        "Test Accuracy": test_accuracy,
        "Correct/Test Total": correct_over_total,
        "Format Success Rate": optional_float(test_summary, "format_success_rate"),
        "Suspicious Final Answer Rate": optional_float(
            test_summary,
            "suspicious_final_answer_rate",
        ),
        "Generated Length Mean": optional_float(test_summary, "generated_length_mean"),
        "Checkpoint Path": optional_str(root_summary, "checkpoint_path") or str(experiment_dir),
        "Test Summary Path": str(test_summary_path) if test_summary_path.exists() else None,
        "Diagnostics Path": str(diagnostics_path) if diagnostics_path is not None and diagnostics_path.exists() else None,
        "Notes": notes,
        "config_path": optional_str(config_summary, "_config_path"),
        "predictions_path": optional_str(test_summary, "predictions_path"),
        "diagnostics_loaded": diagnostics_summary is not None,
    }

    if row["Model"] is not None and row["Model"] != EXPECTED_MODEL_NAME:
        row["Notes"] = f"{row['Notes']}; WARNING unexpected model={row['Model']}"

    return row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method",
        "Model",
        "Train Data",
        "Train Size",
        "Test Size",
        "Reward Type",
        "Analyzer Type",
        "Training Mode",
        "Use LoRA",
        "Gradient Checkpointing",
        "Max Steps",
        "Num Generations",
        "Valid Accuracy",
        "Test Accuracy",
        "Correct/Test Total",
        "Format Success Rate",
        "Suspicious Final Answer Rate",
        "Generated Length Mean",
        "Checkpoint Path",
        "Test Summary Path",
        "Diagnostics Path",
        "Notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    metadata_dir = Path(args.metadata_dir)
    metadata_summary = load_json_if_exists(metadata_dir / "metadata_summary.json")

    output_json = Path(args.output_json) if args.output_json else root_dir / "gsm8k_full_qwen3_1p7b_comparison.json"
    output_csv = Path(args.output_csv) if args.output_csv else root_dir / "gsm8k_full_qwen3_1p7b_comparison.csv"

    rows = [
        collect_row(
            method="Base prompted evaluation",
            train_data="0",
            reward_type="none",
            analyzer_type="none",
            experiment_dir=root_dir / "base_prompted",
            diagnostics_path=None,
            metadata_summary=metadata_summary,
            notes="deterministic official GSM8K full-test evaluation with structured solver prompt",
        ),
        collect_row(
            method="GRPO Answer-only",
            train_data="GSM8K official train full",
            reward_type="answer correctness",
            analyzer_type="none",
            experiment_dir=root_dir / "grpo_answer_only",
            diagnostics_path=None,
            metadata_summary=metadata_summary,
            notes="official GSM8K full-train/full-test",
        ),
        collect_row(
            method="GRPO Bayesian Prompted Analyzer",
            train_data="GSM8K official train full",
            reward_type="full Bayesian posterior",
            analyzer_type="prompted",
            experiment_dir=root_dir / "grpo_bayesian_prompted",
            diagnostics_path=root_dir / "grpo_bayesian_prompted" / "bayesian_reward_diagnostics" / "summary.json",
            metadata_summary=metadata_summary,
            notes="prompted Bayesian baseline; debug logs reused for analyzer data",
        ),
        collect_row(
            method="GRPO Bayesian reward + SFT analyzer",
            train_data="GSM8K official train full",
            reward_type="full Bayesian posterior",
            analyzer_type="learned_sft",
            experiment_dir=root_dir / "grpo_bayesian_sft_analyzer",
            diagnostics_path=root_dir / "grpo_bayesian_sft_analyzer" / "bayesian_reward_diagnostics" / "summary.json",
            metadata_summary=metadata_summary,
            notes="SFT analyzer built from prompted full-train debug logs",
        ),
        collect_row(
            method="GRPO Bayesian reward + SFT+DPO analyzer",
            train_data="GSM8K official train full",
            reward_type="full Bayesian posterior",
            analyzer_type="learned_sft_dpo",
            experiment_dir=root_dir / "grpo_bayesian_sft_dpo_analyzer",
            diagnostics_path=root_dir / "grpo_bayesian_sft_dpo_analyzer" / "bayesian_reward_diagnostics" / "summary.json",
            metadata_summary=metadata_summary,
            notes="DPO analyzer initialized from the new full-train SFT adapter",
        ),
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_dir": str(root_dir),
        "metadata_dir": str(metadata_dir),
        "rows": rows,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return

    write_json(output_json, payload)
    write_csv(output_csv, rows)


if __name__ == "__main__":
    main()
