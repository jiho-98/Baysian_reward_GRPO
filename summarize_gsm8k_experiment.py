#!/usr/bin/env python3
"""Write a compact root-level summary for a GSM8K experiment directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gsm8k_learned_analyzer_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a root-level summary.json for a GSM8K experiment."
    )
    parser.add_argument("--experiment_output_dir", required=True)
    parser.add_argument("--metadata_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--analyzer_type", default="none")
    parser.add_argument("--notes", default="")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--valid_summary_path", default="")
    parser.add_argument("--test_summary_path", required=True)
    parser.add_argument("--summary_path", default="")
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the resolved summary payload without writing it.",
    )
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def metadata_summary_path_for(metadata_dir: Path) -> Path:
    summary_path = metadata_dir / "summary.json"
    if summary_path.exists():
        return summary_path
    return metadata_dir / "metadata_summary.json"


def accuracy(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    value = summary.get("accuracy")
    if value is None:
        return None
    return float(value)


def correct(summary: dict[str, Any] | None) -> int | None:
    if summary is None:
        return None
    value = summary.get("correct")
    if value is None:
        value = summary.get("correct_count")
    if value is None:
        return None
    return int(value)


def total(summary: dict[str, Any] | None) -> int | None:
    if summary is None:
        return None
    value = summary.get("num_examples")
    if value is None:
        return None
    return int(value)


def correct_over_total(summary: dict[str, Any] | None) -> str | None:
    correct_count = correct(summary)
    total_count = total(summary)
    if correct_count is None or total_count is None:
        return None
    return f"{correct_count}/{total_count}"


def optional_float(summary: dict[str, Any] | None, key: str) -> float | None:
    if summary is None:
        return None
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def optional_str(summary: dict[str, Any] | None, key: str) -> str | None:
    if summary is None:
        return None
    value = summary.get(key)
    if value is None:
        return None
    return str(value)


def main() -> None:
    args = parse_args()
    experiment_output_dir = Path(args.experiment_output_dir)
    metadata_dir = Path(args.metadata_dir)
    metadata_summary_path = metadata_summary_path_for(metadata_dir)
    valid_summary_path = Path(args.valid_summary_path) if args.valid_summary_path else None
    test_summary_path = Path(args.test_summary_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else experiment_output_dir / "summary.json"
    )
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else experiment_output_dir

    metadata_summary = load_json_if_exists(metadata_summary_path)
    valid_summary = load_json_if_exists(valid_summary_path) if valid_summary_path else None
    test_summary = load_json_if_exists(test_summary_path)
    training_config_path = first_existing_path(
        [
            experiment_output_dir / "training_config.json",
            experiment_output_dir / "launcher_config.json",
        ]
    )
    training_config = load_json_if_exists(training_config_path) if training_config_path else None

    if test_summary is None:
        raise RuntimeError(f"Missing test summary: {test_summary_path}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "train_data": args.train_data,
        "reward": args.reward,
        "reward_type": args.reward,
        "analyzer_type": args.analyzer_type,
        "notes": args.notes,
        "experiment_output_dir": str(experiment_output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metadata_dir": str(metadata_dir),
        "metadata_summary_path": str(metadata_summary_path),
        "valid_summary_path": str(valid_summary_path) if valid_summary_path else None,
        "test_summary_path": str(test_summary_path),
        "train_size": metadata_summary.get("train_size") if metadata_summary else None,
        "valid_size": metadata_summary.get("valid_size") if metadata_summary else None,
        "test_size": metadata_summary.get("test_size") if metadata_summary else total(test_summary),
        "train_size_full": metadata_summary.get("train_size_full") if metadata_summary else None,
        "test_size_full": metadata_summary.get("test_size_full") if metadata_summary else None,
        "valid_accuracy": accuracy(valid_summary),
        "test_accuracy": accuracy(test_summary),
        "correct": correct(test_summary),
        "test_total": total(test_summary),
        "correct_over_total": correct_over_total(test_summary),
        "metadata_setting": metadata_summary.get("setting") if metadata_summary else None,
        "model_name": optional_str(test_summary, "model_name"),
        "valid_format_success_rate": optional_float(valid_summary, "format_success_rate"),
        "test_format_success_rate": optional_float(test_summary, "format_success_rate"),
        "valid_suspicious_final_answer_rate": optional_float(
            valid_summary,
            "suspicious_final_answer_rate",
        ),
        "test_suspicious_final_answer_rate": optional_float(
            test_summary,
            "suspicious_final_answer_rate",
        ),
        "valid_generated_length_mean": optional_float(valid_summary, "generated_length_mean"),
        "test_generated_length_mean": optional_float(test_summary, "generated_length_mean"),
        "valid_predictions_path": optional_str(valid_summary, "predictions_path"),
        "test_predictions_path": optional_str(test_summary, "predictions_path"),
        "training_config_path": str(training_config_path) if training_config_path else None,
        "use_lora": training_config.get("use_lora") if training_config else None,
        "gradient_checkpointing": (
            training_config.get("gradient_checkpointing") if training_config else None
        ),
        "num_generations": training_config.get("num_generations") if training_config else None,
        "max_steps": training_config.get("max_steps") if training_config else None,
        "training_mode": (
            "full_finetune_grpo"
            if training_config and training_config.get("use_lora") is False
            else "lora_grpo"
            if training_config and training_config.get("use_lora") is True
            else "base_eval"
        ),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return
    write_json(summary_path, payload)


if __name__ == "__main__":
    main()
