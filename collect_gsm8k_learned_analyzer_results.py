#!/usr/bin/env python3
"""Collect GSM8K comparison results including learned-analyzer runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "outputs/gsm8k_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect GSM8K result comparisons for 3K or full-train settings."
    )
    parser.add_argument("--experiments_root", default=DEFAULT_ROOT)
    parser.add_argument("--setting", choices=["3k", "fulltrain"], default="3k")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    parser.add_argument("--base_dir", default="")
    parser.add_argument("--answer_only_dir", default="")
    parser.add_argument("--prompted_dir", default="")
    parser.add_argument("--learned_sft_dir", default="")
    parser.add_argument("--learned_sft_dpo_dir", default="")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def default_paths(root: Path, setting: str) -> dict[str, Path | None]:
    if setting == "fulltrain":
        return {
            "output_json": root / "gsm8k_fulltrain_comparison.json",
            "output_md": root / "gsm8k_fulltrain_comparison.md",
            "base_dir": root / "base_qwen3b",
            "answer_only_dir": root / "grpo_answer_only_qwen3b_fulltrain_n8_steps500",
            "prompted_dir": None,
            "learned_sft_dir": root / "grpo_bayesian_sft_analyzer_qwen3b_fulltrain_n8_steps500_lambda10",
            "learned_sft_dpo_dir": root / "grpo_bayesian_sft_dpo_analyzer_qwen3b_fulltrain_n8_steps500_lambda10",
        }
    return {
        "output_json": root / "gsm8k_learned_analyzer_comparison.json",
        "output_md": root / "gsm8k_learned_analyzer_comparison.md",
        "base_dir": root / "base_qwen3b",
        "answer_only_dir": root / "grpo_answer_only_qwen3b_train3000_n8_steps500",
        "prompted_dir": root / "grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10",
        "learned_sft_dir": root / "grpo_bayesian_sft_analyzer_qwen3b_train3000_n8_steps500_lambda10",
        "learned_sft_dpo_dir": root / "grpo_bayesian_sft_dpo_analyzer_qwen3b_train3000_n8_steps500_lambda10",
    }


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def accuracy(summary: dict[str, Any] | None) -> float | None:
    if summary is None:
        return None
    value = summary.get("accuracy")
    if value is None:
        value = summary.get("test_accuracy")
    if value is None:
        value = summary.get("valid_accuracy")
    if value is None:
        return None
    return float(value)


def correct_total(summary: dict[str, Any] | None) -> str | None:
    if summary is None:
        return None
    value = summary.get("correct_over_total")
    if value is not None:
        return str(value)
    correct = summary.get("correct")
    if correct is None:
        correct = summary.get("correct_count")
    total = summary.get("test_total")
    if total is None:
        total = summary.get("num_examples")
    if correct is None or total is None:
        return None
    return f"{int(correct)}/{int(total)}"


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def root_summary_path(experiment_dir: Path) -> Path:
    return experiment_dir / "summary.json"


def collect_row(
    *,
    method: str,
    train_data: str,
    reward: str,
    analyzer_type: str,
    checkpoint_dir: Path,
    valid_summary_path: Path | None,
    test_summary_path: Path | None,
    notes: str,
) -> dict[str, Any]:
    experiment_summary_path = root_summary_path(checkpoint_dir)
    experiment_summary = load_json_if_exists(experiment_summary_path)
    valid_summary = load_json_if_exists(valid_summary_path)
    test_summary = load_json_if_exists(test_summary_path)

    if experiment_summary is not None:
        row = {
            "Method": experiment_summary.get("method", method),
            "Train Data": experiment_summary.get("train_data", train_data),
            "Valid Accuracy": experiment_summary.get("valid_accuracy"),
            "Test Accuracy": experiment_summary.get("test_accuracy"),
            "Correct/Test Total": experiment_summary.get("correct_over_total"),
            "Reward": experiment_summary.get("reward", reward),
            "Reward Type": experiment_summary.get("reward_type", experiment_summary.get("reward", reward)),
            "Analyzer Type": experiment_summary.get("analyzer_type", analyzer_type),
            "Notes": experiment_summary.get("notes", notes),
            "checkpoint_path": experiment_summary.get("checkpoint_path", str(checkpoint_dir)),
            "experiment_summary_path": str(experiment_summary_path),
            "valid_summary_path": experiment_summary.get("valid_summary_path"),
            "test_summary_path": experiment_summary.get("test_summary_path"),
        }
        if row["Valid Accuracy"] is None:
            row["Valid Accuracy"] = accuracy(valid_summary)
        if row["Test Accuracy"] is None:
            row["Test Accuracy"] = accuracy(test_summary)
        if row["Correct/Test Total"] is None:
            row["Correct/Test Total"] = correct_total(test_summary)
        if row["valid_summary_path"] is None and valid_summary_path is not None:
            row["valid_summary_path"] = str(valid_summary_path)
        if row["test_summary_path"] is None and test_summary_path is not None:
            row["test_summary_path"] = str(test_summary_path)
        return row

    return {
        "Method": method,
        "Train Data": train_data,
        "Valid Accuracy": accuracy(valid_summary),
        "Test Accuracy": accuracy(test_summary),
        "Correct/Test Total": correct_total(test_summary),
        "Reward": reward,
        "Reward Type": reward,
        "Analyzer Type": analyzer_type,
        "Notes": notes,
        "checkpoint_path": str(checkpoint_dir),
        "experiment_summary_path": str(experiment_summary_path) if experiment_summary_path.exists() else None,
        "valid_summary_path": str(valid_summary_path) if valid_summary_path is not None else None,
        "test_summary_path": str(test_summary_path) if test_summary_path is not None else None,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Method",
        "Train Data",
        "Reward",
        "Analyzer Type",
        "Valid Accuracy",
        "Test Accuracy",
        "Correct/Test Total",
        "Notes",
        "checkpoint_path",
        "test_summary_path",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["Method"]),
                    str(item["Train Data"]),
                    str(item["Reward"]),
                    str(item["Analyzer Type"]),
                    fmt(item["Valid Accuracy"]),
                    fmt(item["Test Accuracy"]),
                    str(item["Correct/Test Total"] or "NA"),
                    str(item["Notes"]),
                    str(item["checkpoint_path"]),
                    str(item["test_summary_path"] or "NA"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.experiments_root)
    defaults = default_paths(root, args.setting)

    output_json = Path(args.output_json) if args.output_json else Path(defaults["output_json"])
    output_md = Path(args.output_md) if args.output_md else Path(defaults["output_md"])
    base_dir = Path(args.base_dir) if args.base_dir else Path(defaults["base_dir"])
    answer_only_dir = Path(args.answer_only_dir) if args.answer_only_dir else Path(defaults["answer_only_dir"])
    prompted_dir = Path(args.prompted_dir) if args.prompted_dir else (
        Path(defaults["prompted_dir"]) if defaults["prompted_dir"] is not None else None
    )
    learned_sft_dir = Path(args.learned_sft_dir) if args.learned_sft_dir else Path(defaults["learned_sft_dir"])
    learned_sft_dpo_dir = (
        Path(args.learned_sft_dpo_dir)
        if args.learned_sft_dpo_dir
        else Path(defaults["learned_sft_dpo_dir"])
    )

    rows = [
        collect_row(
            method="Base Qwen2.5-3B-Instruct",
            train_data="0",
            reward="none",
            analyzer_type="none",
            checkpoint_dir=base_dir,
            valid_summary_path=(base_dir / "valid" / "summary.json") if args.setting == "3k" else None,
            test_summary_path=base_dir / "test" / "summary.json",
            notes="deterministic base eval reused on official GSM8K test",
        ),
        collect_row(
            method="GRPO Answer-only on GSM8K-3K"
            if args.setting == "3k"
            else "GRPO Answer-only fulltrain",
            train_data="GSM8K-3K" if args.setting == "3k" else "GSM8K official train full",
            reward="answer correctness",
            analyzer_type="none",
            checkpoint_dir=answer_only_dir,
            valid_summary_path=(
                answer_only_dir / "valid" / "summary.json"
                if args.setting == "3k"
                else None
            ),
            test_summary_path=answer_only_dir / "test" / "summary.json",
            notes="deterministic official-test evaluation",
        ),
    ]

    if prompted_dir is not None:
        rows.append(
            collect_row(
                method="GRPO Bayesian Prompted Analyzer on GSM8K-3K",
                train_data="GSM8K-3K",
                reward="full Bayesian posterior",
                analyzer_type="prompted",
                checkpoint_dir=prompted_dir,
                valid_summary_path=prompted_dir / "valid" / "summary.json",
                test_summary_path=prompted_dir / "test" / "summary.json",
                notes="prompted Bayesian baseline",
            )
        )

    rows.extend(
        [
            collect_row(
                method="GRPO Bayesian Learned SFT Analyzer on GSM8K-3K"
                if args.setting == "3k"
                else "GRPO Bayesian SFT Analyzer fulltrain",
                train_data="GSM8K-3K" if args.setting == "3k" else "GSM8K official train full",
                reward="full Bayesian posterior",
                analyzer_type="learned_sft",
                checkpoint_dir=learned_sft_dir,
                valid_summary_path=(
                    learned_sft_dir / "valid" / "summary.json"
                    if args.setting == "3k"
                    else None
                ),
                test_summary_path=learned_sft_dir / "test" / "summary.json",
                notes="learned SFT analyzer",
            ),
            collect_row(
                method="GRPO Bayesian Learned SFT+DPO Analyzer on GSM8K-3K"
                if args.setting == "3k"
                else "GRPO Bayesian SFT+DPO Analyzer fulltrain",
                train_data="GSM8K-3K" if args.setting == "3k" else "GSM8K official train full",
                reward="full Bayesian posterior",
                analyzer_type="learned_sft_dpo",
                checkpoint_dir=learned_sft_dpo_dir,
                valid_summary_path=(
                    learned_sft_dpo_dir / "valid" / "summary.json"
                    if args.setting == "3k"
                    else None
                ),
                test_summary_path=learned_sft_dpo_dir / "test" / "summary.json",
                notes="learned SFT+DPO analyzer",
            ),
        ]
    )

    if args.setting == "fulltrain":
        for row in rows:
            row["Valid Accuracy"] = None
            row["valid_summary_path"] = None

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "setting": args.setting,
        "experiments_root": str(root),
        "rows": rows,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    write_json(output_json, payload)
    write_text(output_md, markdown(rows))


if __name__ == "__main__":
    main()
