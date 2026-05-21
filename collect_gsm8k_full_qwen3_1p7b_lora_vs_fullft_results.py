#!/usr/bin/env python3
"""Compare LoRA vs full-finetune GSM8K full-train Qwen3-1.7B results."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LORA_ROOT = "outputs/gsm8k_full_qwen3_1p7b"
DEFAULT_FULLFT_ROOT = "outputs/gsm8k_full_qwen3_1p7b_fullft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA and full-finetune Qwen3 GSM8K full-train result tables."
    )
    parser.add_argument("--lora_root", default=DEFAULT_LORA_ROOT)
    parser.add_argument("--fullft_root", default=DEFAULT_FULLFT_ROOT)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_rows(root: Path) -> list[dict[str, Any]]:
    payload_path = root / "gsm8k_full_qwen3_1p7b_comparison.json"
    if not payload_path.exists():
        return []
    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = list(payload.get("rows", []))
    for row in rows:
        row["Experiment Root"] = str(root)
    return rows


def infer_mode(row: dict[str, Any], fallback: str) -> str:
    training_mode = row.get("Training Mode")
    if training_mode:
        return str(training_mode)
    use_lora = row.get("Use LoRA")
    if use_lora is True:
        return "lora_grpo"
    if use_lora is False:
        return "full_finetune_grpo"
    return fallback


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "Method",
        "Training Mode",
        "Model",
        "Train Data",
        "Train Size",
        "Test Size",
        "Reward Type",
        "Analyzer Type",
        "Test Accuracy",
        "Correct/Test Total",
        "Format Success Rate",
        "Generated Length Mean",
        "Checkpoint Path",
        "Diagnostics Path",
        "Experiment Root",
        "Notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    lora_root = Path(args.lora_root)
    fullft_root = Path(args.fullft_root)

    rows = []
    for row in load_rows(lora_root):
        row["Training Mode"] = infer_mode(row, "lora_grpo")
        rows.append(row)
    for row in load_rows(fullft_root):
        row["Training Mode"] = infer_mode(row, "full_finetune_grpo")
        rows.append(row)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lora_root": str(lora_root),
        "fullft_root": str(fullft_root),
        "rows": rows,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return

    output_json = (
        Path(args.output_json)
        if args.output_json
        else lora_root / "gsm8k_full_qwen3_1p7b_lora_vs_fullft.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else lora_root / "gsm8k_full_qwen3_1p7b_lora_vs_fullft.csv"
    )
    write_json(output_json, payload)
    write_csv(output_csv, rows)


if __name__ == "__main__":
    main()
