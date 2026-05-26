#!/usr/bin/env python3
"""Prepare AIME 1983-2024 Kaggle data as JSONL metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    ensure_output_dir,
    normalize_answer_details,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_DATASET_NAME = "hemishveeraboina/aime-problem-set-1983-2024"
DEFAULT_OUTPUT_DIR = "outputs/aime_experiments/metadata_1983_2024_kaggle"
DEFAULT_CSV_NAME = "AIME_Dataset_1983_2024.csv"
PROBLEM_KEYS = ("Question", "Problem", "problem", "question")
ANSWER_KEYS = ("Answer", "answer", "Final Answer", "final_answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle AIME metadata JSONL.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--file_path", default=DEFAULT_CSV_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--local_csv_path",
        default="",
        help="Use an already downloaded CSV instead of downloading through kagglehub.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def clean_answer(value: Any) -> str:
    text = clean_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    normalized = normalize_answer_details(text).normalized
    return normalized or text


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def clean_problem(value: Any) -> str:
    text = clean_text(value)
    if text.endswith(" Solution"):
        text = text[: -len(" Solution")].rstrip()
    return text


def parse_year_problem_number(aime_id: str, record: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    year_text = clean_text(record.get("Year"))
    number_text = clean_text(record.get("Problem Number"))
    part = clean_text(record.get("Part")) or None

    year = int(float(year_text)) if year_text else None
    problem_number = int(float(number_text)) if number_text else None
    if year is not None and problem_number is not None:
        return year, problem_number, part

    pieces = aime_id.replace("-", "_").split("_")
    if pieces and pieces[0].isdigit():
        year = int(pieces[0])
    if pieces and pieces[-1].isdigit():
        problem_number = int(pieces[-1])
    if len(pieces) >= 2:
        part_candidate = "_".join(pieces[1:-1])
        part = part or (part_candidate if part_candidate else None)
    return year, problem_number, part


def load_dataframe(args: argparse.Namespace):
    import pandas as pd

    if str(args.local_csv_path or "").strip():
        return pd.read_csv(args.local_csv_path)

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required. Install with `pip install kagglehub[pandas-datasets]`."
        ) from exc

    dataset_dir = Path(kagglehub.dataset_download(args.dataset_name))
    csv_path = dataset_dir / args.file_path
    if not csv_path.exists():
        available = sorted(str(path.relative_to(dataset_dir)) for path in dataset_dir.rglob("*") if path.is_file())
        raise RuntimeError(
            f"Could not find {args.file_path!r} in {dataset_dir}. Available files: {available}"
        )
    return pd.read_csv(csv_path)


def row_from_record(record: dict[str, Any], row_index: int) -> dict[str, Any]:
    problem = clean_problem(first_present(record, PROBLEM_KEYS))
    answer = clean_answer(first_present(record, ANSWER_KEYS))
    if not problem:
        raise RuntimeError(f"Empty AIME question at row {row_index}")
    if not answer:
        raise RuntimeError(f"Empty AIME answer at row {row_index}")

    aime_id = clean_text(record.get("ID")) or f"aime-row-{row_index}"
    year, problem_number, part = parse_year_problem_number(aime_id, record)

    return {
        "problem": problem,
        "answer": answer,
        "source": "aime",
        "domain": "math",
        "difficulty_bucket": "aime",
        "llama8b_solve_rate": None,
        "raw_answer": first_present(record, ANSWER_KEYS),
        "aime_id": aime_id,
        "year": year,
        "problem_number": problem_number,
        "part": part,
    }


def main() -> None:
    args = parse_args()
    df = load_dataframe(args)

    has_problem = any(key in df.columns for key in PROBLEM_KEYS)
    has_answer = any(key in df.columns for key in ANSWER_KEYS)
    missing = []
    if not has_problem:
        missing.append(f"one of {PROBLEM_KEYS}")
    if not has_answer:
        missing.append(f"one of {ANSWER_KEYS}")
    if missing:
        raise RuntimeError(f"Missing required AIME columns: {missing}; got columns={list(df.columns)}")

    rows = [row_from_record(record, idx) for idx, record in enumerate(df.to_dict("records"), start=1)]
    output_dir = ensure_output_dir(args.output_dir)

    metadata_path = output_dir / "aime_1983_2024_metadata.jsonl"
    selected_test_path = output_dir / "selected_test_metadata.jsonl"
    write_jsonl(metadata_path, rows)
    write_jsonl(selected_test_path, rows)

    years = [row["year"] for row in rows if row.get("year") is not None]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "file_path": args.file_path,
        "num_examples": len(rows),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "metadata_path": str(metadata_path),
        "selected_test_metadata_path": str(selected_test_path),
        "metadata_sha256": sha256_file(metadata_path),
        "selected_test_sha256": sha256_file(selected_test_path),
        "first_5_ids": [row["aime_id"] for row in rows[:5]],
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "metadata_summary.json", summary)

    print("[INFO] prepared AIME metadata successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
