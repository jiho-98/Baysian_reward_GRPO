#!/usr/bin/env python3
"""Prepare a fixed fair Big-Math train/eval split for large-scale GRPO comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    DEFAULT_DATASET_NAME,
    ensure_output_dir,
    filter_by_solve_rate,
    filter_nonempty_problem_answer,
    metadata_row_from_example,
    sha256_file,
    summarize_selected_train,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed Big-Math train/eval metadata for fair GRPO comparisons."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--eval_size", type=int, required=True)
    parser.add_argument("--min_solve_rate", type=float, default=0.2)
    parser.add_argument("--max_solve_rate", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def problem_preview(problem: str, limit: int = 160) -> str:
    single_line = str(problem or "").replace("\n", " ").strip()
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 3]}..."


def build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    total_before: int,
    after_problem_answer: int,
    after_solve_rate: int,
    rows_scanned: int,
    duplicates_skipped: int,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_path: Path,
    eval_path: Path,
) -> dict[str, Any]:
    train_summary = summarize_selected_train(train_rows)
    eval_summary = summarize_selected_train(eval_rows)
    return {
        "dataset_name": args.dataset_name,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "min_solve_rate": args.min_solve_rate,
        "max_solve_rate": args.max_solve_rate,
        "dataset_size_before_filtering": total_before,
        "dataset_size_after_problem_answer_filtering": after_problem_answer,
        "dataset_size_after_solve_rate_filtering": after_solve_rate,
        "rows_scanned_after_shuffle": rows_scanned,
        "duplicates_skipped_before_fill": duplicates_skipped,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "train_metadata_path": str(train_path),
        "eval_metadata_path": str(eval_path),
        "train_sha256": sha256_file(train_path),
        "eval_sha256": sha256_file(eval_path),
        "train_solve_rate_summary": train_summary["solve_rate_summary"],
        "eval_solve_rate_summary": eval_summary["solve_rate_summary"],
        "train_source_distribution": train_summary["source_distribution"],
        "eval_source_distribution": eval_summary["source_distribution"],
        "train_difficulty_bucket_distribution": train_summary["difficulty_distribution"],
        "eval_difficulty_bucket_distribution": eval_summary["difficulty_distribution"],
        "train_first_5_problem_previews": [problem_preview(row["problem"]) for row in train_rows[:5]],
        "eval_first_5_problem_previews": [problem_preview(row["problem"]) for row in eval_rows[:5]],
    }


def main() -> None:
    args = parse_args()
    if args.train_size <= 0:
        raise SystemExit("--train_size must be positive.")
    if args.eval_size < 0:
        raise SystemExit("--eval_size must be non-negative.")
    if args.min_solve_rate > args.max_solve_rate:
        raise SystemExit("--min_solve_rate must be <= --max_solve_rate.")

    try:
        from datasets import load_dataset
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
    total_needed = args.train_size + args.eval_size
    selected_rows: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    rows_scanned = 0
    duplicates_skipped = 0

    for index in range(len(dataset)):
        row = metadata_row_from_example(dataset[index])
        rows_scanned += 1
        problem = str(row["problem"]).strip()
        if problem in seen_problems:
            duplicates_skipped += 1
            continue
        seen_problems.add(problem)
        selected_rows.append(row)
        if len(selected_rows) >= total_needed:
            break

    if len(selected_rows) < total_needed:
        raise RuntimeError(
            "Not enough unique filtered problems to satisfy the requested fair split: "
            f"needed {total_needed}, found {len(selected_rows)}."
        )

    train_rows = selected_rows[: args.train_size]
    eval_rows = selected_rows[args.train_size : total_needed]
    overlap = {row["problem"] for row in train_rows} & {row["problem"] for row in eval_rows}
    if overlap:
        raise RuntimeError(f"Train/eval overlap detected after selection: {len(overlap)} problems.")

    output_dir = ensure_output_dir(args.output_dir)
    train_path = output_dir / "selected_train_metadata.jsonl"
    eval_path = output_dir / "selected_eval_metadata.jsonl"
    summary_path = output_dir / "metadata_summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        total_before=total_before,
        after_problem_answer=after_problem_answer,
        after_solve_rate=after_solve_rate,
        rows_scanned=rows_scanned,
        duplicates_skipped=duplicates_skipped,
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_path=train_path,
        eval_path=eval_path,
    )
    write_json(summary_path, summary)

    print("[INFO] fair metadata prepared successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
