#!/usr/bin/env python3
"""Prepare a contamination-filtered Big-Math rollout prompt pool."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    DEFAULT_DATASET_NAME,
    ensure_output_dir,
    metadata_row_from_example,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = "outputs/bigmath_rollout_pools/no_gsm8k_math_eval_overlap_seed42_b12"
DEFAULT_EVAL_METADATA_PATHS = [
    "outputs/gsm8k_experiments/metadata_fulltrain_seed42/selected_test_metadata.jsonl",
    "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered Big-Math rollout prompt batches."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=12)
    parser.add_argument(
        "--eval_metadata_path",
        action="append",
        default=None,
        help=(
            "Evaluation metadata JSONL to remove from the pool. "
            "May be passed multiple times. Defaults to GSM8K test and MATH-500 test."
        ),
    )
    return parser.parse_args()


def normalize_problem_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    return normalized


def has_nonempty_problem_answer(raw_row: dict[str, Any]) -> bool:
    return bool(str(raw_row.get("problem", "") or "").strip()) and bool(
        str(raw_row.get("answer", "") or "").strip()
    )


def source_is_excluded(source: str) -> bool:
    normalized = str(source or "").lower().strip()
    return "gsm8k" in normalized or normalized in {"math", "hendrycks_math", "math-500", "math500"}


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def load_eval_problem_sets(paths: list[str]) -> tuple[set[str], set[str], dict[str, int]]:
    exact: set[str] = set()
    normalized: set[str] = set()
    counts: dict[str, int] = {}
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise RuntimeError(f"Evaluation metadata path does not exist: {path}")
        rows = load_jsonl_rows(path)
        counts[str(path)] = len(rows)
        for row in rows:
            problem = str(row.get("problem", "") or "")
            if problem:
                exact.add(problem)
                normalized.add(normalize_problem_text(problem))
    return exact, normalized, counts


def problem_preview(problem: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(problem or "")).strip()
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def source_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source", "") or "unknown") for row in rows))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive.")
    if args.num_batches <= 0:
        raise SystemExit("--num_batches must be positive.")

    eval_paths = args.eval_metadata_path or DEFAULT_EVAL_METADATA_PATHS
    eval_exact, eval_normalized, eval_counts = load_eval_problem_sets(eval_paths)
    target_count = args.batch_size * args.num_batches

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install it with `pip install datasets`.") from exc

    dataset = load_dataset(args.dataset_name, split="train")
    total_before = len(dataset)
    indexes = list(range(total_before))
    random.Random(args.seed).shuffle(indexes)

    selected: list[dict[str, Any]] = []
    seen_normalized: set[str] = set()
    counters: Counter[str] = Counter()

    for index in indexes:
        raw_row = dict(dataset[index])
        counters["scanned_after_shuffle"] += 1

        if not has_nonempty_problem_answer(raw_row):
            counters["dropped_empty_problem_or_answer"] += 1
            continue

        row = metadata_row_from_example(raw_row)
        source = str(row.get("source", "") or "")
        if source_is_excluded(source):
            counters["dropped_source_gsm8k_or_math"] += 1
            continue

        problem = str(row.get("problem", "") or "").strip()
        normalized_problem = normalize_problem_text(problem)
        if problem in eval_exact or normalized_problem in eval_normalized:
            counters["dropped_eval_overlap"] += 1
            continue

        if normalized_problem in seen_normalized:
            counters["dropped_duplicate_normalized_problem"] += 1
            continue

        seen_normalized.add(normalized_problem)
        selected.append(row)
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        raise RuntimeError(
            f"Not enough filtered Big-Math rows: needed {target_count}, selected {len(selected)}."
        )

    output_dir = ensure_output_dir(args.output_dir)
    selected_path = output_dir / "selected_rollout_prompts.jsonl"
    train_path = output_dir / "selected_train_metadata.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(train_path, selected)

    batch_paths: list[str] = []
    for batch_index in range(args.num_batches):
        start = batch_index * args.batch_size
        end = start + args.batch_size
        batch_rows = selected[start:end]
        batch_path = output_dir / f"rollout_batch_{batch_index:02d}.jsonl"
        write_jsonl(batch_path, batch_rows)
        batch_paths.append(str(batch_path))

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "target_count": target_count,
        "selected_count": len(selected),
        "dataset_size_before_filtering": total_before,
        "filtering": {
            "keep_nonempty_problem_answer": True,
            "remove_gsm8k_source": True,
            "remove_math_source": True,
            "remove_eval_exact_or_normalized_problem_overlap": True,
            "dedup_by_normalized_problem_text": True,
            "shuffle_seed": args.seed,
        },
        "eval_metadata_counts": eval_counts,
        "counts": dict(counters),
        "source_distribution": source_distribution(selected),
        "output_dir": str(output_dir),
        "selected_rollout_prompts_path": str(selected_path),
        "selected_train_metadata_path": str(train_path),
        "batch_paths": batch_paths,
        "selected_rollout_prompts_sha256": sha256_file(selected_path),
        "selected_train_sha256": sha256_file(train_path),
        "batch_sha256": {path: sha256_file(Path(path)) for path in batch_paths},
        "first_5_problem_previews": [problem_preview(row["problem"]) for row in selected[:5]],
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "metadata_summary.json", summary)

    print("[INFO] prepared Big-Math rollout pool successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
