#!/usr/bin/env python3
"""Prepare BARL-style Big-Math rollout batches for cross-domain math evals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_DATASET = "SynthLabsAI/Big-Math-RL-Verified"
EVAL_DATASETS = [
    "math-ai/olympiadbench",
    "math-ai/minervamath",
    "math-ai/aime26",
]
SAMPLING_PROTOCOL = "barl_style_rollout_batch_sampling"
EXCLUDED_SOURCES = ["amc_aime"]
PROBLEM_KEYS = (
    "problem",
    "question",
    "prompt",
    "input",
    "problem_text",
    "Question",
    "Problem",
)
ANSWER_KEYS = (
    "answer",
    "final_answer",
    "gold_answer",
    "ground_truth",
    "target",
    "label",
    "Answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create BARL-style filtered Big-Math rollout batches."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--num_rollout_batches", type=int, default=12)
    return parser.parse_args()


def normalize_problem_text(text: str) -> str:
    """Conservative deterministic normalization for exact/near-exact overlap checks."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.lower().strip()
    normalized = normalized.replace("\r", "\n").replace("\t", " ")
    normalized = re.sub(r"\\(?:,|!|;|:)", "", normalized)
    normalized = normalized.replace("\\ ", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def first_present(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            value = clean_text(row.get(key))
            if value:
                return value
    return ""


def source_label(row: dict[str, Any]) -> str:
    return clean_text(row.get("source")) or "unknown"


def source_is_amc_aime(source: str) -> bool:
    return source.strip().lower() == "amc_aime"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
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


def dataset_splits(dataset: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset, "items"):
        return [(str(name), split) for name, split in dataset.items()]
    return [("train", dataset)]


def extract_eval_problem(row: dict[str, Any]) -> str:
    problem = first_present(row, PROBLEM_KEYS)
    if problem:
        return problem
    for value in row.values():
        if isinstance(value, dict):
            nested = extract_eval_problem(value)
            if nested:
                return nested
    return ""


def load_eval_normalized_problems(load_dataset: Any) -> tuple[set[str], dict[str, Any]]:
    normalized_problems: set[str] = set()
    summary: dict[str, Any] = {}
    for dataset_name in EVAL_DATASETS:
        dataset = load_dataset(dataset_name)
        dataset_count = 0
        split_counts: dict[str, int] = {}
        missing_problem = 0
        for split_name, split in dataset_splits(dataset):
            split_count = 0
            for example in split:
                problem = extract_eval_problem(dict(example))
                if not problem:
                    missing_problem += 1
                    continue
                normalized = normalize_problem_text(problem)
                if normalized:
                    normalized_problems.add(normalized)
                    split_count += 1
            split_counts[split_name] = split_count
            dataset_count += split_count
        summary[dataset_name] = {
            "num_problem_texts": dataset_count,
            "split_counts": split_counts,
            "missing_problem_rows": missing_problem,
        }
    return normalized_problems, summary


def bigmath_metadata_row(
    *,
    raw_row: dict[str, Any],
    original_idx: int,
    global_idx: int,
    rollout_batch_id: int,
    within_batch_idx: int,
    normalized_problem: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "problem": first_present(raw_row, PROBLEM_KEYS),
        "answer": first_present(raw_row, ANSWER_KEYS),
        "source": source_label(raw_row),
        "domain": clean_text(raw_row.get("domain")),
        "llama8b_solve_rate": raw_row.get("llama8b_solve_rate"),
        "difficulty_bucket": clean_text(raw_row.get("difficulty_bucket")),
        "original_idx": original_idx,
        "normalized_problem": normalized_problem,
        "rollout_batch_id": rollout_batch_id,
        "within_batch_idx": within_batch_idx,
        "global_idx": global_idx,
        "sampling_seed": args.seed,
        "rollout_batch_size": args.rollout_batch_size,
        "num_rollout_batches": args.num_rollout_batches,
        "total_prompt_budget": args.rollout_batch_size * args.num_rollout_batches,
        "sampling_protocol": SAMPLING_PROTOCOL,
        "source_dataset": SOURCE_DATASET,
    }


def compact_removed(row: dict[str, Any], original_idx: int, reason: str) -> dict[str, Any]:
    problem = first_present(row, PROBLEM_KEYS)
    return {
        "original_idx": original_idx,
        "source": source_label(row),
        "reason": reason,
        "normalized_problem": normalize_problem_text(problem),
        "problem_preview": re.sub(r"\s+", " ", problem).strip()[:240],
    }


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def validate_outputs(
    *,
    output_dir: Path,
    selected_path: Path,
    train_path: Path,
    batch_dir: Path,
    eval_normalized_problems: set[str],
    args: argparse.Namespace,
) -> None:
    target_count = args.rollout_batch_size * args.num_rollout_batches
    assert line_count(train_path) == target_count

    selected_sha = sha256_file(selected_path)
    train_sha = sha256_file(train_path)
    assert selected_sha == train_sha

    rows = load_rows_from_jsonl(train_path)
    batch_ids = {int(row["rollout_batch_id"]) for row in rows}
    assert batch_ids == set(range(args.num_rollout_batches))

    normalized = [str(row["normalized_problem"]) for row in rows]
    assert len(normalized) == len(set(normalized))
    assert not (set(normalized) & eval_normalized_problems)

    for batch_id in range(args.num_rollout_batches):
        batch_path = batch_dir / f"rollout_batch_{batch_id:02d}.jsonl"
        assert line_count(batch_path) == args.rollout_batch_size

    print("[VALIDATION] selected_train_metadata rows:", line_count(train_path))
    print("[VALIDATION] selected/train sha256:", selected_sha)
    print("[VALIDATION] rollout_batch_id range:", min(batch_ids), max(batch_ids))
    print("[VALIDATION] no eval overlap and no train duplicate normalized_problem")
    print("[VALIDATION] final source distribution:")
    print(json.dumps(Counter(row["source"] for row in rows), ensure_ascii=False, indent=2, sort_keys=True))
    print("[VALIDATION] first 3 examples:")
    print(json.dumps(rows[:3], ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.rollout_batch_size <= 0:
        raise SystemExit("--rollout_batch_size must be positive.")
    if args.num_rollout_batches <= 0:
        raise SystemExit("--num_rollout_batches must be positive.")

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: The `datasets` package is required. Install it with `pip install datasets` "
            "or install this project's requirements.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    batch_dir = output_dir / "rollout_batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    target_count = args.rollout_batch_size * args.num_rollout_batches
    eval_normalized_problems, eval_summary = load_eval_normalized_problems(load_dataset)

    dataset = load_dataset(SOURCE_DATASET, split="train")
    num_candidates_before_filter = len(dataset)
    source_distribution_before_filter: Counter[str] = Counter()
    for example in dataset:
        source_distribution_before_filter[source_label(dict(example))] += 1

    indexes = list(range(num_candidates_before_filter))
    random.Random(args.seed).shuffle(indexes)

    selected_rows: list[dict[str, Any]] = []
    selected_normalized: set[str] = set()
    source_distribution_after_filter: Counter[str] = Counter()
    source_distribution_selected: Counter[str] = Counter()
    excluded_sources: list[dict[str, Any]] = []
    eval_overlap_removed: list[dict[str, Any]] = []
    train_dedup_removed: list[dict[str, Any]] = []

    num_removed_empty = 0
    num_removed_source_amc_aime = 0
    num_removed_eval_overlap = 0
    num_removed_train_duplicate = 0

    for original_idx in indexes:
        raw_row = dict(dataset[original_idx])
        problem = first_present(raw_row, PROBLEM_KEYS)
        answer = first_present(raw_row, ANSWER_KEYS)
        source = source_label(raw_row)

        if not problem or not answer:
            num_removed_empty += 1
            continue

        if source_is_amc_aime(source):
            num_removed_source_amc_aime += 1
            excluded_sources.append(compact_removed(raw_row, original_idx, "source_amc_aime"))
            continue

        normalized_problem = normalize_problem_text(problem)
        if normalized_problem in eval_normalized_problems:
            num_removed_eval_overlap += 1
            eval_overlap_removed.append(compact_removed(raw_row, original_idx, "eval_overlap"))
            continue

        if normalized_problem in selected_normalized:
            num_removed_train_duplicate += 1
            train_dedup_removed.append(compact_removed(raw_row, original_idx, "train_duplicate"))
            continue

        source_distribution_after_filter[source] += 1

        if len(selected_rows) < target_count:
            global_idx = len(selected_rows)
            rollout_batch_id = global_idx // args.rollout_batch_size
            within_batch_idx = global_idx % args.rollout_batch_size
            selected_row = bigmath_metadata_row(
                raw_row=raw_row,
                original_idx=original_idx,
                global_idx=global_idx,
                rollout_batch_id=rollout_batch_id,
                within_batch_idx=within_batch_idx,
                normalized_problem=normalized_problem,
                args=args,
            )
            selected_rows.append(selected_row)
            selected_normalized.add(normalized_problem)
            source_distribution_selected[source] += 1

        if len(selected_rows) >= target_count:
            break

    if len(selected_rows) != target_count:
        raise RuntimeError(f"Selected {len(selected_rows)} examples, expected {target_count}.")

    selected_path = output_dir / "selected_rollout_prompts.jsonl"
    train_path = output_dir / "selected_train_metadata.jsonl"
    write_jsonl(selected_path, selected_rows)
    write_jsonl(train_path, selected_rows)

    batch_paths: list[str] = []
    for batch_id in range(args.num_rollout_batches):
        start = batch_id * args.rollout_batch_size
        end = start + args.rollout_batch_size
        batch_path = batch_dir / f"rollout_batch_{batch_id:02d}.jsonl"
        write_jsonl(batch_path, selected_rows[start:end])
        batch_paths.append(str(batch_path))

    write_json(output_dir / "source_distribution_before_filter.json", dict(source_distribution_before_filter))
    write_json(output_dir / "source_distribution_after_filter.json", dict(source_distribution_after_filter))
    write_json(output_dir / "source_distribution_selected.json", dict(source_distribution_selected))
    write_json(output_dir / "excluded_sources.json", excluded_sources)
    write_json(output_dir / "eval_overlap_removed.json", eval_overlap_removed)
    write_json(output_dir / "train_dedup_removed.json", train_dedup_removed)

    selected_sha = sha256_file(selected_path)
    train_sha = sha256_file(train_path)
    summary = {
        "source_dataset": SOURCE_DATASET,
        "sampling_protocol": SAMPLING_PROTOCOL,
        "rollout_batch_size": args.rollout_batch_size,
        "num_rollout_batches": args.num_rollout_batches,
        "total_prompt_budget": target_count,
        "sampling_seed": args.seed,
        "sampling_without_replacement": True,
        "source_level_excluded_sources": EXCLUDED_SOURCES,
        "kept_sources_note": (
            "GSM8K and MATH sources are kept because this experiment evaluates only "
            "on OlympiadBench, MinervaMath, and AIME26."
        ),
        "eval_sets_deduped_against": EVAL_DATASETS,
        "eval_dataset_problem_counts": eval_summary,
        "dedup_key": "normalized_problem_text",
        "train_internal_dedup": True,
        "eval_overlap_removed": True,
        "selected_rollout_prompts_path": str(selected_path),
        "selected_train_metadata_path": str(train_path),
        "selected_train_metadata_is_alias_of_selected_rollout_prompts": True,
        "rollout_batch_dir": str(batch_dir),
        "rollout_batch_paths": batch_paths,
        "num_candidates_before_filter": num_candidates_before_filter,
        "num_removed_empty_problem_or_answer": num_removed_empty,
        "num_removed_source_amc_aime": num_removed_source_amc_aime,
        "num_removed_eval_overlap": num_removed_eval_overlap,
        "num_removed_train_duplicate": num_removed_train_duplicate,
        "num_selected": len(selected_rows),
        "source_distribution_before_filter": dict(source_distribution_before_filter),
        "source_distribution_after_filter": dict(source_distribution_after_filter),
        "source_distribution_selected": dict(source_distribution_selected),
        "selected_rollout_prompts_sha256": selected_sha,
        "selected_train_metadata_sha256": train_sha,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "metadata_summary.json", summary)

    validate_outputs(
        output_dir=output_dir,
        selected_path=selected_path,
        train_path=train_path,
        batch_dir=batch_dir,
        eval_normalized_problems=eval_normalized_problems,
        args=args,
    )


if __name__ == "__main__":
    main()
