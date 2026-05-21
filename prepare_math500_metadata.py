#!/usr/bin/env python3
"""Prepare fixed MATH-500 train/valid/test metadata and solver SFT data."""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    SYSTEM_PROMPT,
    build_user_prompt,
    cleanup_extracted_answer,
    ensure_output_dir,
    extract_last_boxed_content,
    normalize_answer_details,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_DATASET_NAME = "ricdomolm/MATH-500"
DEFAULT_DATASET_CONFIG = ""
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_SAMPLED_OUTPUT_DIR = "outputs/math500_experiments/metadata_train3000_valid500_seed42"
DEFAULT_FULL_OUTPUT_DIR = "outputs/math500_experiments/metadata_fulltrain_seed42"
DEFAULT_SETTING = "sampled_train3000_valid500"
FULL_TRAIN_SETTING = "full_train"

PROBLEM_KEYS = ("problem", "question", "problem_text", "prompt", "input")
ANSWER_KEYS = ("answer", "gold_answer", "ground_truth", "final_answer", "target", "label")
SOLUTION_KEYS = ("solution", "full_solution", "detailed_solution", "explanation", "rationale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed MATH-500 metadata and solver SFT data."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train_split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--test_split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument(
        "--setting",
        choices=[DEFAULT_SETTING, FULL_TRAIN_SETTING],
        default=FULL_TRAIN_SETTING,
        help=(
            "Metadata setting to build. "
            "`sampled_train3000_valid500` samples train/valid from the train split; "
            "`full_train` uses the full train split and no valid split."
        ),
    )
    parser.add_argument(
        "--use_full_train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convenience flag equivalent to `--setting full_train`.",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--train_size", type=int, default=None, help="Sampled setting only. Defaults to all train rows.")
    parser.add_argument("--valid_size", type=int, default=None, help="Sampled setting only. Defaults to 0 rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write_sft_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write solver SFT chat-message JSONL files.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve the split and print the summary without writing files.",
    )
    return parser.parse_args()


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


KNOWN_EMPTY_ANSWER_FIXES = {
    collapse_spaces(
        "For any integer $n>1$, the number of prime numbers greater than $n!+1$ and less than $n!+n$ is: "
        "$\\text{(A) } 0\\quad\\qquad \\text{(B) } 1\\quad\\\\ \\text{(C) } \\frac{n}{2} \\text{ for n even, } "
        "\\frac{n+1}{2} \\text{ for n odd}\\quad\\\\ \\text{(D) } n-1\\quad \\text{(E) } n$"
    ): "0",
    collapse_spaces(
        "You are given a sequence of $58$ terms; each term has the form $P+n$ where $P$ stands for the product "
        "$2 \\times 3 \\times 5 \\times\\ldots \\times 61$ of all prime numbers less than or equal to $61$, and "
        "$n$ takes, successively, the values $2, 3, 4,\\ldots, 59$. Let $N$ be the number of primes appearing in "
        "this sequence. Then $N$ is: $\\textbf{(A)}\\ 0\\qquad \\textbf{(B)}\\ 16\\qquad \\textbf{(C)}\\ 17\\qquad "
        "\\textbf{(D)}\\ 57\\qquad \\textbf{(E)}\\ 58$"
    ): "0",
}


def resolved_setting(args: argparse.Namespace) -> str:
    if args.use_full_train:
        return FULL_TRAIN_SETTING
    return str(args.setting)


def resolve_output_dir(args: argparse.Namespace, setting: str) -> str:
    if str(args.output_dir or "").strip():
        return str(args.output_dir)
    if setting == FULL_TRAIN_SETTING:
        return DEFAULT_FULL_OUTPUT_DIR
    return DEFAULT_SAMPLED_OUTPUT_DIR


def first_present_value(example: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = example.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_math_answer(text: str) -> str:
    normalized = normalize_answer_details(str(text or "")).normalized
    if normalized:
        return normalized
    return str(text or "").strip()


def extract_answer_from_solution(problem: str, raw_solution: str) -> str:
    solution = str(raw_solution or "").strip()
    if not solution:
        return KNOWN_EMPTY_ANSWER_FIXES.get(collapse_spaces(problem), "")

    boxed = cleanup_extracted_answer(extract_last_boxed_content(solution))
    if boxed:
        normalized_boxed = normalize_math_answer(boxed)
        if normalized_boxed:
            return normalized_boxed

    match = re.search(
        r"(?i)(?:the answer is|there are|there is|there will be)\s+(.+?)(?:[.]\s*$|$)",
        solution,
    )
    if match is not None:
        candidate = cleanup_extracted_answer(match.group(1))
        normalized_candidate = normalize_math_answer(candidate)
        if normalized_candidate:
            return normalized_candidate

    return KNOWN_EMPTY_ANSWER_FIXES.get(collapse_spaces(problem), "")


def metadata_row_from_example(example: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    problem = first_present_value(example, PROBLEM_KEYS)
    raw_answer = first_present_value(example, ANSWER_KEYS)
    raw_solution = first_present_value(example, SOLUTION_KEYS)

    if not problem:
        raise RuntimeError(f"Encountered empty problem row: keys={sorted(example.keys())}")
    if not raw_answer:
        raw_answer = extract_answer_from_solution(problem, raw_solution)
    if not raw_answer:
        raise RuntimeError(
            f"Failed to extract final answer for row with problem preview: {collapse_spaces(problem)[:120]}"
        )

    answer = normalize_math_answer(raw_answer)
    if not answer:
        raise RuntimeError(
            f"Normalized answer is empty for problem preview: {collapse_spaces(problem)[:120]}"
        )

    benchmark_name = dataset_name.split("/")[-1].lower()
    row = {
        "problem": problem,
        "answer": answer,
        "source": benchmark_name,
        "domain": str(example.get("subject", "") or "math"),
        "difficulty_bucket": benchmark_name,
        "llama8b_solve_rate": None,
        "raw_answer": raw_answer,
    }
    if raw_solution:
        row["raw_solution"] = raw_solution
        row["reasoning_solution"] = raw_solution
    return row


def default_strategy_text(problem: str) -> str:
    lowered = problem.lower()
    if any(token in lowered for token in ("prove", "show that", "determine whether")):
        return "Identify the governing mathematical structure, derive the necessary relations carefully, and justify the final conclusion."
    if any(token in lowered for token in ("probability", "expected", "random")):
        return "Set up the probabilistic quantities explicitly, compute them step by step, and simplify the final result."
    if any(token in lowered for token in ("geometry", "triangle", "circle", "angle", "area", "perimeter")):
        return "Write down the key geometric relations, derive the needed quantities carefully, and simplify the final answer."
    return "Translate the problem into the relevant mathematical relations, solve step by step, and verify the final answer."


def build_solver_target(row: dict[str, Any]) -> str:
    reasoning = str(row.get("reasoning_solution", "") or "").strip()
    if not reasoning:
        reasoning = "Work through the mathematics carefully and simplify the final result."

    return (
        f"[Strategy]\n{default_strategy_text(str(row['problem']))}\n\n"
        f"[Reasoning]\n{reasoning}\n\n"
        f"[Final Answer]\n{row['answer']}"
    )


def sft_row_from_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(str(row["problem"]))},
            {"role": "assistant", "content": build_solver_target(row)},
        ]
    }


def preview_problem(row: dict[str, Any], limit: int = 120) -> str:
    text = collapse_spaces(str(row.get("problem", "")))
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def row_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        collapse_spaces(str(row.get("problem", ""))),
        normalize_math_answer(str(row.get("answer", ""))),
    )


def assert_disjoint_splits(
    *,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    train_ids = {row_identity(row) for row in train_rows}
    valid_ids = {row_identity(row) for row in valid_rows}
    test_ids = {row_identity(row) for row in test_rows}

    train_valid_overlap = train_ids & valid_ids
    if train_valid_overlap:
        raise RuntimeError(
            f"Detected {len(train_valid_overlap)} overlapping MATH-500 examples between train and valid."
        )

    train_test_overlap = train_ids & test_ids
    if train_test_overlap:
        raise RuntimeError(
            f"Detected {len(train_test_overlap)} overlapping MATH-500 examples between train and test. "
            "The test split must never enter solver or analyzer training data."
        )

    valid_test_overlap = valid_ids & test_ids
    if valid_test_overlap:
        raise RuntimeError(
            f"Detected {len(valid_test_overlap)} overlapping MATH-500 examples between valid and test."
        )


def build_summary(
    *,
    args: argparse.Namespace,
    setting: str,
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    official_train_size: int,
    official_test_size: int,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    sft_train_path: Path | None,
    sft_valid_path: Path | None,
    include_hashes: bool,
) -> dict[str, Any]:
    created_at_utc = datetime.now(timezone.utc).isoformat()
    split_info = {
        "train": f"{args.dataset_name} {args.train_split} split"
        if setting == FULL_TRAIN_SETTING
        else f"{args.dataset_name} {args.train_split} split sampled with random seed {args.seed}",
        "valid": "none"
        if setting == FULL_TRAIN_SETTING
        else f"{args.dataset_name} {args.train_split} split sampled with random seed {args.seed}",
        "test": f"{args.dataset_name} {args.test_split} split",
    }
    summary: dict[str, Any] = {
        "setting": setting,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config or None,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "random_seed": args.seed,
        "created_at_utc": created_at_utc,
        "source_split_info": split_info,
        "official_train_size": official_train_size,
        "official_test_size": official_test_size,
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "test_size": len(test_rows),
        "train_size_full": len(train_rows) if setting == FULL_TRAIN_SETTING else None,
        "test_size_full": len(test_rows) if setting == FULL_TRAIN_SETTING else None,
        "output_dir": str(output_dir),
        "train_metadata_path": str(train_path),
        "valid_metadata_path": str(valid_path),
        "test_metadata_path": str(test_path),
        "train_sha256": sha256_file(train_path) if include_hashes and train_path.exists() else None,
        "valid_sha256": sha256_file(valid_path) if include_hashes and valid_path.exists() else None,
        "test_sha256": sha256_file(test_path) if include_hashes and test_path.exists() else None,
        "train_first_5_problem_previews": [preview_problem(row) for row in train_rows[:5]],
        "valid_first_5_problem_previews": [preview_problem(row) for row in valid_rows[:5]],
        "test_first_5_problem_previews": [preview_problem(row) for row in test_rows[:5]],
        "write_sft_data": bool(args.write_sft_data),
    }
    if sft_train_path is not None and sft_valid_path is not None:
        summary.update(
            {
                "sft_train_path": str(sft_train_path),
                "sft_valid_path": str(sft_valid_path),
                "sft_train_sha256": (
                    sha256_file(sft_train_path)
                    if include_hashes and sft_train_path.exists()
                    else None
                ),
                "sft_valid_sha256": (
                    sha256_file(sft_valid_path)
                    if include_hashes and sft_valid_path.exists()
                    else None
                ),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    setting = resolved_setting(args)
    if args.train_size is not None and args.train_size <= 0:
        raise SystemExit("--train_size must be positive.")
    if args.valid_size is not None and args.valid_size < 0:
        raise SystemExit("--valid_size must be non-negative.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install it with `pip install datasets`.") from exc

    if str(args.dataset_config or "").strip():
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)

    if args.train_split not in dataset or args.test_split not in dataset:
        raise RuntimeError(
            f"Expected splits {args.train_split!r}/{args.test_split!r} in {args.dataset_name}."
        )

    train_split = dataset[args.train_split]
    test_split = dataset[args.test_split]

    if setting == FULL_TRAIN_SETTING:
        train_indices = list(range(len(train_split)))
        valid_indices: list[int] = []
    else:
        train_size = len(train_split) if args.train_size is None else args.train_size
        valid_size = 0 if args.valid_size is None else args.valid_size
        if train_size + valid_size > len(train_split):
            raise RuntimeError(
                f"Requested train+valid={train_size + valid_size}, "
                f"but {args.dataset_name} {args.train_split} has only {len(train_split)} rows."
            )

        rng = random.Random(args.seed)
        shuffled_indices = list(range(len(train_split)))
        rng.shuffle(shuffled_indices)

        train_indices = shuffled_indices[:train_size]
        valid_indices = shuffled_indices[train_size : train_size + valid_size]

    train_rows = [metadata_row_from_example(dict(train_split[index]), args.dataset_name) for index in train_indices]
    valid_rows = [metadata_row_from_example(dict(train_split[index]), args.dataset_name) for index in valid_indices]
    test_rows = [metadata_row_from_example(dict(test_split[index]), args.dataset_name) for index in range(len(test_split))]
    assert_disjoint_splits(train_rows=train_rows, valid_rows=valid_rows, test_rows=test_rows)

    output_dir = Path(resolve_output_dir(args, setting))
    if setting == FULL_TRAIN_SETTING and "full" not in str(output_dir).lower():
        raise RuntimeError(
            "Full-train MATH-500 metadata must be written to a path containing 'full' "
            "so it cannot be confused with any sampled metadata."
        )
    train_path = output_dir / "selected_train_metadata.jsonl"
    valid_path = output_dir / "selected_valid_metadata.jsonl"
    test_path = output_dir / "selected_test_metadata.jsonl"

    sft_train_path: Path | None = None
    sft_valid_path: Path | None = None
    summary = build_summary(
        args=args,
        setting=setting,
        output_dir=output_dir,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        official_train_size=len(train_split),
        official_test_size=len(test_split),
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        sft_train_path=sft_train_path,
        sft_valid_path=sft_valid_path,
        include_hashes=False,
    )
    if args.dry_run:
        print("[DRY RUN] resolved MATH-500 metadata configuration")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    output_dir = ensure_output_dir(str(output_dir))
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    write_jsonl(test_path, test_rows)

    if args.write_sft_data:
        sft_train_path = output_dir / "sft_train_messages.jsonl"
        sft_valid_path = output_dir / "sft_valid_messages.jsonl"
        write_jsonl(sft_train_path, [sft_row_from_metadata(row) for row in train_rows])
        write_jsonl(sft_valid_path, [sft_row_from_metadata(row) for row in valid_rows])

    summary = build_summary(
        args=args,
        setting=setting,
        output_dir=output_dir,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        official_train_size=len(train_split),
        official_test_size=len(test_split),
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        sft_train_path=sft_train_path,
        sft_valid_path=sft_valid_path,
        include_hashes=True,
    )
    summary_path = output_dir / "summary.json"
    legacy_summary_path = output_dir / "metadata_summary.json"
    write_json(summary_path, summary)
    write_json(legacy_summary_path, summary)

    print("[INFO] prepared MATH-500 metadata successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
