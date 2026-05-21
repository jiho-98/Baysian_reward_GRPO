#!/usr/bin/env python3
"""Prepare fixed GSM8K train/valid/test metadata and solver SFT data."""

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
    ensure_output_dir,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_DATASET_NAME = "gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_SAMPLED_OUTPUT_DIR = "outputs/gsm8k_3000_500_seed42"
DEFAULT_FULL_OUTPUT_DIR = "outputs/gsm8k_experiments/metadata_fulltrain_seed42"
DEFAULT_SETTING = "sampled_3k_valid500"
FULL_TRAIN_SETTING = "full_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed GSM8K metadata and solver SFT data."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--setting",
        choices=[DEFAULT_SETTING, FULL_TRAIN_SETTING],
        default=DEFAULT_SETTING,
        help=(
            "Metadata setting to build. "
            "`sampled_3k_valid500` preserves the existing 3000/500 split; "
            "`full_train` uses the full official train split and no valid split."
        ),
    )
    parser.add_argument(
        "--use_full_train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convenience flag equivalent to `--setting full_train`.",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--valid_size", type=int, default=500)
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


def strip_gsm8k_workings(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<<[^<>]*>>", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_gsm8k_final_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"(?i)^(?:answer|final answer)\s*[:=]\s*", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.strip()
    return cleaned


def split_gsm8k_solution(raw_solution: str) -> tuple[str, str]:
    solution_text = str(raw_solution or "").strip()
    if not solution_text:
        return "", ""

    if "####" in solution_text:
        reasoning, _, final_answer = solution_text.rpartition("####")
    else:
        nonempty_lines = [line.strip() for line in solution_text.splitlines() if line.strip()]
        if not nonempty_lines:
            return "", ""
        reasoning = "\n".join(nonempty_lines[:-1]).strip()
        final_answer = nonempty_lines[-1]

    cleaned_reasoning = strip_gsm8k_workings(reasoning)
    cleaned_answer = normalize_gsm8k_final_answer(final_answer)
    return cleaned_reasoning, cleaned_answer


def metadata_row_from_example(example: dict[str, Any]) -> dict[str, Any]:
    question = str(example.get("question", "") or "").strip()
    raw_solution = str(example.get("answer", "") or "").strip()
    reasoning, final_answer = split_gsm8k_solution(raw_solution)
    if not question:
        raise RuntimeError("Encountered empty GSM8K question.")
    if not final_answer:
        raise RuntimeError(f"Failed to extract final answer for GSM8K question: {question[:120]}")

    row = {
        "problem": question,
        "answer": final_answer,
        "source": "gsm8k",
        "domain": "math",
        "difficulty_bucket": "gsm8k",
        "llama8b_solve_rate": None,
        "raw_solution": raw_solution,
    }
    if reasoning:
        row["reasoning_solution"] = reasoning
    return row


def default_strategy_text(problem: str) -> str:
    lowered = problem.lower()
    if any(token in lowered for token in ("how many", "total", "left", "remain", "more", "less")):
        return "Track the quantities carefully, translate the word problem into arithmetic steps, and compute the requested value."
    if any(token in lowered for token in ("each", "per", "every", "equal")):
        return "Set up the quantity relationships explicitly, compute step by step, and verify the final quantity."
    return "Translate the word problem into arithmetic relations, solve step by step, and check the final numeric answer."


def build_solver_target(row: dict[str, Any]) -> str:
    reasoning = str(row.get("reasoning_solution", "") or "").strip()
    if not reasoning:
        reasoning = str(row.get("raw_solution", "") or "").strip()
        reasoning, _ = split_gsm8k_solution(reasoning)
    if not reasoning:
        reasoning = "Compute the required quantity step by step."

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


def row_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        collapse_spaces(str(row.get("problem", ""))),
        normalize_gsm8k_final_answer(str(row.get("answer", ""))),
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
            f"Detected {len(train_valid_overlap)} overlapping GSM8K examples between train and valid."
        )

    train_test_overlap = train_ids & test_ids
    if train_test_overlap:
        raise RuntimeError(
            f"Detected {len(train_test_overlap)} overlapping GSM8K examples between train and test. "
            "The official test split must never enter solver or analyzer training data."
        )

    valid_test_overlap = valid_ids & test_ids
    if valid_test_overlap:
        raise RuntimeError(
            f"Detected {len(valid_test_overlap)} overlapping GSM8K examples between valid and test."
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
        "train": "gsm8k official train split"
        if setting == FULL_TRAIN_SETTING
        else f"gsm8k official train split sampled with random seed {args.seed}",
        "valid": "none"
        if setting == FULL_TRAIN_SETTING
        else f"gsm8k official train split sampled with random seed {args.seed}",
        "test": "gsm8k official test split",
    }
    summary: dict[str, Any] = {
        "setting": setting,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
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
    if args.train_size <= 0:
        raise SystemExit("--train_size must be positive.")
    if args.valid_size < 0:
        raise SystemExit("--valid_size must be non-negative.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install it with `pip install datasets`.") from exc

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if "train" not in dataset or "test" not in dataset:
        raise RuntimeError(
            f"Expected train/test splits in {args.dataset_name} ({args.dataset_config})."
        )

    train_split = dataset["train"]
    test_split = dataset["test"]
    if setting == FULL_TRAIN_SETTING:
        train_indices = list(range(len(train_split)))
        valid_indices: list[int] = []
    else:
        if args.train_size + args.valid_size > len(train_split):
            raise RuntimeError(
                f"Requested train+valid={args.train_size + args.valid_size}, "
                f"but GSM8K train has only {len(train_split)} rows."
            )

        rng = random.Random(args.seed)
        shuffled_indices = list(range(len(train_split)))
        rng.shuffle(shuffled_indices)

        train_indices = shuffled_indices[: args.train_size]
        valid_indices = shuffled_indices[args.train_size : args.train_size + args.valid_size]

    train_rows = [metadata_row_from_example(train_split[index]) for index in train_indices]
    valid_rows = [metadata_row_from_example(train_split[index]) for index in valid_indices]
    test_rows = [metadata_row_from_example(test_split[index]) for index in range(len(test_split))]
    assert_disjoint_splits(train_rows=train_rows, valid_rows=valid_rows, test_rows=test_rows)

    output_dir = Path(resolve_output_dir(args, setting))
    if setting == FULL_TRAIN_SETTING and "full" not in str(output_dir).lower():
        raise RuntimeError(
            "Full-train GSM8K metadata must be written to a path containing 'full' "
            "so it cannot be confused with the 3K/500 metadata."
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
        print("[DRY RUN] resolved GSM8K metadata configuration")
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

    print("[INFO] prepared GSM8K metadata successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
