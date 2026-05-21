#!/usr/bin/env python3
"""Build GSM8K learned-analyzer SFT data from prompted Bayesian logs."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from gsm8k_learned_analyzer_utils import (
    add_task_prefix,
    assert_disjoint_question_splits,
    build_messages,
    discover_debug_jsonl,
    load_jsonl,
    build_runtime_evidence_prompt,
    build_runtime_prior_group_examples,
    build_simple_evidence_prompt,
    build_simple_prior_prompt,
    count_tags,
    extract_rollout_records,
    rebalance_two_pools,
    sample_rows,
    summarize_runtime_filtering,
    validate_runtime_sft_example,
    write_json,
    write_jsonl,
)


DEFAULT_LOG_DIR = (
    "outputs/gsm8k_experiments/"
    "grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10"
)
DEFAULT_OUTPUT_DIR = "outputs/gsm8k_learned_analyzer/sft_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simple and runtime-compatible analyzer SFT datasets from GSM8K prompted Bayesian logs."
    )
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--evidence_fraction", type=float, default=0.8)
    parser.add_argument("--clean_fraction", type=float, default=0.7)
    parser.add_argument("--hard_case_top_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def runtime_evidence_example(record: dict[str, Any]) -> dict[str, Any]:
    prompt = add_task_prefix(
        build_runtime_evidence_prompt(
            record["question"],
            record["strategy"],
            record["rollout_solution"],
            record["predicted_answer"],
            float(record["answer_correctness"]),
        ),
        "evidence_judge",
    )
    example = {
        "example_id": f"runtime-evidence:{record['record_id']}",
        "question_id": record["question_id"],
        "split": record["split"],
        "task": "evidence_judge",
        "quality_tier": "hard" if record["hard_case_tags"] else "clean",
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "rollout_id": record["group_key"]["rollout_id"],
        "hard_case_tags": list(record["hard_case_tags"]),
        "teacher_target": record["runtime_evidence_target"],
        "messages": build_messages(prompt, record["runtime_evidence_target"]),
    }
    validate_runtime_sft_example(example)
    return example


def simple_evidence_example(record: dict[str, Any]) -> dict[str, Any]:
    prompt = build_simple_evidence_prompt(
        record["question"],
        record["gold_answer"],
        record["strategy"],
        record["rollout_solution"],
        record["predicted_answer"],
        float(record["answer_correctness"]),
    )
    return {
        "example_id": f"simple-evidence:{record['record_id']}",
        "question_id": record["question_id"],
        "split": record["split"],
        "task": "evidence_judge",
        "quality_tier": "hard" if record["hard_case_tags"] else "clean",
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "rollout_id": record["group_key"]["rollout_id"],
        "hard_case_tags": list(record["hard_case_tags"]),
        "teacher_target": record["simple_evidence_target"],
        "messages": build_messages(prompt, record["simple_evidence_target"]),
    }


def simple_prior_example(record: dict[str, Any]) -> dict[str, Any]:
    prompt = build_simple_prior_prompt(
        record["question"],
        record["strategy"],
    )
    return {
        "example_id": f"simple-prior:{record['record_id']}",
        "question_id": record["question_id"],
        "split": record["split"],
        "task": "prior_judge",
        "quality_tier": "hard" if record["hard_case_tags"] else "clean",
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "rollout_id": record["group_key"]["rollout_id"],
        "hard_case_tags": list(record["hard_case_tags"]),
        "teacher_target": record["simple_prior_target"],
        "messages": build_messages(prompt, record["simple_prior_target"]),
    }


def split_pool(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    return train_rows, valid_rows


def choose_task_mix(
    *,
    clean_rows: list[dict[str, Any]],
    hard_rows: list[dict[str, Any]],
    clean_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not clean_rows:
        return list(hard_rows)
    if not hard_rows:
        return list(clean_rows)

    clean_fraction = min(max(clean_fraction, 0.0), 1.0)
    if clean_fraction >= 1.0:
        hard_take = 0
    elif clean_fraction <= 0.0:
        hard_take = len(hard_rows)
    else:
        max_hard_for_all_clean = int(
            round(len(clean_rows) * (1.0 - clean_fraction) / clean_fraction)
        )
        if max_hard_for_all_clean <= 0:
            hard_take = 0
        else:
            hard_take = min(len(hard_rows), max_hard_for_all_clean)

    selected = []
    selected.extend(clean_rows)
    selected.extend(sample_rows(hard_rows, hard_take, rng))
    selected.sort(key=lambda row: str(row["example_id"]))
    return selected


def build_variant_split(
    *,
    evidence_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    clean_fraction: float,
    evidence_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_clean = [row for row in evidence_rows if row["quality_tier"] == "clean"]
    evidence_hard = [row for row in evidence_rows if row["quality_tier"] == "hard"]
    prior_clean = [row for row in prior_rows if row["quality_tier"] == "clean"]
    prior_hard = [row for row in prior_rows if row["quality_tier"] == "hard"]

    evidence_selected = choose_task_mix(
        clean_rows=evidence_clean,
        hard_rows=evidence_hard,
        clean_fraction=clean_fraction,
        rng=rng,
    )
    prior_selected = choose_task_mix(
        clean_rows=prior_clean,
        hard_rows=prior_hard,
        clean_fraction=clean_fraction,
        rng=rng,
    )
    evidence_selected, prior_selected = rebalance_two_pools(
        evidence_selected,
        prior_selected,
        evidence_fraction,
        rng,
    )
    unified = sorted(
        evidence_selected + prior_selected,
        key=lambda row: str(row["example_id"]),
    )
    return evidence_selected, prior_selected, unified


def build_runtime_split(
    *,
    evidence_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    evidence_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_selected, prior_selected = rebalance_two_pools(
        list(evidence_rows),
        list(prior_rows),
        evidence_fraction,
        rng,
    )
    unified = sorted(
        evidence_selected + prior_selected,
        key=lambda row: str(row["example_id"]),
    )
    return evidence_selected, prior_selected, unified


def task_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["task"]) for row in rows)
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_debug_rows = load_jsonl(discover_debug_jsonl(args.log_dir))

    rollout_records = extract_rollout_records(
        args.log_dir,
        val_ratio=args.val_ratio,
        hard_case_top_fraction=args.hard_case_top_fraction,
    )
    source_split_assertion = assert_disjoint_question_splits(
        rollout_records,
        label="rollout_records",
    )
    write_jsonl(output_dir / "rollout_records.jsonl", rollout_records)

    simple_evidence_pool = [
        simple_evidence_example(record)
        for record in rollout_records
        if record["clean_evidence"]
    ]
    simple_prior_pool = [
        simple_prior_example(record)
        for record in rollout_records
        if record["clean_prior_group"]
    ]
    runtime_evidence_pool = [
        runtime_evidence_example(record)
        for record in rollout_records
        if record["clean_evidence"]
    ]
    runtime_prior_pool = build_runtime_prior_group_examples(rollout_records)

    simple_evidence_train_pool, simple_evidence_valid_pool = split_pool(simple_evidence_pool)
    simple_prior_train_pool, simple_prior_valid_pool = split_pool(simple_prior_pool)
    runtime_evidence_train_pool, runtime_evidence_valid_pool = split_pool(runtime_evidence_pool)
    runtime_prior_train_pool, runtime_prior_valid_pool = split_pool(runtime_prior_pool)

    simple_evidence_train, simple_prior_train, simple_unified_train = build_variant_split(
        evidence_rows=simple_evidence_train_pool,
        prior_rows=simple_prior_train_pool,
        clean_fraction=args.clean_fraction,
        evidence_fraction=args.evidence_fraction,
        rng=rng,
    )
    simple_evidence_valid, simple_prior_valid, simple_unified_valid = build_variant_split(
        evidence_rows=simple_evidence_valid_pool,
        prior_rows=simple_prior_valid_pool,
        clean_fraction=args.clean_fraction,
        evidence_fraction=args.evidence_fraction,
        rng=rng,
    )
    runtime_evidence_train, runtime_prior_train, runtime_unified_train = build_runtime_split(
        evidence_rows=runtime_evidence_train_pool,
        prior_rows=runtime_prior_train_pool,
        evidence_fraction=args.evidence_fraction,
        rng=rng,
    )
    runtime_evidence_valid, runtime_prior_valid, runtime_unified_valid = build_runtime_split(
        evidence_rows=runtime_evidence_valid_pool,
        prior_rows=runtime_prior_valid_pool,
        evidence_fraction=args.evidence_fraction,
        rng=rng,
    )

    artifacts: dict[str, list[dict[str, Any]]] = {
        "simple_evidence_train.jsonl": simple_evidence_train,
        "simple_evidence_valid.jsonl": simple_evidence_valid,
        "simple_prior_train.jsonl": simple_prior_train,
        "simple_prior_valid.jsonl": simple_prior_valid,
        "simple_unified_train.jsonl": simple_unified_train,
        "simple_unified_valid.jsonl": simple_unified_valid,
        "runtime_evidence_train.jsonl": runtime_evidence_train,
        "runtime_evidence_valid.jsonl": runtime_evidence_valid,
        "runtime_prior_train.jsonl": runtime_prior_train,
        "runtime_prior_valid.jsonl": runtime_prior_valid,
        "runtime_unified_train.jsonl": runtime_unified_train,
        "runtime_unified_valid.jsonl": runtime_unified_valid,
    }
    for row in runtime_evidence_pool:
        validate_runtime_sft_example(row)
    for row in runtime_prior_pool:
        validate_runtime_sft_example(row)
    for filename, rows in artifacts.items():
        write_jsonl(output_dir / filename, rows)

    split_assertions = {
        "source_rollout_records": source_split_assertion,
        "simple_unified": assert_disjoint_question_splits(
            simple_unified_train + simple_unified_valid,
            label="simple_unified",
        ),
        "runtime_unified": assert_disjoint_question_splits(
            runtime_unified_train + runtime_unified_valid,
            label="runtime_unified",
        ),
    }
    runtime_filtering = summarize_runtime_filtering(
        raw_debug_rows,
        rollout_records,
        evidence_fraction=args.evidence_fraction,
    )

    summary = {
        "config": {
            "log_dir": args.log_dir,
            "val_ratio": args.val_ratio,
            "evidence_fraction": args.evidence_fraction,
            "clean_fraction": args.clean_fraction,
            "hard_case_top_fraction": args.hard_case_top_fraction,
            "seed": args.seed,
        },
        "source": {
            "rollout_records": len(rollout_records),
            "clean_evidence_rollouts": sum(
                1 for record in rollout_records if record["clean_evidence"]
            ),
            "clean_prior_rollouts": sum(
                1 for record in rollout_records if record["clean_prior_group"]
            ),
            "question_ids": len({record["question_id"] for record in rollout_records}),
            "train_question_ids": len(
                {record["question_id"] for record in rollout_records if record["split"] == "train"}
            ),
            "valid_question_ids": len(
                {record["question_id"] for record in rollout_records if record["split"] == "valid"}
            ),
            "hard_case_tag_distribution": count_tags(rollout_records, "hard_case_tags"),
        },
        "split_assertions": split_assertions,
        "simple": {
            "pool": {
                "evidence": len(simple_evidence_pool),
                "prior": len(simple_prior_pool),
            },
            "train": {
                "evidence": len(simple_evidence_train),
                "prior": len(simple_prior_train),
                "unified": len(simple_unified_train),
                "task_distribution": task_counter(simple_unified_train),
            },
            "valid": {
                "evidence": len(simple_evidence_valid),
                "prior": len(simple_prior_valid),
                "unified": len(simple_unified_valid),
                "task_distribution": task_counter(simple_unified_valid),
            },
        },
        "runtime": {
            "filtering": runtime_filtering,
            "pool": {
                "evidence": len(runtime_evidence_pool),
                "prior": len(runtime_prior_pool),
            },
            "train": {
                "evidence": len(runtime_evidence_train),
                "prior": len(runtime_prior_train),
                "unified": len(runtime_unified_train),
                "task_distribution": task_counter(runtime_unified_train),
            },
            "valid": {
                "evidence": len(runtime_evidence_valid),
                "prior": len(runtime_prior_valid),
                "unified": len(runtime_unified_valid),
                "task_distribution": task_counter(runtime_unified_valid),
            },
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[INFO] wrote SFT data to {output_dir}")
    print(summary["simple"]["train"])
    print(summary["runtime"]["train"])


if __name__ == "__main__":
    main()
