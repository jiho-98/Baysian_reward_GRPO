#!/usr/bin/env python3
"""Prepare unified analyzer DPO pairs from Bayesian debug logs.

This script mines schema-perfect chosen/rejected JSON pairs for the unified
analyzer. It keeps the analyzer prompt/body identical to the current offline
recompute path so Base Qwen -> DPO and v0 SFT -> DPO can share the exact same
data and evaluation flow.

Outputs:
- prior_dpo_pool.jsonl
- evidence_dpo_pool.jsonl
- prior_dpo_train.jsonl
- prior_dpo_val.jsonl
- evidence_dpo_train.jsonl
- evidence_dpo_val.jsonl
- unified_dpo_train.jsonl
- unified_dpo_val.jsonl
- summary.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from prepare_analyzer_training_data import (
    JUDGE_JSON_SYSTEM_PROMPT,
    assign_split,
    build_evidence_judge_prompt,
    build_prior_judge_prompt,
    filtered_rows_for_args,
    load_jsonl_rows,
    relative_path_str,
    stable_hash,
)
from prepare_analyzer_v11_calibration_data import (
    apply_prior_corrections,
    compute_group_metrics,
    correction_evidence_target,
    evidence_signal_map_for_group,
)


DEFAULT_OUTPUT_DIR = "outputs/unified_analyzer_dpo_v0"

TASK_PREFIXES = {
    "evidence_judge": (
        "[TASK=evidence_judge]\n"
        "Return the evidence-judge JSON schema only.\n"
        "Do not output prior-judge fields.\n\n"
    ),
    "prior_judge": (
        "[TASK=prior_judge]\n"
        "Return the prior-judge JSON schema only.\n"
        "Do not output evidence-judge fields.\n\n"
    ),
}

BUCKET_ORDER = (
    "wrong_top_correction",
    "low_prior_correct",
    "high_prior_wrong",
    "good_strategy_bad_execution",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare unified analyzer DPO preference data."
    )
    parser.add_argument(
        "--input_debug_jsonl",
        action="append",
        required=True,
        help="Path to bayesian_reward_debug.jsonl. Repeat to pass multiple files.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected_prior_lambda", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--low_prior_threshold", type=int, default=2)
    parser.add_argument("--high_prior_threshold", type=int, default=3)
    parser.add_argument("--high_likelihood_threshold", type=float, default=0.8)
    parser.add_argument("--low_likelihood_threshold", type=float, default=0.3)

    parser.add_argument(
        "--target_prior_train_pairs",
        type=int,
        default=800,
        help="Upper bound on sampled prior DPO train pairs.",
    )
    parser.add_argument(
        "--target_prior_val_pairs",
        type=int,
        default=120,
        help="Upper bound on sampled prior DPO val pairs.",
    )
    parser.add_argument(
        "--target_evidence_train_pairs",
        type=int,
        default=800,
        help="Upper bound on sampled evidence DPO train pairs.",
    )
    parser.add_argument(
        "--target_evidence_val_pairs",
        type=int,
        default=120,
        help="Upper bound on sampled evidence DPO val pairs.",
    )

    parser.add_argument("--wrong_top_ratio", type=float, default=0.35)
    parser.add_argument("--low_prior_correct_ratio", type=float, default=0.30)
    parser.add_argument("--high_prior_wrong_ratio", type=float, default=0.25)
    parser.add_argument("--good_strategy_bad_execution_ratio", type=float, default=0.10)

    parser.add_argument(
        "--no_task_prefix",
        action="store_true",
        help="Disable the [TASK=...] prefix. By default the current analyzer prefix is kept.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def normalize_ratio_map(args: argparse.Namespace) -> dict[str, float]:
    raw = {
        "wrong_top_correction": max(0.0, float(args.wrong_top_ratio)),
        "low_prior_correct": max(0.0, float(args.low_prior_correct_ratio)),
        "high_prior_wrong": max(0.0, float(args.high_prior_wrong_ratio)),
        "good_strategy_bad_execution": max(
            0.0, float(args.good_strategy_bad_execution_ratio)
        ),
    }
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("At least one bucket ratio must be positive.")
    return {bucket: value / total for bucket, value in raw.items()}


def maybe_add_task_prefix(prompt: str, task: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    return TASK_PREFIXES[task] + prompt


def prompt_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_JSON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def original_evidence_target(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_validity": int(row.get("step_validity", 0) or 0),
        "proof_completeness": int(row.get("proof_completeness", 0) or 0),
        "strategy_compliance": int(row.get("strategy_compliance", 0) or 0),
        "consistency": int(row.get("consistency", 0) or 0),
        "error_type": str(row.get("error_type", "") or ""),
        "key_strength": str(row.get("key_strength", "") or ""),
        "key_weakness": str(row.get("key_weakness", "") or ""),
        "critical_failure_step": str(row.get("critical_failure_step", "") or ""),
        "judge_confidence": float(row.get("judge_confidence", 0.0) or 0.0),
    }


def original_prior_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0) or 0))
    return {
        "priors": [
            {
                "rollout_id": int(row.get("group_rollout_id", 0) or 0),
                "suitability": int(row.get("prior_suitability", 0) or 0),
                "reason": str(row.get("prior_reason", "") or ""),
                "risk_flag": str(row.get("prior_risk_flag", "") or ""),
            }
            for row in ordered_rows
        ]
    }


def corrected_prior_target(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0) or 0))
    return {
        "priors": [
            {
                "rollout_id": int(row.get("group_rollout_id", 0) or 0),
                "suitability": int(row.get("prior_suitability", 0) or 0),
                "reason": str(row.get("prior_reason", "") or ""),
                "risk_flag": str(row.get("prior_risk_flag", "") or ""),
            }
            for row in ordered_rows
        ]
    }


def build_group_key(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "global_reward_call_index": first.get("global_reward_call_index"),
        "group_index_within_call": first.get("group_index_within_call"),
    }


def group_rows_by_call(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("global_reward_call_index"), row.get("group_index_within_call"))
        grouped[key].append(row)
    return [
        sorted(group, key=lambda row: int(row.get("group_rollout_id", 0) or 0))
        for _, group in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def choose_evidence_signal(signals: set[str]) -> Optional[str]:
    if "wrong_high_posterior_top" in signals:
        return "wrong_high_posterior_top"
    if "correct_available_but_not_top" in signals:
        return "correct_available_but_not_top"
    if "low_prior_correct" in signals:
        return "low_prior_correct"
    if "high_prior_wrong_bad_strategy" in signals:
        return "high_prior_wrong_bad_strategy"
    if "high_prior_wrong_good_execution" in signals:
        return "high_prior_wrong_good_execution"
    return None


def bucket_name_for_evidence_signal(signal: str) -> str:
    if signal in {"wrong_high_posterior_top", "correct_available_but_not_top"}:
        return "wrong_top_correction"
    if signal == "low_prior_correct":
        return "low_prior_correct"
    if signal == "high_prior_wrong_bad_strategy":
        return "high_prior_wrong"
    if signal == "high_prior_wrong_good_execution":
        return "good_strategy_bad_execution"
    raise ValueError(f"Unsupported evidence signal: {signal}")


def bucket_tags_for_prior_group(
    changed_rollout_ids: list[int],
    prior_signals: dict[int, set[str]],
    wrong_top: bool,
) -> list[str]:
    tags: set[str] = set()
    if wrong_top and changed_rollout_ids:
        tags.add("wrong_top_correction")
    for rollout_id in changed_rollout_ids:
        signals = prior_signals.get(rollout_id, set())
        if "low_prior_correct" in signals:
            tags.add("low_prior_correct")
        if "high_prior_wrong_bad_strategy" in signals:
            tags.add("high_prior_wrong")
    return [bucket for bucket in BUCKET_ORDER if bucket in tags]


def primary_bucket(bucket_tags: list[str]) -> str:
    for bucket in BUCKET_ORDER:
        if bucket in bucket_tags:
            return bucket
    raise ValueError("Pair has no supported sampling bucket.")


def json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def build_prior_pair(
    rows: list[dict[str, Any]],
    corrected_rows: list[dict[str, Any]],
    *,
    source_file: Path,
    split: str,
    bucket_tags: list[str],
    task_prefix_enabled: bool,
    changed_rollout_ids: list[int],
    prior_signals: dict[int, set[str]],
) -> dict[str, Any]:
    problem = str(rows[0].get("problem", "") or "")
    rollouts = [
        {
            "rollout_id": int(row.get("group_rollout_id", 0) or 0),
            "strategy": str(row.get("parsed_strategy", "") or ""),
        }
        for row in rows
    ]
    prompt = build_prior_judge_prompt(problem, rollouts, len(rollouts))
    prompt = maybe_add_task_prefix(prompt, "prior_judge", task_prefix_enabled)
    rejected_target = original_prior_target(rows)
    chosen_target = corrected_prior_target(corrected_rows)

    signature = json.dumps(
        {
            "problem": problem,
            "group_key": build_group_key(rows),
            "changed_rollout_ids": changed_rollout_ids,
            "bucket_tags": bucket_tags,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "pair_id": stable_hash(signature),
        "task": "prior_judge",
        "split": split,
        "source_file": relative_path_str(source_file),
        "problem_key": stable_hash(problem),
        "group_key": build_group_key(rows),
        "bucket_name": primary_bucket(bucket_tags),
        "bucket_tags": bucket_tags,
        "pair_kind": "prior_group_calibration",
        "problem": problem,
        "num_rollouts": len(rollouts),
        "changed_rollout_ids": changed_rollout_ids,
        "rollout_signals": {
            str(rollout_id): sorted(prior_signals.get(rollout_id, set()))
            for rollout_id in changed_rollout_ids
        },
        "prompt": prompt,
        "prompt_messages": prompt_messages(prompt),
        "chosen": json_text(chosen_target),
        "rejected": json_text(rejected_target),
        "chosen_target": chosen_target,
        "rejected_target": rejected_target,
    }


def build_evidence_pair(
    row: dict[str, Any],
    *,
    source_file: Path,
    split: str,
    signal: str,
    task_prefix_enabled: bool,
    all_signals: set[str],
) -> dict[str, Any]:
    problem = str(row.get("problem", "") or "")
    strategy = str(row.get("parsed_strategy", "") or "")
    reasoning = str(row.get("parsed_reasoning", "") or "")
    final_answer = str(row.get("parsed_final_answer", "") or "")
    answer_correctness = float(row.get("answer_correctness", 0.0) or 0.0)
    prompt = build_evidence_judge_prompt(
        problem,
        strategy,
        reasoning,
        final_answer,
        answer_correctness,
    )
    prompt = maybe_add_task_prefix(prompt, "evidence_judge", task_prefix_enabled)
    rejected_target = original_evidence_target(row)
    chosen_target = correction_evidence_target(row, signal)
    signature = json.dumps(
        {
            "problem": problem,
            "group_key": {
                "global_reward_call_index": row.get("global_reward_call_index"),
                "group_index_within_call": row.get("group_index_within_call"),
                "group_rollout_id": row.get("group_rollout_id"),
            },
            "signal": signal,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "pair_id": stable_hash(signature),
        "task": "evidence_judge",
        "split": split,
        "source_file": relative_path_str(source_file),
        "problem_key": stable_hash(problem),
        "group_key": {
            "global_reward_call_index": row.get("global_reward_call_index"),
            "group_index_within_call": row.get("group_index_within_call"),
            "group_rollout_id": row.get("group_rollout_id"),
        },
        "bucket_name": bucket_name_for_evidence_signal(signal),
        "bucket_tags": [bucket_name_for_evidence_signal(name) for name in [signal] if name],
        "pair_kind": "evidence_rollout_calibration",
        "problem": problem,
        "rollout_id": int(row.get("group_rollout_id", 0) or 0),
        "answer_correctness": answer_correctness,
        "error_type": str(row.get("error_type", "") or ""),
        "failure_signal": signal,
        "failure_signals": sorted(all_signals),
        "prompt": prompt,
        "prompt_messages": prompt_messages(prompt),
        "chosen": json_text(chosen_target),
        "rejected": json_text(rejected_target),
        "chosen_target": chosen_target,
        "rejected_target": rejected_target,
    }


def is_prior_group_usable(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if row.get("prior_suitability") is None:
            return False
        if not str(row.get("parsed_strategy", "") or "").strip():
            return False
    return True


def is_evidence_row_usable(row: dict[str, Any]) -> bool:
    if not str(row.get("parsed_strategy", "") or "").strip():
        return False
    if not str(row.get("parsed_reasoning", "") or "").strip():
        return False
    if not str(row.get("parsed_final_answer", "") or "").strip():
        return False
    return True


def bucket_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("bucket_name", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def task_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("task", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def allocate_bucket_targets(target_total: int, ratios: dict[str, float]) -> dict[str, int]:
    if target_total <= 0:
        return {bucket: 0 for bucket in BUCKET_ORDER}

    exact = {bucket: ratios.get(bucket, 0.0) * target_total for bucket in BUCKET_ORDER}
    base = {bucket: int(exact[bucket]) for bucket in BUCKET_ORDER}
    remainder = target_total - sum(base.values())
    ranked = sorted(
        BUCKET_ORDER,
        key=lambda bucket: (exact[bucket] - base[bucket], exact[bucket], bucket),
        reverse=True,
    )
    for bucket in ranked[:remainder]:
        base[bucket] += 1
    return base


def sample_pairs(
    rows: list[dict[str, Any]],
    *,
    target_total: int,
    ratios: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = list(rows)
    if target_total <= 0 or not rows:
        return []
    if len(rows) <= target_total:
        return sorted(rows, key=lambda row: row["pair_id"])

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["bucket_name"])].append(row)

    for bucket_rows in by_bucket.values():
        rng.shuffle(bucket_rows)

    bucket_targets = allocate_bucket_targets(target_total, ratios)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for bucket in BUCKET_ORDER:
        available = by_bucket.get(bucket, [])
        take = min(bucket_targets[bucket], len(available))
        for row in available[:take]:
            selected.append(row)
            selected_ids.add(str(row["pair_id"]))

    if len(selected) < target_total:
        remaining = [row for row in rows if str(row["pair_id"]) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_total - len(selected)])

    return sorted(selected, key=lambda row: row["pair_id"])


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    return train_rows, val_rows


def shuffle_rows(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    copied = list(rows)
    rng.shuffle(copied)
    return copied


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = normalize_ratio_map(args)
    task_prefix_enabled = not args.no_task_prefix

    prior_pairs_by_id: dict[str, dict[str, Any]] = {}
    evidence_pairs_by_id: dict[str, dict[str, Any]] = {}
    source_row_counts_before_filter: dict[str, int] = {}
    source_row_counts_after_filter: dict[str, int] = {}
    source_group_counts: dict[str, int] = {}
    skipped_prior_groups = 0
    skipped_evidence_rows = 0
    evidence_signal_counts: Counter[str] = Counter()
    prior_signal_counts: Counter[str] = Counter()

    for raw_path in args.input_debug_jsonl:
        path = Path(raw_path)
        raw_rows = load_jsonl_rows(path)
        rows = filtered_rows_for_args(raw_rows, args)
        source_key = relative_path_str(path)
        source_row_counts_before_filter[source_key] = len(raw_rows)
        source_row_counts_after_filter[source_key] = len(rows)

        groups = group_rows_by_call(rows)
        source_group_counts[source_key] = len(groups)

        for group_rows in groups:
            if not group_rows:
                continue

            split = assign_split(str(group_rows[0].get("problem", "") or ""), args.val_ratio)
            metrics = compute_group_metrics(group_rows)
            wrong_top_rollout_ids = {metrics.top_rollout_id} if metrics.wrong_top else set()

            signals_by_rollout = evidence_signal_map_for_group(
                group_rows,
                low_prior_threshold=args.low_prior_threshold,
                high_prior_threshold=args.high_prior_threshold,
                high_likelihood_threshold=args.high_likelihood_threshold,
                low_likelihood_threshold=args.low_likelihood_threshold,
                wrong_top_rollout_ids=wrong_top_rollout_ids,
                wrong_top_present=bool(wrong_top_rollout_ids),
            )
            for signals in signals_by_rollout.values():
                evidence_signal_counts.update(signals)

            prior_signals = {
                rollout_id: {
                    signal
                    for signal in signals
                    if signal in {"low_prior_correct", "high_prior_wrong_bad_strategy"}
                }
                for rollout_id, signals in signals_by_rollout.items()
                if any(
                    signal in {"low_prior_correct", "high_prior_wrong_bad_strategy"}
                    for signal in signals
                )
            }
            for signals in prior_signals.values():
                prior_signal_counts.update(signals)

            if is_prior_group_usable(group_rows):
                corrected_rows, changed_rollout_ids = apply_prior_corrections(
                    group_rows,
                    prior_signals,
                )
                bucket_tags = bucket_tags_for_prior_group(
                    changed_rollout_ids,
                    prior_signals,
                    metrics.wrong_top,
                )
                if changed_rollout_ids and bucket_tags:
                    pair = build_prior_pair(
                        group_rows,
                        corrected_rows,
                        source_file=path,
                        split=split,
                        bucket_tags=bucket_tags,
                        task_prefix_enabled=task_prefix_enabled,
                        changed_rollout_ids=changed_rollout_ids,
                        prior_signals=prior_signals,
                    )
                    if pair["chosen"] != pair["rejected"]:
                        prior_pairs_by_id[str(pair["pair_id"])] = pair
            else:
                skipped_prior_groups += 1

            for row in group_rows:
                if not is_evidence_row_usable(row):
                    skipped_evidence_rows += 1
                    continue
                rollout_id = int(row.get("group_rollout_id", 0) or 0)
                signals = signals_by_rollout.get(rollout_id, set())
                signal = choose_evidence_signal(signals)
                if signal is None:
                    continue
                pair = build_evidence_pair(
                    row,
                    source_file=path,
                    split=split,
                    signal=signal,
                    task_prefix_enabled=task_prefix_enabled,
                    all_signals=signals,
                )
                if pair["chosen"] != pair["rejected"]:
                    evidence_pairs_by_id[str(pair["pair_id"])] = pair

    prior_pool = sorted(prior_pairs_by_id.values(), key=lambda row: row["pair_id"])
    evidence_pool = sorted(evidence_pairs_by_id.values(), key=lambda row: row["pair_id"])

    prior_pool_train, prior_pool_val = split_rows(prior_pool)
    evidence_pool_train, evidence_pool_val = split_rows(evidence_pool)

    prior_train = sample_pairs(
        prior_pool_train,
        target_total=args.target_prior_train_pairs,
        ratios=ratios,
        rng=rng,
    )
    prior_val = sample_pairs(
        prior_pool_val,
        target_total=args.target_prior_val_pairs,
        ratios=ratios,
        rng=rng,
    )
    evidence_train = sample_pairs(
        evidence_pool_train,
        target_total=args.target_evidence_train_pairs,
        ratios=ratios,
        rng=rng,
    )
    evidence_val = sample_pairs(
        evidence_pool_val,
        target_total=args.target_evidence_val_pairs,
        ratios=ratios,
        rng=rng,
    )

    unified_train = shuffle_rows(prior_train + evidence_train, rng)
    unified_val = shuffle_rows(prior_val + evidence_val, rng)

    write_jsonl(output_dir / "prior_dpo_pool.jsonl", prior_pool)
    write_jsonl(output_dir / "evidence_dpo_pool.jsonl", evidence_pool)
    write_jsonl(output_dir / "prior_dpo_train.jsonl", prior_train)
    write_jsonl(output_dir / "prior_dpo_val.jsonl", prior_val)
    write_jsonl(output_dir / "evidence_dpo_train.jsonl", evidence_train)
    write_jsonl(output_dir / "evidence_dpo_val.jsonl", evidence_val)
    write_jsonl(output_dir / "unified_dpo_train.jsonl", unified_train)
    write_jsonl(output_dir / "unified_dpo_val.jsonl", unified_val)

    summary = {
        "input_files": [relative_path_str(Path(path)) for path in args.input_debug_jsonl],
        "config": {
            "expected_prior_lambda": args.expected_prior_lambda,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "low_prior_threshold": args.low_prior_threshold,
            "high_prior_threshold": args.high_prior_threshold,
            "high_likelihood_threshold": args.high_likelihood_threshold,
            "low_likelihood_threshold": args.low_likelihood_threshold,
            "task_prefix_enabled": task_prefix_enabled,
            "target_prior_train_pairs": args.target_prior_train_pairs,
            "target_prior_val_pairs": args.target_prior_val_pairs,
            "target_evidence_train_pairs": args.target_evidence_train_pairs,
            "target_evidence_val_pairs": args.target_evidence_val_pairs,
            "bucket_ratios": ratios,
        },
        "source_row_counts_before_filter": source_row_counts_before_filter,
        "source_row_counts_after_filter": source_row_counts_after_filter,
        "source_group_counts": source_group_counts,
        "signal_counts": {
            "evidence": dict(sorted(evidence_signal_counts.items())),
            "prior": dict(sorted(prior_signal_counts.items())),
        },
        "skips": {
            "prior_groups": skipped_prior_groups,
            "evidence_rows": skipped_evidence_rows,
        },
        "pool": {
            "prior_total": len(prior_pool),
            "prior_train": len(prior_pool_train),
            "prior_val": len(prior_pool_val),
            "evidence_total": len(evidence_pool),
            "evidence_train": len(evidence_pool_train),
            "evidence_val": len(evidence_pool_val),
            "prior_bucket_distribution": bucket_counter(prior_pool),
            "evidence_bucket_distribution": bucket_counter(evidence_pool),
        },
        "selected": {
            "prior_train": len(prior_train),
            "prior_val": len(prior_val),
            "evidence_train": len(evidence_train),
            "evidence_val": len(evidence_val),
            "unified_train": len(unified_train),
            "unified_val": len(unified_val),
            "prior_train_bucket_distribution": bucket_counter(prior_train),
            "prior_val_bucket_distribution": bucket_counter(prior_val),
            "evidence_train_bucket_distribution": bucket_counter(evidence_train),
            "evidence_val_bucket_distribution": bucket_counter(evidence_val),
            "unified_train_task_distribution": task_counter(unified_train),
            "unified_val_task_distribution": task_counter(unified_val),
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(f"[INFO] wrote DPO data to {output_dir}")
    print(
        "[INFO] prior pool "
        f"train={len(prior_pool_train)} val={len(prior_pool_val)} selected_train={len(prior_train)} selected_val={len(prior_val)}"
    )
    print(
        "[INFO] evidence pool "
        f"train={len(evidence_pool_train)} val={len(evidence_pool_val)} selected_train={len(evidence_train)} selected_val={len(evidence_val)}"
    )
    print(
        "[INFO] unified DPO "
        f"train={len(unified_train)} val={len(unified_val)}"
    )


if __name__ == "__main__":
    main()
