#!/usr/bin/env python3
"""Prepare analyzer v1.1 calibration data from train traces only.

This script builds four train-time buckets:

1. anchor_clean: existing clean teacher labels loaded from disk
2. stable_bootstrap: high-confidence train-trace consensus subset
3. retention: v1 trace patterns that improve posterior preference over v0
4. correction: automatic calibration targets mined from reward failures

It never reads eval metadata or eval predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from prepare_analyzer_training_data import (
    JUDGE_JSON_SYSTEM_PROMPT,
    assign_split,
    build_evidence_judge_prompt,
    build_messages,
    build_prior_judge_prompt,
    load_jsonl_rows,
    stable_hash,
)


COMPACT_SYSTEM_PROMPT = (
    "You are a JSON-only evaluation API. "
    "Return exactly one valid JSON object and nothing else."
)

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

CORRECT_ERROR_TYPES = {"correct_complete", "correct_weak_proof", "lucky_correct"}
RAISEABLE_CORRECT_ERRORS = {"correct_complete", "correct_weak_proof"}
GOOD_EXECUTION_ERRORS = {
    "arithmetic_error",
    "algebraic_error",
    "finalization_error",
    "valid_but_incomplete",
}
BAD_STRATEGY_ERRORS = {
    "wrong_direction",
    "invalid_assumption",
    "strategy_mismatch",
    "no_meaningful_solution",
}

DEFAULT_ANCHOR_DIR = "outputs/analyzer_training_data_v1"
DEFAULT_OUTPUT_DIR = "outputs/analyzer_v11_small"
DEFAULT_ANCHOR_EVIDENCE_TRAIN = "evidence_clean_train.jsonl"
DEFAULT_ANCHOR_EVIDENCE_VAL = "evidence_clean_val.jsonl"
DEFAULT_ANCHOR_PRIOR_TRAIN = "prior_clean_train.jsonl"
DEFAULT_ANCHOR_PRIOR_VAL = "prior_clean_val.jsonl"


@dataclass
class RunSpec:
    run_name: str
    path: Path
    role: str


@dataclass
class GroupMetrics:
    top_rollout_id: int
    top_answer_correctness: float
    top_bayesian_reward: float
    any_correct: bool
    mass_on_correct: float
    entropy: float
    top1_top2_gap: float
    wrong_top: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare analyzer v1.1 calibration data from train traces only."
    )
    parser.add_argument("--anchor_input_dir", default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--anchor_evidence_train_file", default=DEFAULT_ANCHOR_EVIDENCE_TRAIN)
    parser.add_argument("--anchor_evidence_val_file", default=DEFAULT_ANCHOR_EVIDENCE_VAL)
    parser.add_argument("--anchor_prior_train_file", default=DEFAULT_ANCHOR_PRIOR_TRAIN)
    parser.add_argument("--anchor_prior_val_file", default=DEFAULT_ANCHOR_PRIOR_VAL)
    parser.add_argument("--baseline_debug_jsonl", required=True)
    parser.add_argument(
        "--target_debug_jsonl",
        required=True,
        help="Primary v1 train trace used to emit retention/correction targets.",
    )
    parser.add_argument(
        "--support_debug_jsonl",
        action="append",
        default=None,
        help="Optional extra train traces used only for extra retention/correction triggers.",
    )
    parser.add_argument(
        "--train_metadata_jsonl",
        action="append",
        default=None,
        help="Optional train metadata JSONL keyed by problem text.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_judge_confidence", type=float, default=0.8)
    parser.add_argument("--low_prior_threshold", type=int, default=2)
    parser.add_argument("--high_prior_threshold", type=int, default=3)
    parser.add_argument("--high_likelihood_threshold", type=float, default=0.8)
    parser.add_argument("--low_likelihood_threshold", type=float, default=0.3)
    parser.add_argument("--retention_mass_margin", type=float, default=0.05)
    parser.add_argument("--bootstrap_prior_delta", type=int, default=1)
    parser.add_argument("--bootstrap_score_delta", type=int, default=1)

    parser.add_argument("--anchor_ratio", type=float, default=0.60)
    parser.add_argument("--bootstrap_ratio", type=float, default=0.15)
    parser.add_argument("--retention_ratio", type=float, default=0.10)
    parser.add_argument("--correction_ratio", type=float, default=0.15)
    parser.add_argument("--evidence_weight", type=float, default=0.60)
    parser.add_argument("--prior_weight", type=float, default=0.40)
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional cap for unified_train after upsampling.",
    )
    parser.add_argument(
        "--include_bucket_val",
        action="store_true",
        help="Include bootstrap/retention/correction validation rows in unified_val.",
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


def load_metadata_lookup(raw_paths: list[str] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not raw_paths:
        return lookup
    for raw_path in raw_paths:
        for row in load_jsonl_rows(Path(raw_path)):
            problem = str(row.get("problem", "") or "")
            if problem and problem not in lookup:
                lookup[problem] = row
    return lookup


def safe_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def normalize_ratio_map(raw_map: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in raw_map.values())
    if total <= 0.0:
        raise ValueError("Mixture ratios must sum to a positive value.")
    return {key: max(0.0, float(value)) / total for key, value in raw_map.items()}


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    return train_rows, val_rows


def argmax_index(values: list[float]) -> int:
    if not values:
        raise ValueError("argmax_index requires a non-empty list.")
    return max(range(len(values)), key=lambda idx: values[idx])


def compute_entropy(values: list[float]) -> float:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in values:
        prob = max(0.0, float(value)) / total
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    return entropy


def compute_group_metrics(rows: list[dict[str, Any]]) -> GroupMetrics:
    rewards = [float(row.get("bayesian_reward", 0.0)) for row in rows]
    top_index = argmax_index(rewards)
    sorted_rewards = sorted(rewards, reverse=True)
    top1 = sorted_rewards[0]
    top2 = sorted_rewards[1] if len(sorted_rewards) > 1 else 0.0
    return GroupMetrics(
        top_rollout_id=int(rows[top_index].get("group_rollout_id", top_index) or top_index),
        top_answer_correctness=float(rows[top_index].get("answer_correctness", 0.0)),
        top_bayesian_reward=float(rows[top_index].get("bayesian_reward", 0.0)),
        any_correct=any(float(row.get("answer_correctness", 0.0)) == 1.0 for row in rows),
        mass_on_correct=sum(
            float(row.get("bayesian_reward", 0.0))
            for row in rows
            if float(row.get("answer_correctness", 0.0)) == 1.0
        ),
        entropy=compute_entropy(rewards),
        top1_top2_gap=float(top1 - top2),
        wrong_top=(
            any(float(row.get("answer_correctness", 0.0)) == 1.0 for row in rows)
            and float(rows[top_index].get("answer_correctness", 0.0)) == 0.0
        ),
    )


def load_trace_groups(path: Path) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    rows = load_jsonl_rows(path)
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("global_reward_call_index"), row.get("group_index_within_call"))
        groups[key].append(row)
    return {
        key: sorted(group_rows, key=lambda row: int(row.get("group_rollout_id", 0) or 0))
        for key, group_rows in groups.items()
    }


def evidence_fields(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("step_validity", 0) or 0),
        int(row.get("proof_completeness", 0) or 0),
        int(row.get("strategy_compliance", 0) or 0),
        int(row.get("consistency", 0) or 0),
    )


def evidence_family(error_type: str) -> str:
    if error_type in CORRECT_ERROR_TYPES:
        return "correct"
    if error_type in GOOD_EXECUTION_ERRORS:
        return "good_strategy_bad_execution"
    if error_type in BAD_STRATEGY_ERRORS:
        return "bad_strategy"
    return error_type


def row_passes_common_quality(row: dict[str, Any], min_judge_confidence: float) -> bool:
    if row.get("evidence_judge_fallback_used"):
        return False
    if row.get("prior_judge_fallback_used"):
        return False
    if not row.get("format_valid", False):
        return False
    if float(row.get("judge_confidence", 0.0)) < min_judge_confidence:
        return False
    if not str(row.get("parsed_strategy", "") or "").strip():
        return False
    if not str(row.get("parsed_reasoning", "") or "").strip():
        return False
    if not str(row.get("parsed_final_answer", "") or "").strip():
        return False
    return True


def is_stable_bootstrap_group(
    groups: list[list[dict[str, Any]]],
    min_judge_confidence: float,
    bootstrap_prior_delta: int,
    bootstrap_score_delta: int,
) -> bool:
    if not groups:
        return False
    group_size = len(groups[0])
    if any(len(group) != group_size for group in groups):
        return False

    for row_index in range(group_size):
        trace_rows = [group[row_index] for group in groups]
        if not all(row_passes_common_quality(row, min_judge_confidence) for row in trace_rows):
            return False

        prior_values = [int(row.get("prior_suitability", 0) or 0) for row in trace_rows]
        if max(prior_values) - min(prior_values) > bootstrap_prior_delta:
            return False

        base_scores = evidence_fields(trace_rows[0])
        for compare_row in trace_rows[1:]:
            compare_scores = evidence_fields(compare_row)
            if max(abs(lhs - rhs) for lhs, rhs in zip(base_scores, compare_scores)) > bootstrap_score_delta:
                return False
            if evidence_family(str(compare_row.get("error_type", "") or "")) != evidence_family(
                str(trace_rows[0].get("error_type", "") or "")
            ):
                return False

    metrics = [compute_group_metrics(group) for group in groups]
    return all(metric.top_answer_correctness == 1.0 for metric in metrics)


def base_label_metadata(
    row: dict[str, Any],
    *,
    source_file: Path,
    label_source: str,
    label_semantics: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "label_source": label_source,
        "label_semantics": label_semantics,
        "source_file": safe_relative_path(source_file),
        "prior_lambda": row.get("prior_lambda"),
        "prior_mode": row.get("prior_mode"),
        "learned_analyzer_model_name": row.get("learned_analyzer_model_name"),
        "learned_analyzer_adapter_path": row.get("learned_analyzer_adapter_path"),
        "learned_evidence_analyzer_model_name": row.get("learned_evidence_analyzer_model_name"),
        "learned_evidence_analyzer_adapter_path": row.get("learned_evidence_analyzer_adapter_path"),
        "learned_analyzer_task_prefix": row.get("learned_analyzer_task_prefix"),
        "evidence_judge_model": row.get("evidence_judge_model"),
        "prior_judge_model": row.get("prior_judge_model"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    if extra:
        metadata.update(extra)
    return metadata


def clamp_score(value: int, low: int = 0, high: int = 4) -> int:
    return max(low, min(high, int(value)))


def build_evidence_example_from_target(
    row: dict[str, Any],
    *,
    source_file: Path,
    split: str,
    quality_tier: str,
    label_source: str,
    label_semantics: str,
    teacher_target: dict[str, Any],
    bucket_name: str,
    failure_type: str,
    label_extra: Optional[dict[str, Any]] = None,
    example_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    problem = str(row.get("problem", "") or "")
    strategy = str(row.get("parsed_strategy", "") or "")
    reasoning = str(row.get("parsed_reasoning", "") or "")
    final_answer = str(row.get("parsed_final_answer", "") or "")
    prompt = build_evidence_judge_prompt(
        problem,
        strategy,
        reasoning,
        final_answer,
        float(row.get("answer_correctness", 0.0)),
    )
    signature = json.dumps(
        {
            "problem": problem,
            "strategy": strategy,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "answer_correctness": float(row.get("answer_correctness", 0.0)),
            "teacher_target": teacher_target,
            "bucket_name": bucket_name,
            "failure_type": failure_type,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    label_metadata = base_label_metadata(
        row,
        source_file=source_file,
        label_source=label_source,
        label_semantics=label_semantics,
        extra=label_extra,
    )
    example = {
        "example_id": stable_hash(signature),
        "task": "evidence_judge",
        "split": split,
        "quality_tier": quality_tier,
        "bucket_name": bucket_name,
        "failure_type": failure_type,
        "source_file": safe_relative_path(source_file),
        "label_source": label_source,
        "label_semantics": label_semantics,
        "label_metadata": label_metadata,
        "problem_key": stable_hash(problem),
        "group_key": {
            "global_reward_call_index": row.get("global_reward_call_index"),
            "group_index_within_call": row.get("group_index_within_call"),
            "group_rollout_id": row.get("group_rollout_id"),
        },
        "problem": problem,
        "strategy": strategy,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "answer_correctness": float(row.get("answer_correctness", 0.0)),
        "teacher_target": teacher_target,
        "teacher_metadata": {
            "original_error_type": str(row.get("error_type", "") or ""),
            "original_scores": {
                "step_validity": int(row.get("step_validity", 0) or 0),
                "proof_completeness": int(row.get("proof_completeness", 0) or 0),
                "strategy_compliance": int(row.get("strategy_compliance", 0) or 0),
                "consistency": int(row.get("consistency", 0) or 0),
            },
            "likelihood": float(row.get("likelihood", 0.0) or 0.0),
            "bayesian_reward": float(row.get("bayesian_reward", 0.0) or 0.0),
        },
        "prompt": prompt,
        "messages": build_messages(prompt, teacher_target),
    }
    if example_extra:
        example.update(example_extra)
    return example


def build_prior_example_from_rows(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    split: str,
    quality_tier: str,
    label_source: str,
    label_semantics: str,
    bucket_name: str,
    failure_type: str,
    label_extra: Optional[dict[str, Any]] = None,
    example_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0) or 0))
    problem = str(ordered_rows[0].get("problem", "") or "")
    rollouts = [
        {
            "rollout_id": int(row.get("group_rollout_id", 0) or 0),
            "strategy": str(row.get("parsed_strategy", "") or ""),
        }
        for row in ordered_rows
    ]
    target = {
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
    prompt = build_prior_judge_prompt(problem, rollouts, len(rollouts))
    signature = json.dumps(
        {
            "problem": problem,
            "target": target,
            "bucket_name": bucket_name,
            "failure_type": failure_type,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    label_metadata = base_label_metadata(
        ordered_rows[0],
        source_file=source_file,
        label_source=label_source,
        label_semantics=label_semantics,
        extra=label_extra,
    )
    example = {
        "example_id": stable_hash(signature),
        "task": "prior_judge",
        "split": split,
        "quality_tier": quality_tier,
        "bucket_name": bucket_name,
        "failure_type": failure_type,
        "source_file": safe_relative_path(source_file),
        "label_source": label_source,
        "label_semantics": label_semantics,
        "label_metadata": label_metadata,
        "problem_key": stable_hash(problem),
        "group_key": {
            "global_reward_call_index": ordered_rows[0].get("global_reward_call_index"),
            "group_index_within_call": ordered_rows[0].get("group_index_within_call"),
        },
        "problem": problem,
        "num_rollouts": len(rollouts),
        "candidates": [
            {
                "rollout_id": int(row.get("group_rollout_id", 0) or 0),
                "strategy": str(row.get("parsed_strategy", "") or ""),
                "teacher_suitability": int(row.get("prior_suitability", 0) or 0),
                "teacher_probability": float(row.get("prior_probability", 0.0) or 0.0),
                "teacher_reason": str(row.get("prior_reason", "") or ""),
                "risk_flag": str(row.get("prior_risk_flag", "") or ""),
            }
            for row in ordered_rows
        ],
        "teacher_target": target,
        "prompt": prompt,
        "messages": build_messages(prompt, target),
    }
    if example_extra:
        example.update(example_extra)
    return example


def correction_evidence_target(row: dict[str, Any], signal: str) -> dict[str, Any]:
    error_type = str(row.get("error_type", "") or "")
    original_scores = {
        "step_validity": int(row.get("step_validity", 0) or 0),
        "proof_completeness": int(row.get("proof_completeness", 0) or 0),
        "strategy_compliance": int(row.get("strategy_compliance", 0) or 0),
        "consistency": int(row.get("consistency", 0) or 0),
    }
    answer_correctness = float(row.get("answer_correctness", 0.0) or 0.0)

    if answer_correctness == 1.0:
        if error_type == "correct_complete":
            scores = {
                "step_validity": 4,
                "proof_completeness": 4,
                "strategy_compliance": max(4, original_scores["strategy_compliance"]),
                "consistency": 4,
            }
            return {
                **scores,
                "error_type": "correct_complete",
                "key_strength": "Valid strategy with a correct and complete solution.",
                "key_weakness": "",
                "critical_failure_step": "",
                "judge_confidence": max(0.95, float(row.get("judge_confidence", 0.0) or 0.0)),
            }
        if error_type == "correct_weak_proof":
            scores = {
                "step_validity": max(3, original_scores["step_validity"]),
                "proof_completeness": max(2, original_scores["proof_completeness"]),
                "strategy_compliance": max(3, original_scores["strategy_compliance"]),
                "consistency": max(3, original_scores["consistency"]),
            }
            return {
                **scores,
                "error_type": "correct_weak_proof",
                "key_strength": "The strategy is valid and reaches the correct answer.",
                "key_weakness": "The proof is correct but lighter than a complete derivation.",
                "critical_failure_step": "",
                "judge_confidence": max(0.95, float(row.get("judge_confidence", 0.0) or 0.0)),
            }
        scores = {
            "step_validity": clamp_score(max(2, min(3, original_scores["step_validity"]))),
            "proof_completeness": clamp_score(max(1, min(2, original_scores["proof_completeness"]))),
            "strategy_compliance": clamp_score(max(1, min(2, original_scores["strategy_compliance"]))),
            "consistency": clamp_score(max(2, min(3, original_scores["consistency"]))),
        }
        return {
            **scores,
            "error_type": "lucky_correct",
            "key_strength": "The final answer is correct.",
            "key_weakness": "The reasoning is weaker than the correctness of the final answer suggests.",
            "critical_failure_step": "A key justification is missing or under-supported.",
            "judge_confidence": max(0.9, float(row.get("judge_confidence", 0.0) or 0.0)),
        }

    if error_type in BAD_STRATEGY_ERRORS:
        if error_type == "wrong_direction":
            scores = {"step_validity": 0, "proof_completeness": 0, "strategy_compliance": 0, "consistency": 1}
        elif error_type == "invalid_assumption":
            scores = {"step_validity": 1, "proof_completeness": 1, "strategy_compliance": 1, "consistency": 1}
        elif error_type == "strategy_mismatch":
            scores = {"step_validity": 1, "proof_completeness": 1, "strategy_compliance": 0, "consistency": 1}
        else:
            scores = {"step_validity": 0, "proof_completeness": 0, "strategy_compliance": 0, "consistency": 0}
        return {
            **scores,
            "error_type": error_type,
            "key_strength": "",
            "key_weakness": "The apparent strategy is not a valid route to the solution.",
            "critical_failure_step": "The trajectory commits to an invalid or mismatched direction early.",
            "judge_confidence": max(0.95, float(row.get("judge_confidence", 0.0) or 0.0)),
        }

    if error_type == "finalization_error":
        scores = {"step_validity": 3, "proof_completeness": 3, "strategy_compliance": 4, "consistency": 3}
        return {
            **scores,
            "error_type": "finalization_error",
            "key_strength": "The overall strategy is appropriate for the problem.",
            "key_weakness": "The derivation is mostly sound but the final answer handling is wrong.",
            "critical_failure_step": "The answer is lost during the last conversion or finalization step.",
            "judge_confidence": max(0.95, float(row.get("judge_confidence", 0.0) or 0.0)),
        }

    if error_type == "valid_but_incomplete":
        scores = {"step_validity": 2, "proof_completeness": 2, "strategy_compliance": 3, "consistency": 2}
        return {
            **scores,
            "error_type": "valid_but_incomplete",
            "key_strength": "The strategy points in a useful direction.",
            "key_weakness": "A key derivation step is missing before the answer is completed.",
            "critical_failure_step": "The solution stops before establishing the needed final relation.",
            "judge_confidence": max(0.95, float(row.get("judge_confidence", 0.0) or 0.0)),
        }

    if error_type in {"arithmetic_error", "algebraic_error"}:
        scores = {"step_validity": 2, "proof_completeness": 2, "strategy_compliance": 3, "consistency": 2}
        return {
            **scores,
            "error_type": error_type,
            "key_strength": "The strategy is broadly appropriate.",
            "key_weakness": "A computation or symbolic manipulation error breaks the solution.",
            "critical_failure_step": "A local arithmetic or algebraic step changes the result incorrectly.",
            "judge_confidence": max(0.95, float(row.get("judge_confidence", 0.0) or 0.0)),
        }

    if error_type == "format_error":
        scores = {"step_validity": 1, "proof_completeness": 1, "strategy_compliance": 1, "consistency": 1}
        return {
            **scores,
            "error_type": "format_error",
            "key_strength": "",
            "key_weakness": "The response does not cleanly deliver a usable final answer.",
            "critical_failure_step": "The final answer is not stated in a reliably parseable form.",
            "judge_confidence": max(0.9, float(row.get("judge_confidence", 0.0) or 0.0)),
        }

    return {
        "step_validity": 0,
        "proof_completeness": 0,
        "strategy_compliance": 0,
        "consistency": 0,
        "error_type": "no_meaningful_solution",
        "key_strength": "",
        "key_weakness": "The trajectory does not provide a meaningful valid solution.",
        "critical_failure_step": "No coherent mathematical path is established.",
        "judge_confidence": max(0.9, float(row.get("judge_confidence", 0.0) or 0.0)),
    }


def apply_prior_corrections(rows: list[dict[str, Any]], rollout_signals: dict[int, set[str]]) -> tuple[list[dict[str, Any]], list[int]]:
    corrected_rows: list[dict[str, Any]] = []
    changed_rollout_ids: list[int] = []

    for row in rows:
        updated = dict(row)
        rollout_id = int(row.get("group_rollout_id", 0) or 0)
        signals = rollout_signals.get(rollout_id, set())
        original_suitability = int(row.get("prior_suitability", 0) or 0)
        new_suitability = original_suitability
        new_reason = str(row.get("prior_reason", "") or "")
        new_risk_flag = str(row.get("prior_risk_flag", "") or "")

        if "low_prior_correct" in signals:
            error_type = str(row.get("error_type", "") or "")
            if error_type in RAISEABLE_CORRECT_ERRORS:
                new_suitability = max(original_suitability, 3)
                new_reason = "Valid direct strategy that matches the problem structure and supports a correct solution."
                new_risk_flag = "none"

        if "high_prior_wrong_bad_strategy" in signals:
            error_type = str(row.get("error_type", "") or "")
            cap = 2 if error_type == "strategy_mismatch" else 1
            new_suitability = min(new_suitability, cap)
            new_reason = "The strategy looks plausible on the surface but does not match a valid solution path."
            new_risk_flag = "strategy appears plausible but follows an invalid direction"

        if new_suitability != original_suitability or new_reason != str(row.get("prior_reason", "") or ""):
            updated["prior_suitability"] = new_suitability
            updated["prior_reason"] = new_reason
            updated["prior_risk_flag"] = new_risk_flag
            changed_rollout_ids.append(rollout_id)
        corrected_rows.append(updated)

    return corrected_rows, sorted(set(changed_rollout_ids))


def with_task_prefix(example: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(example)
    task = str(enriched["task"])
    prefix = TASK_PREFIXES[task]

    if task == "evidence_judge":
        compact_prompt = (
            prefix
            + "Return one JSON object with keys: "
            + "step_validity, proof_completeness, strategy_compliance, consistency, "
            + "error_type, key_strength, key_weakness, critical_failure_step, judge_confidence.\n"
            + "The four score fields are integers 0-4 and judge_confidence is a float in [0,1].\n"
            + "The deterministic correctness flag is authoritative.\n\n"
            + f"Problem:\n{enriched['problem']}\n\n"
            + f"Strategy:\n{enriched['strategy']}\n\n"
            + f"Reasoning:\n{enriched['reasoning']}\n\n"
            + f"Final Answer:\n{enriched['final_answer']}\n\n"
            + f"Deterministic correctness flag:\n{int(enriched['answer_correctness'])}\n"
        )
    elif task == "prior_judge":
        candidates_block = "\n\n".join(
            f"Rollout {candidate['rollout_id']}:\n{candidate['strategy']}"
            for candidate in enriched["candidates"]
        )
        compact_prompt = (
            prefix
            + "Return one JSON object with key 'priors'.\n"
            + "Each item in 'priors' must contain: rollout_id, suitability, reason, risk_flag.\n"
            + "Suitability is an integer from 0 to 4 and every rollout_id must appear exactly once.\n\n"
            + f"Problem:\n{enriched['problem']}\n\n"
            + f"Candidate strategies:\n{candidates_block}\n"
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    enriched["task_prefix"] = prefix
    enriched["prompt"] = compact_prompt
    enriched["messages"] = [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": compact_prompt},
        {
            "role": "assistant",
            "content": json.dumps(enriched["teacher_target"], ensure_ascii=False),
        },
    ]
    return enriched


def with_mixture_source(example: dict[str, Any], source_name: str) -> dict[str, Any]:
    updated = deepcopy(example)
    updated["mixture_source_name"] = source_name
    return updated


def load_prefixed_rows(path: Path, source_name: str) -> list[dict[str, Any]]:
    return [with_mixture_source(with_task_prefix(row), source_name) for row in load_jsonl_rows(path)]


def prefixed_rows(rows: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    return [with_mixture_source(with_task_prefix(row), source_name) for row in rows]


def sample_with_replacement(rows: list[dict[str, Any]], num_required: int, rng: random.Random) -> list[dict[str, Any]]:
    if num_required <= 0:
        return []
    if not rows:
        raise RuntimeError("Cannot sample from an empty dataset.")

    selected: list[dict[str, Any]] = []
    while len(selected) < num_required:
        block = list(rows)
        rng.shuffle(block)
        take = min(len(block), num_required - len(selected))
        selected.extend(deepcopy(block[:take]))
    return selected


def annotate_sampling(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["mixture_split"] = split_name
        updated["sampling_instance_index"] = index
        annotated.append(updated)
    return annotated


def bucket_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key, "unknown")) for row in rows)
    return dict(sorted(counter.items()))


def task_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(f"{row.get('mixture_source_name', 'unknown')}::{row['task']}" for row in rows)
    return dict(sorted(counter.items()))


def evidence_signal_map_for_group(
    rows: list[dict[str, Any]],
    *,
    low_prior_threshold: int,
    high_prior_threshold: int,
    high_likelihood_threshold: float,
    low_likelihood_threshold: float,
    wrong_top_rollout_ids: set[int],
    wrong_top_present: bool,
) -> dict[int, set[str]]:
    rollout_signals: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        rollout_id = int(row.get("group_rollout_id", 0) or 0)
        answer_correctness = float(row.get("answer_correctness", 0.0) or 0.0)
        prior_suitability = int(row.get("prior_suitability", 0) or 0)
        likelihood = float(row.get("likelihood", 0.0) or 0.0)
        error_type = str(row.get("error_type", "") or "")

        if answer_correctness == 1.0 and prior_suitability <= low_prior_threshold and likelihood >= high_likelihood_threshold:
            if error_type in RAISEABLE_CORRECT_ERRORS:
                rollout_signals[rollout_id].add("low_prior_correct")
            elif error_type == "lucky_correct":
                rollout_signals[rollout_id].add("low_prior_lucky_correct")

        if answer_correctness == 0.0 and prior_suitability >= high_prior_threshold and likelihood <= low_likelihood_threshold:
            if error_type in BAD_STRATEGY_ERRORS:
                rollout_signals[rollout_id].add("high_prior_wrong_bad_strategy")
            elif error_type in GOOD_EXECUTION_ERRORS:
                rollout_signals[rollout_id].add("high_prior_wrong_good_execution")
            else:
                rollout_signals[rollout_id].add("high_prior_wrong_other")

        if rollout_id in wrong_top_rollout_ids:
            rollout_signals[rollout_id].add("wrong_high_posterior_top")
        if wrong_top_present and answer_correctness == 1.0:
            rollout_signals[rollout_id].add("correct_available_but_not_top")

    return rollout_signals


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    bucket_ratios = normalize_ratio_map(
        {
            "anchor_clean": args.anchor_ratio,
            "stable_bootstrap": args.bootstrap_ratio,
            "retention": args.retention_ratio,
            "calibration_correction": args.correction_ratio,
        }
    )
    if args.evidence_weight <= 0.0 or args.prior_weight <= 0.0:
        raise ValueError("Both --evidence_weight and --prior_weight must be positive.")

    anchor_dir = Path(args.anchor_input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anchor_evidence_train = load_prefixed_rows(
        anchor_dir / args.anchor_evidence_train_file,
        source_name="anchor_clean",
    )
    anchor_evidence_val = load_prefixed_rows(
        anchor_dir / args.anchor_evidence_val_file,
        source_name="anchor_clean",
    )
    anchor_prior_train = load_prefixed_rows(
        anchor_dir / args.anchor_prior_train_file,
        source_name="anchor_clean",
    )
    anchor_prior_val = load_prefixed_rows(
        anchor_dir / args.anchor_prior_val_file,
        source_name="anchor_clean",
    )

    baseline_spec = RunSpec("v0_baseline", Path(args.baseline_debug_jsonl), "baseline")
    target_spec = RunSpec("v1_target", Path(args.target_debug_jsonl), "target")
    support_specs = [
        RunSpec(f"support_{index + 1}", Path(raw_path), "support")
        for index, raw_path in enumerate(args.support_debug_jsonl or [])
    ]
    all_specs = [baseline_spec, target_spec] + support_specs
    groups_by_run = {spec.run_name: load_trace_groups(spec.path) for spec in all_specs}

    common_keys = set.intersection(*(set(groups.keys()) for groups in groups_by_run.values()))
    if not common_keys:
        raise RuntimeError("No common train-trace groups were found across the provided debug files.")

    metadata_lookup = load_metadata_lookup(args.train_metadata_jsonl)

    bucket_evidence_rows: dict[str, list[dict[str, Any]]] = {
        "stable_bootstrap": [],
        "retention": [],
        "calibration_correction": [],
    }
    bucket_prior_rows: dict[str, list[dict[str, Any]]] = {
        "stable_bootstrap": [],
        "retention": [],
        "calibration_correction": [],
    }
    evidence_seen: dict[str, set[str]] = {key: set() for key in bucket_evidence_rows}
    prior_seen: dict[str, set[str]] = {key: set() for key in bucket_prior_rows}

    correction_signal_counts: Counter[str] = Counter()
    retention_signal_counts: Counter[str] = Counter()
    bootstrap_signal_counts: Counter[str] = Counter()

    for group_key in sorted(common_keys):
        baseline_rows = groups_by_run[baseline_spec.run_name][group_key]
        target_rows = groups_by_run[target_spec.run_name][group_key]
        support_rows = [groups_by_run[spec.run_name][group_key] for spec in support_specs]
        all_group_rows = [baseline_rows, target_rows] + support_rows

        split = assign_split(str(target_rows[0].get("problem", "") or ""), args.val_ratio)
        group_metrics_by_run = {
            spec.run_name: compute_group_metrics(groups_by_run[spec.run_name][group_key])
            for spec in all_specs
        }
        baseline_metrics = group_metrics_by_run[baseline_spec.run_name]
        candidate_metrics = {
            spec.run_name: group_metrics_by_run[spec.run_name]
            for spec in [target_spec] + support_specs
        }

        problem = str(target_rows[0].get("problem", "") or "")
        difficulty_bucket = None
        if problem in metadata_lookup:
            raw_bucket = metadata_lookup[problem].get("difficulty_bucket")
            if raw_bucket is not None:
                difficulty_bucket = str(raw_bucket)

        correction_reasons: set[str] = set()
        retention_reasons: set[str] = set()
        wrong_top_rollout_ids: set[int] = set()

        for run_name, metrics in candidate_metrics.items():
            if metrics.wrong_top:
                correction_reasons.add("wrong_high_posterior")
                wrong_top_rollout_ids.add(metrics.top_rollout_id)
            if baseline_metrics.top_answer_correctness == 1.0 and metrics.top_answer_correctness == 0.0:
                correction_reasons.add("regression_top_vs_v0")
            if baseline_metrics.mass_on_correct > metrics.mass_on_correct + args.retention_mass_margin:
                correction_reasons.add("regression_mass_vs_v0")
            if metrics.top_answer_correctness == 1.0 and baseline_metrics.top_answer_correctness == 0.0:
                retention_reasons.add("retention_top_vs_v0")
            if metrics.mass_on_correct > baseline_metrics.mass_on_correct + args.retention_mass_margin:
                retention_reasons.add("retention_mass_vs_v0")

        evidence_signals = evidence_signal_map_for_group(
            target_rows,
            low_prior_threshold=args.low_prior_threshold,
            high_prior_threshold=args.high_prior_threshold,
            high_likelihood_threshold=args.high_likelihood_threshold,
            low_likelihood_threshold=args.low_likelihood_threshold,
            wrong_top_rollout_ids=wrong_top_rollout_ids,
            wrong_top_present=bool(wrong_top_rollout_ids),
        )
        prior_signals = {
            rollout_id: {signal for signal in signals if signal in {"low_prior_correct", "high_prior_wrong_bad_strategy"}}
            for rollout_id, signals in evidence_signals.items()
            if any(signal in {"low_prior_correct", "high_prior_wrong_bad_strategy"} for signal in signals)
        }

        if any("low_prior_correct" in signals for signals in evidence_signals.values()):
            correction_reasons.add("low_prior_correct")
        if any("high_prior_wrong_bad_strategy" in signals for signals in evidence_signals.values()):
            correction_reasons.add("high_prior_wrong_bad_strategy")
        if any("high_prior_wrong_good_execution" in signals for signals in evidence_signals.values()):
            correction_reasons.add("high_prior_wrong_good_execution")

        bootstrap_eligible = not correction_reasons and not retention_reasons and is_stable_bootstrap_group(
            all_group_rows,
            min_judge_confidence=args.min_judge_confidence,
            bootstrap_prior_delta=args.bootstrap_prior_delta,
            bootstrap_score_delta=args.bootstrap_score_delta,
        )

        if bootstrap_eligible:
            bucket_name = "stable_bootstrap"
            bootstrap_signal_counts.update(["stable_group"])
            label_source = "train_trace_consensus_bootstrap"
            label_semantics = "high_confidence_train_trace_consensus"
            label_extra = {
                "bucket_name": bucket_name,
                "difficulty_bucket": difficulty_bucket,
                "consensus_runs": [spec.run_name for spec in all_specs],
            }

            prior_example = build_prior_example_from_rows(
                target_rows,
                source_file=target_spec.path,
                split=split,
                quality_tier="bootstrap",
                label_source=label_source,
                label_semantics=label_semantics,
                bucket_name=bucket_name,
                failure_type="stable_consensus",
                label_extra=label_extra,
                example_extra={"difficulty_bucket": difficulty_bucket},
            )
            if prior_example["example_id"] not in prior_seen[bucket_name]:
                prior_seen[bucket_name].add(prior_example["example_id"])
                bucket_prior_rows[bucket_name].append(prior_example)

            for row in target_rows:
                evidence_example = build_evidence_example_from_target(
                    row,
                    source_file=target_spec.path,
                    split=split,
                    quality_tier="bootstrap",
                    label_source=label_source,
                    label_semantics=label_semantics,
                    teacher_target={
                        "step_validity": int(row.get("step_validity", 0) or 0),
                        "proof_completeness": int(row.get("proof_completeness", 0) or 0),
                        "strategy_compliance": int(row.get("strategy_compliance", 0) or 0),
                        "consistency": int(row.get("consistency", 0) or 0),
                        "error_type": str(row.get("error_type", "") or ""),
                        "key_strength": str(row.get("key_strength", "") or ""),
                        "key_weakness": str(row.get("key_weakness", "") or ""),
                        "critical_failure_step": str(row.get("critical_failure_step", "") or ""),
                        "judge_confidence": float(row.get("judge_confidence", 0.0) or 0.0),
                    },
                    bucket_name=bucket_name,
                    failure_type="stable_consensus",
                    label_extra=label_extra,
                    example_extra={"difficulty_bucket": difficulty_bucket},
                )
                if evidence_example["example_id"] not in evidence_seen[bucket_name]:
                    evidence_seen[bucket_name].add(evidence_example["example_id"])
                    bucket_evidence_rows[bucket_name].append(evidence_example)
            continue

        corrected_prior_rows = target_rows
        changed_rollout_ids: list[int] = []
        if correction_reasons:
            corrected_prior_rows, changed_rollout_ids = apply_prior_corrections(
                target_rows,
                prior_signals,
            )

        if retention_reasons:
            bucket_name = "retention"
            retention_signal_counts.update(retention_reasons)
            label_source = "train_trace_v1_retention"
            label_semantics = "train_trace_retention"
            label_extra = {
                "bucket_name": bucket_name,
                "difficulty_bucket": difficulty_bucket,
                "retention_reasons": sorted(retention_reasons),
                "comparison_runs": list(candidate_metrics.keys()),
            }

            if not changed_rollout_ids:
                prior_example = build_prior_example_from_rows(
                    target_rows,
                    source_file=target_spec.path,
                    split=split,
                    quality_tier="retention",
                    label_source=label_source,
                    label_semantics=label_semantics,
                    bucket_name=bucket_name,
                    failure_type="retention_group",
                    label_extra=label_extra,
                    example_extra={"difficulty_bucket": difficulty_bucket},
                )
                if prior_example["example_id"] not in prior_seen[bucket_name]:
                    prior_seen[bucket_name].add(prior_example["example_id"])
                    bucket_prior_rows[bucket_name].append(prior_example)

            for row in target_rows:
                if float(row.get("answer_correctness", 0.0) or 0.0) != 1.0:
                    continue
                evidence_example = build_evidence_example_from_target(
                    row,
                    source_file=target_spec.path,
                    split=split,
                    quality_tier="retention",
                    label_source=label_source,
                    label_semantics=label_semantics,
                    teacher_target=correction_evidence_target(row, "retention_positive"),
                    bucket_name=bucket_name,
                    failure_type="retention_positive",
                    label_extra=label_extra,
                    example_extra={"difficulty_bucket": difficulty_bucket},
                )
                if evidence_example["example_id"] not in evidence_seen[bucket_name]:
                    evidence_seen[bucket_name].add(evidence_example["example_id"])
                    bucket_evidence_rows[bucket_name].append(evidence_example)

        if correction_reasons:
            bucket_name = "calibration_correction"
            correction_signal_counts.update(correction_reasons)
            label_source = "train_trace_calibration_correction"
            label_semantics = "automatic_reward_failure_calibration"
            label_extra = {
                "bucket_name": bucket_name,
                "difficulty_bucket": difficulty_bucket,
                "correction_reasons": sorted(correction_reasons),
                "comparison_runs": list(candidate_metrics.keys()),
                "wrong_top_rollout_ids": sorted(wrong_top_rollout_ids),
            }

            if changed_rollout_ids:
                prior_example = build_prior_example_from_rows(
                    corrected_prior_rows,
                    source_file=target_spec.path,
                    split=split,
                    quality_tier="calibration_correction",
                    label_source=label_source,
                    label_semantics=label_semantics,
                    bucket_name=bucket_name,
                    failure_type="prior_calibration_correction",
                    label_extra={**label_extra, "changed_rollout_ids": changed_rollout_ids},
                    example_extra={
                        "difficulty_bucket": difficulty_bucket,
                        "rollout_signals": {
                            rollout_id: sorted(signals)
                            for rollout_id, signals in prior_signals.items()
                            if signals
                        },
                    },
                )
                if prior_example["example_id"] not in prior_seen[bucket_name]:
                    prior_seen[bucket_name].add(prior_example["example_id"])
                    bucket_prior_rows[bucket_name].append(prior_example)

            for row in target_rows:
                rollout_id = int(row.get("group_rollout_id", 0) or 0)
                signals = evidence_signals.get(rollout_id, set())
                if not signals:
                    continue
                preferred_signal = (
                    "high_prior_wrong_bad_strategy"
                    if "high_prior_wrong_bad_strategy" in signals
                    else "high_prior_wrong_good_execution"
                    if "high_prior_wrong_good_execution" in signals
                    else "low_prior_correct"
                    if "low_prior_correct" in signals
                    else "wrong_high_posterior_top"
                    if "wrong_high_posterior_top" in signals
                    else "correct_available_but_not_top"
                    if "correct_available_but_not_top" in signals
                    else sorted(signals)[0]
                )
                evidence_example = build_evidence_example_from_target(
                    row,
                    source_file=target_spec.path,
                    split=split,
                    quality_tier="calibration_correction",
                    label_source=label_source,
                    label_semantics=label_semantics,
                    teacher_target=correction_evidence_target(row, preferred_signal),
                    bucket_name=bucket_name,
                    failure_type=preferred_signal,
                    label_extra={**label_extra, "rollout_id": rollout_id},
                    example_extra={
                        "difficulty_bucket": difficulty_bucket,
                        "failure_signals": sorted(signals),
                    },
                )
                if evidence_example["example_id"] not in evidence_seen[bucket_name]:
                    evidence_seen[bucket_name].add(evidence_example["example_id"])
                    bucket_evidence_rows[bucket_name].append(evidence_example)

    for bucket_name in bucket_evidence_rows:
        bucket_evidence_rows[bucket_name].sort(key=lambda row: row["example_id"])
    for bucket_name in bucket_prior_rows:
        bucket_prior_rows[bucket_name].sort(key=lambda row: row["example_id"])

    bucket_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for bucket_name in bucket_evidence_rows:
        evidence_train, evidence_val = split_rows(bucket_evidence_rows[bucket_name])
        prior_train, prior_val = split_rows(bucket_prior_rows[bucket_name])
        bucket_splits[bucket_name] = {
            "evidence_train": evidence_train,
            "evidence_val": evidence_val,
            "prior_train": prior_train,
            "prior_val": prior_val,
        }
        write_jsonl(output_dir / f"{bucket_name}_evidence_train.jsonl", evidence_train)
        write_jsonl(output_dir / f"{bucket_name}_evidence_val.jsonl", evidence_val)
        write_jsonl(output_dir / f"{bucket_name}_prior_train.jsonl", prior_train)
        write_jsonl(output_dir / f"{bucket_name}_prior_val.jsonl", prior_val)

    evidence_ratio = args.evidence_weight / (args.evidence_weight + args.prior_weight)
    prior_ratio = 1.0 - evidence_ratio

    source_evidence_train = {
        "anchor_clean": anchor_evidence_train,
        "stable_bootstrap": prefixed_rows(bucket_splits["stable_bootstrap"]["evidence_train"], "stable_bootstrap"),
        "retention": prefixed_rows(bucket_splits["retention"]["evidence_train"], "retention"),
        "calibration_correction": prefixed_rows(
            bucket_splits["calibration_correction"]["evidence_train"],
            "calibration_correction",
        ),
    }
    source_prior_train = {
        "anchor_clean": anchor_prior_train,
        "stable_bootstrap": prefixed_rows(bucket_splits["stable_bootstrap"]["prior_train"], "stable_bootstrap"),
        "retention": prefixed_rows(bucket_splits["retention"]["prior_train"], "retention"),
        "calibration_correction": prefixed_rows(
            bucket_splits["calibration_correction"]["prior_train"],
            "calibration_correction",
        ),
    }

    if args.max_train_examples is not None:
        train_size = int(args.max_train_examples)
    else:
        candidate_sizes: list[float] = []
        for bucket_name, ratio in bucket_ratios.items():
            if ratio <= 0.0:
                continue
            evidence_rows = source_evidence_train[bucket_name]
            prior_rows = source_prior_train[bucket_name]
            if evidence_rows:
                candidate_sizes.append(len(evidence_rows) / max(1e-9, evidence_ratio * ratio))
            if prior_rows:
                candidate_sizes.append(len(prior_rows) / max(1e-9, prior_ratio * ratio))
        train_size = math.ceil(max(candidate_sizes)) if candidate_sizes else 0

    num_evidence = round(train_size * evidence_ratio)
    num_prior = train_size - num_evidence

    def allocate(total: int) -> dict[str, int]:
        raw = {bucket: total * ratio for bucket, ratio in bucket_ratios.items()}
        allocated = {bucket: int(math.floor(value)) for bucket, value in raw.items()}
        remainder = total - sum(allocated.values())
        fractional_order = sorted(
            bucket_ratios.keys(),
            key=lambda bucket: raw[bucket] - allocated[bucket],
            reverse=True,
        )
        for bucket in fractional_order[:remainder]:
            allocated[bucket] += 1
        return allocated

    evidence_allocation = allocate(num_evidence)
    prior_allocation = allocate(num_prior)

    unified_train: list[dict[str, Any]] = []
    for bucket_name, count in evidence_allocation.items():
        if count <= 0:
            continue
        unified_train.extend(sample_with_replacement(source_evidence_train[bucket_name], count, rng))
    for bucket_name, count in prior_allocation.items():
        if count <= 0:
            continue
        unified_train.extend(sample_with_replacement(source_prior_train[bucket_name], count, rng))
    rng.shuffle(unified_train)
    unified_train = annotate_sampling(unified_train, "train")

    unified_val = list(anchor_evidence_val) + list(anchor_prior_val)
    if args.include_bucket_val:
        unified_val.extend(prefixed_rows(bucket_splits["stable_bootstrap"]["evidence_val"], "stable_bootstrap"))
        unified_val.extend(prefixed_rows(bucket_splits["stable_bootstrap"]["prior_val"], "stable_bootstrap"))
        unified_val.extend(prefixed_rows(bucket_splits["retention"]["evidence_val"], "retention"))
        unified_val.extend(prefixed_rows(bucket_splits["retention"]["prior_val"], "retention"))
        unified_val.extend(
            prefixed_rows(
                bucket_splits["calibration_correction"]["evidence_val"],
                "calibration_correction",
            )
        )
        unified_val.extend(
            prefixed_rows(
                bucket_splits["calibration_correction"]["prior_val"],
                "calibration_correction",
            )
        )
    unified_val = annotate_sampling(unified_val, "val")

    write_jsonl(output_dir / "unified_train.jsonl", unified_train)
    write_jsonl(output_dir / "unified_val.jsonl", unified_val)
    write_jsonl(output_dir / "anchor_evidence_val_marked.jsonl", anchor_evidence_val)
    write_jsonl(output_dir / "anchor_prior_val_marked.jsonl", anchor_prior_val)

    summary = {
        "config": {
            "anchor_input_dir": safe_relative_path(anchor_dir),
            "baseline_debug_jsonl": safe_relative_path(baseline_spec.path),
            "target_debug_jsonl": safe_relative_path(target_spec.path),
            "support_debug_jsonl": [safe_relative_path(spec.path) for spec in support_specs],
            "train_metadata_jsonl": [safe_relative_path(Path(path)) for path in (args.train_metadata_jsonl or [])],
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "min_judge_confidence": args.min_judge_confidence,
            "low_prior_threshold": args.low_prior_threshold,
            "high_prior_threshold": args.high_prior_threshold,
            "high_likelihood_threshold": args.high_likelihood_threshold,
            "low_likelihood_threshold": args.low_likelihood_threshold,
            "retention_mass_margin": args.retention_mass_margin,
            "bootstrap_prior_delta": args.bootstrap_prior_delta,
            "bootstrap_score_delta": args.bootstrap_score_delta,
            "bucket_ratios": bucket_ratios,
            "evidence_weight": args.evidence_weight,
            "prior_weight": args.prior_weight,
            "include_bucket_val": args.include_bucket_val,
            "max_train_examples": args.max_train_examples,
        },
        "counts": {
            "common_groups": len(common_keys),
            "anchor_evidence_train": len(anchor_evidence_train),
            "anchor_evidence_val": len(anchor_evidence_val),
            "anchor_prior_train": len(anchor_prior_train),
            "anchor_prior_val": len(anchor_prior_val),
            "stable_bootstrap_evidence_train": len(bucket_splits["stable_bootstrap"]["evidence_train"]),
            "stable_bootstrap_evidence_val": len(bucket_splits["stable_bootstrap"]["evidence_val"]),
            "stable_bootstrap_prior_train": len(bucket_splits["stable_bootstrap"]["prior_train"]),
            "stable_bootstrap_prior_val": len(bucket_splits["stable_bootstrap"]["prior_val"]),
            "retention_evidence_train": len(bucket_splits["retention"]["evidence_train"]),
            "retention_evidence_val": len(bucket_splits["retention"]["evidence_val"]),
            "retention_prior_train": len(bucket_splits["retention"]["prior_train"]),
            "retention_prior_val": len(bucket_splits["retention"]["prior_val"]),
            "correction_evidence_train": len(bucket_splits["calibration_correction"]["evidence_train"]),
            "correction_evidence_val": len(bucket_splits["calibration_correction"]["evidence_val"]),
            "correction_prior_train": len(bucket_splits["calibration_correction"]["prior_train"]),
            "correction_prior_val": len(bucket_splits["calibration_correction"]["prior_val"]),
            "unified_train": len(unified_train),
            "unified_val": len(unified_val),
        },
        "signal_counts": {
            "bootstrap": dict(sorted(bootstrap_signal_counts.items())),
            "retention": dict(sorted(retention_signal_counts.items())),
            "correction": dict(sorted(correction_signal_counts.items())),
        },
        "bucket_distributions": {
            "stable_bootstrap_evidence_failure_types": bucket_counts(bucket_evidence_rows["stable_bootstrap"], "failure_type"),
            "retention_evidence_failure_types": bucket_counts(bucket_evidence_rows["retention"], "failure_type"),
            "correction_evidence_failure_types": bucket_counts(bucket_evidence_rows["calibration_correction"], "failure_type"),
        },
        "mixture": {
            "effective_num_evidence_examples": num_evidence,
            "effective_num_prior_examples": num_prior,
            "evidence_allocation": evidence_allocation,
            "prior_allocation": prior_allocation,
            "train_mixture_source_counts": bucket_counts(unified_train, "mixture_source_name"),
            "val_mixture_source_counts": bucket_counts(unified_val, "mixture_source_name"),
            "train_task_mixture_counts": task_bucket_counts(unified_train),
            "val_task_mixture_counts": task_bucket_counts(unified_val),
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(f"[INFO] wrote analyzer v1.1 calibration data to {output_dir}")
    print(
        "[INFO] buckets "
        f"bootstrap_evidence={len(bucket_splits['stable_bootstrap']['evidence_train'])} "
        f"bootstrap_prior={len(bucket_splits['stable_bootstrap']['prior_train'])} | "
        f"retention_evidence={len(bucket_splits['retention']['evidence_train'])} "
        f"retention_prior={len(bucket_splits['retention']['prior_train'])} | "
        f"correction_evidence={len(bucket_splits['calibration_correction']['evidence_train'])} "
        f"correction_prior={len(bucket_splits['calibration_correction']['prior_train'])}"
    )
    print(
        "[INFO] unified "
        f"train={len(unified_train)} val={len(unified_val)} "
        f"(evidence={num_evidence}, prior={num_prior})"
    )


if __name__ == "__main__":
    main()
