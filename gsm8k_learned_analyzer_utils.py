#!/usr/bin/env python3
"""Shared helpers for the GSM8K learned-analyzer distillation pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JSON_ONLY_SYSTEM_PROMPT = """You are a JSON-only evaluation API.
Return exactly one valid JSON object and nothing else.
Do not include markdown.
Do not include code blocks.
Do not include explanations outside the JSON.
Use double quotes for all keys and string values.
Do not use LaTeX notation inside JSON string values.
Do not use backslashes inside JSON string values.
Write short plain-English reason fields only.
The first character of your response must be {.
The last character of your response must be }."""


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


ALLOWED_ERROR_TYPES = {
    "correct_complete",
    "correct_weak_proof",
    "lucky_correct",
    "finalization_error",
    "valid_but_incomplete",
    "arithmetic_error",
    "algebraic_error",
    "invalid_assumption",
    "strategy_mismatch",
    "wrong_direction",
    "format_error",
    "no_meaningful_solution",
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def maybe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def clamp_int_0_to_4(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = 0
    return max(0, min(4, numeric))


def clamp01(value: Any) -> float:
    numeric = maybe_float(value)
    if numeric is None:
        return 0.0
    return max(0.0, min(1.0, numeric))


def discover_debug_jsonl(log_dir_or_path: str | Path) -> Path:
    path = Path(log_dir_or_path)
    if path.is_file():
        return path
    candidate = path / "bayesian_reward_debug.jsonl"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find bayesian_reward_debug.jsonl under {path}"
    )


def question_id_for(question: str, gold_answer: str) -> str:
    return stable_hash(json.dumps([question, gold_answer], ensure_ascii=False))


def question_split(question_id: str, val_ratio: float) -> str:
    bucket = int(question_id[:8], 16) / 0xFFFFFFFF
    return "valid" if bucket < float(val_ratio) else "train"


def relative_path(path: str | Path) -> str:
    return str(Path(path))


def build_simple_prior_prompt(question: str, strategy: str) -> str:
    return f"""
Return ONLY one valid JSON object.
Do not include markdown.
Do not include code blocks.
Do not include explanations outside the JSON.

You are judging whether a solver strategy is promising before seeing the full rollout.

Return exactly these keys:
- strategy_relevance: integer 0 to 4
- problem_fit: integer 0 to 4
- risk_of_error: integer 0 to 4 where larger means higher risk
- prior_score: integer 0 to 4
- brief_reason: short plain-English string

Rules:
- Judge only from the problem and the strategy.
- Do not use hindsight from correctness or the gold answer.
- Do not solve the full problem.
- The first character must be {{ and the last character must be }}.

Problem:
{question}

Strategy:
{strategy or "(empty strategy)"}
""".strip()


def build_simple_evidence_prompt(
    question: str,
    gold_answer: str,
    strategy: str,
    rollout_solution: str,
    predicted_answer: str,
    answer_correctness: float,
) -> str:
    return f"""
Return ONLY one valid JSON object.
Do not include markdown.
Do not include code blocks.
Do not include explanations outside the JSON.

You are judging the quality of a solver rollout for a math problem.

Return exactly these keys:
- answer_correctness: integer 0 or 1
- step_validity: integer 0 to 4
- proof_completeness: integer 0 to 4
- strategy_compliance: integer 0 to 4
- consistency: integer 0 to 4
- error_type: one of {", ".join(sorted(ALLOWED_ERROR_TYPES))}
- likelihood_score: float in [0, 1]
- brief_reason: short plain-English string

Rules:
- The deterministic correctness flag is authoritative.
- Do not override the correctness flag.
- likelihood_score should reflect how trustworthy the rollout is overall.
- The first character must be {{ and the last character must be }}.

Problem:
{question}

Gold answer:
{gold_answer}

Strategy:
{strategy or "(empty strategy)"}

Rollout solution:
{rollout_solution or "(empty rollout)"}

Predicted final answer:
{predicted_answer or "(empty answer)"}

Deterministic correctness flag:
{int(answer_correctness)}
""".strip()


def build_messages(prompt: str, target: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]


def add_task_prefix(prompt: str, task: str) -> str:
    return TASK_PREFIXES[task] + prompt


def build_runtime_evidence_prompt(
    question: str,
    strategy: str,
    reasoning: str,
    final_answer: str,
    answer_correctness: float,
) -> str:
    from prepare_analyzer_training_data import build_evidence_judge_prompt

    return build_evidence_judge_prompt(
        question,
        strategy,
        reasoning,
        final_answer,
        answer_correctness,
    )


def build_runtime_prior_prompt(
    question: str,
    candidates: list[dict[str, Any]],
) -> str:
    from prepare_analyzer_training_data import build_prior_judge_prompt

    return build_prior_judge_prompt(question, candidates, len(candidates))


def extract_prompted_prior_item(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = row.get("parsed_prior_judge_json")
    if not isinstance(parsed, dict):
        return None
    priors = parsed.get("priors")
    if not isinstance(priors, list):
        return None
    rollout_id = int(row.get("group_rollout_id", 0) or 0)
    for item in priors:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("rollout_id"))
        except (TypeError, ValueError):
            continue
        if item_id == rollout_id:
            return item
    return None


def derive_simple_prior_target(row: dict[str, Any]) -> dict[str, Any]:
    prior_item = extract_prompted_prior_item(row) or {}
    suitability = clamp_int_0_to_4(
        prior_item.get("suitability", row.get("prior_suitability", 0))
    )
    risk_flag = str(
        prior_item.get("risk_flag", row.get("prior_risk_flag", "") or "")
    ).strip()
    reason = str(prior_item.get("reason", row.get("prior_reason", "") or "")).strip()
    risk_of_error = 0 if risk_flag in {"", "none", "low"} else 2
    risk_of_error = max(risk_of_error, 4 - suitability)
    return {
        "strategy_relevance": suitability,
        "problem_fit": suitability,
        "risk_of_error": clamp_int_0_to_4(risk_of_error),
        "prior_score": suitability,
        "brief_reason": reason or "Teacher prior judgment.",
    }


def derive_runtime_prior_target(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        group_rows,
        key=lambda item: int(item.get("group_rollout_id", 0) or 0),
    )
    return {
        "priors": [
            {
                "rollout_id": int(row.get("group_rollout_id", 0) or 0),
                "suitability": clamp_int_0_to_4(row.get("prior_suitability", 0)),
                "reason": str(row.get("prior_reason", "") or ""),
                "risk_flag": str(row.get("prior_risk_flag", "") or ""),
            }
            for row in ordered
        ]
    }


def derive_simple_evidence_target(row: dict[str, Any]) -> dict[str, Any]:
    reason = str(row.get("key_weakness", "") or "").strip()
    if not reason:
        reason = str(row.get("key_strength", "") or "").strip()
    if not reason:
        reason = str(row.get("critical_failure_step", "") or "").strip()
    if not reason:
        reason = "Teacher evidence judgment."
    return {
        "answer_correctness": int(float(row.get("answer_correctness", 0.0) or 0.0)),
        "step_validity": clamp_int_0_to_4(row.get("step_validity", 0)),
        "proof_completeness": clamp_int_0_to_4(row.get("proof_completeness", 0)),
        "strategy_compliance": clamp_int_0_to_4(row.get("strategy_compliance", 0)),
        "consistency": clamp_int_0_to_4(row.get("consistency", 0)),
        "error_type": str(row.get("error_type", "") or "format_error"),
        "likelihood_score": clamp01(row.get("likelihood", 0.0)),
        "brief_reason": reason,
    }


def derive_runtime_evidence_target(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_validity": clamp_int_0_to_4(row.get("step_validity", 0)),
        "proof_completeness": clamp_int_0_to_4(row.get("proof_completeness", 0)),
        "strategy_compliance": clamp_int_0_to_4(row.get("strategy_compliance", 0)),
        "consistency": clamp_int_0_to_4(row.get("consistency", 0)),
        "error_type": str(row.get("error_type", "") or "format_error"),
        "key_strength": str(row.get("key_strength", "") or ""),
        "key_weakness": str(row.get("key_weakness", "") or ""),
        "critical_failure_step": str(row.get("critical_failure_step", "") or ""),
        "judge_confidence": clamp01(row.get("judge_confidence", 0.0)),
    }


def build_record_id(question_id: str, rollout_id: int) -> str:
    return f"{question_id}:{rollout_id}"


def assert_disjoint_question_splits(rows: list[dict[str, Any]], *, label: str) -> dict[str, int]:
    train_ids = {
        str(row["question_id"])
        for row in rows
        if str(row.get("split", "")) == "train"
    }
    valid_ids = {
        str(row["question_id"])
        for row in rows
        if str(row.get("split", "")) == "valid"
    }
    overlap = sorted(train_ids & valid_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise AssertionError(
            f"{label}: question_id split leakage detected for {len(overlap)} ids: {preview}"
        )
    return {
        "train_question_ids": len(train_ids),
        "valid_question_ids": len(valid_ids),
        "intersection_size": 0,
    }


def _top_bottom_index_sets(values: list[float], fraction: float) -> tuple[set[int], set[int]]:
    if not values:
        return set(), set()
    count = max(1, math.ceil(len(values) * min(max(fraction, 0.0), 0.5)))
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    return set(ordered[-count:]), set(ordered[:count])


def _group_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        int(row.get("global_reward_call_index", 0) or 0),
        int(row.get("group_index_within_call", 0) or 0),
    )


def _clean_prior_group(group_rows: list[dict[str, Any]]) -> bool:
    if any(row.get("prior_judge_fallback_used") for row in group_rows):
        return False
    if any(not isinstance(row.get("parsed_prior_judge_json"), dict) for row in group_rows):
        return False
    scores = [maybe_float(row.get("prior_probability")) for row in group_rows]
    if any(score is None for score in scores):
        return False
    return True


def _clean_evidence_row(row: dict[str, Any]) -> bool:
    if row.get("evidence_judge_fallback_used"):
        return False
    if row.get("prior_judge_fallback_used"):
        return False
    if row.get("parse_failure_reasons"):
        return False
    if not isinstance(row.get("parsed_judge_json"), dict):
        return False
    if not isinstance(row.get("parsed_prior_judge_json"), dict):
        return False
    if not bool(row.get("format_valid", False)):
        return False
    if not bool(row.get("strategy_section_present", False)):
        return False
    if not bool(row.get("reasoning_section_present", False)):
        return False
    if not bool(row.get("final_answer_section_present", False)):
        return False
    if maybe_float(row.get("likelihood")) is None:
        return False
    if maybe_float(row.get("bayesian_reward")) is None:
        return False
    if maybe_float(row.get("prior_probability")) is None:
        return False
    if str(row.get("error_type", "") or "") not in ALLOWED_ERROR_TYPES:
        return False
    return True


def evidence_drop_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("prior_judge_fallback_used") or row.get("evidence_judge_fallback_used"):
        reasons.append("fallback")
    if row.get("parse_failure_reasons"):
        reasons.append("parse_failure")
    if not isinstance(row.get("parsed_judge_json"), dict):
        reasons.append("missing_evidence_json")
    if not isinstance(row.get("parsed_prior_judge_json"), dict):
        reasons.append("missing_prior_json")
    if maybe_float(row.get("likelihood")) is None or maybe_float(row.get("bayesian_reward")) is None:
        reasons.append("nan_posterior_or_likelihood")
    if maybe_float(row.get("prior_probability")) is None:
        reasons.append("nan_prior_probability")
    score_fields = (
        row.get("step_validity"),
        row.get("proof_completeness"),
        row.get("strategy_compliance"),
        row.get("consistency"),
    )
    if any(maybe_float(value) is None for value in score_fields):
        reasons.append("missing_score_field")
    elif any(not (0.0 <= float(value) <= 4.0) for value in score_fields):
        reasons.append("score_out_of_range")
    if str(row.get("error_type", "") or "") not in ALLOWED_ERROR_TYPES:
        reasons.append("invalid_error_type")
    missing_field = False
    if not bool(row.get("format_valid", False)):
        missing_field = True
    if not bool(row.get("strategy_section_present", False)):
        missing_field = True
    if not bool(row.get("reasoning_section_present", False)):
        missing_field = True
    if not bool(row.get("final_answer_section_present", False)):
        missing_field = True
    if missing_field:
        reasons.append("missing_field")
    return sorted(set(reasons))


def prior_group_drop_reasons(group_rows: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if any(row.get("prior_judge_fallback_used") for row in group_rows):
        reasons.append("fallback")
    if any(not isinstance(row.get("parsed_prior_judge_json"), dict) for row in group_rows):
        reasons.append("parse_failure")
    if any(maybe_float(row.get("prior_probability")) is None for row in group_rows):
        reasons.append("nan_prior_probability")
    if any(row.get("prior_suitability") is None for row in group_rows):
        reasons.append("missing_score_field")
    else:
        suitabilities = [int(row.get("prior_suitability", 0) or 0) for row in group_rows]
        if any(not (0 <= value <= 4) for value in suitabilities):
            reasons.append("score_out_of_range")
    if any(not bool(row.get("strategy_section_present", False)) for row in group_rows):
        reasons.append("missing_field")
    return sorted(set(reasons))


def summarize_runtime_filtering(
    raw_rows: list[dict[str, Any]],
    rollout_records: list[dict[str, Any]],
    *,
    evidence_fraction: float,
) -> dict[str, Any]:
    def evidence_parse_failed(row: dict[str, Any]) -> bool:
        return bool(row.get("parse_failure_reasons")) or not isinstance(
            row.get("parsed_judge_json"), dict
        ) or not isinstance(row.get("parsed_prior_judge_json"), dict)

    def evidence_has_fallback(row: dict[str, Any]) -> bool:
        return bool(row.get("prior_judge_fallback_used")) or bool(
            row.get("evidence_judge_fallback_used")
        )

    def evidence_has_nan_posterior_or_likelihood(row: dict[str, Any]) -> bool:
        return maybe_float(row.get("likelihood")) is None or maybe_float(
            row.get("bayesian_reward")
        ) is None

    def evidence_has_nan_prior(row: dict[str, Any]) -> bool:
        return maybe_float(row.get("prior_probability")) is None

    def evidence_missing_score_field(row: dict[str, Any]) -> bool:
        score_fields = (
            row.get("step_validity"),
            row.get("proof_completeness"),
            row.get("strategy_compliance"),
            row.get("consistency"),
        )
        return any(maybe_float(value) is None for value in score_fields)

    def evidence_score_out_of_range(row: dict[str, Any]) -> bool:
        score_fields = (
            row.get("step_validity"),
            row.get("proof_completeness"),
            row.get("strategy_compliance"),
            row.get("consistency"),
        )
        return any(
            maybe_float(value) is not None and not (0.0 <= float(value) <= 4.0)
            for value in score_fields
        )

    def evidence_invalid_error_type(row: dict[str, Any]) -> bool:
        return str(row.get("error_type", "") or "") not in ALLOWED_ERROR_TYPES

    def evidence_missing_field(row: dict[str, Any]) -> bool:
        return (
            not bool(row.get("format_valid", False))
            or not bool(row.get("strategy_section_present", False))
            or not bool(row.get("reasoning_section_present", False))
            or not bool(row.get("final_answer_section_present", False))
        )

    def prior_parse_failed(group_rows: list[dict[str, Any]]) -> bool:
        return any(
            not isinstance(row.get("parsed_prior_judge_json"), dict)
            for row in group_rows
        )

    def prior_has_fallback(group_rows: list[dict[str, Any]]) -> bool:
        return any(row.get("prior_judge_fallback_used") for row in group_rows)

    def prior_has_nan_probability(group_rows: list[dict[str, Any]]) -> bool:
        return any(maybe_float(row.get("prior_probability")) is None for row in group_rows)

    def prior_missing_score_field(group_rows: list[dict[str, Any]]) -> bool:
        return any(row.get("prior_suitability") is None for row in group_rows)

    def prior_score_out_of_range(group_rows: list[dict[str, Any]]) -> bool:
        return any(
            row.get("prior_suitability") is not None
            and not (0 <= int(row.get("prior_suitability", 0) or 0) <= 4)
            for row in group_rows
        )

    def prior_missing_field(group_rows: list[dict[str, Any]]) -> bool:
        return any(not bool(row.get("strategy_section_present", False)) for row in group_rows)

    evidence_flag_counts: Counter[str] = Counter()
    evidence_primary_counts: Counter[str] = Counter()
    for row in raw_rows:
        reasons = evidence_drop_reasons(row)
        for reason in reasons:
            evidence_flag_counts[reason] += 1
        if reasons:
            evidence_primary_counts[reasons[0]] += 1

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[_group_key(row)].append(row)

    prior_flag_counts: Counter[str] = Counter()
    prior_primary_counts: Counter[str] = Counter()
    for group_rows in grouped.values():
        reasons = prior_group_drop_reasons(group_rows)
        for reason in reasons:
            prior_flag_counts[reason] += 1
        if reasons:
            prior_primary_counts[reasons[0]] += 1

    evidence_step_rows = list(raw_rows)
    evidence_parse_removed = sum(1 for row in evidence_step_rows if evidence_parse_failed(row))
    evidence_step_rows = [row for row in evidence_step_rows if not evidence_parse_failed(row)]
    evidence_fallback_removed = sum(1 for row in evidence_step_rows if evidence_has_fallback(row))
    evidence_step_rows = [row for row in evidence_step_rows if not evidence_has_fallback(row)]
    evidence_nan_removed = sum(
        1 for row in evidence_step_rows if evidence_has_nan_posterior_or_likelihood(row)
    )
    evidence_step_rows = [
        row for row in evidence_step_rows if not evidence_has_nan_posterior_or_likelihood(row)
    ]
    evidence_nan_prior_removed = sum(1 for row in evidence_step_rows if evidence_has_nan_prior(row))
    evidence_step_rows = [row for row in evidence_step_rows if not evidence_has_nan_prior(row)]
    evidence_missing_score_removed = sum(
        1 for row in evidence_step_rows if evidence_missing_score_field(row)
    )
    evidence_step_rows = [row for row in evidence_step_rows if not evidence_missing_score_field(row)]
    evidence_score_removed = sum(1 for row in evidence_step_rows if evidence_score_out_of_range(row))
    evidence_step_rows = [row for row in evidence_step_rows if not evidence_score_out_of_range(row)]
    evidence_invalid_error_type_removed = sum(
        1 for row in evidence_step_rows if evidence_invalid_error_type(row)
    )
    evidence_step_rows = [
        row for row in evidence_step_rows if not evidence_invalid_error_type(row)
    ]
    evidence_missing_field_removed = sum(1 for row in evidence_step_rows if evidence_missing_field(row))
    evidence_step_rows = [row for row in evidence_step_rows if not evidence_missing_field(row)]

    prior_step_groups = list(grouped.values())
    prior_parse_removed = sum(1 for rows in prior_step_groups if prior_parse_failed(rows))
    prior_step_groups = [rows for rows in prior_step_groups if not prior_parse_failed(rows)]
    prior_fallback_removed = sum(1 for rows in prior_step_groups if prior_has_fallback(rows))
    prior_step_groups = [rows for rows in prior_step_groups if not prior_has_fallback(rows)]
    prior_nan_removed = sum(1 for rows in prior_step_groups if prior_has_nan_probability(rows))
    prior_step_groups = [rows for rows in prior_step_groups if not prior_has_nan_probability(rows)]
    prior_missing_score_removed = sum(
        1 for rows in prior_step_groups if prior_missing_score_field(rows)
    )
    prior_step_groups = [rows for rows in prior_step_groups if not prior_missing_score_field(rows)]
    prior_score_removed = sum(1 for rows in prior_step_groups if prior_score_out_of_range(rows))
    prior_step_groups = [rows for rows in prior_step_groups if not prior_score_out_of_range(rows)]
    prior_missing_field_removed = sum(1 for rows in prior_step_groups if prior_missing_field(rows))
    prior_step_groups = [rows for rows in prior_step_groups if not prior_missing_field(rows)]

    clean_evidence_records = [row for row in rollout_records if row["clean_evidence"]]
    clean_prior_groups = build_runtime_prior_group_examples(rollout_records)
    train_evidence_records = [row for row in clean_evidence_records if row["split"] == "train"]
    valid_evidence_records = [row for row in clean_evidence_records if row["split"] == "valid"]
    train_prior_groups = [row for row in clean_prior_groups if row["split"] == "train"]
    valid_prior_groups = [row for row in clean_prior_groups if row["split"] == "valid"]

    expected_train_evidence = 0
    expected_valid_evidence = 0
    if train_prior_groups and evidence_fraction > 0:
        expected_train_evidence = int(
            round(len(train_prior_groups) * evidence_fraction / max(1.0 - evidence_fraction, 1e-8))
        )
    if valid_prior_groups and evidence_fraction > 0:
        expected_valid_evidence = int(
            round(len(valid_prior_groups) * evidence_fraction / max(1.0 - evidence_fraction, 1e-8))
        )

    selected_train_evidence = min(len(train_evidence_records), expected_train_evidence) if train_prior_groups else len(train_evidence_records)
    selected_valid_evidence = min(len(valid_evidence_records), expected_valid_evidence) if valid_prior_groups else len(valid_evidence_records)

    return {
        "raw_record_count": len(raw_rows),
        "max_possible_rollouts_if_full_3k_train_seen": 3000 * 8,
        "observed_debug_groups": len(grouped),
        "observed_debug_rollouts": len(raw_rows),
        "why_not_24000": "bayesian_reward_debug.jsonl contains 500 training reward calls x 8 rollouts, not all 3000 train questions",
        "evidence_filtering": {
            "kept_clean_rollouts": len(clean_evidence_records),
            "dropped_rollouts": len(raw_rows) - len(clean_evidence_records),
            "drop_flag_counts": dict(sorted(evidence_flag_counts.items())),
            "drop_primary_reason_counts": dict(sorted(evidence_primary_counts.items())),
            "stepwise_drop_counts": {
                "raw_record_count": len(raw_rows),
                "parse_failure_removed": evidence_parse_removed,
                "after_parse_failure": len(raw_rows) - evidence_parse_removed,
                "fallback_removed": evidence_fallback_removed,
                "after_fallback": len(raw_rows)
                - evidence_parse_removed
                - evidence_fallback_removed,
                "nan_posterior_or_likelihood_removed": evidence_nan_removed,
                "after_nan_posterior_or_likelihood": len(raw_rows)
                - evidence_parse_removed
                - evidence_fallback_removed
                - evidence_nan_removed,
                "nan_prior_probability_removed": evidence_nan_prior_removed,
                "missing_score_field_removed": evidence_missing_score_removed,
                "score_out_of_range_removed": evidence_score_removed,
                "invalid_error_type_removed": evidence_invalid_error_type_removed,
                "missing_field_removed": evidence_missing_field_removed,
                "kept_clean_rollouts": len(evidence_step_rows),
            },
        },
        "prior_filtering": {
            "raw_groups": len(grouped),
            "kept_clean_groups": len(clean_prior_groups),
            "dropped_groups": len(grouped) - len(clean_prior_groups),
            "drop_flag_counts": dict(sorted(prior_flag_counts.items())),
            "drop_primary_reason_counts": dict(sorted(prior_primary_counts.items())),
            "stepwise_drop_counts": {
                "raw_group_count": len(grouped),
                "parse_failure_removed": prior_parse_removed,
                "after_parse_failure": len(grouped) - prior_parse_removed,
                "fallback_removed": prior_fallback_removed,
                "after_fallback": len(grouped)
                - prior_parse_removed
                - prior_fallback_removed,
                "nan_prior_probability_removed": prior_nan_removed,
                "after_nan_prior_probability": len(grouped)
                - prior_parse_removed
                - prior_fallback_removed
                - prior_nan_removed,
                "missing_score_field_removed": prior_missing_score_removed,
                "score_out_of_range_removed": prior_score_removed,
                "missing_field_removed": prior_missing_field_removed,
                "kept_clean_groups": len(prior_step_groups),
            },
        },
        "train_valid_split": {
            "train_clean_evidence_rollouts": len(train_evidence_records),
            "valid_clean_evidence_rollouts": len(valid_evidence_records),
            "train_clean_prior_groups": len(train_prior_groups),
            "valid_clean_prior_groups": len(valid_prior_groups),
        },
        "ratio_rebalancing": {
            "evidence_fraction_target": evidence_fraction,
            "train_evidence_selected": selected_train_evidence,
            "train_evidence_dropped_for_ratio": len(train_evidence_records) - selected_train_evidence,
            "valid_evidence_selected": selected_valid_evidence,
            "valid_evidence_dropped_for_ratio": len(valid_evidence_records) - selected_valid_evidence,
        },
    }


def validate_runtime_sft_example(row: dict[str, Any]) -> None:
    task = str(row.get("task", ""))
    if task not in {"evidence_judge", "prior_judge"}:
        raise AssertionError(f"Unexpected runtime SFT task: {task}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise AssertionError("Runtime SFT row must contain 3 chat messages.")
    user_prompt = str(messages[1].get("content", "") or "")
    if not user_prompt.startswith(TASK_PREFIXES[task]):
        raise AssertionError(f"Runtime SFT prompt missing {task} task prefix.")
    assistant = json.loads(str(messages[-1]["content"]))
    if task == "evidence_judge":
        required = {
            "step_validity",
            "proof_completeness",
            "strategy_compliance",
            "consistency",
            "error_type",
            "key_strength",
            "key_weakness",
            "critical_failure_step",
            "judge_confidence",
        }
        if set(assistant.keys()) != required:
            raise AssertionError(f"Runtime evidence target keys mismatch: {sorted(assistant.keys())}")
    else:
        priors = assistant.get("priors")
        if not isinstance(priors, list) or not priors:
            raise AssertionError("Runtime prior target must contain non-empty priors list.")
        for item in priors:
            if set(item.keys()) != {"rollout_id", "suitability", "reason", "risk_flag"}:
                raise AssertionError(f"Runtime prior item keys mismatch: {sorted(item.keys())}")


def validate_runtime_dpo_pair(row: dict[str, Any]) -> None:
    task = str(row.get("task", ""))
    if task not in {"evidence_judge", "prior_judge"}:
        raise AssertionError("Runtime DPO pair task must be evidence_judge or prior_judge.")
    prompt_messages = row.get("prompt_messages")
    if not isinstance(prompt_messages, list) or len(prompt_messages) < 2:
        raise AssertionError("Runtime DPO pair must include prompt_messages.")
    user_prompt = str(prompt_messages[1].get("content", "") or "")
    if not user_prompt.startswith(TASK_PREFIXES[task]):
        raise AssertionError(f"Runtime DPO prompt missing {task} task prefix.")
    for key in ("chosen", "rejected"):
        parsed = json.loads(str(row[key]))
        if task == "evidence_judge":
            required = {
                "step_validity",
                "proof_completeness",
                "strategy_compliance",
                "consistency",
                "error_type",
                "key_strength",
                "key_weakness",
                "critical_failure_step",
                "judge_confidence",
            }
            if set(parsed.keys()) != required:
                raise AssertionError(f"Runtime evidence DPO {key} keys mismatch: {sorted(parsed.keys())}")
        else:
            priors = parsed.get("priors")
            if not isinstance(priors, list) or not priors:
                raise AssertionError(f"Runtime prior DPO {key} missing priors list.")
            for item in priors:
                if set(item.keys()) != {"rollout_id", "suitability", "reason", "risk_flag"}:
                    raise AssertionError(f"Runtime prior DPO {key} item keys mismatch: {sorted(item.keys())}")


def extract_rollout_records(
    debug_jsonl_path: str | Path,
    *,
    val_ratio: float = 0.1,
    hard_case_top_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    path = discover_debug_jsonl(debug_jsonl_path)
    raw_rows = load_jsonl(path)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[_group_key(row)].append(row)

    records: list[dict[str, Any]] = []
    group_tags_by_key: dict[tuple[int, int], list[str]] = {}
    rollout_tags_by_key: dict[tuple[int, int, int], list[str]] = {}

    for group_key, group_rows in grouped.items():
        ordered = sorted(
            group_rows,
            key=lambda item: int(item.get("group_rollout_id", 0) or 0),
        )
        question = str(ordered[0].get("problem", "") or "")
        gold_answer = str(ordered[0].get("gold_answer", "") or "")
        qid = question_id_for(question, gold_answer)

        priors = [maybe_float(row.get("prior_probability")) or 0.0 for row in ordered]
        likelihoods = [maybe_float(row.get("likelihood")) or 0.0 for row in ordered]
        posteriors = [maybe_float(row.get("bayesian_reward")) or 0.0 for row in ordered]
        correctness = [
            1 if float(row.get("answer_correctness", 0.0) or 0.0) == 1.0 else 0
            for row in ordered
        ]

        high_prior, low_prior = _top_bottom_index_sets(priors, hard_case_top_fraction)
        high_likelihood, low_likelihood = _top_bottom_index_sets(
            likelihoods, hard_case_top_fraction
        )
        top_index = max(range(len(ordered)), key=lambda index: posteriors[index])
        group_tags: set[str] = set()
        if any(correctness) and correctness[top_index] == 0:
            group_tags.add("posterior_top1_wrong")
        if (
            any(correctness)
            and any(value == 0 for value in correctness)
            and (group_tags or any(index in high_likelihood for index, value in enumerate(correctness) if value == 0))
        ):
            group_tags.add("medium_reward_conflict")
        group_tags_by_key[group_key] = sorted(group_tags)

        for index, row in enumerate(ordered):
            rollout_tags: set[str] = set(group_tags)
            if correctness[index] == 0 and index in high_likelihood:
                rollout_tags.add("wrong_high_likelihood")
            if correctness[index] == 1 and index in low_likelihood:
                rollout_tags.add("correct_low_likelihood")
            if correctness[index] == 0 and index in high_prior:
                rollout_tags.add("wrong_high_prior")
            if correctness[index] == 1 and index in low_prior:
                rollout_tags.add("correct_low_prior")
            rollout_key = (
                group_key[0],
                group_key[1],
                int(row.get("group_rollout_id", 0) or 0),
            )
            rollout_tags_by_key[rollout_key] = sorted(rollout_tags)

        for row in ordered:
            rollout_id = int(row.get("group_rollout_id", 0) or 0)
            record_key = build_record_id(qid, rollout_id)
            prompted_prior_item = extract_prompted_prior_item(row)
            clean_evidence = _clean_evidence_row(row)
            clean_prior_group = _clean_prior_group(ordered)
            record = {
                "record_id": record_key,
                "question_id": qid,
                "problem_hash": stable_hash(question),
                "learned_lookup_key": build_record_id(stable_hash(question), rollout_id),
                "split": question_split(qid, val_ratio),
                "source_debug_jsonl": relative_path(path),
                "group_key": {
                    "global_reward_call_index": group_key[0],
                    "group_index_within_call": group_key[1],
                    "rollout_id": rollout_id,
                },
                "question": question,
                "gold_answer": gold_answer,
                "strategy": str(row.get("parsed_strategy", "") or ""),
                "rollout_solution": str(row.get("parsed_reasoning", "") or ""),
                "predicted_answer": str(row.get("parsed_final_answer", "") or ""),
                "raw_completion": str(row.get("raw_completion", "") or ""),
                "prompted_prior_json": prompted_prior_item,
                "prompted_prior_group_json": row.get("parsed_prior_judge_json"),
                "prompted_evidence_json": row.get("parsed_judge_json"),
                "likelihood_score": maybe_float(row.get("likelihood")),
                "posterior_score": maybe_float(row.get("bayesian_reward")),
                "prior_probability": maybe_float(row.get("prior_probability")),
                "prior_suitability": (
                    int(row.get("prior_suitability", 0) or 0)
                    if row.get("prior_suitability") is not None
                    else None
                ),
                "answer_correctness": float(row.get("answer_correctness", 0.0) or 0.0),
                "prior_fallback": bool(row.get("prior_judge_fallback_used")),
                "evidence_fallback": bool(row.get("evidence_judge_fallback_used")),
                "prior_parse_success": clean_prior_group,
                "evidence_parse_success": bool(
                    isinstance(row.get("parsed_judge_json"), dict)
                    and not row.get("evidence_judge_fallback_used")
                ),
                "clean_evidence": clean_evidence,
                "clean_prior_group": clean_prior_group,
                "hard_case_tags": rollout_tags_by_key[
                    (group_key[0], group_key[1], rollout_id)
                ],
                "group_hard_case_tags": group_tags_by_key[group_key],
                "error_type": str(row.get("error_type", "") or ""),
                "step_validity": clamp_int_0_to_4(row.get("step_validity", 0)),
                "proof_completeness": clamp_int_0_to_4(
                    row.get("proof_completeness", 0)
                ),
                "strategy_compliance": clamp_int_0_to_4(
                    row.get("strategy_compliance", 0)
                ),
                "consistency": clamp_int_0_to_4(row.get("consistency", 0)),
                "judge_confidence": clamp01(row.get("judge_confidence", 0.0)),
                "brief_reason": (
                    str(row.get("key_weakness", "") or "").strip()
                    or str(row.get("key_strength", "") or "").strip()
                    or str(row.get("critical_failure_step", "") or "").strip()
                ),
                "simple_prior_target": derive_simple_prior_target(row),
                "simple_evidence_target": derive_simple_evidence_target(row),
                "runtime_evidence_target": derive_runtime_evidence_target(row),
            }
            records.append(record)

    records.sort(key=lambda item: item["record_id"])
    return records


def build_runtime_prior_group_examples(
    rollout_records: list[dict[str, Any]],
    *,
    use_task_prefix: bool = True,
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in rollout_records:
        key = (
            str(record["question_id"]),
            int(record["group_key"]["global_reward_call_index"]),
            int(record["group_key"]["group_index_within_call"]),
        )
        by_group[key].append(record)

    group_examples: list[dict[str, Any]] = []
    for key, group_rows in by_group.items():
        ordered = sorted(group_rows, key=lambda item: int(item["group_key"]["rollout_id"]))
        if not ordered or not all(row["clean_prior_group"] for row in ordered):
            continue
        candidates = [
            {
                "rollout_id": int(row["group_key"]["rollout_id"]),
                "strategy": str(row["strategy"]),
            }
            for row in ordered
        ]
        prompt = build_runtime_prior_prompt(ordered[0]["question"], candidates)
        if use_task_prefix:
            prompt = add_task_prefix(prompt, "prior_judge")
        target_rows = [
            {
                "group_rollout_id": row["group_key"]["rollout_id"],
                "prior_suitability": row["prior_suitability"],
                "prior_reason": (
                    str((row.get("prompted_prior_json") or {}).get("reason", "") or "")
                ),
                "prior_risk_flag": (
                    str((row.get("prompted_prior_json") or {}).get("risk_flag", "") or "")
                ),
            }
            for row in ordered
        ]
        teacher_target = derive_runtime_prior_target(target_rows)
        group_examples.append(
            {
                "example_id": stable_hash(json.dumps(key)),
                "question_id": ordered[0]["question_id"],
                "split": ordered[0]["split"],
                "task": "prior_judge",
                "quality_tier": (
                    "hard" if any(row["hard_case_tags"] for row in ordered) else "clean"
                ),
                "question": ordered[0]["question"],
                "gold_answer": ordered[0]["gold_answer"],
                "group_rollouts": ordered,
                "prompt": prompt,
                "teacher_target": teacher_target,
                "messages": build_messages(prompt, teacher_target),
                "hard_case_tags": sorted(
                    {tag for row in ordered for tag in row["hard_case_tags"]}
                ),
            }
        )
    group_examples.sort(key=lambda item: item["example_id"])
    return group_examples


def count_tags(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for tag in row.get(field, []):
            counter[str(tag)] += 1
    return dict(sorted(counter.items()))


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    indexes = list(range(len(rows)))
    rng.shuffle(indexes)
    chosen = sorted(indexes[:count])
    return [rows[index] for index in chosen]


def balanced_counts(
    clean_count: int,
    hard_count: int,
    clean_fraction: float,
) -> tuple[int, int]:
    if clean_count <= 0 and hard_count <= 0:
        return 0, 0
    clean_fraction = min(max(clean_fraction, 0.0), 1.0)
    hard_fraction = 1.0 - clean_fraction
    if clean_count <= 0:
        return 0, hard_count
    if hard_count <= 0 or hard_fraction <= 0:
        return clean_count, 0
    if clean_fraction <= 0:
        return 0, hard_count
    total = min(
        clean_count / clean_fraction,
        hard_count / hard_fraction,
    )
    clean_take = min(clean_count, int(round(total * clean_fraction)))
    hard_take = min(hard_count, int(round(total * hard_fraction)))
    if clean_take == 0 and clean_count > 0:
        clean_take = min(clean_count, 1)
    if hard_take == 0 and hard_count > 0 and hard_fraction > 0:
        hard_take = min(hard_count, 1)
    return clean_take, hard_take


def rebalance_two_pools(
    evidence_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    evidence_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not evidence_rows or not prior_rows:
        return list(evidence_rows), list(prior_rows)
    evidence_fraction = min(max(evidence_fraction, 0.0), 1.0)
    prior_fraction = 1.0 - evidence_fraction
    if evidence_fraction <= 0 or prior_fraction <= 0:
        return list(evidence_rows), list(prior_rows)
    total = min(
        len(evidence_rows) / evidence_fraction,
        len(prior_rows) / prior_fraction,
    )
    evidence_take = min(len(evidence_rows), int(round(total * evidence_fraction)))
    prior_take = min(len(prior_rows), int(round(total * prior_fraction)))
    return (
        sample_rows(evidence_rows, evidence_take, rng),
        sample_rows(prior_rows, prior_take, rng),
    )
