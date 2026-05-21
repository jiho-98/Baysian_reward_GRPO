#!/usr/bin/env python3
"""Build GSM8K learned-analyzer DPO data from prompted Bayesian logs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gsm8k_learned_analyzer_utils import (
    add_task_prefix,
    assert_disjoint_question_splits,
    build_messages,
    build_record_id,
    build_runtime_evidence_prompt,
    build_runtime_prior_group_examples,
    build_simple_evidence_prompt,
    build_simple_prior_prompt,
    clamp01,
    clamp_int_0_to_4,
    count_tags,
    extract_rollout_records,
    sample_rows,
    stable_hash,
    validate_runtime_dpo_pair,
    write_json,
    write_jsonl,
)


DEFAULT_LOG_DIR = (
    "outputs/gsm8k_experiments/"
    "grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10"
)
DEFAULT_OUTPUT_DIR = "outputs/gsm8k_learned_analyzer/dpo_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simple and runtime-compatible analyzer DPO data from GSM8K prompted Bayesian logs."
    )
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--learned_posterior_debug_jsonl", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hard_case_top_fraction", type=float, default=0.25)
    parser.add_argument("--target_evidence_train_pairs", type=int, default=4000)
    parser.add_argument("--target_prior_train_pairs", type=int, default=1000)
    parser.add_argument("--target_evidence_valid_pairs", type=int, default=400)
    parser.add_argument("--target_prior_valid_pairs", type=int, default=100)
    parser.add_argument("--synthetic_companion_rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def boost_int_score(value: Any, delta: int) -> int:
    return clamp_int_0_to_4(clamp_int_0_to_4(value) + delta)


def load_learned_rollout_map(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))

    learned: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem = str(row.get("problem", "") or "")
        rollout_id = int(row.get("rollout_id", 0) or 0)
        key = build_record_id(stable_hash(problem), rollout_id)
        learned[key] = row
    return learned


def simple_prior_target_from_learned(teacher: dict[str, Any], learned_row: dict[str, Any]) -> dict[str, Any]:
    prior_score = clamp_int_0_to_4(learned_row.get("learned_prior_suitability", teacher["prior_score"]))
    return {
        "strategy_relevance": prior_score,
        "problem_fit": prior_score,
        "risk_of_error": clamp_int_0_to_4(4 - prior_score),
        "prior_score": prior_score,
        "brief_reason": "SFT analyzer prior judgment.",
    }


def simple_evidence_target_from_learned(teacher: dict[str, Any], learned_row: dict[str, Any]) -> dict[str, Any]:
    scores = learned_row.get("learned_evidence_scores", {}) or {}
    return {
        "answer_correctness": int(teacher["answer_correctness"]),
        "step_validity": clamp_int_0_to_4(scores.get("step_validity", teacher["step_validity"])),
        "proof_completeness": clamp_int_0_to_4(scores.get("proof_completeness", teacher["proof_completeness"])),
        "strategy_compliance": clamp_int_0_to_4(scores.get("strategy_compliance", teacher["strategy_compliance"])),
        "consistency": clamp_int_0_to_4(scores.get("consistency", teacher["consistency"])),
        "error_type": str(scores.get("error_type", teacher["error_type"]) or teacher["error_type"]),
        "likelihood_score": clamp01(learned_row.get("learned_likelihood", teacher["likelihood_score"])),
        "brief_reason": "SFT analyzer evidence judgment.",
    }


def runtime_evidence_target_from_learned(teacher: dict[str, Any], learned_row: dict[str, Any]) -> dict[str, Any]:
    scores = learned_row.get("learned_evidence_scores", {}) or {}
    return {
        "step_validity": clamp_int_0_to_4(scores.get("step_validity", teacher["step_validity"])),
        "proof_completeness": clamp_int_0_to_4(scores.get("proof_completeness", teacher["proof_completeness"])),
        "strategy_compliance": clamp_int_0_to_4(scores.get("strategy_compliance", teacher["strategy_compliance"])),
        "consistency": clamp_int_0_to_4(scores.get("consistency", teacher["consistency"])),
        "error_type": str(scores.get("error_type", teacher["error_type"]) or teacher["error_type"]),
        "key_strength": str(teacher.get("key_strength", "") or ""),
        "key_weakness": str(teacher.get("key_weakness", "") or "SFT analyzer disagreement."),
        "critical_failure_step": str(teacher.get("critical_failure_step", "") or ""),
        "judge_confidence": clamp01(learned_row.get("learned_likelihood", teacher.get("judge_confidence", 0.0))),
    }


def perturb_simple_prior_target(
    teacher: dict[str, Any],
    record: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    base = dict(teacher)
    prior_score = clamp_int_0_to_4(teacher["prior_score"])
    hash_bit = int(stable_hash(record["record_id"])[0], 16) % 2
    if prior_score >= 3:
        new_score = clamp_int_0_to_4(prior_score - 2)
        style = "low_prior_on_teacher_high_prior"
        base["brief_reason"] = "Undervalues a strategy that looks directly applicable."
    elif prior_score <= 1:
        new_score = clamp_int_0_to_4(prior_score + 2)
        style = "high_prior_on_teacher_low_prior"
        base["brief_reason"] = "Overrates a vague or weak strategy."
    else:
        new_score = 0 if hash_bit == 0 else 4
        style = "polarized_prior_misjudge"
        base["brief_reason"] = "Polarizes a borderline strategy instead of judging it cautiously."
    base["prior_score"] = new_score
    base["strategy_relevance"] = new_score
    base["problem_fit"] = new_score
    if new_score >= 3:
        base["risk_of_error"] = clamp_int_0_to_4(min(base["risk_of_error"], 1))
    elif new_score <= 1:
        base["risk_of_error"] = clamp_int_0_to_4(max(base["risk_of_error"], 3))
    else:
        base["risk_of_error"] = clamp_int_0_to_4(4 - new_score)
    return base, style


def perturb_simple_evidence_target(
    teacher: dict[str, Any],
    record: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    base = dict(teacher)
    tags = set(record.get("hard_case_tags", []))
    answer_correct = int(teacher["answer_correctness"])
    if "wrong_high_likelihood" in tags:
        base["step_validity"] = boost_int_score(teacher["step_validity"], 1)
        base["proof_completeness"] = boost_int_score(teacher["proof_completeness"], 1)
        base["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], 1)
        base["consistency"] = boost_int_score(teacher["consistency"], 1)
        base["likelihood_score"] = clamp01(teacher["likelihood_score"] + 0.35)
        base["error_type"] = "finalization_error"
        base["brief_reason"] = "Overcredits an incorrect but plausible-looking rollout."
        style = "wrong_but_plausible_high_likelihood"
    elif "correct_low_likelihood" in tags:
        base["step_validity"] = boost_int_score(teacher["step_validity"], -1)
        base["proof_completeness"] = boost_int_score(teacher["proof_completeness"], -2)
        base["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], -1)
        base["consistency"] = boost_int_score(teacher["consistency"], -1)
        base["likelihood_score"] = clamp01(teacher["likelihood_score"] - 0.35)
        base["error_type"] = "correct_weak_proof"
        base["brief_reason"] = "Underrates a correct rollout despite mostly valid reasoning."
        style = "correct_but_underrated"
    elif "wrong_high_prior" in tags:
        base["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], 2)
        base["step_validity"] = boost_int_score(teacher["step_validity"], 1)
        base["likelihood_score"] = clamp01(teacher["likelihood_score"] + 0.25)
        base["error_type"] = "valid_but_incomplete"
        base["brief_reason"] = "Lets high prior bias leak into evidence confidence."
        style = "high_prior_wrong_reward_leakage"
    elif "correct_low_prior" in tags:
        base["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], -2)
        base["likelihood_score"] = clamp01(teacher["likelihood_score"] - 0.25)
        base["brief_reason"] = "Underrates a correct rollout because the strategy looked less promising."
        style = "low_prior_correct_underrated"
    elif "posterior_top1_wrong" in tags or "medium_reward_conflict" in tags:
        if answer_correct == 0:
            base["step_validity"] = boost_int_score(teacher["step_validity"], 1)
            base["proof_completeness"] = boost_int_score(teacher["proof_completeness"], 1)
            base["likelihood_score"] = clamp01(teacher["likelihood_score"] + 0.2)
            base["error_type"] = "valid_but_incomplete"
            base["brief_reason"] = "Creates a misleading near-miss judgment that can steal posterior mass."
            style = "posterior_conflict_wrong_boosted"
        else:
            base["proof_completeness"] = boost_int_score(teacher["proof_completeness"], -1)
            base["likelihood_score"] = clamp01(teacher["likelihood_score"] - 0.2)
            base["error_type"] = "correct_weak_proof"
            base["brief_reason"] = "Shaves confidence off the correct rollout in a conflicted group."
            style = "posterior_conflict_correct_underrated"
    elif answer_correct == 1:
        base["step_validity"] = boost_int_score(teacher["step_validity"], -1)
        base["proof_completeness"] = boost_int_score(teacher["proof_completeness"], -1)
        base["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], -1)
        base["consistency"] = boost_int_score(teacher["consistency"], -1)
        base["likelihood_score"] = clamp01(teacher["likelihood_score"] - 0.2)
        base["error_type"] = "correct_weak_proof"
        base["brief_reason"] = "Undervalues a valid rollout."
        style = "generic_correct_underrated"
    else:
        base["step_validity"] = boost_int_score(teacher["step_validity"], 1)
        base["proof_completeness"] = boost_int_score(teacher["proof_completeness"], 1)
        base["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], 1)
        base["consistency"] = boost_int_score(teacher["consistency"], 1)
        base["likelihood_score"] = clamp01(teacher["likelihood_score"] + 0.2)
        base["error_type"] = "finalization_error"
        base["brief_reason"] = "Reward leakage: overrates an incorrect rollout."
        style = "generic_wrong_overrated"
    return base, style


def perturb_runtime_evidence_target(
    teacher: dict[str, Any],
    record: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    perturbed = dict(teacher)
    tags = set(record.get("hard_case_tags", []))
    answer_correct = float(record["answer_correctness"]) == 1.0
    if "wrong_high_likelihood" in tags:
        perturbed["step_validity"] = boost_int_score(teacher["step_validity"], 1)
        perturbed["proof_completeness"] = boost_int_score(teacher["proof_completeness"], 1)
        perturbed["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], 1)
        perturbed["consistency"] = boost_int_score(teacher["consistency"], 1)
        perturbed["error_type"] = "finalization_error"
        perturbed["key_weakness"] = "Overcredits an incorrect but plausible-looking rollout."
        perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) + 0.35)
        style = "wrong_but_plausible_high_likelihood"
    elif "correct_low_likelihood" in tags:
        perturbed["step_validity"] = boost_int_score(teacher["step_validity"], -1)
        perturbed["proof_completeness"] = boost_int_score(teacher["proof_completeness"], -2)
        perturbed["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], -1)
        perturbed["consistency"] = boost_int_score(teacher["consistency"], -1)
        perturbed["error_type"] = "correct_weak_proof"
        perturbed["key_weakness"] = "Underrates a correct rollout despite mostly valid reasoning."
        perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) - 0.35)
        style = "correct_but_underrated"
    elif "wrong_high_prior" in tags:
        perturbed["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], 2)
        perturbed["step_validity"] = boost_int_score(teacher["step_validity"], 1)
        perturbed["error_type"] = "valid_but_incomplete"
        perturbed["key_weakness"] = "Lets high prior bias leak into evidence confidence."
        perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) + 0.25)
        style = "high_prior_wrong_reward_leakage"
    elif "correct_low_prior" in tags:
        perturbed["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], -2)
        perturbed["key_weakness"] = "Underrates a correct rollout because the strategy looked less promising."
        perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) - 0.25)
        style = "low_prior_correct_underrated"
    elif "posterior_top1_wrong" in tags or "medium_reward_conflict" in tags:
        if answer_correct:
            perturbed["proof_completeness"] = boost_int_score(teacher["proof_completeness"], -1)
            perturbed["error_type"] = "correct_weak_proof"
            perturbed["key_weakness"] = "Shaves confidence off the correct rollout in a conflicted group."
            perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) - 0.2)
            style = "posterior_conflict_correct_underrated"
        else:
            perturbed["step_validity"] = boost_int_score(teacher["step_validity"], 1)
            perturbed["proof_completeness"] = boost_int_score(teacher["proof_completeness"], 1)
            perturbed["error_type"] = "valid_but_incomplete"
            perturbed["key_weakness"] = "Creates a misleading near-miss judgment that can steal posterior mass."
            perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) + 0.2)
            style = "posterior_conflict_wrong_boosted"
    elif answer_correct:
        perturbed["step_validity"] = boost_int_score(teacher["step_validity"], -1)
        perturbed["proof_completeness"] = boost_int_score(teacher["proof_completeness"], -1)
        perturbed["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], -1)
        perturbed["consistency"] = boost_int_score(teacher["consistency"], -1)
        perturbed["error_type"] = "correct_weak_proof"
        perturbed["key_weakness"] = "Undervalues a valid rollout."
        perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) - 0.2)
        style = "generic_correct_underrated"
    else:
        perturbed["step_validity"] = boost_int_score(teacher["step_validity"], 1)
        perturbed["proof_completeness"] = boost_int_score(teacher["proof_completeness"], 1)
        perturbed["strategy_compliance"] = boost_int_score(teacher["strategy_compliance"], 1)
        perturbed["consistency"] = boost_int_score(teacher["consistency"], 1)
        perturbed["error_type"] = "finalization_error"
        perturbed["key_weakness"] = "Reward leakage: overrates an incorrect rollout."
        perturbed["judge_confidence"] = clamp01(float(teacher.get("judge_confidence", 0.0)) + 0.2)
        style = "generic_wrong_overrated"
    return perturbed, style


def runtime_prior_target_from_group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "priors": [
            {
                "rollout_id": int(row["group_key"]["rollout_id"]),
                "suitability": clamp_int_0_to_4(row["prior_suitability"]),
                "reason": str((row.get("prompted_prior_json") or {}).get("reason", "") or ""),
                "risk_flag": str((row.get("prompted_prior_json") or {}).get("risk_flag", "") or ""),
            }
            for row in sorted(group_rows, key=lambda item: int(item["group_key"]["rollout_id"]))
        ]
    }


def runtime_prior_target_from_learned(
    group_rows: list[dict[str, Any]],
    learned_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    priors: list[dict[str, Any]] = []
    changed = False
    for row in sorted(group_rows, key=lambda item: int(item["group_key"]["rollout_id"])):
        learned_row = learned_map.get(row["learned_lookup_key"])
        if learned_row is None:
            return None
        learned_suitability = clamp_int_0_to_4(
            learned_row.get("learned_prior_suitability", row["prior_suitability"])
        )
        if learned_suitability != row["prior_suitability"]:
            changed = True
        priors.append(
            {
                "rollout_id": int(row["group_key"]["rollout_id"]),
                "suitability": learned_suitability,
                "reason": "SFT analyzer prior judgment.",
                "risk_flag": "" if learned_suitability >= 2 else "risky",
            }
        )
    if not changed:
        return None
    return {"priors": priors}


def perturb_runtime_prior_target(
    group_rows: list[dict[str, Any]],
    focus_rollout_id: int,
) -> tuple[dict[str, Any], str]:
    teacher = runtime_prior_target_from_group(group_rows)
    priors = []
    focus_item = next(
        item for item in teacher["priors"] if int(item["rollout_id"]) == int(focus_rollout_id)
    )
    focus_score = int(focus_item["suitability"])
    if focus_score >= 3:
        new_score = clamp_int_0_to_4(focus_score - 2)
        style = "low_prior_on_teacher_high_prior"
        reason_text = "Undervalues a strategy that looks directly applicable."
        risk_flag = "too_risky"
    elif focus_score <= 1:
        new_score = clamp_int_0_to_4(focus_score + 2)
        style = "high_prior_on_teacher_low_prior"
        reason_text = "Overrates a vague or weak strategy."
        risk_flag = "none"
    else:
        new_score = 0 if int(focus_rollout_id) % 2 == 0 else 4
        style = "polarized_prior_misjudge"
        reason_text = "Polarizes a borderline strategy instead of judging it cautiously."
        risk_flag = "too_risky" if new_score == 0 else "none"
    for item in teacher["priors"]:
        updated = dict(item)
        if int(updated["rollout_id"]) == int(focus_rollout_id):
            updated["suitability"] = new_score
            updated["reason"] = reason_text
            updated["risk_flag"] = risk_flag
        priors.append(updated)
    return {"priors": priors}, style


def select_prior_focus_rollout_ids(group_rows: list[dict[str, Any]]) -> list[int]:
    ordered = sorted(group_rows, key=lambda row: int(row["group_key"]["rollout_id"]))
    by_score = sorted(
        ordered,
        key=lambda row: (
            int(row["prior_suitability"] if row["prior_suitability"] is not None else 0),
            int(row["group_key"]["rollout_id"]),
        ),
    )
    candidate_ids: list[int] = []
    for row in (by_score[0], by_score[-1], ordered[len(ordered) // 2]):
        rollout_id = int(row["group_key"]["rollout_id"])
        if rollout_id not in candidate_ids:
            candidate_ids.append(rollout_id)
    return candidate_ids


def pair_row(
    *,
    pair_id: str,
    split: str,
    task: str,
    bucket_name: str,
    question_id: str,
    prompt_messages: list[dict[str, str]],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    rejected_source: str,
    rejected_style: str,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "split": split,
        "task": task,
        "bucket_name": bucket_name,
        "question_id": question_id,
        "prompt_messages": prompt_messages,
        "chosen": json_text(chosen),
        "rejected": json_text(rejected),
        "rejected_source": rejected_source,
        "rejected_style": rejected_style,
    }


def bucket_for_record(record: dict[str, Any]) -> str:
    return record["hard_case_tags"][0] if record["hard_case_tags"] else "clean"


def build_simple_pairs(
    records: list[dict[str, Any]],
    learned_map: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
    synthetic_companion_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_pairs: list[dict[str, Any]] = []
    prior_pairs: list[dict[str, Any]] = []
    for record in records:
        if record["clean_evidence"]:
            prompt = build_simple_evidence_prompt(
                record["question"],
                record["gold_answer"],
                record["strategy"],
                record["rollout_solution"],
                record["predicted_answer"],
                float(record["answer_correctness"]),
            )
            chosen = dict(record["simple_evidence_target"])
            learned_row = learned_map.get(record["learned_lookup_key"])
            rejected_source = "synthetic"
            rejected_style = ""
            if learned_row is not None:
                candidate = simple_evidence_target_from_learned(chosen, learned_row)
                if json_text(candidate) != json_text(chosen):
                    rejected = candidate
                    rejected_source = "learned_sft"
                    rejected_style = "real_learned_mistake"
                else:
                    rejected, rejected_style = perturb_simple_evidence_target(chosen, record)
            else:
                rejected, rejected_style = perturb_simple_evidence_target(chosen, record)
            evidence_pairs.append(
                pair_row(
                    pair_id=f"simple-evidence:{record['record_id']}",
                    split=record["split"],
                    task="evidence_judge",
                    bucket_name=bucket_for_record(record),
                    question_id=record["question_id"],
                    prompt_messages=build_messages(prompt, chosen)[:-1],
                    chosen=chosen,
                    rejected=rejected,
                    rejected_source=rejected_source,
                    rejected_style=rejected_style,
                )
            )
            if rejected_source == "learned_sft" and rng.random() < synthetic_companion_rate:
                synthetic_rejected, synthetic_style = perturb_simple_evidence_target(chosen, record)
                if json_text(synthetic_rejected) != json_text(chosen):
                    evidence_pairs.append(
                        pair_row(
                            pair_id=f"simple-evidence:{record['record_id']}:synthetic",
                            split=record["split"],
                            task="evidence_judge",
                            bucket_name=bucket_for_record(record),
                            question_id=record["question_id"],
                            prompt_messages=build_messages(prompt, chosen)[:-1],
                            chosen=chosen,
                            rejected=synthetic_rejected,
                            rejected_source="synthetic",
                            rejected_style=synthetic_style,
                        )
                    )

        if record["clean_prior_group"]:
            prompt = build_simple_prior_prompt(record["question"], record["strategy"])
            chosen = dict(record["simple_prior_target"])
            learned_row = learned_map.get(record["learned_lookup_key"])
            rejected_source = "synthetic"
            rejected_style = ""
            if learned_row is not None:
                candidate = simple_prior_target_from_learned(chosen, learned_row)
                if json_text(candidate) != json_text(chosen):
                    rejected = candidate
                    rejected_source = "learned_sft"
                    rejected_style = "real_learned_mistake"
                else:
                    rejected, rejected_style = perturb_simple_prior_target(chosen, record)
            else:
                rejected, rejected_style = perturb_simple_prior_target(chosen, record)
            prior_pairs.append(
                pair_row(
                    pair_id=f"simple-prior:{record['record_id']}",
                    split=record["split"],
                    task="prior_judge",
                    bucket_name=bucket_for_record(record),
                    question_id=record["question_id"],
                    prompt_messages=build_messages(prompt, chosen)[:-1],
                    chosen=chosen,
                    rejected=rejected,
                    rejected_source=rejected_source,
                    rejected_style=rejected_style,
                )
            )
            if rejected_source == "learned_sft" and rng.random() < synthetic_companion_rate:
                synthetic_rejected, synthetic_style = perturb_simple_prior_target(chosen, record)
                if json_text(synthetic_rejected) != json_text(chosen):
                    prior_pairs.append(
                        pair_row(
                            pair_id=f"simple-prior:{record['record_id']}:synthetic",
                            split=record["split"],
                            task="prior_judge",
                            bucket_name=bucket_for_record(record),
                            question_id=record["question_id"],
                            prompt_messages=build_messages(prompt, chosen)[:-1],
                            chosen=chosen,
                            rejected=synthetic_rejected,
                            rejected_source="synthetic",
                            rejected_style=synthetic_style,
                        )
                    )
    return evidence_pairs, prior_pairs


def build_runtime_pairs(
    records: list[dict[str, Any]],
    learned_map: dict[str, dict[str, Any]],
    *,
    rng: random.Random,
    synthetic_companion_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_pairs: list[dict[str, Any]] = []
    prior_pairs: list[dict[str, Any]] = []

    for record in records:
        if not record["clean_evidence"]:
            continue
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
        chosen = dict(record["runtime_evidence_target"])
        learned_row = learned_map.get(record["learned_lookup_key"])
        rejected_source = "synthetic"
        rejected_style = ""
        if learned_row is not None:
            candidate = runtime_evidence_target_from_learned(chosen, learned_row)
            if json_text(candidate) != json_text(chosen):
                rejected = candidate
                rejected_source = "learned_sft"
                rejected_style = "real_learned_mistake"
            else:
                rejected, rejected_style = perturb_runtime_evidence_target(chosen, record)
        else:
            rejected, rejected_style = perturb_runtime_evidence_target(chosen, record)
        evidence_pairs.append(
            pair_row(
                pair_id=f"runtime-evidence:{record['record_id']}",
                split=record["split"],
                task="evidence_judge",
                bucket_name=bucket_for_record(record),
                question_id=record["question_id"],
                prompt_messages=build_messages(prompt, chosen)[:-1],
                chosen=chosen,
                rejected=rejected,
                rejected_source=rejected_source,
                rejected_style=rejected_style,
            )
        )
        if rejected_source == "learned_sft" and rng.random() < synthetic_companion_rate:
            synthetic_rejected, synthetic_style = perturb_runtime_evidence_target(chosen, record)
            if json_text(synthetic_rejected) != json_text(chosen):
                evidence_pairs.append(
                    pair_row(
                        pair_id=f"runtime-evidence:{record['record_id']}:synthetic",
                        split=record["split"],
                        task="evidence_judge",
                        bucket_name=bucket_for_record(record),
                        question_id=record["question_id"],
                        prompt_messages=build_messages(prompt, chosen)[:-1],
                        chosen=chosen,
                        rejected=synthetic_rejected,
                        rejected_source="synthetic",
                        rejected_style=synthetic_style,
                    )
                )

    groups = build_runtime_prior_group_examples(records)
    for group in groups:
        group_rows = list(group["group_rollouts"])
        candidate_rollout_ids = select_prior_focus_rollout_ids(group_rows)
        for focus_rollout_id in candidate_rollout_ids:
            chosen = runtime_prior_target_from_group(group_rows)
            learned_target = runtime_prior_target_from_learned(group_rows, learned_map)
            rejected_source = "synthetic"
            rejected_style = ""
            if learned_target is not None and json_text(learned_target) != json_text(chosen):
                rejected = learned_target
                rejected_source = "learned_sft"
                rejected_style = "real_learned_mistake"
            else:
                rejected, rejected_style = perturb_runtime_prior_target(group_rows, focus_rollout_id)
            prior_pairs.append(
                pair_row(
                    pair_id=f"runtime-prior:{group['example_id']}:{focus_rollout_id}",
                    split=group["split"],
                    task="prior_judge",
                    bucket_name=group["hard_case_tags"][0] if group["hard_case_tags"] else "clean",
                    question_id=group["question_id"],
                    prompt_messages=build_messages(group["prompt"], chosen)[:-1],
                    chosen=chosen,
                    rejected=rejected,
                    rejected_source=rejected_source,
                    rejected_style=rejected_style,
                )
            )
            if rejected_source == "learned_sft" and rng.random() < synthetic_companion_rate:
                synthetic_rejected, synthetic_style = perturb_runtime_prior_target(
                    group_rows,
                    focus_rollout_id,
                )
                if json_text(synthetic_rejected) != json_text(chosen):
                    prior_pairs.append(
                        pair_row(
                            pair_id=f"runtime-prior:{group['example_id']}:{focus_rollout_id}:synthetic",
                            split=group["split"],
                            task="prior_judge",
                            bucket_name=group["hard_case_tags"][0] if group["hard_case_tags"] else "clean",
                            question_id=group["question_id"],
                            prompt_messages=build_messages(group["prompt"], chosen)[:-1],
                            chosen=chosen,
                            rejected=synthetic_rejected,
                            rejected_source="synthetic",
                            rejected_style=synthetic_style,
                        )
                    )
    return evidence_pairs, prior_pairs


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    return train_rows, valid_rows


def trim(rows: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    return sample_rows(rows, target, rng)


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["task"]) for row in rows)
    return dict(sorted(counter.items()))


def rejected_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["rejected_source"]) for row in rows)
    return dict(sorted(counter.items()))


def rejected_style_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("rejected_style", "") or "unknown") for row in rows)
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = extract_rollout_records(
        args.log_dir,
        val_ratio=args.val_ratio,
        hard_case_top_fraction=args.hard_case_top_fraction,
    )
    learned_map = load_learned_rollout_map(args.learned_posterior_debug_jsonl)

    source_split_assertion = assert_disjoint_question_splits(
        records,
        label="rollout_records",
    )

    simple_evidence_pairs, simple_prior_pairs = build_simple_pairs(
        records,
        learned_map,
        rng=rng,
        synthetic_companion_rate=args.synthetic_companion_rate,
    )
    runtime_evidence_pairs, runtime_prior_pairs = build_runtime_pairs(
        records,
        learned_map,
        rng=rng,
        synthetic_companion_rate=args.synthetic_companion_rate,
    )

    simple_evidence_train_pool, simple_evidence_valid_pool = split_rows(simple_evidence_pairs)
    simple_prior_train_pool, simple_prior_valid_pool = split_rows(simple_prior_pairs)
    runtime_evidence_train_pool, runtime_evidence_valid_pool = split_rows(runtime_evidence_pairs)
    runtime_prior_train_pool, runtime_prior_valid_pool = split_rows(runtime_prior_pairs)

    simple_evidence_train = trim(simple_evidence_train_pool, args.target_evidence_train_pairs, rng)
    simple_prior_train = trim(simple_prior_train_pool, args.target_prior_train_pairs, rng)
    simple_evidence_valid = trim(simple_evidence_valid_pool, args.target_evidence_valid_pairs, rng)
    simple_prior_valid = trim(simple_prior_valid_pool, args.target_prior_valid_pairs, rng)
    runtime_evidence_train = trim(runtime_evidence_train_pool, args.target_evidence_train_pairs, rng)
    runtime_prior_train = trim(runtime_prior_train_pool, args.target_prior_train_pairs, rng)
    runtime_evidence_valid = trim(runtime_evidence_valid_pool, args.target_evidence_valid_pairs, rng)
    runtime_prior_valid = trim(runtime_prior_valid_pool, args.target_prior_valid_pairs, rng)

    simple_unified_train = sorted(
        simple_evidence_train + simple_prior_train,
        key=lambda row: str(row["pair_id"]),
    )
    simple_unified_valid = sorted(
        simple_evidence_valid + simple_prior_valid,
        key=lambda row: str(row["pair_id"]),
    )
    runtime_unified_train = sorted(
        runtime_evidence_train + runtime_prior_train,
        key=lambda row: str(row["pair_id"]),
    )
    runtime_unified_valid = sorted(
        runtime_evidence_valid + runtime_prior_valid,
        key=lambda row: str(row["pair_id"]),
    )

    artifacts = {
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
    for row in runtime_evidence_pairs:
        validate_runtime_dpo_pair(row)
    for row in runtime_prior_pairs:
        validate_runtime_dpo_pair(row)
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

    summary = {
        "config": {
            "log_dir": args.log_dir,
            "learned_posterior_debug_jsonl": args.learned_posterior_debug_jsonl,
            "val_ratio": args.val_ratio,
            "hard_case_top_fraction": args.hard_case_top_fraction,
            "target_evidence_train_pairs": args.target_evidence_train_pairs,
            "target_prior_train_pairs": args.target_prior_train_pairs,
            "target_evidence_valid_pairs": args.target_evidence_valid_pairs,
            "target_prior_valid_pairs": args.target_prior_valid_pairs,
            "synthetic_companion_rate": args.synthetic_companion_rate,
            "seed": args.seed,
        },
        "source": {
            "rollout_records": len(records),
            "hard_case_tag_distribution": count_tags(records, "hard_case_tags"),
            "learned_override_records": len(learned_map),
        },
        "split_assertions": split_assertions,
        "simple": {
            "train": {
                "evidence": len(simple_evidence_train),
                "prior": len(simple_prior_train),
                "unified": len(simple_unified_train),
                "task_distribution": task_counts(simple_unified_train),
                "rejected_source_distribution": rejected_source_counts(simple_unified_train),
                "rejected_style_distribution": rejected_style_counts(simple_unified_train),
            },
            "valid": {
                "evidence": len(simple_evidence_valid),
                "prior": len(simple_prior_valid),
                "unified": len(simple_unified_valid),
                "task_distribution": task_counts(simple_unified_valid),
                "rejected_source_distribution": rejected_source_counts(simple_unified_valid),
                "rejected_style_distribution": rejected_style_counts(simple_unified_valid),
            },
        },
        "runtime": {
            "train": {
                "evidence": len(runtime_evidence_train),
                "prior": len(runtime_prior_train),
                "unified": len(runtime_unified_train),
                "task_distribution": task_counts(runtime_unified_train),
                "rejected_source_distribution": rejected_source_counts(runtime_unified_train),
                "rejected_style_distribution": rejected_style_counts(runtime_unified_train),
            },
            "valid": {
                "evidence": len(runtime_evidence_valid),
                "prior": len(runtime_prior_valid),
                "unified": len(runtime_unified_valid),
                "task_distribution": task_counts(runtime_unified_valid),
                "rejected_source_distribution": rejected_source_counts(runtime_unified_valid),
                "rejected_style_distribution": rejected_style_counts(runtime_unified_valid),
            },
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[INFO] wrote DPO data to {output_dir}")
    print(summary["runtime"]["train"])


if __name__ == "__main__":
    main()
