#!/usr/bin/env python3
"""Prepare analyzer training data from Bayesian reward debug logs.

This script builds training-ready JSONL files for two analyzer subtasks:

1. prior judge:
   problem + candidate strategies -> suitability JSON
2. evidence judge:
   problem + strategy + reasoning + final answer + correctness -> evidence JSON

It also keeps a stricter clean split and a larger bootstrap split for
evidence labels so later training can choose between high precision and
broader coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


DEFAULT_OUTPUT_DIR = "outputs/analyzer_training_data"
DEFAULT_INPUT_GLOB = "outputs/**/bayesian_reward_debug.jsonl"

DATASET_MODES = ("teacher_clean", "learned_bootstrap")

JUDGE_JSON_SYSTEM_PROMPT = """You are a JSON-only evaluation API.
Return exactly one valid JSON object and nothing else.
Do not include markdown.
Do not include ```json.
Do not include explanations before or after the JSON.
Use double quotes for all keys and string values.
Do not use LaTeX notation inside JSON string values.
Do not use backslashes inside JSON string values.
Write all reason/key fields in plain English text only.
The first character of your response must be {.
The last character of your response must be }."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare analyzer training data from bayesian_reward_debug.jsonl logs."
    )
    parser.add_argument(
        "--input_debug_jsonl",
        action="append",
        default=None,
        help="Path to a bayesian_reward_debug.jsonl file. Repeat to pass multiple files.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset_mode",
        choices=DATASET_MODES,
        default="teacher_clean",
        help=(
            "teacher_clean = legacy prompted-teacher extraction used for analyzer v0. "
            "learned_bootstrap = high-confidence bootstrap extraction from learned analyzer traces."
        ),
    )
    parser.add_argument(
        "--output_tag",
        default="v1",
        help="Version tag used in bootstrap-mode output filenames, e.g. v1 -> evidence_bootstrap_v1_train.jsonl.",
    )
    parser.add_argument(
        "--label_source",
        default=None,
        help=(
            "Optional explicit label source tag. "
            "Defaults to prompted_teacher or learned_analyzer_v0_bootstrap depending on --dataset_mode."
        ),
    )
    parser.add_argument(
        "--solver_run",
        default=None,
        help="Optional solver run tag to stamp into bootstrap metadata, e.g. lambda07.",
    )
    parser.add_argument(
        "--expected_prior_lambda",
        type=float,
        default=None,
        help="Optional prior_lambda filter applied to input rows before selection/hard-case mining.",
    )
    parser.add_argument(
        "--comparison_debug_jsonl",
        action="append",
        default=None,
        help=(
            "Optional comparison bayesian_reward_debug.jsonl path(s), "
            "used for cross-run disagreement hard-case mining."
        ),
    )
    parser.add_argument(
        "--metadata_jsonl",
        action="append",
        default=None,
        help=(
            "Optional metadata JSONL path(s) keyed by problem text. "
            "Used for difficulty-aware hard-case tags such as medium_reward_conflict."
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Problem-level validation split ratio.",
    )
    parser.add_argument(
        "--min_judge_confidence",
        type=float,
        default=0.75,
        help="Minimum teacher confidence for clean/bootstrap evidence labels.",
    )
    parser.add_argument(
        "--min_prior_distinct_suitabilities",
        type=int,
        default=2,
        help="Minimum number of distinct suitability labels in a prior group.",
    )
    parser.add_argument(
        "--hard_case_top_fraction",
        type=float,
        default=0.25,
        help="Top/bottom fraction used for prior/likelihood conflict hard-case mining.",
    )
    parser.add_argument(
        "--high_reward_ratio",
        type=float,
        default=0.8,
        help="Relative-to-group-max reward threshold for incorrect_high_reward style hard cases.",
    )
    return parser.parse_args()


def discover_input_paths(cli_paths: list[str] | None) -> list[Path]:
    if cli_paths:
        paths = [Path(raw_path) for raw_path in cli_paths]
    else:
        paths = sorted(Path(".").glob(DEFAULT_INPUT_GLOB))

    resolved: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if path.stat().st_size == 0:
            continue
        resolved.append(path)
    if not resolved:
        raise RuntimeError("No non-empty bayesian_reward_debug.jsonl files were found.")
    return resolved


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(parsed)
    return rows


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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def build_problem_answer_key(problem_text: str, gold_answer: str) -> str:
    payload = {
        "problem": problem_text,
        "gold_answer": gold_answer,
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def resolve_label_source(args: argparse.Namespace) -> str:
    if args.label_source:
        return str(args.label_source)
    if args.dataset_mode == "learned_bootstrap":
        return "learned_analyzer_v0_bootstrap"
    return "prompted_teacher"


def resolve_label_semantics(args: argparse.Namespace) -> str:
    if args.dataset_mode == "learned_bootstrap":
        return "high_confidence_bootstrap_trace"
    return "clean_teacher_label"


def resolve_selected_quality_tier(args: argparse.Namespace) -> str:
    if args.dataset_mode == "learned_bootstrap":
        return "bootstrap"
    return "clean"


def build_output_spec(args: argparse.Namespace) -> dict[str, str]:
    if args.dataset_mode == "teacher_clean":
        return {
            "selected_evidence_train": "evidence_clean_train.jsonl",
            "selected_evidence_val": "evidence_clean_val.jsonl",
            "selected_evidence_summary_key": "evidence_clean",
            "legacy_evidence_bootstrap_train": "evidence_bootstrap_train.jsonl",
            "legacy_evidence_bootstrap_val": "evidence_bootstrap_val.jsonl",
            "legacy_evidence_bootstrap_summary_key": "evidence_bootstrap",
            "evidence_hard_train": "evidence_hard_cases_train.jsonl",
            "evidence_hard_val": "evidence_hard_cases_val.jsonl",
            "evidence_hard_all": "evidence_hard_cases.jsonl",
            "evidence_hard_summary_key": "evidence_hard_cases",
            "prior_selected_train": "prior_clean_train.jsonl",
            "prior_selected_val": "prior_clean_val.jsonl",
            "prior_selected_summary_key": "prior_clean",
            "prior_pair_train": "prior_pairwise_train.jsonl",
            "prior_pair_val": "prior_pairwise_val.jsonl",
            "prior_pair_all": "prior_pairwise.jsonl",
            "prior_pair_summary_key": "prior_pairwise",
        }

    tag = args.output_tag
    return {
        "selected_evidence_train": f"evidence_bootstrap_{tag}_train.jsonl",
        "selected_evidence_val": f"evidence_bootstrap_{tag}_val.jsonl",
        "selected_evidence_summary_key": f"evidence_bootstrap_{tag}",
        "evidence_hard_train": f"evidence_hard_cases_{tag}_train.jsonl",
        "evidence_hard_val": f"evidence_hard_cases_{tag}_val.jsonl",
        "evidence_hard_all": f"evidence_hard_cases_{tag}.jsonl",
        "evidence_hard_summary_key": f"evidence_hard_cases_{tag}",
        "prior_selected_train": f"prior_bootstrap_{tag}_train.jsonl",
        "prior_selected_val": f"prior_bootstrap_{tag}_val.jsonl",
        "prior_selected_summary_key": f"prior_bootstrap_{tag}",
        "prior_pair_train": f"prior_pairwise_bootstrap_{tag}_train.jsonl",
        "prior_pair_val": f"prior_pairwise_bootstrap_{tag}_val.jsonl",
        "prior_pair_all": f"prior_pairwise_bootstrap_{tag}.jsonl",
        "prior_pair_summary_key": f"prior_pairwise_bootstrap_{tag}",
        "prior_hard_train": f"prior_hard_cases_{tag}_train.jsonl",
        "prior_hard_val": f"prior_hard_cases_{tag}_val.jsonl",
        "prior_hard_all": f"prior_hard_cases_{tag}.jsonl",
        "prior_hard_summary_key": f"prior_hard_cases_{tag}",
        "posterior_hard_train": f"posterior_hard_cases_{tag}_train.jsonl",
        "posterior_hard_val": f"posterior_hard_cases_{tag}_val.jsonl",
        "posterior_hard_all": f"posterior_hard_cases_{tag}.jsonl",
        "posterior_hard_summary_key": f"posterior_hard_cases_{tag}",
    }


def assign_split(problem_text: str, val_ratio: float) -> str:
    digest = stable_hash(problem_text)
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def format_required_rollout_ids(num_rollouts: int) -> str:
    return ", ".join(str(rollout_id) for rollout_id in range(num_rollouts))


def build_prior_judge_prompt(problem_text: str, rollouts: list[dict[str, Any]], num_rollouts: int) -> str:
    strategies_block = "\n\n".join(
        f"Rollout {rollout['rollout_id']}:\n{rollout['strategy'] or '(empty strategy)'}"
        for rollout in rollouts
    )
    required_rollout_ids = format_required_rollout_ids(num_rollouts)
    return f"""
You are a mathematical strategy evaluator.

You will be given a math problem and several candidate strategies generated by a solver.
Your task is to evaluate how promising each strategy is before seeing the actual solution.

Important rules:
- Do not solve the full problem.
- Do not look at reasoning or final answers.
- Evaluate only whether the strategy is appropriate for the problem type.
- Similar strategies may receive similar scores.
- The strategies do not need to be unique.
- Use suitability as an integer from 0 to 4.
- 0 = unsuitable or vague
- 1 = weak
- 2 = partially useful
- 3 = good
- 4 = very strong/directly appropriate
- Do not output prior_probability.
- The code will compute prior probabilities from suitability scores.
- Do not use LaTeX notation in any JSON string field.
- Do not use backslashes.
- Write reason and risk_flag in plain English only.
- You must evaluate every rollout_id from 0 to {num_rollouts - 1}.
- Return exactly {num_rollouts} items in the "priors" list.
- For num_rollouts = {num_rollouts}, return exactly rollout_id {required_rollout_ids}.
- Do not omit any rollout_id.
- Do not merge similar strategies.
- Even if two strategies are identical or very similar, evaluate both separately.
- If a strategy is weak, still include it with low suitability.
- Each rollout_id must appear exactly once.
- Do not invent rollout IDs.
- Do not return fewer than {num_rollouts} rows.
- Do not return more than {num_rollouts} rows.
- suitability must be an integer from 0 to 4.

Return only valid JSON.
The first character must be {{ and the last character must be }}.

JSON schema:
{{
  "priors": [
    {{
      "rollout_id": 0,
      "suitability": 3,
      "reason": "Uses the structure of the problem directly.",
      "risk_flag": "none"
    }}
  ]
}}

Problem:
{problem_text}

Candidate strategies:
{strategies_block}
""".strip()


def build_evidence_judge_prompt(
    problem_text: str,
    strategy: str,
    reasoning: str,
    final_answer: str,
    answer_correctness: float,
) -> str:
    return f"""
Return ONLY a valid JSON object.
Do not include markdown.
Do not include ```json.
Do not include explanations outside JSON.
The first character of your response must be {{ and the last character must be }}.

You are a strict mathematical reasoning judge.

You are given:
1. A math problem
2. The solver's stated strategy
3. The solver's reasoning trajectory
4. The solver's final answer
5. A deterministic correctness flag

The correctness flag is authoritative.
Do not override it.
Your task is not to solve the problem from scratch.
Your task is to evaluate the reliability of the reasoning trajectory.

Important rules:
- Do not reward verbosity.
- Do not reward a solution merely because it sounds fluent.
- Penalize hidden gaps, invalid assumptions, unsupported jumps, contradictions, and strategy mismatch.
- If the final answer is correct but the reasoning is flawed, classify it as lucky_correct or correct_weak_proof.
- If the final answer is incorrect but the reasoning is mostly valid, classify it as finalization_error or valid_but_incomplete.
- If the approach is fundamentally unsuitable, classify it as wrong_direction.
- Use the provided correctness flag as the final authority on answer correctness.

Error type consistency:
- If deterministic correctness flag is 0, error_type must NOT be correct_complete, correct_weak_proof, or lucky_correct.
- If deterministic correctness flag is 1, error_type should be one of correct_complete, correct_weak_proof, or lucky_correct unless there is a format issue.
- Treat the deterministic correctness flag as authoritative.

Evaluate the trajectory using the following rubric.

[Step Validity: 0-4]
0 = no meaningful valid reasoning
1 = major invalid step early in the solution
2 = some correct steps but a key mathematical error exists
3 = mostly valid with minor flaws
4 = all major steps are mathematically valid

[Proof Completeness: 0-4]
0 = no proof or only final answer
1 = central derivation is missing
2 = key idea exists but important steps are omitted
3 = almost complete with minor omissions
4 = complete and self-contained

[Strategy Compliance: 0-4]
0 = does not follow the stated strategy
1 = mentions the strategy but mostly deviates
2 = partially follows the strategy
3 = mostly follows the strategy
4 = the strategy is central to the solution

[Consistency: 0-4]
0 = severe contradictions in variables, assumptions, or final answer
1 = frequent inconsistencies
2 = some inconsistencies but partially traceable
3 = mostly consistent
4 = fully internally consistent

[Error Type]
Choose exactly one:
correct_complete,
correct_weak_proof,
lucky_correct,
finalization_error,
valid_but_incomplete,
arithmetic_error,
algebraic_error,
invalid_assumption,
strategy_mismatch,
wrong_direction,
format_error,
no_meaningful_solution

Return JSON only.

JSON schema:
{{
  "step_validity": 0,
  "proof_completeness": 0,
  "strategy_compliance": 0,
  "consistency": 0,
  "error_type": "",
  "key_strength": "",
  "key_weakness": "",
  "critical_failure_step": "",
  "judge_confidence": 0.0
}}

Problem:
{problem_text}

Strategy:
{strategy or "(empty strategy)"}

Reasoning:
{reasoning or "(empty reasoning)"}

Final Answer:
{final_answer or "(empty final answer)"}

Deterministic correctness flag:
{int(answer_correctness)}

Final output rules:
- Return ONLY one valid JSON object.
- Do not include markdown.
- Do not include a code block.
- Do not include any explanation outside the JSON.
- Do not use LaTeX notation in any JSON string value.
- Do not use backslashes.
- Write mathematical expressions in plain English text.
- Use exactly the keys shown in the JSON schema.
- The response must start with {{ and end with }}.
""".strip()


def build_messages(prompt: str, target: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_JSON_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]


def filtered_rows_for_args(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.expected_prior_lambda is None:
        return rows

    filtered: list[dict[str, Any]] = []
    for row in rows:
        raw_value = row.get("prior_lambda")
        if raw_value is None:
            continue
        try:
            prior_lambda = float(raw_value)
        except (TypeError, ValueError):
            continue
        if abs(prior_lambda - float(args.expected_prior_lambda)) <= 1e-9:
            filtered.append(row)
    return filtered


def load_metadata_lookup(raw_paths: list[str] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not raw_paths:
        return lookup

    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {path}")
        for row in load_jsonl_rows(path):
            problem = str(row.get("problem", "") or "")
            if problem and problem not in lookup:
                lookup[problem] = row
    return lookup


def argmax_index(values: list[float]) -> int:
    if not values:
        raise ValueError("argmax_index requires a non-empty list.")
    return max(range(len(values)), key=lambda index: values[index])


def top_bottom_index_sets(values: list[float], fraction: float) -> tuple[set[int], set[int]]:
    if not values:
        return set(), set()

    n = len(values)
    clamped_fraction = min(max(float(fraction), 0.0), 0.5)
    k = max(1, math.ceil(n * clamped_fraction))
    ascending = sorted(range(n), key=lambda index: values[index])
    low = set(ascending[:k])
    high = set(ascending[-k:])
    return high, low


def build_group_identity(rows: list[dict[str, Any]], source_file: Path) -> dict[str, Any]:
    first = rows[0]
    return {
        "source_file": relative_path_str(source_file),
        "global_reward_call_index": first.get("global_reward_call_index"),
        "group_index_within_call": first.get("group_index_within_call"),
        "problem": str(first.get("problem", "") or ""),
        "gold_answer": str(first.get("gold_answer", "") or ""),
    }


def build_label_metadata(
    row: dict[str, Any],
    *,
    source_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "label_source": resolve_label_source(args),
        "label_semantics": resolve_label_semantics(args),
        "source_file": relative_path_str(source_file),
    }

    if args.solver_run:
        metadata["solver_run"] = str(args.solver_run)
    if args.expected_prior_lambda is not None:
        metadata["prior_lambda_filter"] = float(args.expected_prior_lambda)

    for key in (
        "prior_lambda",
        "prior_mode",
        "learned_analyzer_model_name",
        "learned_analyzer_adapter_path",
        "learned_evidence_analyzer_model_name",
        "learned_evidence_analyzer_adapter_path",
        "learned_analyzer_task_prefix",
        "evidence_judge_model",
        "prior_judge_model",
    ):
        value = row.get(key)
        if value is None:
            continue
        metadata[key] = value

    return metadata


def build_rollout_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = {
        "rollout_id": int(row.get("group_rollout_id", row.get("completion_index_within_call", 0)) or 0),
        "strategy": str(row.get("parsed_strategy", "") or ""),
        "reasoning": str(row.get("parsed_reasoning", "") or ""),
        "final_answer": str(row.get("parsed_final_answer", "") or ""),
        "answer_correctness": float(row.get("answer_correctness", 0.0)),
        "prior_suitability": (
            int(row["prior_suitability"])
            if row.get("prior_suitability") is not None
            else None
        ),
        "prior_probability": (
            float(row["prior_probability"])
            if row.get("prior_probability") is not None
            else None
        ),
        "likelihood": float(row.get("likelihood", 0.0)),
        "bayesian_reward": float(row.get("bayesian_reward", 0.0)),
        "error_type": str(row.get("error_type", "") or ""),
        "judge_confidence": float(row.get("judge_confidence", 0.0)),
        "normalized_predicted_answer": str(row.get("normalized_predicted_answer", "") or ""),
    }
    return snapshot


def summarize_group_for_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0)))
    rewards = [float(row.get("bayesian_reward", 0.0)) for row in ordered_rows]
    top_index = argmax_index(rewards)
    top_row = ordered_rows[top_index]
    return {
        "problem_key": build_problem_answer_key(
            str(top_row.get("problem", "") or ""),
            str(top_row.get("gold_answer", "") or ""),
        ),
        "top_rollout_id": int(top_row.get("group_rollout_id", top_index)),
        "top_bayesian_reward": float(top_row.get("bayesian_reward", 0.0)),
        "top_answer_correctness": float(top_row.get("answer_correctness", 0.0)),
        "top_normalized_predicted_answer": str(top_row.get("normalized_predicted_answer", "") or ""),
        "top_prior_suitability": (
            int(top_row["prior_suitability"])
            if top_row.get("prior_suitability") is not None
            else None
        ),
        "prior_lambda": (
            float(top_row["prior_lambda"])
            if top_row.get("prior_lambda") is not None
            else None
        ),
        "group_size": len(ordered_rows),
        "source_file": str(top_row.get("_source_file", "") or ""),
    }


def load_comparison_group_lookup(raw_paths: list[str] | None) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not raw_paths:
        return lookup

    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Comparison debug file does not exist: {path}")
        rows = load_jsonl_rows(path)
        for row in rows:
            row["_source_file"] = relative_path_str(path)

        groups_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            problem_key = build_problem_answer_key(
                str(row.get("problem", "") or ""),
                str(row.get("gold_answer", "") or ""),
            )
            groups_by_problem[problem_key].append(row)

        for summaries in groups_by_problem.values():
            summary = summarize_group_for_comparison(summaries)
            lookup[summary["problem_key"]].append(summary)

    return lookup


def evidence_quality_issues(row: dict[str, Any], *, min_judge_confidence: float) -> list[str]:
    issues: list[str] = []
    if row.get("evidence_judge_fallback_used"):
        issues.append("evidence_fallback")
    if not row.get("format_valid", False):
        issues.append("format_invalid")
    if not row.get("strategy_section_present", False):
        issues.append("strategy_section_missing")
    if not row.get("reasoning_section_present", False):
        issues.append("reasoning_section_missing")
    if not row.get("final_answer_section_present", False):
        issues.append("final_answer_section_missing")
    if row.get("parse_failure_reasons"):
        issues.append("judge_parse_failure")
    if float(row.get("judge_confidence", 0.0)) < min_judge_confidence:
        issues.append("low_teacher_confidence")
    if not str(row.get("parsed_strategy", "")).strip():
        issues.append("empty_strategy")
    if not str(row.get("parsed_reasoning", "")).strip():
        issues.append("empty_reasoning")
    if not str(row.get("parsed_final_answer", "")).strip():
        issues.append("empty_final_answer")
    return issues


def prior_group_quality_issues(
    rows: list[dict[str, Any]], *, min_prior_distinct_suitabilities: int
) -> list[str]:
    issues: list[str] = []
    suitabilities = {int(row.get("prior_suitability", 0)) for row in rows}
    if any(row.get("prior_judge_fallback_used") for row in rows):
        issues.append("prior_fallback")
    if any(row.get("prior_missing_from_judge") for row in rows):
        issues.append("prior_missing_from_judge")
    if any(not row.get("format_valid", False) for row in rows):
        issues.append("format_invalid")
    if any(not row.get("strategy_section_present", False) for row in rows):
        issues.append("strategy_section_missing")
    if any(not str(row.get("parsed_strategy", "")).strip() for row in rows):
        issues.append("empty_strategy")
    if len(suitabilities) < min_prior_distinct_suitabilities:
        issues.append("low_suitability_diversity")
    return issues


def build_evidence_example(
    row: dict[str, Any],
    *,
    source_file: Path,
    quality_tier: str,
    val_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target = {
        "step_validity": int(row["step_validity"]),
        "proof_completeness": int(row["proof_completeness"]),
        "strategy_compliance": int(row["strategy_compliance"]),
        "consistency": int(row["consistency"]),
        "error_type": str(row["error_type"]),
        "key_strength": str(row.get("key_strength", "") or ""),
        "key_weakness": str(row.get("key_weakness", "") or ""),
        "critical_failure_step": str(row.get("critical_failure_step", "") or ""),
        "judge_confidence": float(row.get("judge_confidence", 0.0)),
    }
    problem = str(row["problem"])
    strategy = str(row.get("parsed_strategy", "") or "")
    reasoning = str(row.get("parsed_reasoning", "") or "")
    final_answer = str(row.get("parsed_final_answer", "") or "")
    prompt = build_evidence_judge_prompt(
        problem,
        strategy,
        reasoning,
        final_answer,
        float(row["answer_correctness"]),
    )
    signature = json.dumps(
        {
            "problem": problem,
            "strategy": strategy,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "answer_correctness": float(row["answer_correctness"]),
            "target": target,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    label_metadata = build_label_metadata(row, source_file=source_file, args=args)
    return {
        "example_id": stable_hash(signature),
        "task": "evidence_judge",
        "split": assign_split(problem, val_ratio),
        "quality_tier": quality_tier,
        "source_file": relative_path_str(source_file),
        "label_source": label_metadata["label_source"],
        "label_semantics": label_metadata["label_semantics"],
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
        "answer_correctness": float(row["answer_correctness"]),
        "teacher_target": target,
        "teacher_metadata": {
            "bayesian_reward": float(row.get("bayesian_reward", 0.0)),
            "likelihood": float(row.get("likelihood", 0.0)),
            "judge_label_inconsistency": bool(row.get("judge_label_inconsistency", False)),
            "evidence_source": str(row.get("evidence_source", "") or ""),
            "verification_method": str(row.get("verification_method", "") or ""),
        },
        "prompt": prompt,
        "messages": build_messages(prompt, target),
    }


def build_prior_example(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    val_ratio: float,
    quality_tier: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0)))
    problem = str(ordered_rows[0]["problem"])
    rollouts = [
        {
            "rollout_id": int(row["group_rollout_id"]),
            "strategy": str(row.get("parsed_strategy", "") or ""),
        }
        for row in ordered_rows
    ]
    target = {
        "priors": [
            {
                "rollout_id": int(row["group_rollout_id"]),
                "suitability": int(row["prior_suitability"]),
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
            "rollouts": rollouts,
            "target": target,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    label_metadata = build_label_metadata(ordered_rows[0], source_file=source_file, args=args)
    return {
        "example_id": stable_hash(signature),
        "task": "prior_judge",
        "split": assign_split(problem, val_ratio),
        "quality_tier": quality_tier,
        "source_file": relative_path_str(source_file),
        "label_source": label_metadata["label_source"],
        "label_semantics": label_metadata["label_semantics"],
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
                "rollout_id": int(row["group_rollout_id"]),
                "strategy": str(row.get("parsed_strategy", "") or ""),
                "teacher_suitability": int(row["prior_suitability"]),
                "teacher_probability": float(row.get("prior_probability", 0.0)),
                "teacher_reason": str(row.get("prior_reason", "") or ""),
                "risk_flag": str(row.get("prior_risk_flag", "") or ""),
            }
            for row in ordered_rows
        ],
        "teacher_target": target,
        "prompt": prompt,
        "messages": build_messages(prompt, target),
    }


def build_prior_pair_examples(group_example: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(group_example["candidates"])
    pairs: list[dict[str, Any]] = []
    for chosen in candidates:
        for rejected in candidates:
            if int(chosen["teacher_suitability"]) <= int(rejected["teacher_suitability"]):
                continue
            pair_signature = json.dumps(
                {
                    "problem": group_example["problem"],
                    "chosen_rollout_id": chosen["rollout_id"],
                    "rejected_rollout_id": rejected["rollout_id"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            pairs.append(
                {
                    "pair_id": stable_hash(pair_signature),
                    "task": "prior_pairwise_preference",
                    "split": group_example["split"],
                    "source_file": group_example["source_file"],
                    "quality_tier": group_example["quality_tier"],
                    "label_source": group_example["label_source"],
                    "label_semantics": group_example["label_semantics"],
                    "label_metadata": group_example["label_metadata"],
                    "problem_key": group_example["problem_key"],
                    "problem": group_example["problem"],
                    "chosen_rollout_id": int(chosen["rollout_id"]),
                    "chosen_strategy": str(chosen["strategy"]),
                    "chosen_suitability": int(chosen["teacher_suitability"]),
                    "rejected_rollout_id": int(rejected["rollout_id"]),
                    "rejected_strategy": str(rejected["strategy"]),
                    "rejected_suitability": int(rejected["teacher_suitability"]),
                    "suitability_margin": int(chosen["teacher_suitability"]) - int(rejected["teacher_suitability"]),
                }
            )
    return pairs


def build_evidence_hard_case_example(
    row: dict[str, Any],
    *,
    source_file: Path,
    hard_case_tags: list[str],
    val_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    example = build_evidence_example(
        row,
        source_file=source_file,
        quality_tier="hard_case",
        val_ratio=val_ratio,
        args=args,
    )
    example["hard_case_tags"] = sorted(set(hard_case_tags))
    return example


def build_prior_hard_case_example(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    hard_case_tags: list[str],
    flagged_rollouts: list[dict[str, Any]],
    difficulty_bucket: Optional[str],
    val_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0)))
    problem = str(ordered_rows[0].get("problem", "") or "")
    gold_answer = str(ordered_rows[0].get("gold_answer", "") or "")
    identity = build_group_identity(ordered_rows, source_file)
    case_id = stable_hash(json.dumps(identity, ensure_ascii=False, sort_keys=True))
    label_metadata = build_label_metadata(ordered_rows[0], source_file=source_file, args=args)
    return {
        "example_id": case_id,
        "task": "prior_hard_case_group",
        "split": assign_split(problem, val_ratio),
        "source_file": relative_path_str(source_file),
        "label_source": label_metadata["label_source"],
        "label_semantics": label_metadata["label_semantics"],
        "label_metadata": label_metadata,
        "problem_key": stable_hash(problem),
        "group_key": {
            "global_reward_call_index": ordered_rows[0].get("global_reward_call_index"),
            "group_index_within_call": ordered_rows[0].get("group_index_within_call"),
        },
        "problem": problem,
        "gold_answer": gold_answer,
        "difficulty_bucket": difficulty_bucket,
        "hard_case_tags": sorted(set(hard_case_tags)),
        "flagged_rollouts": flagged_rollouts,
        "rollouts": [build_rollout_snapshot(row) for row in ordered_rows],
    }


def build_posterior_hard_case_example(
    rows: list[dict[str, Any]],
    *,
    source_file: Path,
    hard_case_tags: list[str],
    difficulty_bucket: Optional[str],
    comparison_summaries: list[dict[str, Any]],
    val_ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0)))
    problem = str(ordered_rows[0].get("problem", "") or "")
    gold_answer = str(ordered_rows[0].get("gold_answer", "") or "")
    rewards = [float(row.get("bayesian_reward", 0.0)) for row in ordered_rows]
    top_index = argmax_index(rewards)
    top_row = ordered_rows[top_index]
    identity = build_group_identity(ordered_rows, source_file)
    case_id = stable_hash(json.dumps(identity, ensure_ascii=False, sort_keys=True))
    label_metadata = build_label_metadata(ordered_rows[0], source_file=source_file, args=args)
    return {
        "example_id": case_id,
        "task": "posterior_hard_case_group",
        "split": assign_split(problem, val_ratio),
        "source_file": relative_path_str(source_file),
        "label_source": label_metadata["label_source"],
        "label_semantics": label_metadata["label_semantics"],
        "label_metadata": label_metadata,
        "problem_key": stable_hash(problem),
        "group_key": {
            "global_reward_call_index": ordered_rows[0].get("global_reward_call_index"),
            "group_index_within_call": ordered_rows[0].get("group_index_within_call"),
        },
        "problem": problem,
        "gold_answer": gold_answer,
        "difficulty_bucket": difficulty_bucket,
        "hard_case_tags": sorted(set(hard_case_tags)),
        "top_rollout_id": int(top_row.get("group_rollout_id", top_index)),
        "top_answer_correctness": float(top_row.get("answer_correctness", 0.0)),
        "top_bayesian_reward": float(top_row.get("bayesian_reward", 0.0)),
        "top_normalized_predicted_answer": str(top_row.get("normalized_predicted_answer", "") or ""),
        "comparison_summaries": comparison_summaries,
        "rollouts": [build_rollout_snapshot(row) for row in ordered_rows],
    }


def analyze_group_hard_cases(
    rows: list[dict[str, Any]],
    *,
    difficulty_bucket: Optional[str],
    comparison_summaries: list[dict[str, Any]],
    hard_case_top_fraction: float,
    high_reward_ratio: float,
) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: int(row.get("group_rollout_id", 0)))
    rewards = [float(row.get("bayesian_reward", 0.0)) for row in ordered_rows]
    likelihoods = [float(row.get("likelihood", 0.0)) for row in ordered_rows]
    corrects = [float(row.get("answer_correctness", 0.0)) for row in ordered_rows]
    error_types = [str(row.get("error_type", "") or "") for row in ordered_rows]
    priors_present = all(row.get("prior_probability") is not None for row in ordered_rows)
    prior_probabilities = [float(row.get("prior_probability", 0.0)) for row in ordered_rows]

    top_reward_index = argmax_index(rewards)
    max_reward = rewards[top_reward_index]
    any_correct = any(value == 1.0 for value in corrects)
    any_incorrect = any(value == 0.0 for value in corrects)

    evidence_tags_by_rollout: dict[int, set[str]] = defaultdict(set)
    prior_tags_by_rollout: dict[int, set[str]] = defaultdict(set)
    posterior_group_tags: set[str] = set()
    prior_group_tags: set[str] = set()

    if corrects[top_reward_index] == 0.0:
        posterior_group_tags.add("posterior_top_incorrect")

    for index, row in enumerate(ordered_rows):
        reward = rewards[index]
        answer_correct = corrects[index]
        error_type = error_types[index]

        if answer_correct == 0.0 and max_reward > 0.0 and reward >= high_reward_ratio * max_reward:
            evidence_tags_by_rollout[index].add("incorrect_high_reward")
        if answer_correct == 1.0 and any_incorrect and reward < max_reward:
            evidence_tags_by_rollout[index].add("correct_not_top_reward")
        if (
            answer_correct == 0.0
            and error_type in {"wrong_direction", "strategy_mismatch"}
            and max_reward > 0.0
            and reward >= high_reward_ratio * max_reward
        ):
            evidence_tags_by_rollout[index].add("off_target_high_reward")

    if any("incorrect_high_reward" in tags for tags in evidence_tags_by_rollout.values()):
        posterior_group_tags.add("incorrect_high_reward_present")
    if any("correct_not_top_reward" in tags for tags in evidence_tags_by_rollout.values()):
        posterior_group_tags.add("correct_not_top_reward_present")
    if any("off_target_high_reward" in tags for tags in evidence_tags_by_rollout.values()):
        posterior_group_tags.add("off_target_high_reward_present")

    if priors_present:
        high_prior_indexes, low_prior_indexes = top_bottom_index_sets(prior_probabilities, hard_case_top_fraction)
        high_likelihood_indexes, low_likelihood_indexes = top_bottom_index_sets(likelihoods, hard_case_top_fraction)

        for index, row in enumerate(ordered_rows):
            answer_correct = corrects[index]

            if answer_correct == 1.0 and index in low_prior_indexes:
                prior_tags_by_rollout[index].add("low_prior_correct_rollout")
            if answer_correct == 0.0 and index in high_prior_indexes:
                prior_tags_by_rollout[index].add("high_prior_wrong_rollout")
            if index in high_prior_indexes and index in low_likelihood_indexes:
                prior_tags_by_rollout[index].add("high_prior_low_likelihood_rollout")
            if index in low_prior_indexes and index in high_likelihood_indexes:
                prior_tags_by_rollout[index].add("low_prior_high_likelihood_rollout")

        if any(prior_tags_by_rollout.values()):
            prior_group_tags.add("prior_evidence_conflict")

    top_row = ordered_rows[top_reward_index]
    top_answer = str(top_row.get("normalized_predicted_answer", "") or "")
    top_correctness = float(top_row.get("answer_correctness", 0.0))

    comparison_disagreement = False
    for summary in comparison_summaries:
        if (
            str(summary.get("top_normalized_predicted_answer", "") or "") != top_answer
            or float(summary.get("top_answer_correctness", 0.0)) != top_correctness
        ):
            comparison_disagreement = True
            break
    if comparison_disagreement:
        posterior_group_tags.add("lambda_selection_disagreement")

    if difficulty_bucket == "medium":
        has_reward_conflict = (
            "posterior_top_incorrect" in posterior_group_tags
            or comparison_disagreement
            or any(evidence_tags_by_rollout.values())
            or any(prior_tags_by_rollout.values())
            or (any_correct and any_incorrect)
        )
        if has_reward_conflict:
            posterior_group_tags.add("medium_reward_conflict")

    return {
        "ordered_rows": ordered_rows,
        "evidence_tags_by_rollout": {
            int(ordered_rows[index].get("group_rollout_id", index)): sorted(tags)
            for index, tags in evidence_tags_by_rollout.items()
            if tags
        },
        "prior_tags_by_rollout": {
            int(ordered_rows[index].get("group_rollout_id", index)): sorted(tags)
            for index, tags in prior_tags_by_rollout.items()
            if tags
        },
        "posterior_group_tags": sorted(posterior_group_tags),
        "prior_group_tags": sorted(prior_group_tags),
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    return train_rows, val_rows


def count_by_key(rows: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value: Any = row
        for key in key_path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def count_tag_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for tag in row.get(key, []):
            counter[str(tag)] += 1
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    input_paths = discover_input_paths(args.input_debug_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_spec = build_output_spec(args)
    selected_quality_tier = resolve_selected_quality_tier(args)
    metadata_lookup = load_metadata_lookup(args.metadata_jsonl)
    comparison_lookup = load_comparison_group_lookup(args.comparison_debug_jsonl)

    selected_evidence_examples: list[dict[str, Any]] = []
    legacy_bootstrap_evidence_examples: list[dict[str, Any]] = []
    evidence_hard_case_by_id: dict[str, dict[str, Any]] = {}
    prior_group_examples: list[dict[str, Any]] = []
    prior_pair_examples: list[dict[str, Any]] = []
    prior_hard_case_by_id: dict[str, dict[str, Any]] = {}
    posterior_hard_case_by_id: dict[str, dict[str, Any]] = {}

    selected_evidence_seen: set[str] = set()
    legacy_bootstrap_evidence_seen: set[str] = set()
    prior_group_seen: set[str] = set()
    prior_pair_seen: set[str] = set()

    evidence_issue_counts: Counter[str] = Counter()
    prior_issue_counts: Counter[str] = Counter()
    source_row_counts_before_filter: dict[str, int] = {}
    source_row_counts_after_filter: dict[str, int] = {}
    source_group_counts: dict[str, int] = {}

    for path in input_paths:
        raw_rows = load_jsonl_rows(path)
        rows = filtered_rows_for_args(raw_rows, args)
        source_key = relative_path_str(path)
        source_row_counts_before_filter[source_key] = len(raw_rows)
        source_row_counts_after_filter[source_key] = len(rows)

        groups_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        evidence_groups_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        has_prior_labels = any("prior_suitability" in row for row in rows)

        for row in rows:
            issues = evidence_quality_issues(
                row,
                min_judge_confidence=args.min_judge_confidence,
            )
            evidence_issue_counts.update(issues)

            bootstrap_ok = not issues
            clean_ok = bootstrap_ok and not row.get("judge_label_inconsistency", False)

            if row.get("judge_label_inconsistency", False):
                evidence_issue_counts.update(["label_inconsistency"])
                hard_case = build_evidence_hard_case_example(
                    row,
                    source_file=path,
                    hard_case_tags=["label_inconsistency"],
                    val_ratio=args.val_ratio,
                    args=args,
                )
                evidence_hard_case_by_id[hard_case["example_id"]] = hard_case

            if args.dataset_mode == "teacher_clean" and bootstrap_ok:
                example = build_evidence_example(
                    row,
                    source_file=path,
                    quality_tier="bootstrap",
                    val_ratio=args.val_ratio,
                    args=args,
                )
                if example["example_id"] not in legacy_bootstrap_evidence_seen:
                    legacy_bootstrap_evidence_seen.add(example["example_id"])
                    legacy_bootstrap_evidence_examples.append(example)

            if clean_ok:
                example = build_evidence_example(
                    row,
                    source_file=path,
                    quality_tier=selected_quality_tier,
                    val_ratio=args.val_ratio,
                    args=args,
                )
                if example["example_id"] not in selected_evidence_seen:
                    selected_evidence_seen.add(example["example_id"])
                    selected_evidence_examples.append(example)

            if has_prior_labels:
                group_key = (
                    row.get("global_reward_call_index"),
                    row.get("group_index_within_call"),
                )
                groups_by_key[group_key].append(row)

            if row.get("global_reward_call_index") is not None:
                evidence_group_key = (
                    row.get("global_reward_call_index"),
                    row.get("group_index_within_call"),
                )
                evidence_groups_by_key[evidence_group_key].append(row)

        source_group_counts[source_key] = len(evidence_groups_by_key)

        for group_rows in evidence_groups_by_key.values():
            if not group_rows:
                continue

            ordered_rows = sorted(group_rows, key=lambda row: int(row.get("group_rollout_id", 0)))
            problem = str(ordered_rows[0].get("problem", "") or "")
            gold_answer = str(ordered_rows[0].get("gold_answer", "") or "")
            problem_answer_key = build_problem_answer_key(problem, gold_answer)
            difficulty_bucket = None
            if problem in metadata_lookup:
                raw_bucket = metadata_lookup[problem].get("difficulty_bucket")
                if raw_bucket is not None:
                    difficulty_bucket = str(raw_bucket)

            hard_case_analysis = analyze_group_hard_cases(
                ordered_rows,
                difficulty_bucket=difficulty_bucket,
                comparison_summaries=comparison_lookup.get(problem_answer_key, []),
                hard_case_top_fraction=args.hard_case_top_fraction,
                high_reward_ratio=args.high_reward_ratio,
            )

            combined_rollout_tags: dict[int, set[str]] = defaultdict(set)
            for rollout_id, tags in hard_case_analysis["evidence_tags_by_rollout"].items():
                combined_rollout_tags[int(rollout_id)].update(tags)
            for rollout_id, tags in hard_case_analysis["prior_tags_by_rollout"].items():
                combined_rollout_tags[int(rollout_id)].update(tags)

            for row in ordered_rows:
                rollout_id = int(row.get("group_rollout_id", 0) or 0)
                tags = sorted(combined_rollout_tags.get(rollout_id, set()))
                if not tags:
                    continue
                hard_case = build_evidence_hard_case_example(
                    row,
                    source_file=path,
                    hard_case_tags=tags,
                    val_ratio=args.val_ratio,
                    args=args,
                )
                existing = evidence_hard_case_by_id.get(hard_case["example_id"])
                if existing is None:
                    evidence_hard_case_by_id[hard_case["example_id"]] = hard_case
                else:
                    existing_tags = set(existing.get("hard_case_tags", []))
                    existing_tags.update(hard_case["hard_case_tags"])
                    existing["hard_case_tags"] = sorted(existing_tags)

            if args.dataset_mode == "learned_bootstrap":
                if hard_case_analysis["prior_tags_by_rollout"] or hard_case_analysis["prior_group_tags"]:
                    flagged_rollouts: list[dict[str, Any]] = []
                    for row in ordered_rows:
                        rollout_id = int(row.get("group_rollout_id", 0) or 0)
                        rollout_tags = hard_case_analysis["prior_tags_by_rollout"].get(rollout_id, [])
                        if not rollout_tags:
                            continue
                        snapshot = build_rollout_snapshot(row)
                        snapshot["hard_case_tags"] = rollout_tags
                        flagged_rollouts.append(snapshot)

                    prior_hard_case = build_prior_hard_case_example(
                        ordered_rows,
                        source_file=path,
                        hard_case_tags=hard_case_analysis["prior_group_tags"]
                        + sorted({tag for tags in hard_case_analysis["prior_tags_by_rollout"].values() for tag in tags}),
                        flagged_rollouts=flagged_rollouts,
                        difficulty_bucket=difficulty_bucket,
                        val_ratio=args.val_ratio,
                        args=args,
                    )
                    existing = prior_hard_case_by_id.get(prior_hard_case["example_id"])
                    if existing is None:
                        prior_hard_case_by_id[prior_hard_case["example_id"]] = prior_hard_case
                    else:
                        existing_tags = set(existing.get("hard_case_tags", []))
                        existing_tags.update(prior_hard_case["hard_case_tags"])
                        existing["hard_case_tags"] = sorted(existing_tags)

                if hard_case_analysis["posterior_group_tags"]:
                    posterior_hard_case = build_posterior_hard_case_example(
                        ordered_rows,
                        source_file=path,
                        hard_case_tags=hard_case_analysis["posterior_group_tags"],
                        difficulty_bucket=difficulty_bucket,
                        comparison_summaries=comparison_lookup.get(problem_answer_key, []),
                        val_ratio=args.val_ratio,
                        args=args,
                    )
                    existing = posterior_hard_case_by_id.get(posterior_hard_case["example_id"])
                    if existing is None:
                        posterior_hard_case_by_id[posterior_hard_case["example_id"]] = posterior_hard_case
                    else:
                        existing_tags = set(existing.get("hard_case_tags", []))
                        existing_tags.update(posterior_hard_case["hard_case_tags"])
                        existing["hard_case_tags"] = sorted(existing_tags)

        if has_prior_labels:
            for group_rows in groups_by_key.values():
                issues = prior_group_quality_issues(
                    group_rows,
                    min_prior_distinct_suitabilities=args.min_prior_distinct_suitabilities,
                )
                prior_issue_counts.update(issues)
                if issues:
                    continue
                group_example = build_prior_example(
                    group_rows,
                    source_file=path,
                    val_ratio=args.val_ratio,
                    quality_tier=selected_quality_tier,
                    args=args,
                )
                if group_example["example_id"] in prior_group_seen:
                    continue
                prior_group_seen.add(group_example["example_id"])
                prior_group_examples.append(group_example)

                for pair_example in build_prior_pair_examples(group_example):
                    if pair_example["pair_id"] in prior_pair_seen:
                        continue
                    prior_pair_seen.add(pair_example["pair_id"])
                    prior_pair_examples.append(pair_example)

    selected_evidence_examples.sort(key=lambda row: row["example_id"])
    legacy_bootstrap_evidence_examples.sort(key=lambda row: row["example_id"])
    evidence_hard_case_examples = sorted(evidence_hard_case_by_id.values(), key=lambda row: row["example_id"])
    prior_group_examples.sort(key=lambda row: row["example_id"])
    prior_pair_examples.sort(key=lambda row: row["pair_id"])
    prior_hard_case_examples = sorted(prior_hard_case_by_id.values(), key=lambda row: row["example_id"])
    posterior_hard_case_examples = sorted(posterior_hard_case_by_id.values(), key=lambda row: row["example_id"])

    selected_evidence_train, selected_evidence_val = split_rows(selected_evidence_examples)
    legacy_bootstrap_train, legacy_bootstrap_val = split_rows(legacy_bootstrap_evidence_examples)
    evidence_hard_case_train, evidence_hard_case_val = split_rows(evidence_hard_case_examples)
    prior_group_train, prior_group_val = split_rows(prior_group_examples)
    prior_pair_train, prior_pair_val = split_rows(prior_pair_examples)
    prior_hard_case_train, prior_hard_case_val = split_rows(prior_hard_case_examples)
    posterior_hard_case_train, posterior_hard_case_val = split_rows(posterior_hard_case_examples)

    write_jsonl(output_dir / output_spec["selected_evidence_train"], selected_evidence_train)
    write_jsonl(output_dir / output_spec["selected_evidence_val"], selected_evidence_val)
    write_jsonl(output_dir / output_spec["evidence_hard_train"], evidence_hard_case_train)
    write_jsonl(output_dir / output_spec["evidence_hard_val"], evidence_hard_case_val)
    write_jsonl(output_dir / output_spec["evidence_hard_all"], evidence_hard_case_examples)
    write_jsonl(output_dir / output_spec["prior_selected_train"], prior_group_train)
    write_jsonl(output_dir / output_spec["prior_selected_val"], prior_group_val)
    write_jsonl(output_dir / output_spec["prior_pair_train"], prior_pair_train)
    write_jsonl(output_dir / output_spec["prior_pair_val"], prior_pair_val)
    write_jsonl(output_dir / output_spec["prior_pair_all"], prior_pair_examples)

    if args.dataset_mode == "teacher_clean":
        write_jsonl(output_dir / output_spec["legacy_evidence_bootstrap_train"], legacy_bootstrap_train)
        write_jsonl(output_dir / output_spec["legacy_evidence_bootstrap_val"], legacy_bootstrap_val)
    else:
        write_jsonl(output_dir / output_spec["prior_hard_train"], prior_hard_case_train)
        write_jsonl(output_dir / output_spec["prior_hard_val"], prior_hard_case_val)
        write_jsonl(output_dir / output_spec["prior_hard_all"], prior_hard_case_examples)
        write_jsonl(output_dir / output_spec["posterior_hard_train"], posterior_hard_case_train)
        write_jsonl(output_dir / output_spec["posterior_hard_val"], posterior_hard_case_val)
        write_jsonl(output_dir / output_spec["posterior_hard_all"], posterior_hard_case_examples)

    outputs_summary: dict[str, Any] = {
        output_spec["selected_evidence_summary_key"]: {
            "train": len(selected_evidence_train),
            "val": len(selected_evidence_val),
            "error_type_distribution": count_by_key(
                selected_evidence_examples, ("teacher_target", "error_type")
            ),
            "label_source_distribution": count_by_key(selected_evidence_examples, ("label_source",)),
        },
        output_spec["evidence_hard_summary_key"]: {
            "train": len(evidence_hard_case_train),
            "val": len(evidence_hard_case_val),
            "tag_distribution": count_tag_distribution(
                evidence_hard_case_examples, "hard_case_tags"
            ),
        },
        output_spec["prior_selected_summary_key"]: {
            "train": len(prior_group_train),
            "val": len(prior_group_val),
            "suitability_distribution": count_by_key(
                [
                    candidate
                    for example in prior_group_examples
                    for candidate in example["candidates"]
                ],
                ("teacher_suitability",),
            ),
            "label_source_distribution": count_by_key(prior_group_examples, ("label_source",)),
        },
        output_spec["prior_pair_summary_key"]: {
            "train": len(prior_pair_train),
            "val": len(prior_pair_val),
        },
    }

    if args.dataset_mode == "teacher_clean":
        outputs_summary[output_spec["legacy_evidence_bootstrap_summary_key"]] = {
            "train": len(legacy_bootstrap_train),
            "val": len(legacy_bootstrap_val),
            "error_type_distribution": count_by_key(
                legacy_bootstrap_evidence_examples, ("teacher_target", "error_type")
            ),
        }
    else:
        outputs_summary[output_spec["prior_hard_summary_key"]] = {
            "train": len(prior_hard_case_train),
            "val": len(prior_hard_case_val),
            "tag_distribution": count_tag_distribution(
                prior_hard_case_examples, "hard_case_tags"
            ),
        }
        outputs_summary[output_spec["posterior_hard_summary_key"]] = {
            "train": len(posterior_hard_case_train),
            "val": len(posterior_hard_case_val),
            "tag_distribution": count_tag_distribution(
                posterior_hard_case_examples, "hard_case_tags"
            ),
        }

    summary = {
        "input_files": [relative_path_str(path) for path in input_paths],
        "comparison_files": [relative_path_str(Path(path)) for path in (args.comparison_debug_jsonl or [])],
        "metadata_files": [relative_path_str(Path(path)) for path in (args.metadata_jsonl or [])],
        "source_row_counts_before_filter": source_row_counts_before_filter,
        "source_row_counts_after_filter": source_row_counts_after_filter,
        "source_group_counts": source_group_counts,
        "config": {
            "dataset_mode": args.dataset_mode,
            "output_tag": args.output_tag,
            "label_source": resolve_label_source(args),
            "label_semantics": resolve_label_semantics(args),
            "solver_run": args.solver_run,
            "expected_prior_lambda": args.expected_prior_lambda,
            "val_ratio": args.val_ratio,
            "min_judge_confidence": args.min_judge_confidence,
            "min_prior_distinct_suitabilities": args.min_prior_distinct_suitabilities,
            "hard_case_top_fraction": args.hard_case_top_fraction,
            "high_reward_ratio": args.high_reward_ratio,
        },
        "selection": {
            "selected_quality_tier": selected_quality_tier,
            "evidence_issue_counts": dict(sorted(evidence_issue_counts.items())),
            "prior_issue_counts": dict(sorted(prior_issue_counts.items())),
        },
        "outputs": outputs_summary,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"[INFO] wrote analyzer training data to {output_dir}")
    print(
        "[INFO] selected evidence "
        f"train={len(selected_evidence_train)} val={len(selected_evidence_val)} | "
        "evidence hard "
        f"train={len(evidence_hard_case_train)} val={len(evidence_hard_case_val)}"
    )
    if args.dataset_mode == "teacher_clean":
        print(
            "[INFO] legacy bootstrap evidence "
            f"train={len(legacy_bootstrap_train)} val={len(legacy_bootstrap_val)}"
        )
    else:
        print(
            "[INFO] prior bootstrap "
            f"train={len(prior_group_train)} val={len(prior_group_val)} | "
            "prior hard "
            f"train={len(prior_hard_case_train)} val={len(prior_hard_case_val)} | "
            "posterior hard "
            f"train={len(posterior_hard_case_train)} val={len(posterior_hard_case_val)}"
        )
    print(
        "[INFO] prior pairwise "
        f"train={len(prior_pair_train)} val={len(prior_pair_val)}"
    )


if __name__ == "__main__":
    main()
