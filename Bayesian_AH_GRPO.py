#!/usr/bin/env python3
"""Experiment B: Uniform-prior Bayesian evidence GRPO on Big-Math-RL-Verified.

This script intentionally mirrors Answer_only_GRPO.py as closely as possible.
The only intended experimental difference is the reward function:

- Answer_only_GRPO.py: reward = answer_correctness
- Bayesian_GRPO.py: reward = answer-heavy Bayesian evidence likelihood
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

from Answer_only_GRPO import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAME,
    add_bool_arg,
    align_gold_answers,
    align_values,
    build_grpo_training_components,
    build_peft_config,
    build_training_config_payload,
    create_grpo_config,
    create_grpo_trainer,
    default_bf16_enabled,
    difficulty_bucket_from_solve_rate,
    ensure_output_dir,
    extract_section,
    extract_text_from_completion,
    generate_smoke_outputs,
    import_torch,
    load_dataset_rows,
    load_tokenizer,
    maybe_warn_on_smoke_metrics,
    parse_completion_sections,
    parse_solve_rate,
    render_prompt,
    run_parser_self_test,
    run_smoke_test,
    set_seed,
    summarize_selected_train,
    verify_answer,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = "outputs/grpo_bayesian_uniform_qwen3b_bigmath_n4_steps200"
DEFAULT_TRAIN_METADATA_PATH = (
    "outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_train_metadata.jsonl"
)
DEFAULT_EVAL_METADATA_PATH = (
    "outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_eval_metadata.jsonl"
)
DEFAULT_REWARD_DEBUG_JSONL = "bayesian_reward_debug.jsonl"

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

LIKELIHOOD_WEIGHTS = {
    "answer_correctness": 0.80,
    "step_validity": 0.07,
    "proof_completeness": 0.08,
    "strategy_compliance": 0.02,
    "consistency": 0.03,
}


@dataclass
class EvidenceAssessment:
    step_validity: int
    proof_completeness: int
    strategy_compliance: int
    consistency: int
    step_validity_norm: float
    proof_completeness_norm: float
    strategy_compliance_norm: float
    consistency_norm: float
    error_type: str
    judge_confidence: float
    key_strength: str
    key_weakness: str
    critical_failure_step: str
    evidence_source: str
    judge_label_inconsistency: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform-prior Bayesian evidence GRPO training for Big-Math-RL-Verified."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--min_solve_rate", type=float, default=0.2)
    parser.add_argument("--max_solve_rate", type=float, default=0.8)
    parser.add_argument("--train_size", type=int, default=None, help="Number of train rows to use. Defaults to all rows.")
    parser.add_argument("--eval_size", type=int, default=None, help="Number of eval rows to use. Defaults to all rows.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--progress_interval_percent",
        type=int,
        default=10,
        help="Print compact training progress updates every N percent of max_steps.",
    )

    add_bool_arg(parser, "use_lora", True, "Enable LoRA adapters.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    add_bool_arg(parser, "bf16", True, "Use bf16 when supported.")
    add_bool_arg(parser, "gradient_checkpointing", True, "Enable gradient checkpointing.")

    add_bool_arg(parser, "smoke_test_only", False, "Run prompt-format smoke test only.")
    parser.add_argument("--smoke_test_examples", type=int, default=8)
    parser.add_argument("--smoke_test_generations", type=int, default=4)
    add_bool_arg(parser, "run_pretrain_smoke", False, "Run smoke test before GRPO training.")
    parser.add_argument("--min_smoke_success_rate", type=float, default=0.8)
    add_bool_arg(parser, "parser_self_test", False, "Run parser self-test cases and exit.")

    parser.add_argument("--format_bonus", type=float, default=0.0)

    add_bool_arg(
        parser,
        "use_fixed_metadata",
        True,
        "Load the fixed selected_train/selected_eval metadata for a fair comparison.",
    )
    parser.add_argument("--train_metadata_path", default=DEFAULT_TRAIN_METADATA_PATH)
    parser.add_argument("--eval_metadata_path", default=DEFAULT_EVAL_METADATA_PATH)

    parser.add_argument("--evidence_judge_model", default=None)
    parser.add_argument("--judge_max_new_tokens", type=int, default=768)
    parser.add_argument("--evidence_judge_temperature", type=float, default=0.0)
    add_bool_arg(
        parser,
        "disable_llm_judge",
        False,
        "Disable the LLM judge and use heuristic evidence fallback for debugging only.",
    )
    parser.add_argument("--reward_debug_jsonl", default=None)

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            rows.append(dict(row))
    return rows


def normalize_metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    solve_rate = parse_solve_rate(row.get("llama8b_solve_rate"))
    difficulty_bucket = str(row.get("difficulty_bucket", "") or "").strip()
    if not difficulty_bucket:
        difficulty_bucket = difficulty_bucket_from_solve_rate(solve_rate)
    return {
        "problem": str(row.get("problem", "") or ""),
        "answer": str(row.get("answer", "") or ""),
        "source": str(row.get("source", "") or ""),
        "domain": str(row.get("domain", "") or ""),
        "llama8b_solve_rate": solve_rate,
        "difficulty_bucket": difficulty_bucket,
    }


def build_training_row(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    training_row = dict(row)
    training_row["prompt"] = render_prompt(row["problem"], tokenizer)
    return training_row


def print_counter(label: str, counts: Counter[str]) -> None:
    printable = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    print(f"[INFO] {label}: {printable}")


def solve_rate_summary(rows: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    solve_rates = [row["llama8b_solve_rate"] for row in rows if row["llama8b_solve_rate"] is not None]
    return {
        "min": min(solve_rates) if solve_rates else None,
        "max": max(solve_rates) if solve_rates else None,
        "mean": statistics.fmean(solve_rates) if solve_rates else None,
    }


def log_metadata_summary(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"[INFO] {label} count: {len(rows)}")
    for index, row in enumerate(rows[:3], start=1):
        snippet = str(row.get("problem", "") or "").replace("\n", " ").strip()
        print(f"[INFO] {label} example {index}: {snippet[:80]}")
    summary = summarize_selected_train(rows) if rows else {
        "solve_rate_summary": {"min": None, "max": None, "mean": None},
        "source_distribution": {},
        "difficulty_distribution": {},
    }
    sr = summary["solve_rate_summary"]
    print(
        f"[INFO] {label} solve-rate min/max/mean: "
        f"{sr['min']} / {sr['max']} / {sr['mean']}"
    )
    print_counter(f"{label} source distribution", Counter(summary["source_distribution"]))
    print_counter(
        f"{label} difficulty bucket distribution",
        Counter(summary["difficulty_distribution"]),
    )


def load_fixed_metadata_datasets(
    args: argparse.Namespace,
    tokenizer: Any,
    output_dir: Path,
):
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("datasets is required. Install it with `pip install datasets`.") from exc

    train_path = Path(args.train_metadata_path)
    eval_path = Path(args.eval_metadata_path)

    full_train_rows = [normalize_metadata_row(row) for row in load_jsonl_rows(train_path)]
    full_eval_rows = [normalize_metadata_row(row) for row in load_jsonl_rows(eval_path)]
    if not full_train_rows:
        raise RuntimeError(f"Fixed train metadata is empty: {train_path}")
    train_size = len(full_train_rows) if args.train_size is None else args.train_size
    eval_size = len(full_eval_rows) if args.eval_size is None else args.eval_size
    if train_size < 0 or eval_size < 0:
        raise SystemExit("--train_size and --eval_size must be non-negative.")
    if eval_size > 0 and not full_eval_rows:
        raise RuntimeError(f"Fixed eval metadata is empty: {eval_path}")

    if train_size > len(full_train_rows):
        print(
            f"[WARN] Requested train_size={train_size} exceeds fixed metadata size={len(full_train_rows)}. "
            "Clipping to available rows."
        )
    if eval_size > len(full_eval_rows):
        print(
            f"[WARN] Requested eval_size={eval_size} exceeds fixed metadata size={len(full_eval_rows)}. "
            "Clipping to available rows."
        )

    train_rows = full_train_rows[: min(train_size, len(full_train_rows))]
    eval_rows = full_eval_rows[: min(eval_size, len(full_eval_rows))]
    args.train_size = len(train_rows)
    args.eval_size = len(eval_rows)

    write_jsonl(output_dir / "selected_train_metadata.jsonl", train_rows)
    write_jsonl(output_dir / "selected_eval_metadata.jsonl", eval_rows)

    log_metadata_summary("selected fixed train metadata", train_rows)
    log_metadata_summary("selected fixed eval metadata", eval_rows)

    train_dataset = Dataset.from_list([build_training_row(row, tokenizer) for row in train_rows])
    eval_dataset = Dataset.from_list([build_training_row(row, tokenizer) for row in eval_rows]) if eval_rows else None
    smoke_rows = train_rows[: min(len(train_rows), max(args.smoke_test_examples, 1))]

    train_summary = summarize_selected_train(train_rows)
    eval_summary = summarize_selected_train(eval_rows) if eval_rows else {
        "solve_rate_summary": {"min": None, "max": None, "mean": None},
        "source_distribution": {},
        "difficulty_distribution": {},
    }

    dataset_stats = {
        "dataset_mode": "fixed_metadata",
        "dataset_size_before_filtering": len(full_train_rows) + len(full_eval_rows),
        "dataset_size_after_problem_answer_filtering": len(full_train_rows) + len(full_eval_rows),
        "dataset_size_after_solve_rate_filtering": len(full_train_rows) + len(full_eval_rows),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "selected_train_solve_rate_summary": train_summary["solve_rate_summary"],
        "selected_train_source_distribution": train_summary["source_distribution"],
        "selected_train_difficulty_distribution": train_summary["difficulty_distribution"],
        "selected_eval_solve_rate_summary": eval_summary["solve_rate_summary"],
        "selected_eval_source_distribution": eval_summary["source_distribution"],
        "selected_eval_difficulty_distribution": eval_summary["difficulty_distribution"],
        "fixed_train_metadata_full_rows": len(full_train_rows),
        "fixed_eval_metadata_full_rows": len(full_eval_rows),
    }
    metadata_context = {
        "used_fixed_metadata": True,
        "fixed_train_metadata_path": str(train_path),
        "fixed_eval_metadata_path": str(eval_path),
        "fixed_train_metadata_sha256": sha256_file(train_path),
        "fixed_eval_metadata_sha256": sha256_file(eval_path),
    }
    return train_dataset, eval_dataset, smoke_rows, dataset_stats, metadata_context


def load_training_data(
    args: argparse.Namespace,
    tokenizer: Any,
    output_dir: Path,
):
    train_path = Path(args.train_metadata_path)
    eval_path = Path(args.eval_metadata_path)
    if args.use_fixed_metadata and train_path.exists() and eval_path.exists():
        return load_fixed_metadata_datasets(args, tokenizer, output_dir)

    if args.use_fixed_metadata:
        print(
            "[WARN] Fixed metadata not found. Falling back to fresh Big-Math sampling. "
            "This is not a perfectly fair comparison."
        )
    train_dataset, eval_dataset, smoke_rows, dataset_stats = load_dataset_rows(args, tokenizer, output_dir)
    metadata_context = {
        "used_fixed_metadata": False,
        "fixed_train_metadata_path": str(train_path),
        "fixed_eval_metadata_path": str(eval_path),
        "fixed_train_metadata_sha256": sha256_file(train_path) if train_path.exists() else None,
        "fixed_eval_metadata_sha256": sha256_file(eval_path) if eval_path.exists() else None,
    }
    return train_dataset, eval_dataset, smoke_rows, dataset_stats, metadata_context


def render_chat_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    rendered_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        content = str(message.get("content", ""))
        rendered_parts.append(f"{role}:\n{content}")
    rendered_parts.append("Assistant:\n")
    return "\n\n".join(rendered_parts)


def repair_json_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
    return cleaned


def safe_json_parse(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(repair_json_text(text))
    except json.JSONDecodeError:
        return None


def diagnose_json_output_failure(text: str) -> list[str]:
    if not text or not str(text).strip():
        return ["empty_output"]

    stripped = str(text).strip()
    reasons: list[str] = []
    if "```" in stripped:
        reasons.append("contains_code_block")
    if "{" not in stripped:
        reasons.append("missing_open_brace")
    if "}" not in stripped:
        reasons.append("missing_close_brace")
    if stripped and not stripped.startswith("{"):
        reasons.append("prefix_text_before_json")
    if stripped and not stripped.endswith("}"):
        reasons.append("suffix_text_after_json")
    if any(char in stripped for char in ("“", "”", "’")):
        reasons.append("contains_smart_quotes")
    if "'" in stripped and '"' not in stripped:
        reasons.append("likely_single_quotes_only")
    if "\\" in stripped:
        reasons.append("contains_backslashes")
    return reasons or ["json_decode_error"]


def clamp_score_0_to_4(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = 0
    return max(0, min(4, numeric))


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def conservative_evidence_fallback(source: str, error_type: str = "format_error") -> EvidenceAssessment:
    return EvidenceAssessment(
        step_validity=0,
        proof_completeness=0,
        strategy_compliance=0,
        consistency=0,
        step_validity_norm=0.0,
        proof_completeness_norm=0.0,
        strategy_compliance_norm=0.0,
        consistency_norm=0.0,
        error_type=error_type,
        judge_confidence=0.0,
        key_strength="",
        key_weakness="",
        critical_failure_step="",
        evidence_source=source,
        judge_label_inconsistency=False,
    )


def heuristic_evidence_assessment(answer_correctness: float) -> EvidenceAssessment:
    if answer_correctness == 1.0:
        return EvidenceAssessment(
            step_validity=2,
            proof_completeness=2,
            strategy_compliance=2,
            consistency=2,
            step_validity_norm=0.5,
            proof_completeness_norm=0.5,
            strategy_compliance_norm=0.5,
            consistency_norm=0.5,
            error_type="correct_weak_proof",
            judge_confidence=0.5,
            key_strength="Deterministically correct final answer.",
            key_weakness="Heuristic mode does not inspect reasoning semantics.",
            critical_failure_step="",
            evidence_source="heuristic_debug",
            judge_label_inconsistency=False,
        )
    return EvidenceAssessment(
        step_validity=0,
        proof_completeness=0,
        strategy_compliance=0,
        consistency=0,
        step_validity_norm=0.0,
        proof_completeness_norm=0.0,
        strategy_compliance_norm=0.0,
        consistency_norm=0.0,
        error_type="no_meaningful_solution",
        judge_confidence=0.5,
        key_strength="",
        key_weakness="Deterministically incorrect final answer.",
        critical_failure_step="",
        evidence_source="heuristic_debug",
        judge_label_inconsistency=False,
    )


def parse_evidence_assessment(parsed: Any) -> tuple[Optional[EvidenceAssessment], list[str]]:
    if not isinstance(parsed, dict):
        return None, ["parsed_json_not_object"]

    required_keys = (
        "step_validity",
        "proof_completeness",
        "strategy_compliance",
        "consistency",
        "error_type",
    )
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        return None, [f"missing_key:{key}" for key in missing]

    error_type = str(parsed.get("error_type", "")).strip()
    if error_type not in ALLOWED_ERROR_TYPES:
        return None, [f"invalid_error_type:{error_type or 'empty'}"]

    step_validity = clamp_score_0_to_4(parsed.get("step_validity"))
    proof_completeness = clamp_score_0_to_4(parsed.get("proof_completeness"))
    strategy_compliance = clamp_score_0_to_4(parsed.get("strategy_compliance"))
    consistency = clamp_score_0_to_4(parsed.get("consistency"))

    return (
        EvidenceAssessment(
            step_validity=step_validity,
            proof_completeness=proof_completeness,
            strategy_compliance=strategy_compliance,
            consistency=consistency,
            step_validity_norm=step_validity / 4.0,
            proof_completeness_norm=proof_completeness / 4.0,
            strategy_compliance_norm=strategy_compliance / 4.0,
            consistency_norm=consistency / 4.0,
            error_type=error_type,
            judge_confidence=clamp01(parsed.get("judge_confidence", 0.0)),
            key_strength=str(parsed.get("key_strength", "") or ""),
            key_weakness=str(parsed.get("key_weakness", "") or ""),
            critical_failure_step=str(parsed.get("critical_failure_step", "") or ""),
            evidence_source="llm_judge",
            judge_label_inconsistency=False,
        ),
        [],
    )


def enforce_evidence_label_consistency(
    evidence: EvidenceAssessment,
    *,
    answer_correctness: float,
    problem_text: str,
) -> EvidenceAssessment:
    incorrect_disallowed = {"correct_complete", "correct_weak_proof", "lucky_correct"}
    correct_disallowed = {
        "finalization_error",
        "valid_but_incomplete",
        "arithmetic_error",
        "algebraic_error",
        "invalid_assumption",
        "wrong_direction",
        "no_meaningful_solution",
    }

    if answer_correctness == 0.0 and evidence.error_type in incorrect_disallowed:
        remapped = "finalization_error" if evidence.step_validity >= 2 else "wrong_direction"
        print(
            "[WARN] Evidence label inconsistency: "
            f"answer_correctness=0 but error_type={evidence.error_type!r} for problem="
            f"{problem_text[:80]!r}. Remapping to {remapped}."
        )
        return replace(evidence, error_type=remapped, judge_label_inconsistency=True)

    if answer_correctness == 1.0 and evidence.error_type in correct_disallowed:
        print(
            "[WARN] Evidence label inconsistency: "
            f"answer_correctness=1 but error_type={evidence.error_type!r} for problem="
            f"{problem_text[:80]!r}. Remapping to 'correct_weak_proof'."
        )
        return replace(evidence, error_type="correct_weak_proof", judge_label_inconsistency=True)

    return evidence


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


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


class BayesianRewardScorer:
    def __init__(self, args: argparse.Namespace, output_dir: Path) -> None:
        self.args = args
        self.output_dir = output_dir
        self.debug_path = Path(args.reward_debug_jsonl) if args.reward_debug_jsonl else output_dir / DEFAULT_REWARD_DEBUG_JSONL
        self.judge_model_name = args.evidence_judge_model or args.model_name
        self.judge_model: Any = None
        self.judge_tokenizer: Any = None
        self.reward_call_index = 0

    def _load_judge(self) -> None:
        if self.args.disable_llm_judge:
            return
        if self.judge_model is not None and self.judge_tokenizer is not None:
            return

        torch = import_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "transformers is required for the evidence judge. Install it with `pip install transformers`."
            ) from exc

        print(f"[INFO] loading evidence judge model: {self.judge_model_name}")
        try:
            self.judge_tokenizer = AutoTokenizer.from_pretrained(
                self.judge_model_name,
                trust_remote_code=True,
            )
            if self.judge_tokenizer.pad_token is None:
                self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token
            self.judge_tokenizer.padding_side = "left"

            model_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.bfloat16 if self.args.bf16 else torch.float16

            self.judge_model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_name,
                **model_kwargs,
            )
            if not torch.cuda.is_available():
                self.judge_model.to("cpu")
            self.judge_model.eval()
        except Exception as exc:  # pragma: no cover - depends on env
            message = str(exc).lower()
            if "out of memory" in message or "cuda out of memory" in message:
                raise RuntimeError(
                    "Failed to load the evidence judge model due to OOM. "
                    "Try --disable_llm_judge for debugging, use a smaller judge model, "
                    "or reduce batch/generation settings."
                ) from exc
            raise

    def _call_evidence_judge(
        self,
        *,
        problem: str,
        strategy: str,
        reasoning: str,
        final_answer: str,
        answer_correctness: float,
    ) -> tuple[EvidenceAssessment, dict[str, Any]]:
        if self.args.disable_llm_judge:
            evidence = heuristic_evidence_assessment(answer_correctness)
            return evidence, {
                "raw_judge_output": None,
                "repaired_json_candidate": None,
                "parsed_judge_json": None,
                "parse_failure_reasons": [],
                "evidence_judge_fallback_used": True,
            }

        self._load_judge()
        assert self.judge_model is not None
        assert self.judge_tokenizer is not None

        prompt = build_evidence_judge_prompt(
            problem_text=problem,
            strategy=strategy,
            reasoning=reasoning,
            final_answer=final_answer,
            answer_correctness=answer_correctness,
        )
        messages = [
            {"role": "system", "content": JUDGE_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = render_chat_prompt(messages, self.judge_tokenizer)

        last_raw_output = None
        last_repaired = None
        last_parsed_json = None
        aggregated_reasons: list[str] = []

        for attempt in range(1, 3):
            outputs = generate_smoke_outputs(
                model=self.judge_model,
                tokenizer=self.judge_tokenizer,
                prompt=rendered_prompt,
                num_generations=1,
                max_new_tokens=self.args.judge_max_new_tokens,
                temperature=self.args.evidence_judge_temperature,
                top_p=self.args.top_p,
            )
            raw_output = outputs[0] if outputs else ""
            repaired = repair_json_text(raw_output)
            parsed_json = safe_json_parse(raw_output)

            last_raw_output = raw_output
            last_repaired = repaired
            last_parsed_json = parsed_json

            if parsed_json is None:
                reasons = diagnose_json_output_failure(raw_output)
                aggregated_reasons.extend(reasons)
                print(f"[WARN] evidence judge JSON parse failed on attempt {attempt}: {reasons}")
                continue

            evidence, reasons = parse_evidence_assessment(parsed_json)
            if evidence is None:
                aggregated_reasons.extend(reasons)
                print(f"[WARN] evidence judge schema validation failed on attempt {attempt}: {reasons}")
                continue

            evidence = enforce_evidence_label_consistency(
                evidence,
                answer_correctness=answer_correctness,
                problem_text=problem,
            )
            return evidence, {
                "raw_judge_output": raw_output,
                "repaired_json_candidate": repaired,
                "parsed_judge_json": parsed_json,
                "parse_failure_reasons": aggregated_reasons,
                "evidence_judge_fallback_used": False,
            }

        fallback = conservative_evidence_fallback("llm_judge_fallback", error_type="format_error")
        return fallback, {
            "raw_judge_output": last_raw_output,
            "repaired_json_candidate": last_repaired,
            "parsed_judge_json": last_parsed_json,
            "parse_failure_reasons": sorted(set(aggregated_reasons)) or ["llm_judge_failed_after_retry"],
            "evidence_judge_fallback_used": True,
        }

    def _compute_reward(
        self,
        *,
        answer_correctness: float,
        evidence: EvidenceAssessment,
    ) -> tuple[float, dict[str, float]]:
        components = {
            "answer_correctness": LIKELIHOOD_WEIGHTS["answer_correctness"] * answer_correctness,
            "step_validity": LIKELIHOOD_WEIGHTS["step_validity"] * evidence.step_validity_norm,
            "proof_completeness": LIKELIHOOD_WEIGHTS["proof_completeness"] * evidence.proof_completeness_norm,
            "strategy_compliance": LIKELIHOOD_WEIGHTS["strategy_compliance"] * evidence.strategy_compliance_norm,
            "consistency": LIKELIHOOD_WEIGHTS["consistency"] * evidence.consistency_norm,
        }
        reward = sum(components.values())
        return float(reward), components

    def __call__(self, completions, answer=None, problem=None, **kwargs):
        self.reward_call_index += 1

        gold_answers = answer if answer is not None else kwargs.get("answer")
        if gold_answers is None:
            gold_answers = kwargs.get("solution")
        problem_texts = problem if problem is not None else kwargs.get("problem")

        completion_texts = [extract_text_from_completion(completion) for completion in completions]
        aligned_golds = align_gold_answers(gold_answers, len(completion_texts))
        aligned_problems = align_values(problem_texts, len(completion_texts))

        rewards: list[float] = []
        for index, (completion_text, gold_answer, problem_text) in enumerate(
            zip(completion_texts, aligned_golds, aligned_problems)
        ):
            parsed_sections = parse_completion_sections(completion_text)
            parsed_strategy = extract_section(completion_text, "Strategy", ["Reasoning", "Final Answer"]).strip()
            parsed_reasoning = extract_section(completion_text, "Reasoning", ["Final Answer"]).strip()
            parsed_final_answer = parsed_sections["parsed_final_answer"]

            verification = verify_answer(parsed_final_answer, gold_answer, problem_text=problem_text)
            answer_correctness = 1.0 if verification["correct"] else 0.0
            format_valid = bool(parsed_sections["exact_format_success"])
            suspicious_final_answer = bool(parsed_sections["suspicious_final_answer"])

            if not format_valid or suspicious_final_answer or not parsed_final_answer:
                evidence = conservative_evidence_fallback("format_guard", error_type="format_error")
                evidence_debug = {
                    "raw_judge_output": None,
                    "repaired_json_candidate": None,
                    "parsed_judge_json": None,
                    "parse_failure_reasons": ["format_invalid_or_empty_final_answer"],
                    "evidence_judge_fallback_used": True,
                }
            else:
                evidence, evidence_debug = self._call_evidence_judge(
                    problem=problem_text,
                    strategy=parsed_strategy,
                    reasoning=parsed_reasoning,
                    final_answer=parsed_final_answer,
                    answer_correctness=answer_correctness,
                )

            reward, reward_components = self._compute_reward(
                answer_correctness=answer_correctness,
                evidence=evidence,
            )
            rewards.append(reward)

            debug_row = {
                "global_reward_call_index": self.reward_call_index,
                "completion_index_within_call": index,
                "problem": problem_text,
                "gold_answer": gold_answer,
                "raw_completion": completion_text,
                "parsed_strategy": parsed_strategy,
                "parsed_reasoning": parsed_reasoning,
                "parsed_final_answer": parsed_final_answer,
                "normalized_predicted_answer": verification["normalized_predicted_answer"],
                "normalized_gold_answer": verification["normalized_gold_answer"],
                "verification_method": verification["verification_method"],
                "answer_correctness": answer_correctness,
                "step_validity": evidence.step_validity,
                "proof_completeness": evidence.proof_completeness,
                "strategy_compliance": evidence.strategy_compliance,
                "consistency": evidence.consistency,
                "step_validity_norm": evidence.step_validity_norm,
                "proof_completeness_norm": evidence.proof_completeness_norm,
                "strategy_compliance_norm": evidence.strategy_compliance_norm,
                "consistency_norm": evidence.consistency_norm,
                "error_type": evidence.error_type,
                "judge_confidence": evidence.judge_confidence,
                "key_strength": evidence.key_strength,
                "key_weakness": evidence.key_weakness,
                "critical_failure_step": evidence.critical_failure_step,
                "judge_label_inconsistency": evidence.judge_label_inconsistency,
                "evidence_judge_fallback_used": evidence_debug["evidence_judge_fallback_used"],
                "evidence_source": evidence.evidence_source,
                "format_valid": format_valid,
                "strategy_section_present": parsed_sections["strategy_section_present"],
                "reasoning_section_present": parsed_sections["reasoning_section_present"],
                "final_answer_section_present": parsed_sections["final_answer_section_present"],
                "suspicious_final_answer": suspicious_final_answer,
                "bayesian_reward": reward,
                "reward_components": reward_components,
                "raw_judge_output": evidence_debug["raw_judge_output"],
                "repaired_json_candidate": evidence_debug["repaired_json_candidate"],
                "parsed_judge_json": evidence_debug["parsed_judge_json"],
                "parse_failure_reasons": evidence_debug["parse_failure_reasons"],
            }
            append_jsonl_row(self.debug_path, debug_row)

        return rewards


def build_bayesian_training_config_payload(
    *,
    args: argparse.Namespace,
    dataset_stats: dict[str, Any],
    dropped_grpo_config_kwargs: list[str],
    dropped_trainer_kwargs: list[str],
    metadata_context: dict[str, Any],
) -> dict[str, Any]:
    payload = build_training_config_payload(
        args=args,
        dataset_stats=dataset_stats,
        dropped_grpo_config_kwargs=dropped_grpo_config_kwargs,
        dropped_trainer_kwargs=dropped_trainer_kwargs,
    )
    payload.update(
        {
            "reward_type": "uniform_prior_bayesian_evidence",
            "prior_mode": "uniform",
            "prior_lambda": 0.0,
            "likelihood_weights": dict(LIKELIHOOD_WEIGHTS),
            "evidence_judge_model": args.evidence_judge_model,
            "judge_max_new_tokens": args.judge_max_new_tokens,
            "evidence_judge_temperature": args.evidence_judge_temperature,
            "disable_llm_judge": args.disable_llm_judge,
            "reward_debug_jsonl": str(args.reward_debug_jsonl),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "format_bonus_ignored_in_reward": True,
            "progress_interval_percent": args.progress_interval_percent,
            "use_fixed_metadata": metadata_context["used_fixed_metadata"],
            "use_fixed_metadata_requested": args.use_fixed_metadata,
            "fixed_train_metadata_path": metadata_context["fixed_train_metadata_path"],
            "fixed_eval_metadata_path": metadata_context["fixed_eval_metadata_path"],
            "fixed_train_metadata_sha256": metadata_context["fixed_train_metadata_sha256"],
            "fixed_eval_metadata_sha256": metadata_context["fixed_eval_metadata_sha256"],
        }
    )
    return payload


def attach_percent_progress_callback(trainer: Any, progress_interval_percent: int, expected_total_steps: int) -> None:
    interval = max(1, min(100, int(progress_interval_percent)))
    if not hasattr(trainer, "add_callback"):
        print("[WARN] Trainer does not support add_callback; progress callback is disabled.")
        return

    try:
        from transformers import TrainerCallback
    except ImportError:
        print("[WARN] transformers TrainerCallback is unavailable; progress callback is disabled.")
        return

    class PercentProgressCallback(TrainerCallback):
        def __init__(self, interval_percent: int, fallback_total_steps: int) -> None:
            self.interval_percent = interval_percent
            self.fallback_total_steps = fallback_total_steps
            self.milestones: list[tuple[int, int]] = []
            self.next_milestone_index = 0
            self.total_steps = 0
            self.finished_reported = False

        def _resolve_total_steps(self, state: Any) -> int:
            state_total = int(getattr(state, "max_steps", 0) or 0)
            return state_total if state_total > 0 else max(0, int(self.fallback_total_steps))

        def _build_milestones(self, total_steps: int) -> list[tuple[int, int]]:
            step_to_percent: dict[int, int] = {}
            target_percents = list(range(self.interval_percent, 101, self.interval_percent))
            if 100 not in target_percents:
                target_percents.append(100)
            for percent in target_percents:
                step = max(1, math.ceil(total_steps * percent / 100.0))
                step_to_percent[step] = max(percent, step_to_percent.get(step, 0))
            return sorted(step_to_percent.items())

        def on_train_begin(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True):
                return control
            self.total_steps = self._resolve_total_steps(state)
            if self.total_steps <= 0:
                print("[INFO] training progress callback enabled, but total step count is unknown.", flush=True)
                return control
            self.milestones = self._build_milestones(self.total_steps)
            checkpoint_preview = ", ".join(
                f"{percent}%={step}/{self.total_steps}" for step, percent in self.milestones
            )
            print(f"[INFO] training progress checkpoints: {checkpoint_preview}", flush=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True) or not self.milestones:
                return control
            current_step = int(getattr(state, "global_step", 0) or 0)
            while self.next_milestone_index < len(self.milestones):
                target_step, target_percent = self.milestones[self.next_milestone_index]
                if current_step < target_step:
                    break
                print(
                    f"[PROGRESS] {target_percent}% ({min(current_step, self.total_steps)}/{self.total_steps} steps)",
                    flush=True,
                )
                self.next_milestone_index += 1
                if target_percent >= 100:
                    self.finished_reported = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if (
                getattr(state, "is_world_process_zero", True)
                and self.total_steps > 0
                and not self.finished_reported
            ):
                final_step = int(getattr(state, "global_step", 0) or 0)
                print(
                    f"[PROGRESS] 100% ({min(final_step, self.total_steps)}/{self.total_steps} steps)",
                    flush=True,
                )
            return control

    trainer.add_callback(PercentProgressCallback(interval, expected_total_steps))


def main() -> None:
    args = parse_args()
    args.bf16 = bool(args.bf16 and default_bf16_enabled())
    if args.evidence_judge_model is None:
        args.evidence_judge_model = args.model_name

    if args.parser_self_test:
        success = run_parser_self_test()
        if not success:
            raise SystemExit(1)
        return

    if args.train_size is not None and args.train_size <= 0:
        raise SystemExit("--train_size must be positive.")
    if args.eval_size is not None and args.eval_size < 0:
        raise SystemExit("--eval_size must be non-negative.")
    if args.smoke_test_examples <= 0:
        raise SystemExit("--smoke_test_examples must be positive.")
    if args.smoke_test_generations <= 0:
        raise SystemExit("--smoke_test_generations must be positive.")
    if args.num_generations <= 0:
        raise SystemExit("--num_generations must be positive.")
    if args.min_solve_rate > args.max_solve_rate:
        raise SystemExit("--min_solve_rate must be <= --max_solve_rate.")
    if args.format_bonus != 0.0:
        print(
            "[WARN] --format_bonus is ignored in Bayesian_GRPO.py so that the reward remains "
            "the intended answer-heavy Bayesian evidence likelihood."
        )

    output_dir = ensure_output_dir(args.output_dir)
    if args.reward_debug_jsonl is None:
        args.reward_debug_jsonl = str(output_dir / DEFAULT_REWARD_DEBUG_JSONL)
    Path(args.reward_debug_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reward_debug_jsonl).write_text("", encoding="utf-8")

    set_seed(args.seed)

    tokenizer = load_tokenizer(args.model_name)
    train_dataset, eval_dataset, smoke_rows, dataset_stats, metadata_context = load_training_data(
        args,
        tokenizer,
        output_dir,
    )

    if args.smoke_test_only or args.run_pretrain_smoke:
        smoke_metrics = run_smoke_test(
            args=args,
            tokenizer=tokenizer,
            smoke_rows=smoke_rows,
            output_dir=output_dir,
        )
        maybe_warn_on_smoke_metrics(smoke_metrics, args.min_smoke_success_rate)
        if args.smoke_test_only:
            print(f"[DONE] smoke test artifacts saved to {output_dir}")
            return

    GRPOConfig, GRPOTrainer = build_grpo_training_components(args)
    peft_config = build_peft_config(args)
    training_args, dropped_grpo_config_kwargs = create_grpo_config(args, GRPOConfig)
    bayesian_scorer = BayesianRewardScorer(args=args, output_dir=output_dir)

    def bayesian_evidence_reward(completions, **kwargs):
        return bayesian_scorer(completions, **kwargs)

    bayesian_evidence_reward.__name__ = "bayesian_evidence_reward"

    trainer, dropped_trainer_kwargs = create_grpo_trainer(
        args=args,
        GRPOTrainer=GRPOTrainer,
        training_args=training_args,
        reward_fn=bayesian_evidence_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    attach_percent_progress_callback(
        trainer,
        progress_interval_percent=args.progress_interval_percent,
        expected_total_steps=args.max_steps,
    )

    training_config = build_bayesian_training_config_payload(
        args=args,
        dataset_stats=dataset_stats,
        dropped_grpo_config_kwargs=dropped_grpo_config_kwargs,
        dropped_trainer_kwargs=dropped_trainer_kwargs,
        metadata_context=metadata_context,
    )
    write_json(output_dir / "training_config.json", training_config)
    print("[INFO] final resolved configuration:")
    print(json.dumps(training_config, ensure_ascii=False, indent=2, sort_keys=True))

    if args.resume_from_checkpoint:
        print(f"[INFO] Resuming training from checkpoint: {args.resume_from_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] saved to {output_dir}")


if __name__ == "__main__":
    main()

# Parser self-test:
# CUDA_VISIBLE_DEVICES=3 python3 Bayesian_GRPO.py \
#   --parser_self_test \
#   --model_name Qwen/Qwen2.5-3B-Instruct \
#   --output_dir outputs/parser_self_test_bayesian
#
# Smoke test only:
# CUDA_VISIBLE_DEVICES=3 python3 Bayesian_GRPO.py \
#   --smoke_test_only \
#   --model_name Qwen/Qwen2.5-3B-Instruct \
#   --use_fixed_metadata \
#   --train_metadata_path outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_train_metadata.jsonl \
#   --eval_metadata_path outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_eval_metadata.jsonl \
#   --train_size 16 \
#   --eval_size 4 \
#   --num_generations 4 \
#   --smoke_test_examples 4 \
#   --smoke_test_generations 2 \
#   --min_solve_rate 0.2 \
#   --max_solve_rate 0.8 \
#   --output_dir outputs/smoke_bayesian_qwen3b_bigmath
#
# Fair n=4 Bayesian GRPO run:
# CUDA_VISIBLE_DEVICES=3 nohup python3 Bayesian_GRPO.py \
#   --model_name Qwen/Qwen2.5-3B-Instruct \
#   --evidence_judge_model Qwen/Qwen2.5-3B-Instruct \
#   --use_fixed_metadata \
#   --train_metadata_path outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_train_metadata.jsonl \
#   --eval_metadata_path outputs/grpo_answer_only_qwen3b_bigmath_n4_steps200/selected_eval_metadata.jsonl \
#   --train_size 1000 \
#   --eval_size 100 \
#   --num_generations 4 \
#   --max_steps 200 \
#   --max_completion_length 1024 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 5e-6 \
#   --min_solve_rate 0.2 \
#   --max_solve_rate 0.8 \
#   --run_pretrain_smoke \
#   --min_smoke_success_rate 0.8 \
#   --output_dir outputs/grpo_bayesian_uniform_qwen3b_bigmath_n4_steps200 \
#   > train_bayesian_uniform_qwen3b_n4.log 2>&1 &
