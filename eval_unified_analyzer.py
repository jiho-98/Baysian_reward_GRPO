#!/usr/bin/env python3
"""Offline validation gate for the unified analyzer."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_EVIDENCE_VAL_PATH = "outputs/unified_analyzer_sft_v0/evidence_clean_val_marked.jsonl"
DEFAULT_PRIOR_VAL_PATH = "outputs/unified_analyzer_sft_v0/prior_clean_val_marked.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/unified_analyzer_eval_v0"

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

CORRECT_ERROR_TYPES = {"correct_complete", "correct_weak_proof", "lucky_correct"}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline validation for unified analyzer.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--evidence_val_path", default=DEFAULT_EVIDENCE_VAL_PATH)
    parser.add_argument("--prior_val_path", default=DEFAULT_PRIOR_VAL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_examples_per_task", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("torch is required. Install it with `pip install torch`.") from exc
    return torch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
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


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "transformers is required. Install it with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str, adapter_path: Optional[str], bf16: bool):
    torch = import_torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "transformers is required. Install it with `pip install transformers`."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if (bf16 and torch.cuda.is_available()) else "auto",
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = base_model

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError("peft is required to load LoRA adapters. Install it with `pip install peft`.") from exc
        model = PeftModel.from_pretrained(base_model, adapter_path)

    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model


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

    rendered_parts = []
    for message in messages:
        rendered_parts.append(f"{message['role'].capitalize()}:\n{message['content']}")
    rendered_parts.append("Assistant:\n")
    return "\n\n".join(rendered_parts)


def add_task_prefix(example: dict[str, Any]) -> dict[str, Any]:
    task = str(example["task"])
    prefix = TASK_PREFIXES[task]
    updated = dict(example)
    updated_messages = []
    for message in example["messages"]:
        copied = dict(message)
        if copied.get("role") == "user":
            content = str(copied.get("content", ""))
            if not content.startswith(prefix):
                copied["content"] = prefix + content
        updated_messages.append(copied)
    updated["messages"] = updated_messages
    return updated


def safe_json_parse(text: str) -> Optional[dict[str, Any]]:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    candidates = [cleaned]

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(cleaned[first_obj : last_obj + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def generate_response(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    torch = import_torch()
    prompt = render_chat_prompt(messages, tokenizer)
    batch = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        batch = {key: value.to(model.device) for key, value in batch.items()}
    with torch.no_grad():
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = batch["input_ids"].shape[1]
    new_tokens = generated[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def validate_evidence_schema(parsed: dict[str, Any], answer_correctness: float) -> tuple[bool, dict[str, Any]]:
    required_keys = (
        "step_validity",
        "proof_completeness",
        "strategy_compliance",
        "consistency",
        "error_type",
        "key_strength",
        "key_weakness",
        "critical_failure_step",
        "judge_confidence",
    )
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        return False, {"missing_keys": missing}

    try:
        scores = {
            "step_validity": int(parsed["step_validity"]),
            "proof_completeness": int(parsed["proof_completeness"]),
            "strategy_compliance": int(parsed["strategy_compliance"]),
            "consistency": int(parsed["consistency"]),
        }
        judge_confidence = float(parsed["judge_confidence"])
    except (TypeError, ValueError):
        return False, {"invalid_numeric_field": True}

    if any(value < 0 or value > 4 for value in scores.values()):
        return False, {"score_out_of_range": scores}
    if judge_confidence < 0.0 or judge_confidence > 1.0:
        return False, {"judge_confidence_out_of_range": judge_confidence}

    error_type = str(parsed["error_type"])
    if error_type not in ALLOWED_ERROR_TYPES:
        return False, {"invalid_error_type": error_type}

    consistency_ok = (
        (answer_correctness == 1.0 and error_type in CORRECT_ERROR_TYPES)
        or (answer_correctness == 0.0 and error_type not in CORRECT_ERROR_TYPES)
    )
    return True, {
        "scores": scores,
        "error_type": error_type,
        "judge_confidence": judge_confidence,
        "consistency_ok": consistency_ok,
    }


def validate_prior_schema(parsed: dict[str, Any], num_rollouts: int) -> tuple[bool, dict[str, Any]]:
    priors = parsed.get("priors")
    if not isinstance(priors, list):
        return False, {"priors_not_list": True}

    by_rollout_id: dict[int, dict[str, Any]] = {}
    duplicate_ids: list[int] = []
    invalid_scores = 0
    for row in priors:
        if not isinstance(row, dict):
            continue
        try:
            rollout_id = int(row.get("rollout_id"))
            suitability = int(row.get("suitability"))
        except (TypeError, ValueError):
            invalid_scores += 1
            continue
        if suitability < 0 or suitability > 4:
            invalid_scores += 1
            continue
        if rollout_id in by_rollout_id:
            duplicate_ids.append(rollout_id)
        by_rollout_id[rollout_id] = row

    missing_ids = [rollout_id for rollout_id in range(num_rollouts) if rollout_id not in by_rollout_id]
    coverage_ok = not missing_ids and not duplicate_ids and invalid_scores == 0 and len(priors) == num_rollouts
    return coverage_ok, {
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "invalid_scores": invalid_scores,
        "rows_returned": len(priors),
        "by_rollout_id": by_rollout_id,
    }


def evaluate_evidence_task(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    parse_success = 0
    schema_success = 0
    consistency_success = 0
    score_maes: list[float] = []
    error_type_match = 0

    for row in rows:
        example = add_task_prefix(row)
        prompt_messages = example["messages"][:-1]
        raw_output = generate_response(model, tokenizer, prompt_messages, max_new_tokens)
        parsed = safe_json_parse(raw_output)

        result = {
            "example_id": row["example_id"],
            "task": "evidence_judge",
            "raw_output": raw_output,
            "parsed_json": parsed,
        }

        if parsed is not None:
            parse_success += 1
            valid, debug = validate_evidence_schema(parsed, float(row["answer_correctness"]))
            result["schema_debug"] = debug
            if valid:
                schema_success += 1
                if bool(debug["consistency_ok"]):
                    consistency_success += 1
                teacher = row["teacher_target"]
                fields = ("step_validity", "proof_completeness", "strategy_compliance", "consistency")
                score_maes.append(
                    statistics.fmean(abs(float(parsed[field]) - float(teacher[field])) for field in fields)
                )
                if str(parsed["error_type"]) == str(teacher["error_type"]):
                    error_type_match += 1
        results.append(result)

    write_jsonl(output_dir / "evidence_eval_results.jsonl", results)
    total = len(rows)
    return {
        "num_examples": total,
        "parse_rate": parse_success / total if total else 0.0,
        "schema_valid_rate": schema_success / total if total else 0.0,
        "correctness_error_type_consistency_rate": consistency_success / total if total else 0.0,
        "error_type_accuracy": error_type_match / total if total else 0.0,
        "score_mae": statistics.fmean(score_maes) if score_maes else None,
    }


def evaluate_prior_task(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    output_dir: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    parse_success = 0
    schema_success = 0
    exact_coverage = 0
    suitability_maes: list[float] = []
    all_rollouts_exact = 0

    for row in rows:
        example = add_task_prefix(row)
        prompt_messages = example["messages"][:-1]
        raw_output = generate_response(model, tokenizer, prompt_messages, max_new_tokens)
        parsed = safe_json_parse(raw_output)

        result = {
            "example_id": row["example_id"],
            "task": "prior_judge",
            "raw_output": raw_output,
            "parsed_json": parsed,
        }
        if parsed is not None:
            parse_success += 1
            valid, debug = validate_prior_schema(parsed, int(row["num_rollouts"]))
            result["schema_debug"] = {
                key: value for key, value in debug.items() if key != "by_rollout_id"
            }
            if valid:
                schema_success += 1
                exact_coverage += 1
                teacher_by_rollout = {
                    int(item["rollout_id"]): int(item["suitability"])
                    for item in row["teacher_target"]["priors"]
                }
                predicted_by_rollout = {
                    rollout_id: int(item["suitability"])
                    for rollout_id, item in debug["by_rollout_id"].items()
                }
                maes = [
                    abs(predicted_by_rollout[rollout_id] - teacher_by_rollout[rollout_id])
                    for rollout_id in sorted(teacher_by_rollout)
                ]
                suitability_maes.append(statistics.fmean(maes))
                if all(predicted_by_rollout[rid] == teacher_by_rollout[rid] for rid in teacher_by_rollout):
                    all_rollouts_exact += 1
        results.append(result)

    write_jsonl(output_dir / "prior_eval_results.jsonl", results)
    total = len(rows)
    return {
        "num_examples": total,
        "parse_rate": parse_success / total if total else 0.0,
        "schema_valid_rate": schema_success / total if total else 0.0,
        "rollout_coverage_rate": exact_coverage / total if total else 0.0,
        "suitability_mae": statistics.fmean(suitability_maes) if suitability_maes else None,
        "all_rollouts_exact_rate": all_rollouts_exact / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows = load_jsonl(Path(args.evidence_val_path))
    prior_rows = load_jsonl(Path(args.prior_val_path))
    if args.max_examples_per_task > 0:
        evidence_rows = evidence_rows[: args.max_examples_per_task]
        prior_rows = prior_rows[: args.max_examples_per_task]

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, args.adapter_path, args.bf16)

    evidence_summary = evaluate_evidence_task(
        model=model,
        tokenizer=tokenizer,
        rows=evidence_rows,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
    )
    prior_summary = evaluate_prior_task(
        model=model,
        tokenizer=tokenizer,
        rows=prior_rows,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
    )

    summary = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "evidence_val_path": args.evidence_val_path,
        "prior_val_path": args.prior_val_path,
        "max_examples_per_task": args.max_examples_per_task,
        "evidence": evidence_summary,
        "prior": prior_summary,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[INFO] wrote unified analyzer eval summary to {output_dir / 'summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
