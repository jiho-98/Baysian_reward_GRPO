#!/usr/bin/env python3
"""Build a balanced unified multi-task SFT dataset for the analyzer."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = "outputs/analyzer_training_data_v1"
DEFAULT_OUTPUT_DIR = "outputs/unified_analyzer_sft_v0"

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

EVIDENCE_ERROR_TYPES = (
    "correct_complete, correct_weak_proof, lucky_correct, "
    "finalization_error, valid_but_incomplete, arithmetic_error, algebraic_error, "
    "invalid_assumption, strategy_mismatch, wrong_direction, format_error, no_meaningful_solution"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced unified analyzer multi-task SFT dataset."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--bootstrap_input_dir",
        default=None,
        help="Optional bootstrap data directory to merge with the anchor clean data.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor_evidence_train_file", default="evidence_clean_train.jsonl")
    parser.add_argument("--anchor_evidence_val_file", default="evidence_clean_val.jsonl")
    parser.add_argument("--anchor_prior_train_file", default="prior_clean_train.jsonl")
    parser.add_argument("--anchor_prior_val_file", default="prior_clean_val.jsonl")
    parser.add_argument("--bootstrap_evidence_train_file", default="evidence_bootstrap_v1_train.jsonl")
    parser.add_argument("--bootstrap_evidence_val_file", default="evidence_bootstrap_v1_val.jsonl")
    parser.add_argument("--bootstrap_prior_train_file", default="prior_bootstrap_v1_train.jsonl")
    parser.add_argument("--bootstrap_prior_val_file", default="prior_bootstrap_v1_val.jsonl")
    parser.add_argument(
        "--include_bootstrap_val",
        action="store_true",
        help="Include bootstrap validation rows in unified_val. By default only anchor clean val is used.",
    )
    parser.add_argument("--evidence_weight", type=float, default=0.6)
    parser.add_argument("--prior_weight", type=float, default=0.4)
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Optional cap on the unified train set size after upsampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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
            + "Scoring rules: the four score fields are integers 0-4. "
            + "judge_confidence is a float in [0,1].\n"
            + f"Allowed error_type values: {EVIDENCE_ERROR_TYPES}.\n"
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
            + "Suitability is an integer from 0 to 4.\n"
            + "You must return every rollout_id exactly once.\n\n"
            + f"Problem:\n{enriched['problem']}\n\n"
            + f"Candidate strategies:\n{candidates_block}\n"
        )
    else:
        raise ValueError(f"Unknown task: {task}")

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
    enriched = deepcopy(example)
    enriched["mixture_source_name"] = source_name
    return enriched


def sample_with_replacement(
    rows: list[dict[str, Any]],
    num_required: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
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


def count_tasks(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["task"]) for row in rows)
    return dict(sorted(counter.items()))


def count_mixture_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("mixture_source_name", "unknown")) for row in rows)
    return dict(sorted(counter.items()))


def count_task_source_pairs(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(
        f"{row.get('mixture_source_name', 'unknown')}::{row['task']}"
        for row in rows
    )
    return dict(sorted(counter.items()))


def load_prefixed_rows(path: Path, source_name: str) -> list[dict[str, Any]]:
    return [
        with_mixture_source(with_task_prefix(row), source_name)
        for row in load_jsonl(path)
    ]


def main() -> None:
    args = parse_args()
    if args.evidence_weight <= 0 or args.prior_weight <= 0:
        raise ValueError("Both --evidence_weight and --prior_weight must be positive.")

    rng = random.Random(args.seed)
    input_dir = Path(args.input_dir)
    bootstrap_input_dir = Path(args.bootstrap_input_dir) if args.bootstrap_input_dir else None
    output_dir = Path(args.output_dir)

    anchor_evidence_train = load_prefixed_rows(
        input_dir / args.anchor_evidence_train_file,
        source_name="anchor_clean",
    )
    anchor_evidence_val = load_prefixed_rows(
        input_dir / args.anchor_evidence_val_file,
        source_name="anchor_clean",
    )
    anchor_prior_train = load_prefixed_rows(
        input_dir / args.anchor_prior_train_file,
        source_name="anchor_clean",
    )
    anchor_prior_val = load_prefixed_rows(
        input_dir / args.anchor_prior_val_file,
        source_name="anchor_clean",
    )

    bootstrap_evidence_train: list[dict[str, Any]] = []
    bootstrap_evidence_val: list[dict[str, Any]] = []
    bootstrap_prior_train: list[dict[str, Any]] = []
    bootstrap_prior_val: list[dict[str, Any]] = []
    if bootstrap_input_dir is not None:
        bootstrap_evidence_train = load_prefixed_rows(
            bootstrap_input_dir / args.bootstrap_evidence_train_file,
            source_name="bootstrap_v1",
        )
        bootstrap_evidence_val = load_prefixed_rows(
            bootstrap_input_dir / args.bootstrap_evidence_val_file,
            source_name="bootstrap_v1",
        )
        bootstrap_prior_train = load_prefixed_rows(
            bootstrap_input_dir / args.bootstrap_prior_train_file,
            source_name="bootstrap_v1",
        )
        bootstrap_prior_val = load_prefixed_rows(
            bootstrap_input_dir / args.bootstrap_prior_val_file,
            source_name="bootstrap_v1",
        )

    evidence_train = list(anchor_evidence_train) + list(bootstrap_evidence_train)
    prior_train = list(anchor_prior_train) + list(bootstrap_prior_train)
    evidence_val = list(anchor_evidence_val)
    prior_val = list(anchor_prior_val)
    if args.include_bootstrap_val:
        evidence_val.extend(bootstrap_evidence_val)
        prior_val.extend(bootstrap_prior_val)

    total_weight = args.evidence_weight + args.prior_weight
    evidence_ratio = args.evidence_weight / total_weight
    prior_ratio = args.prior_weight / total_weight

    if args.max_train_examples is not None:
        train_size = int(args.max_train_examples)
    else:
        train_size = math.ceil(
            max(
                len(evidence_train) / evidence_ratio,
                len(prior_train) / prior_ratio,
            )
        )

    num_evidence = round(train_size * evidence_ratio)
    num_prior = train_size - num_evidence

    unified_train = sample_with_replacement(evidence_train, num_evidence, rng) + sample_with_replacement(
        prior_train, num_prior, rng
    )
    rng.shuffle(unified_train)
    unified_train = annotate_sampling(unified_train, split_name="train")

    unified_val = annotate_sampling(
        list(evidence_val) + list(prior_val),
        split_name="val",
    )

    write_jsonl(output_dir / "unified_train.jsonl", unified_train)
    write_jsonl(output_dir / "unified_val.jsonl", unified_val)
    write_jsonl(output_dir / "evidence_clean_val_marked.jsonl", anchor_evidence_val)
    write_jsonl(output_dir / "prior_clean_val_marked.jsonl", anchor_prior_val)
    write_jsonl(output_dir / "anchor_evidence_val_marked.jsonl", anchor_evidence_val)
    write_jsonl(output_dir / "anchor_prior_val_marked.jsonl", anchor_prior_val)
    if bootstrap_input_dir is not None:
        write_jsonl(output_dir / "bootstrap_evidence_val_marked.jsonl", bootstrap_evidence_val)
        write_jsonl(output_dir / "bootstrap_prior_val_marked.jsonl", bootstrap_prior_val)

    summary = {
        "config": {
            "anchor_input_dir": str(input_dir),
            "bootstrap_input_dir": str(bootstrap_input_dir) if bootstrap_input_dir else None,
            "evidence_weight": args.evidence_weight,
            "prior_weight": args.prior_weight,
            "evidence_ratio": evidence_ratio,
            "prior_ratio": prior_ratio,
            "max_train_examples": args.max_train_examples,
            "include_bootstrap_val": args.include_bootstrap_val,
            "seed": args.seed,
        },
        "inputs": {
            "anchor_evidence_train": len(anchor_evidence_train),
            "anchor_evidence_val": len(anchor_evidence_val),
            "anchor_prior_train": len(anchor_prior_train),
            "anchor_prior_val": len(anchor_prior_val),
            "bootstrap_evidence_train": len(bootstrap_evidence_train),
            "bootstrap_evidence_val": len(bootstrap_evidence_val),
            "bootstrap_prior_train": len(bootstrap_prior_train),
            "bootstrap_prior_val": len(bootstrap_prior_val),
            "merged_evidence_train": len(evidence_train),
            "merged_evidence_val": len(evidence_val),
            "merged_prior_train": len(prior_train),
            "merged_prior_val": len(prior_val),
        },
        "outputs": {
            "unified_train": len(unified_train),
            "unified_val": len(unified_val),
            "train_task_source_counts": count_tasks(unified_train),
            "val_task_source_counts": count_tasks(unified_val),
            "train_mixture_source_counts": count_mixture_sources(unified_train),
            "val_mixture_source_counts": count_mixture_sources(unified_val),
            "train_task_mixture_counts": count_task_source_pairs(unified_train),
            "val_task_mixture_counts": count_task_source_pairs(unified_val),
            "effective_num_evidence_examples": num_evidence,
            "effective_num_prior_examples": num_prior,
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(f"[INFO] wrote unified analyzer SFT data to {output_dir}")
    print(
        f"[INFO] unified_train={len(unified_train)} "
        f"(evidence={num_evidence}, prior={num_prior}) | unified_val={len(unified_val)}"
    )


if __name__ == "__main__":
    main()
