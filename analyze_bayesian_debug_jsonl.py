#!/usr/bin/env python3
"""Summarize posterior diagnostics from Bayesian Full GRPO debug JSONL."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Bayesian Full GRPO debug JSONL diagnostics."
    )
    parser.add_argument("--input_debug_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def normalized_entropy(probs: list[float]) -> float:
    if not probs or len(probs) == 1:
        return 0.0
    entropy = 0.0
    for prob in probs:
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy / math.log(len(probs))


def top1_top2_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    sorted_values = sorted(values, reverse=True)
    return sorted_values[0] - sorted_values[1]


def group_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        int(row.get("global_reward_call_index", 0) or 0),
        int(row.get("group_index_within_call", 0) or 0),
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_debug_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Debug JSONL not found: {input_path}")

    rows = load_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows loaded from {input_path}")

    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    group_metric_rows: list[dict[str, Any]] = []
    evidence_parse_success = 0

    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: int(row.get("group_rollout_id", 0) or 0))
        posteriors = [float(row.get("bayesian_reward", 0.0) or 0.0) for row in group]
        answer_correctness = [float(row.get("answer_correctness", 0.0) or 0.0) for row in group]
        any_correct = any(value == 1.0 for value in answer_correctness)

        top_index = max(range(len(group)), key=lambda idx: posteriors[idx])
        top_correct = answer_correctness[top_index] == 1.0
        wrong_top = any_correct and not top_correct
        mass_on_correct = sum(
            posterior
            for posterior, correct in zip(posteriors, answer_correctness)
            if correct == 1.0
        )
        prior_parse_ok = not any(
            as_bool(row.get("prior_judge_fallback_used"))
            or as_bool(row.get("prior_missing_from_judge"))
            for row in group
        )
        posterior_fallback = any(
            as_bool(row.get("posterior_normalization_fallback_used")) for row in group
        )

        evidence_parse_ok_count = 0
        for row in group:
            evidence_parse_ok = not as_bool(row.get("evidence_judge_fallback_used"))
            evidence_parse_success += int(evidence_parse_ok)
            evidence_parse_ok_count += int(evidence_parse_ok)

        group_metric_rows.append(
            {
                "group_key": [key[0], key[1]],
                "problem": str(group[0].get("problem", "")),
                "num_rollouts": len(group),
                "any_correct": any_correct,
                "top_correct": top_correct,
                "wrong_top": wrong_top,
                "mass_on_correct": mass_on_correct,
                "entropy": normalized_entropy(posteriors),
                "top1_top2_gap": top1_top2_gap(posteriors),
                "prior_parse_ok": prior_parse_ok,
                "evidence_parse_ok_count": evidence_parse_ok_count,
                "posterior_fallback": posterior_fallback,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "group_metrics.jsonl", group_metric_rows)

    total_groups = len(group_metric_rows)
    groups_with_correct = [row for row in group_metric_rows if row["any_correct"]]
    prior_parse_success = sum(1 for row in group_metric_rows if row["prior_parse_ok"])
    posterior_fallback_count = sum(1 for row in group_metric_rows if row["posterior_fallback"])
    wrong_top_count = sum(1 for row in group_metric_rows if row["wrong_top"])
    top1_accuracy_when_correct_exists = (
        statistics.fmean(1.0 if row["top_correct"] else 0.0 for row in groups_with_correct)
        if groups_with_correct
        else 0.0
    )

    diagnostics = {
        "prior_parse_rate": prior_parse_success / total_groups if total_groups else 0.0,
        "evidence_parse_rate": evidence_parse_success / len(rows) if rows else 0.0,
        "posterior_fallback_count": posterior_fallback_count,
        "posterior_top1_accuracy_when_correct_exists": top1_accuracy_when_correct_exists,
        "wrong_top_count_when_correct_exists": wrong_top_count,
        "mass_on_correct_mean": (
            statistics.fmean(row["mass_on_correct"] for row in group_metric_rows)
            if group_metric_rows
            else 0.0
        ),
        "entropy_mean": (
            statistics.fmean(row["entropy"] for row in group_metric_rows)
            if group_metric_rows
            else 0.0
        ),
        "top1_top2_gap_mean": (
            statistics.fmean(row["top1_top2_gap"] for row in group_metric_rows)
            if group_metric_rows
            else 0.0
        ),
    }
    summary = {
        "config": {
            "input_debug_jsonl": str(input_path),
            "output_dir": str(output_dir),
        },
        "counts": {
            "num_rows": len(rows),
            "num_groups": total_groups,
            "groups_with_any_correct": len(groups_with_correct),
            "prior_parse_success": prior_parse_success,
            "evidence_parse_success": evidence_parse_success,
        },
        "diagnostics": diagnostics,
    }
    write_json(output_dir / "summary.json", summary)

    print("[INFO] Bayesian debug diagnostics")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
