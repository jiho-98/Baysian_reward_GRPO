#!/usr/bin/env python3
"""Offline posterior recomputation over a fixed rollout pool.

This script is used to test whether strategy prior itself improves posterior
reward behavior. Since both modes use the same fixed rollouts, any difference
comes from the prior calculation, not from generation randomness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROUND_DIGITS = 12


@dataclass(frozen=True)
class RolloutRecord:
    problem_id: str
    rollout_id: Any
    likelihood: float
    stored_prior: float
    answer_correct: float
    error_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute posterior rewards offline for fixed rollouts under "
            "uniform and strategy_suitability priors."
        )
    )
    parser.add_argument("--input_path", required=True, help="Path to rollout_records.jsonl")
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save offline_prior_comparison_metrics.json",
    )
    parser.add_argument(
        "--high_posterior_threshold",
        type=float,
        default=0.2,
        help="Threshold for incorrect_but_high_posterior metrics (default: 0.2)",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        result = float(value)
    elif isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            result = float(text)
        except ValueError:
            return default
    else:
        return default
    if not math.isfinite(result):
        return default
    return result


def mean_or_zero(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def uniform_distribution(size: int) -> list[float]:
    if size <= 0:
        return []
    return [1.0 / size] * size


def is_correct(value: float) -> bool:
    return value > 0.5


def get_error_type(record: dict[str, Any]) -> str:
    error_type = record.get("error_type")
    if isinstance(error_type, str) and error_type.strip():
        return error_type.strip()
    evidence = record.get("evidence")
    if isinstance(evidence, dict):
        nested_error_type = evidence.get("error_type")
        if isinstance(nested_error_type, str) and nested_error_type.strip():
            return nested_error_type.strip()
    return "unknown"


def problem_id_for_record(record: dict[str, Any], line_number: int) -> str:
    for key in ("problem_id", "unique_id"):
        value = record.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return f"missing_problem_id_line_{line_number}"


def rollout_sort_key(record: RolloutRecord) -> tuple[int, float, str]:
    value = record.rollout_id
    if value is None:
        return (2, 0.0, "")
    if isinstance(value, bool):
        return (0, float(value), "")
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return (0, number, "")
        return (2, 0.0, "")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return (2, 0.0, "")
        try:
            return (0, float(text), "")
        except ValueError:
            return (1, 0.0, text)
    return (2, 0.0, str(value))


def normalize_nonnegative(values: list[float]) -> tuple[list[float], bool]:
    if not values:
        return [], False
    if any((not math.isfinite(value)) or value < 0.0 for value in values):
        return uniform_distribution(len(values)), False
    total = sum(values)
    if (not math.isfinite(total)) or total <= 0.0:
        return uniform_distribution(len(values)), False
    return [value / total for value in values], True


def posterior_entropy(values: list[float]) -> float:
    entropy = 0.0
    for value in values:
        entropy -= value * math.log(value + 1e-12)
    return entropy


def top_two_gap(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


def select_first_max_index(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def load_records(path: Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Failed to parse JSON on line {line_number} of {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise SystemExit(
                    f"Expected a JSON object on line {line_number} of {path}, got {type(payload).__name__}."
                )
            records.append(
                RolloutRecord(
                    problem_id=problem_id_for_record(payload, line_number),
                    rollout_id=payload.get("rollout_id"),
                    likelihood=safe_float(payload.get("likelihood"), 0.0),
                    stored_prior=safe_float(payload.get("prior"), 1.0),
                    answer_correct=safe_float(payload.get("answer_correct"), 0.0),
                    error_type=get_error_type(payload),
                )
            )
    return records


def group_records_by_problem(records: list[RolloutRecord]) -> dict[str, list[RolloutRecord]]:
    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        grouped[record.problem_id].append(record)
    for problem_id in grouped:
        grouped[problem_id].sort(key=rollout_sort_key)
    return dict(grouped)


def mode_rollout_count(grouped: dict[str, list[RolloutRecord]]) -> int:
    counts = [len(items) for items in grouped.values()]
    if not counts:
        return 0
    modes = statistics.multimode(counts)
    if 8 in modes:
        return 8
    return min(modes)


def warn_on_unexpected_rollout_counts(grouped: dict[str, list[RolloutRecord]]) -> None:
    expected_mode = mode_rollout_count(grouped)
    for problem_id, items in grouped.items():
        count = len(items)
        if count != expected_mode or count != 8:
            print(
                (
                    f"Warning: problem_id={problem_id} has {count} rollouts "
                    f"(mode_count={expected_mode}, nominal_count=8)."
                ),
                file=sys.stderr,
            )


def compute_distributions(items: list[RolloutRecord], prior_mode: str) -> tuple[list[float], list[float]]:
    size = len(items)
    likelihoods = [record.likelihood for record in items]

    if prior_mode == "uniform":
        priors = uniform_distribution(size)
        posterior, _ = normalize_nonnegative(likelihoods)
        return priors, posterior

    if prior_mode != "strategy_suitability":
        raise ValueError(f"Unsupported prior mode: {prior_mode}")

    normalized_priors, _ = normalize_nonnegative([record.stored_prior for record in items])
    weighted_values = [prior * likelihood for prior, likelihood in zip(normalized_priors, likelihoods)]
    posterior, _ = normalize_nonnegative(weighted_values)
    return normalized_priors, posterior


def analyze_mode(
    grouped: dict[str, list[RolloutRecord]],
    prior_mode: str,
    high_posterior_threshold: float,
) -> dict[str, Any]:
    posterior_top1_hits: list[float] = []
    likelihood_top1_hits: list[float] = []
    outcome_top1_hits: list[float] = []
    prior_only_top1_hits: list[float] = []
    oracle_hits: list[float] = []
    posterior_entropies: list[float] = []
    posterior_gaps: list[float] = []
    posterior_mass_on_correct_values: list[float] = []
    posterior_advantage_unique_counts: list[float] = []
    likelihood_correct_values: list[float] = []
    likelihood_incorrect_values: list[float] = []
    posterior_correct_values: list[float] = []
    posterior_incorrect_values: list[float] = []
    incorrect_but_high_posterior_count = 0
    wrong_direction_high_posterior_count = 0
    incorrect_rollout_count = 0
    num_rollouts_total = 0

    for items in grouped.values():
        if not items:
            continue

        priors, posteriors = compute_distributions(items, prior_mode)
        likelihoods = [record.likelihood for record in items]
        correctness = [is_correct(record.answer_correct) for record in items]

        posterior_choice = items[select_first_max_index(posteriors)]
        likelihood_choice = items[select_first_max_index(likelihoods)]
        outcome_choice = items[select_first_max_index([record.answer_correct for record in items])]
        prior_choice = items[select_first_max_index(priors)]
        oracle_value = 1.0 if any(correctness) else 0.0

        posterior_top1_hits.append(1.0 if is_correct(posterior_choice.answer_correct) else 0.0)
        likelihood_top1_hits.append(1.0 if is_correct(likelihood_choice.answer_correct) else 0.0)
        outcome_top1_hits.append(1.0 if is_correct(outcome_choice.answer_correct) else 0.0)
        prior_only_top1_hits.append(1.0 if is_correct(prior_choice.answer_correct) else 0.0)
        oracle_hits.append(oracle_value)

        posterior_entropies.append(posterior_entropy(posteriors))
        posterior_gaps.append(top_two_gap(posteriors))
        posterior_mass_on_correct_values.append(
            sum(posterior for posterior, correct in zip(posteriors, correctness) if correct)
        )

        mean_posterior = mean_or_zero(posteriors)
        advantages = [round(posterior - mean_posterior, ROUND_DIGITS) for posterior in posteriors]
        posterior_advantage_unique_counts.append(float(len(set(advantages))))

        for record, posterior in zip(items, posteriors):
            num_rollouts_total += 1
            if is_correct(record.answer_correct):
                likelihood_correct_values.append(record.likelihood)
                posterior_correct_values.append(posterior)
            else:
                incorrect_rollout_count += 1
                likelihood_incorrect_values.append(record.likelihood)
                posterior_incorrect_values.append(posterior)
                if posterior >= high_posterior_threshold:
                    incorrect_but_high_posterior_count += 1
                    if record.error_type == "wrong_direction":
                        wrong_direction_high_posterior_count += 1

    avg_likelihood_correct = mean_or_zero(likelihood_correct_values)
    avg_likelihood_incorrect = mean_or_zero(likelihood_incorrect_values)
    avg_posterior_correct = mean_or_zero(posterior_correct_values)
    avg_posterior_incorrect = mean_or_zero(posterior_incorrect_values)

    return {
        "num_problems": len(grouped),
        "num_rollouts_total": num_rollouts_total,
        "posterior_top1_accuracy": mean_or_zero(posterior_top1_hits),
        "likelihood_top1_accuracy": mean_or_zero(likelihood_top1_hits),
        "outcome_top1_accuracy": mean_or_zero(outcome_top1_hits),
        "prior_only_top1_accuracy": mean_or_zero(prior_only_top1_hits),
        "oracle_best_of_n_accuracy": mean_or_zero(oracle_hits),
        "posterior_entropy_mean": mean_or_zero(posterior_entropies),
        "posterior_top1_top2_gap_mean": mean_or_zero(posterior_gaps),
        "posterior_mass_on_correct_mean": mean_or_zero(posterior_mass_on_correct_values),
        "posterior_advantage_unique_values_per_problem_mean": mean_or_zero(
            posterior_advantage_unique_counts
        ),
        "incorrect_but_high_posterior_count": incorrect_but_high_posterior_count,
        "incorrect_but_high_posterior_rate": (
            incorrect_but_high_posterior_count / incorrect_rollout_count if incorrect_rollout_count else 0.0
        ),
        "wrong_direction_high_posterior_count": wrong_direction_high_posterior_count,
        "avg_likelihood_correct": avg_likelihood_correct,
        "avg_likelihood_incorrect": avg_likelihood_incorrect,
        "likelihood_correct_incorrect_gap": avg_likelihood_correct - avg_likelihood_incorrect,
        "avg_posterior_correct": avg_posterior_correct,
        "avg_posterior_incorrect": avg_posterior_incorrect,
        "posterior_correct_incorrect_gap": avg_posterior_correct - avg_posterior_incorrect,
    }


def build_output(
    input_path: Path,
    grouped: dict[str, list[RolloutRecord]],
    uniform_metrics: dict[str, Any],
    strategy_metrics: dict[str, Any],
) -> dict[str, Any]:
    delta_keys = [
        "posterior_top1_accuracy",
        "posterior_entropy_mean",
        "posterior_top1_top2_gap_mean",
        "posterior_mass_on_correct_mean",
        "posterior_advantage_unique_values_per_problem_mean",
        "incorrect_but_high_posterior_count",
        "wrong_direction_high_posterior_count",
    ]

    return {
        "comparison_note": "Same fixed rollout pool. Only prior mode is changed offline.",
        "input_path": str(input_path),
        "num_problems": len(grouped),
        "num_rollouts_total": sum(len(items) for items in grouped.values()),
        "uniform": uniform_metrics,
        "strategy_suitability": strategy_metrics,
        "delta_strategy_minus_uniform": {
            key: strategy_metrics[key] - uniform_metrics[key] for key in delta_keys
        },
    }


def print_summary(
    input_path: Path,
    num_problems: int,
    num_rollouts_total: int,
    uniform_metrics: dict[str, Any],
    strategy_metrics: dict[str, Any],
) -> None:
    delta = strategy_metrics["posterior_top1_accuracy"] - uniform_metrics["posterior_top1_accuracy"]
    print(f"input_path: {input_path}")
    print(f"number_of_problems: {num_problems}")
    print(f"number_of_rollouts: {num_rollouts_total}")
    print(f"uniform_posterior_top1_accuracy: {uniform_metrics['posterior_top1_accuracy']:.6f}")
    print(f"strategy_posterior_top1_accuracy: {strategy_metrics['posterior_top1_accuracy']:.6f}")
    print(f"delta_strategy_minus_uniform: {delta:.6f}")
    print(
        "uniform_posterior_advantage_unique_values_per_problem_mean: "
        f"{uniform_metrics['posterior_advantage_unique_values_per_problem_mean']:.6f}"
    )
    print(
        "strategy_posterior_advantage_unique_values_per_problem_mean: "
        f"{strategy_metrics['posterior_advantage_unique_values_per_problem_mean']:.6f}"
    )
    print(
        "uniform_incorrect_but_high_posterior_count: "
        f"{uniform_metrics['incorrect_but_high_posterior_count']}"
    )
    print(
        "strategy_incorrect_but_high_posterior_count: "
        f"{strategy_metrics['incorrect_but_high_posterior_count']}"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    records = load_records(input_path)
    grouped = group_records_by_problem(records)
    warn_on_unexpected_rollout_counts(grouped)

    uniform_metrics = analyze_mode(grouped, "uniform", args.high_posterior_threshold)
    strategy_metrics = analyze_mode(grouped, "strategy_suitability", args.high_posterior_threshold)
    output_payload = build_output(input_path, grouped, uniform_metrics, strategy_metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_summary(
        input_path=input_path,
        num_problems=output_payload["num_problems"],
        num_rollouts_total=output_payload["num_rollouts_total"],
        uniform_metrics=uniform_metrics,
        strategy_metrics=strategy_metrics,
    )


if __name__ == "__main__":
    main()
