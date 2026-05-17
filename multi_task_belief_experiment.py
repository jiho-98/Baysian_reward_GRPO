#!/usr/bin/env python3
"""Multi-task Bayesian strategy-belief experiment.

This file tests whether a Bayesian controller can learn different reliable
reasoning strategies for different task types. It uses only Python standard
library mock solvers and Python verifiers. No API, GPU, or external package is
required.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


PROBLEM_TYPES = [
    "modular_exponentiation",
    "linear_equation",
    "counting_cases",
]
STRATEGIES = [
    "brute_force",
    "modular_arithmetic",
    "cycle_detection",
    "equation_transform",
    "case_analysis",
    "inclusion_exclusion",
]
MODULI = [5, 7, 11, 13, 17, 19]
DIVISORS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_PER_TASK = 30
NUM_PROBLEMS = 90
PHASE_A_SIZE = 45
RESULT_PATH = Path("results/multi_task_belief_experiment.json")


@dataclass
class Problem:
    problem_id: int
    problem_type: str
    problem: str
    true_answer: int
    params: dict


@dataclass
class Belief:
    """Beta-Bernoulli belief with drift-aware usability."""

    alpha: float = 1.0
    beta: float = 1.0
    strategy_drift: float = 0.0
    total_trials: float = 0.0

    @property
    def success_belief(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def avg_strategy_drift(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.strategy_drift / self.total_trials

    @property
    def usability_belief(self) -> float:
        return self.success_belief * (1.0 - self.avg_strategy_drift)

    def update_assigned(self, correct: bool, compliance_score: float) -> None:
        score = clamp01(compliance_score)
        if correct:
            self.alpha += score
        else:
            self.beta += score
        self.strategy_drift += 1.0 - score
        self.total_trials += 1.0

    def update_actual(self, correct: bool, weight: float) -> None:
        credit = clamp01(weight)
        if correct:
            self.alpha += credit
        else:
            self.beta += credit
        self.total_trials += credit


@dataclass
class RolloutResult:
    problem_id: int
    problem_type: str
    problem: str
    true_answer: int
    assigned_strategy: str
    predicted_answer: int
    correct: bool
    strategy_compliance_score: float
    actual_strategy_weights: Dict[str, float]
    strategy_drift_score: float
    trace: str


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def belief_snapshot(belief: Belief) -> dict:
    return {
        "alpha": belief.alpha,
        "beta": belief.beta,
        "strategy_drift": belief.strategy_drift,
        "total_trials": belief.total_trials,
        "success_belief": belief.success_belief,
        "avg_strategy_drift": belief.avg_strategy_drift,
        "usability_belief": belief.usability_belief,
    }


def nested_belief_snapshot(table: Dict[str, Dict[str, Belief]]) -> dict:
    return {
        problem_type: {
            strategy: belief_snapshot(belief)
            for strategy, belief in strategy_table.items()
        }
        for problem_type, strategy_table in table.items()
    }


def make_empty_beliefs() -> Dict[str, Dict[str, Belief]]:
    return {
        problem_type: {strategy: Belief() for strategy in STRATEGIES}
        for problem_type in PROBLEM_TYPES
    }


def generate_problems(seed: int = 20260525) -> List[Problem]:
    rng = random.Random(seed)
    problems: List[Problem] = []
    next_id = 0

    for _ in range(NUM_PER_TASK):
        a = rng.randint(2, 20)
        b = rng.randint(2, 20)
        n = rng.randint(20, 500)
        m = rng.choice(MODULI)
        true_answer = (pow(a, n, m) + pow(b, n, m)) % m
        problems.append(
            Problem(
                problem_id=next_id,
                problem_type="modular_exponentiation",
                problem=f"Find the remainder when {a}^{n} + {b}^{n} is divided by {m}.",
                true_answer=true_answer,
                params={"a": a, "b": b, "n": n, "m": m},
            )
        )
        next_id += 1

    for _ in range(NUM_PER_TASK):
        a = rng.randint(2, 20)
        x_true = rng.randint(-20, 20)
        b = rng.randint(-50, 50)
        c = a * x_true + b
        problems.append(
            Problem(
                problem_id=next_id,
                problem_type="linear_equation",
                problem=f"Solve for x: {a}x + {b} = {c}.",
                true_answer=x_true,
                params={"a": a, "b": b, "c": c, "x_true": x_true},
            )
        )
        next_id += 1

    for _ in range(NUM_PER_TASK):
        n = rng.randint(30, 200)
        p = rng.choice(DIVISORS)
        q = rng.choice([value for value in DIVISORS if value != p])
        true_answer = sum(1 for i in range(1, n + 1) if i % p == 0 or i % q == 0)
        problems.append(
            Problem(
                problem_id=next_id,
                problem_type="counting_cases",
                problem=f"How many integers from 1 to {n} are divisible by {p} or {q}?",
                true_answer=true_answer,
                params={"N": n, "p": p, "q": q},
            )
        )
        next_id += 1

    rng.shuffle(problems)
    return problems


def power_by_cycle(base: int, exponent: int, modulus: int) -> int:
    if exponent == 0:
        return 1 % modulus

    residues = []
    seen = {}
    value = 1 % modulus
    while True:
        value = (value * base) % modulus
        if value in seen:
            cycle_start = seen[value]
            if exponent - 1 < len(residues):
                return residues[exponent - 1]
            cycle = residues[cycle_start:]
            idx = cycle_start + ((exponent - 1 - cycle_start) % len(cycle))
            return residues[idx]
        seen[value] = len(residues)
        residues.append(value)


def solve_modular_exponentiation(problem: Problem, strategy: str) -> tuple[int, Dict[str, float], str]:
    a = problem.params["a"]
    b = problem.params["b"]
    n = problem.params["n"]
    m = problem.params["m"]

    if strategy == "modular_arithmetic":
        return (pow(a, n, m) + pow(b, n, m)) % m, {"modular_arithmetic": 1.0}, "Used modular exponentiation."
    if strategy == "cycle_detection":
        predicted = (power_by_cycle(a, n, m) + power_by_cycle(b, n, m)) % m
        return predicted, {"cycle_detection": 0.5, "modular_arithmetic": 0.5}, "Detected residue cycles and combined residues."
    if strategy == "brute_force":
        if n <= 25:
            predicted = (a**n + b**n) % m
            trace = "Brute-forced both powers because n <= 25."
        else:
            predicted = (a + b) % m
            trace = "Brute force is too expensive; used naive fallback."
        return predicted, {"brute_force": 1.0}, trace
    if strategy == "case_analysis":
        predicted = ((a * a + b * b) if n % 2 == 0 else (a + b)) % m
        return predicted, {"case_analysis": 1.0}, "Used naive odd/even case heuristic."
    if strategy == "equation_transform":
        return (a + b + n) % m, {"equation_transform": 1.0}, "Irrelevant equation transformation heuristic."
    if strategy == "inclusion_exclusion":
        return (n // max(1, a) + n // max(1, b)) % m, {"inclusion_exclusion": 1.0}, "Irrelevant counting formula heuristic."
    raise ValueError(f"Unknown strategy: {strategy}")


def solve_linear_equation(problem: Problem, strategy: str) -> tuple[int, Dict[str, float], str]:
    a = problem.params["a"]
    b = problem.params["b"]
    c = problem.params["c"]
    x_true = problem.params["x_true"]

    if strategy == "equation_transform":
        return (c - b) // a, {"equation_transform": 1.0}, "Isolated x = (c - b) / a."
    if strategy == "brute_force":
        for candidate in range(-20, 21):
            if a * candidate + b == c:
                return candidate, {"brute_force": 1.0}, "Searched x in [-20, 20]."
        return 0, {"brute_force": 1.0}, "Search range failed."
    if strategy == "case_analysis":
        if x_true in (-1, 0, 1):
            predicted = x_true
        elif c > b:
            predicted = 1
        elif c < b:
            predicted = -1
        else:
            predicted = 0
        return predicted, {"case_analysis": 1.0}, "Used rough sign-based cases."
    if strategy == "modular_arithmetic":
        return (c - b) % a, {"modular_arithmetic": 1.0}, "Used irrelevant modular remainder."
    if strategy == "cycle_detection":
        return c % a, {"cycle_detection": 1.0}, "Used irrelevant cycle-like remainder."
    if strategy == "inclusion_exclusion":
        return a + b - c, {"inclusion_exclusion": 1.0}, "Used irrelevant inclusion-exclusion-like expression."
    raise ValueError(f"Unknown strategy: {strategy}")


def solve_counting_cases(problem: Problem, strategy: str) -> tuple[int, Dict[str, float], str]:
    n = problem.params["N"]
    p = problem.params["p"]
    q = problem.params["q"]

    if strategy == "inclusion_exclusion":
        lcm = abs(p * q) // math.gcd(p, q)
        predicted = n // p + n // q - n // lcm
        return predicted, {"inclusion_exclusion": 1.0}, "Applied inclusion-exclusion with lcm(p, q)."
    if strategy == "brute_force":
        predicted = sum(1 for i in range(1, n + 1) if i % p == 0 or i % q == 0)
        return predicted, {"brute_force": 1.0}, "Counted every integer from 1 to N."
    if strategy == "case_analysis":
        # Sometimes correct when p and q are coprime-ish by accident, but often wrong
        # because it ignores overlap.
        predicted = n // p + n // q
        return predicted, {"case_analysis": 1.0}, "Split into p-divisible and q-divisible cases but ignored overlap."
    if strategy == "modular_arithmetic":
        predicted = (n % p) + (n % q)
        return predicted, {"modular_arithmetic": 1.0}, "Used irrelevant remainder arithmetic."
    if strategy == "cycle_detection":
        predicted = n // max(p, q)
        return predicted, {"cycle_detection": 1.0}, "Used irrelevant periodic shortcut."
    if strategy == "equation_transform":
        return n - p - q, {"equation_transform": 1.0}, "Used irrelevant algebraic transformation."
    raise ValueError(f"Unknown strategy: {strategy}")


def solve_with_strategy(problem: Problem, strategy: str) -> RolloutResult:
    if problem.problem_type == "modular_exponentiation":
        predicted, weights, trace = solve_modular_exponentiation(problem, strategy)
    elif problem.problem_type == "linear_equation":
        predicted, weights, trace = solve_linear_equation(problem, strategy)
    elif problem.problem_type == "counting_cases":
        predicted, weights, trace = solve_counting_cases(problem, strategy)
    else:
        raise ValueError(f"Unknown problem type: {problem.problem_type}")

    compliance_score = 1.0
    correct = predicted == problem.true_answer
    return RolloutResult(
        problem_id=problem.problem_id,
        problem_type=problem.problem_type,
        problem=problem.problem,
        true_answer=problem.true_answer,
        assigned_strategy=strategy,
        predicted_answer=predicted,
        correct=correct,
        strategy_compliance_score=compliance_score,
        actual_strategy_weights=weights,
        strategy_drift_score=1.0 - compliance_score,
        trace=trace,
    )


def update_beliefs(
    assigned_beliefs: Dict[str, Dict[str, Belief]],
    actual_beliefs: Dict[str, Dict[str, Belief]],
    rollout: RolloutResult,
) -> None:
    problem_type = rollout.problem_type
    assigned_beliefs[problem_type][rollout.assigned_strategy].update_assigned(
        rollout.correct,
        rollout.strategy_compliance_score,
    )
    for actual_strategy, weight in rollout.actual_strategy_weights.items():
        actual_beliefs[problem_type][actual_strategy].update_actual(
            rollout.correct,
            weight,
        )


def choose_belief_guided_strategy(
    actual_beliefs: Dict[str, Dict[str, Belief]],
    problem_type: str,
) -> str:
    table = actual_beliefs[problem_type]
    return max(
        STRATEGIES,
        key=lambda strategy: (
            table[strategy].usability_belief,
            table[strategy].success_belief,
            strategy,
        ),
    )


def ranking(table: Dict[str, Belief]) -> List[dict]:
    return [
        {"strategy": strategy, **belief_snapshot(belief)}
        for strategy, belief in sorted(
            table.items(),
            key=lambda item: item[1].usability_belief,
            reverse=True,
        )
    ]


def nested_ranking(table: Dict[str, Dict[str, Belief]]) -> dict:
    return {problem_type: ranking(strategy_table) for problem_type, strategy_table in table.items()}


def accuracy(items: List[dict], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item[key]["correct"]) / len(items)


def per_task_accuracy(items: List[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for item in items:
        grouped[item["problem_type"]].append(item)
    return {
        problem_type: accuracy(problem_items, key)
        for problem_type, problem_items in grouped.items()
    }


def nested_counter_to_dict(counter: Counter) -> dict:
    result = {problem_type: {} for problem_type in PROBLEM_TYPES}
    for (problem_type, strategy), count in counter.items():
        result[problem_type][strategy] = count
    return result


def sample_phase_b_logs(phase_b_logs: List[dict], per_type: int = 3) -> List[dict]:
    samples = []
    for problem_type in PROBLEM_TYPES:
        type_logs = [item for item in phase_b_logs if item["problem_type"] == problem_type]
        samples.extend(type_logs[:per_type])
    return samples


def run_experiment() -> dict:
    problems = generate_problems()
    phase_a = problems[:PHASE_A_SIZE]
    phase_b = problems[PHASE_A_SIZE:]

    assigned_beliefs = make_empty_beliefs()
    actual_beliefs = make_empty_beliefs()

    phase_a_logs = []
    for problem in phase_a:
        for strategy in STRATEGIES:
            rollout = solve_with_strategy(problem, strategy)
            update_beliefs(assigned_beliefs, actual_beliefs, rollout)
            phase_a_logs.append(asdict(rollout))

    rng = random.Random(20260526)
    phase_b_logs = []
    random_counts = Counter()
    guided_counts = Counter()

    for problem in phase_b:
        random_strategy = rng.choice(STRATEGIES)
        guided_strategy = choose_belief_guided_strategy(actual_beliefs, problem.problem_type)

        random_rollout = solve_with_strategy(problem, random_strategy)
        guided_rollout = solve_with_strategy(problem, guided_strategy)

        random_counts[(problem.problem_type, random_strategy)] += 1
        guided_counts[(problem.problem_type, guided_strategy)] += 1

        phase_b_logs.append(
            {
                "problem_id": problem.problem_id,
                "problem_type": problem.problem_type,
                "problem": problem.problem,
                "true_answer": problem.true_answer,
                "random_strategy": random_strategy,
                "belief_guided_strategy": guided_strategy,
                "random_rollout": asdict(random_rollout),
                "belief_guided_rollout": asdict(guided_rollout),
            }
        )

    result = {
        "num_problems": NUM_PROBLEMS,
        "phase_a_size": len(phase_a),
        "phase_b_size": len(phase_b),
        "phase_b_random_accuracy": accuracy(phase_b_logs, "random_rollout"),
        "phase_b_belief_guided_accuracy": accuracy(phase_b_logs, "belief_guided_rollout"),
        "phase_b_random_accuracy_by_problem_type": per_task_accuracy(phase_b_logs, "random_rollout"),
        "phase_b_belief_guided_accuracy_by_problem_type": per_task_accuracy(
            phase_b_logs,
            "belief_guided_rollout",
        ),
        "final_actual_beliefs_ranking_by_problem_type": nested_ranking(actual_beliefs),
        "final_assigned_beliefs_ranking_by_problem_type": nested_ranking(assigned_beliefs),
        "final_assigned_beliefs": nested_belief_snapshot(assigned_beliefs),
        "final_actual_beliefs": nested_belief_snapshot(actual_beliefs),
        "belief_guided_strategy_distribution_by_problem_type": nested_counter_to_dict(guided_counts),
        "random_strategy_distribution_by_problem_type": nested_counter_to_dict(random_counts),
        "sample_phase_b_logs": sample_phase_b_logs(phase_b_logs),
        "phase_a_logs": phase_a_logs,
        "phase_b_logs": phase_b_logs,
    }
    return result


def print_ranking(title: str, rankings: dict) -> None:
    print(title)
    for problem_type in PROBLEM_TYPES:
        print(f"[{problem_type}]")
        print("strategy              success  usability")
        print("-" * 45)
        for item in rankings[problem_type]:
            print(
                f"{item['strategy']:<21} "
                f"{item['success_belief']:<8.3f} "
                f"{item['usability_belief']:<8.3f}"
            )
        print()


def print_accuracy_table(title: str, values: dict) -> None:
    print(title)
    for problem_type in PROBLEM_TYPES:
        print(f"  {problem_type:<24} {values.get(problem_type, 0.0):.3f}")
    print()


def print_summary(result: dict) -> None:
    print(f"Overall random accuracy:        {result['phase_b_random_accuracy']:.3f}")
    print(f"Overall belief-guided accuracy: {result['phase_b_belief_guided_accuracy']:.3f}")
    print()

    print_accuracy_table("Per-task random accuracy", result["phase_b_random_accuracy_by_problem_type"])
    print_accuracy_table(
        "Per-task belief-guided accuracy",
        result["phase_b_belief_guided_accuracy_by_problem_type"],
    )

    print_ranking(
        "Final actual belief ranking per problem_type",
        result["final_actual_beliefs_ranking_by_problem_type"],
    )
    print_ranking(
        "Final assigned belief ranking per problem_type",
        result["final_assigned_beliefs_ranking_by_problem_type"],
    )

    print("Belief-guided selected strategy distribution per problem_type")
    print(json.dumps(result["belief_guided_strategy_distribution_by_problem_type"], indent=2))
    print()
    print("Random selected strategy distribution per problem_type")
    print(json.dumps(result["random_strategy_distribution_by_problem_type"], indent=2))
    print()

    print("Sample Phase B logs")
    print("-" * 60)
    for item in result["sample_phase_b_logs"]:
        random_rollout = item["random_rollout"]
        guided_rollout = item["belief_guided_rollout"]
        print(
            f"{item['problem_type']:<24} id={item['problem_id']:<2} "
            f"true={item['true_answer']:<4} "
            f"random={random_rollout['assigned_strategy']}:{random_rollout['correct']} "
            f"guided={guided_rollout['assigned_strategy']}:{guided_rollout['correct']} | "
            f"{item['problem']}"
        )


def main() -> None:
    result = run_experiment()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result)
    print()
    print(f"Saved JSON result to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
