#!/usr/bin/env python3
"""Multi-problem Bayesian strategy-belief experiment.

This experiment extends the minimal belief demo without any API/GPU dependency.
It uses mock solvers plus a Python verifier to test whether beliefs learned on
early modular-exponentiation problems improve later strategy selection.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


STRATEGIES = [
    "brute_force",
    "modular_arithmetic",
    "cycle_detection",
    "case_analysis",
]
MODULI = [5, 7, 11, 13, 17, 19]
NUM_PROBLEMS = 50
PHASE_A_SIZE = 25
RESULT_PATH = Path("results/multi_problem_belief_experiment.json")


@dataclass
class Problem:
    problem_id: int
    a: int
    b: int
    n: int
    m: int
    problem: str
    true_answer: int


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
        """Update belief for the requested strategy."""
        score = clamp01(compliance_score)
        if correct:
            self.alpha += score
        else:
            self.beta += score
        self.strategy_drift += 1.0 - score
        self.total_trials += 1.0

    def update_actual(self, correct: bool, weight: float) -> None:
        """Update belief for strategies inferred to have actually contributed."""
        credit = clamp01(weight)
        if correct:
            self.alpha += credit
        else:
            self.beta += credit
        self.total_trials += credit


@dataclass
class RolloutResult:
    problem_id: int
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


def belief_table_snapshot(table: Dict[str, Belief]) -> dict:
    return {strategy: belief_snapshot(belief) for strategy, belief in table.items()}


def generate_problems(num_problems: int, seed: int = 20260525) -> List[Problem]:
    rng = random.Random(seed)
    problems = []
    for problem_id in range(num_problems):
        a = rng.randint(2, 20)
        b = rng.randint(2, 20)
        n = rng.randint(20, 500)
        m = rng.choice(MODULI)
        true_answer = (pow(a, n, m) + pow(b, n, m)) % m
        problem_text = f"Find the remainder when {a}^{n} + {b}^{n} is divided by {m}."
        problems.append(
            Problem(
                problem_id=problem_id,
                a=a,
                b=b,
                n=n,
                m=m,
                problem=problem_text,
                true_answer=true_answer,
            )
        )
    return problems


def power_by_cycle(base: int, exponent: int, modulus: int) -> int:
    """Compute base**exponent % modulus by explicit residue-cycle detection."""
    if exponent == 0:
        return 1 % modulus

    residues = []
    seen = {}
    value = 1 % modulus
    exp = 0
    while True:
        exp += 1
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


def solve_with_strategy(problem: Problem, strategy: str) -> RolloutResult:
    """Mock solver for one strategy-conditioned rollout."""
    if strategy == "brute_force":
        if problem.n <= 25:
            predicted = (problem.a**problem.n + problem.b**problem.n) % problem.m
            trace = "Directly computed both powers because n <= 25."
        else:
            predicted = (problem.a + problem.b) % problem.m
            trace = (
                "Brute force is inefficient for this exponent; returned a naive "
                "(a + b) % m fallback."
            )
        weights = {"brute_force": 1.0}

    elif strategy == "modular_arithmetic":
        predicted = (pow(problem.a, problem.n, problem.m) + pow(problem.b, problem.n, problem.m)) % problem.m
        trace = "Used Python modular exponentiation pow(base, n, m) for both terms."
        weights = {"modular_arithmetic": 1.0}

    elif strategy == "cycle_detection":
        a_residue = power_by_cycle(problem.a, problem.n, problem.m)
        b_residue = power_by_cycle(problem.b, problem.n, problem.m)
        predicted = (a_residue + b_residue) % problem.m
        trace = (
            "Detected residue cycles for both bases, then used modular reduction "
            "to combine residues."
        )
        weights = {"cycle_detection": 0.5, "modular_arithmetic": 0.5}

    elif strategy == "case_analysis":
        if problem.n % 2 == 0:
            predicted = (problem.a * problem.a + problem.b * problem.b) % problem.m
            trace = "Naively treated every even exponent like exponent 2."
        else:
            predicted = (problem.a + problem.b) % problem.m
            trace = "Naively treated every odd exponent like exponent 1."
        weights = {"case_analysis": 1.0}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    correct = predicted == problem.true_answer
    compliance_score = 1.0
    return RolloutResult(
        problem_id=problem.problem_id,
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
    assigned_beliefs: Dict[str, Belief],
    actual_beliefs: Dict[str, Belief],
    rollout: RolloutResult,
) -> None:
    """Apply assigned-strategy and actual-strategy belief updates."""
    assigned_beliefs[rollout.assigned_strategy].update_assigned(
        rollout.correct,
        rollout.strategy_compliance_score,
    )
    for actual_strategy, weight in rollout.actual_strategy_weights.items():
        actual_beliefs[actual_strategy].update_actual(rollout.correct, weight)


def choose_belief_guided_strategy(actual_beliefs: Dict[str, Belief]) -> str:
    return max(
        STRATEGIES,
        key=lambda strategy: (
            actual_beliefs[strategy].usability_belief,
            actual_beliefs[strategy].success_belief,
            strategy,
        ),
    )


def ranking(table: Dict[str, Belief]) -> List[dict]:
    ordered = sorted(
        table.items(),
        key=lambda item: item[1].usability_belief,
        reverse=True,
    )
    return [
        {
            "strategy": strategy,
            **belief_snapshot(belief),
        }
        for strategy, belief in ordered
    ]


def accuracy(logs: List[dict]) -> float:
    if not logs:
        return 0.0
    return sum(1 for item in logs if item["rollout"]["correct"]) / len(logs)


def run_experiment() -> dict:
    problems = generate_problems(NUM_PROBLEMS)
    phase_a = problems[:PHASE_A_SIZE]
    phase_b = problems[PHASE_A_SIZE:]

    assigned_beliefs = {strategy: Belief() for strategy in STRATEGIES}
    actual_beliefs = {strategy: Belief() for strategy in STRATEGIES}

    phase_a_logs = []
    for problem in phase_a:
        for strategy in STRATEGIES:
            rollout = solve_with_strategy(problem, strategy)
            update_beliefs(assigned_beliefs, actual_beliefs, rollout)
            phase_a_logs.append(asdict(rollout))

    # Phase B is a held-out comparison: learned beliefs are used for selection,
    # but evaluation rollouts do not update the belief tables.
    rng = random.Random(20260526)
    phase_b_logs = []
    random_selection_counts = Counter()
    guided_selection_counts = Counter()

    for problem in phase_b:
        random_strategy = rng.choice(STRATEGIES)
        guided_strategy = choose_belief_guided_strategy(actual_beliefs)

        random_rollout = solve_with_strategy(problem, random_strategy)
        guided_rollout = solve_with_strategy(problem, guided_strategy)

        random_selection_counts[random_strategy] += 1
        guided_selection_counts[guided_strategy] += 1

        phase_b_logs.append(
            {
                "problem_id": problem.problem_id,
                "problem": problem.problem,
                "true_answer": problem.true_answer,
                "random_strategy": random_strategy,
                "belief_guided_strategy": guided_strategy,
                "random_rollout": asdict(random_rollout),
                "belief_guided_rollout": asdict(guided_rollout),
            }
        )

    random_logs_for_accuracy = [
        {"rollout": item["random_rollout"]} for item in phase_b_logs
    ]
    guided_logs_for_accuracy = [
        {"rollout": item["belief_guided_rollout"]} for item in phase_b_logs
    ]

    result = {
        "num_problems": NUM_PROBLEMS,
        "phase_a_size": len(phase_a),
        "phase_b_size": len(phase_b),
        "phase_b_random_accuracy": accuracy(random_logs_for_accuracy),
        "phase_b_belief_guided_accuracy": accuracy(guided_logs_for_accuracy),
        "phase_a_actual_beliefs_ranking": ranking(actual_beliefs),
        "final_assigned_beliefs": belief_table_snapshot(assigned_beliefs),
        "final_actual_beliefs": belief_table_snapshot(actual_beliefs),
        "belief_guided_strategy_distribution": dict(guided_selection_counts),
        "random_strategy_distribution": dict(random_selection_counts),
        "phase_a_logs": phase_a_logs,
        "phase_b_logs": phase_b_logs,
    }
    return result


def print_belief_table(title: str, table: Dict[str, dict]) -> None:
    print(title)
    print("strategy              alpha   beta    success  drift   usability")
    print("-" * 70)
    ordered = sorted(
        table.items(),
        key=lambda item: item[1]["usability_belief"],
        reverse=True,
    )
    for strategy, belief in ordered:
        print(
            f"{strategy:<21} "
            f"{belief['alpha']:<7.2f} "
            f"{belief['beta']:<7.2f} "
            f"{belief['success_belief']:<8.3f} "
            f"{belief['avg_strategy_drift']:<7.3f} "
            f"{belief['usability_belief']:<8.3f}"
        )
    print()


def print_summary(result: dict) -> None:
    print("Phase A actual_beliefs ranking")
    print("strategy              success  usability")
    print("-" * 45)
    for item in result["phase_a_actual_beliefs_ranking"]:
        print(
            f"{item['strategy']:<21} "
            f"{item['success_belief']:<8.3f} "
            f"{item['usability_belief']:<8.3f}"
        )
    print()

    print(f"Phase B random_strategy accuracy:        {result['phase_b_random_accuracy']:.3f}")
    print(f"Phase B belief_guided_strategy accuracy: {result['phase_b_belief_guided_accuracy']:.3f}")
    print()

    print(f"belief_guided selected: {result['belief_guided_strategy_distribution']}")
    print(f"random selected:        {result['random_strategy_distribution']}")
    print()

    print_belief_table("final assigned_beliefs", result["final_assigned_beliefs"])
    print_belief_table("final actual_beliefs", result["final_actual_beliefs"])

    print("sample logs")
    print("-" * 45)
    for item in result["phase_b_logs"][:5]:
        random_rollout = item["random_rollout"]
        guided_rollout = item["belief_guided_rollout"]
        print(
            f"problem_id={item['problem_id']} | true={item['true_answer']} | "
            f"random={random_rollout['assigned_strategy']}:{random_rollout['correct']} | "
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
