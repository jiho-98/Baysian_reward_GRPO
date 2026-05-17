#!/usr/bin/env python3
"""LLM-based self-data Bayesian reasoning experiment.

Real mode uses an LLM for problem generation, strategy proposal, solving, and
rollout analysis. Python is only used for verification, answer extraction,
belief updates, and logging.

DRY_RUN=1 uses local deterministic stand-ins to test the full experiment
structure without API calls.
"""

from __future__ import annotations
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_API_CALLS = int(os.getenv("MAX_API_CALLS", "200"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
RUN_DIRECT_BASELINE = os.getenv("RUN_DIRECT_BASELINE", "0") == "1"
RESULT_PATH = Path("results/llm_self_data_bayesian_experiment.json")

PROBLEM_TYPES = [
    "modular_exponentiation",
    "linear_equation",
    "counting_cases",
]
ALLOWED_STRATEGY_FAMILIES = [
    "brute_force",
    "modular_arithmetic",
    "cycle_detection",
    "equation_transform",
    "case_analysis",
    "inclusion_exclusion",
    "elimination",
    "direct_formula",
    "other",
]
FAILURE_TYPES = [
    "none",
    "calculation_error",
    "invalid_strategy",
    "strategy_drift",
    "final_answer_extraction_error",
    "incomplete_reasoning",
    "other",
]
MODULI = [5, 7, 11, 13, 17, 19, 23]
DIVISORS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
NUM_PROBLEMS = 30
PHASE_A_SIZE = 15
RNG = random.Random(20260525)


@dataclass
class Belief:
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

    def update_assigned_success(self, correct: bool, compliance_score: float) -> None:
        score = clamp01(compliance_score)
        if correct:
            self.alpha += score
        else:
            self.beta += score
        self.strategy_drift += 1.0 - score
        self.total_trials += 1.0

    def update_actual_success(self, correct: bool, weight: float) -> None:
        credit = clamp01(weight)
        if correct:
            self.alpha += credit
        else:
            self.beta += credit
        self.total_trials += credit

    def update_validity(self, alpha_credit: float, beta_credit: float, trial_credit: float) -> None:
        self.alpha += max(0.0, alpha_credit)
        self.beta += max(0.0, beta_credit)
        self.total_trials += max(0.0, trial_credit)


@dataclass
class StrategySpec:
    strategy_id: str
    strategy_family: str
    description: str


@dataclass
class RolloutLog:
    problem_id: int
    problem_type: str
    problem: str
    true_answer: int
    assigned_strategy_id: str
    assigned_strategy_family: str
    strategy_description: str
    raw_response: str
    predicted_answer: Optional[int]
    correct: bool
    analysis: dict


class APICounter:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.total = 0
        self.problem_generator = 0
        self.strategy_proposer = 0
        self.solver = 0
        self.analyzer = 0
        self.direct_baseline = 0

    def check(self, kind: str) -> None:
        if self.total >= self.max_calls:
            raise RuntimeError(f"MAX_API_CALLS reached before {kind}: {self.total}/{self.max_calls}")
        self.total += 1
        if not hasattr(self, kind):
            raise ValueError(f"Unknown API call kind: {kind}")
        setattr(self, kind, getattr(self, kind) + 1)

    def snapshot(self) -> dict:
        return {
            "total": self.total,
            "problem_generator": self.problem_generator,
            "strategy_proposer": self.strategy_proposer,
            "solver": self.solver,
            "analyzer": self.analyzer,
            "direct_baseline": self.direct_baseline,
            "max_api_calls": self.max_calls,
        }


api_calls = APICounter(MAX_API_CALLS)


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def normalize_family(value: str) -> str:
    family = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    return family if family in ALLOWED_STRATEGY_FAMILIES else "other"


def safe_json_parse(text: str) -> Optional[dict]:
    """Parse JSON even if the model wraps it in text or code fences."""
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(cleaned[first_obj : last_obj + 1])

    first_arr = cleaned.find("[")
    last_arr = cleaned.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(cleaned[first_arr : last_arr + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
        except json.JSONDecodeError:
            continue
    return None


def call_llm(prompt: str, kind: str, temperature: float = 0.2) -> str:
    """Call OpenAI Responses API when available, with Chat Completions fallback."""
    if DRY_RUN:
        raise RuntimeError("call_llm should not be used in DRY_RUN mode.")

    api_calls.check(kind)
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for real mode.")

    try:
        response = client.responses.create(
            model=MODEL,
            input=prompt,
            temperature=temperature,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text
        return str(response)
    except AttributeError:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def json_llm_call(prompt: str, kind: str, retries: int = 2, temperature: float = 0.1) -> Optional[dict]:
    for attempt in range(retries + 1):
        text = call_llm(prompt, kind=kind, temperature=temperature)
        parsed = safe_json_parse(text)
        if parsed is not None:
            return parsed
        time.sleep(0.5 * (attempt + 1))
    return None


def compute_true_answer(spec: dict) -> int:
    problem_type = spec["problem_type"]
    if problem_type == "modular_exponentiation":
        return (pow(int(spec["a"]), int(spec["n"]), int(spec["m"])) + pow(int(spec["b"]), int(spec["n"]), int(spec["m"]))) % int(spec["m"])
    if problem_type == "linear_equation":
        return int(spec["x_true"])
    if problem_type == "counting_cases":
        n = int(spec["N"])
        p = int(spec["p"])
        q = int(spec["q"])
        return sum(1 for i in range(1, n + 1) if i % p == 0 or i % q == 0)
    raise ValueError(f"Unknown problem_type: {problem_type}")


def local_problem_specs() -> List[dict]:
    specs = []
    problem_id = 0
    for _ in range(10):
        a, b = RNG.randint(2, 30), RNG.randint(2, 30)
        n, m = RNG.randint(20, 800), RNG.choice(MODULI)
        specs.append({
            "problem_id": problem_id,
            "problem_type": "modular_exponentiation",
            "a": a,
            "b": b,
            "n": n,
            "m": m,
            "problem": f"Find the remainder when {a}^{n} + {b}^{n} is divided by {m}.",
        })
        problem_id += 1
    for _ in range(10):
        a, x_true, b = RNG.randint(2, 25), RNG.randint(-30, 30), RNG.randint(-80, 80)
        c = a * x_true + b
        specs.append({
            "problem_id": problem_id,
            "problem_type": "linear_equation",
            "a": a,
            "b": b,
            "x_true": x_true,
            "c": c,
            "problem": f"Solve for x: {a}x + {b} = {c}.",
        })
        problem_id += 1
    for _ in range(10):
        n = RNG.randint(40, 300)
        p = RNG.choice(DIVISORS)
        q = RNG.choice([value for value in DIVISORS if value != p])
        specs.append({
            "problem_id": problem_id,
            "problem_type": "counting_cases",
            "N": n,
            "p": p,
            "q": q,
            "problem": f"How many integers from 1 to {n} are divisible by {p} or {q}?",
        })
        problem_id += 1
    RNG.shuffle(specs)
    for idx, spec in enumerate(specs):
        spec["problem_id"] = idx
    return specs


def sanitize_problem_spec(raw: dict, fallback: dict) -> dict:
    problem_type = raw.get("problem_type")
    if problem_type not in PROBLEM_TYPES:
        return fallback
    try:
        if problem_type == "modular_exponentiation":
            a = min(30, max(2, int(raw.get("a", fallback["a"]))))
            b = min(30, max(2, int(raw.get("b", fallback["b"]))))
            n = min(800, max(20, int(raw.get("n", fallback["n"]))))
            m = int(raw.get("m", fallback["m"]))
            if m not in MODULI:
                m = fallback["m"]
            return {
                "problem_type": problem_type,
                "a": a,
                "b": b,
                "n": n,
                "m": m,
                "problem": f"Find the remainder when {a}^{n} + {b}^{n} is divided by {m}.",
            }
        if problem_type == "linear_equation":
            a = min(25, max(2, int(raw.get("a", fallback["a"]))))
            x_true = min(30, max(-30, int(raw.get("x_true", fallback["x_true"]))))
            b = min(80, max(-80, int(raw.get("b", fallback["b"]))))
            c = a * x_true + b
            return {
                "problem_type": problem_type,
                "a": a,
                "b": b,
                "x_true": x_true,
                "c": c,
                "problem": f"Solve for x: {a}x + {b} = {c}.",
            }
        n = min(300, max(40, int(raw.get("N", fallback["N"]))))
        p = int(raw.get("p", fallback["p"]))
        q = int(raw.get("q", fallback["q"]))
        if p not in DIVISORS:
            p = fallback["p"]
        if q not in DIVISORS or q == p:
            q = next(value for value in DIVISORS if value != p)
        return {
            "problem_type": problem_type,
            "N": n,
            "p": p,
            "q": q,
            "problem": f"How many integers from 1 to {n} are divisible by {p} or {q}?",
        }
    except Exception:
        return fallback


def validate_problem_specs(raw_specs: List[dict]) -> List[dict]:
    fallback_specs = local_problem_specs()
    fallback_by_type = defaultdict(list)
    for spec in fallback_specs:
        fallback_by_type[spec["problem_type"]].append(spec)

    raw_by_type = defaultdict(list)
    for item in raw_specs:
        if isinstance(item, dict) and item.get("problem_type") in PROBLEM_TYPES:
            raw_by_type[item["problem_type"]].append(item)

    specs = []
    for problem_type in PROBLEM_TYPES:
        for i in range(10):
            fallback = dict(fallback_by_type[problem_type][i])
            raw = raw_by_type[problem_type][i] if i < len(raw_by_type[problem_type]) else fallback
            spec = sanitize_problem_spec(raw, fallback)
            specs.append(spec)

    RNG.shuffle(specs)
    for problem_id, spec in enumerate(specs):
        spec["problem_id"] = problem_id
        spec["true_answer"] = compute_true_answer(spec)
    return specs


def generate_problem_specs() -> List[dict]:
    if DRY_RUN:
        specs = local_problem_specs()
        for spec in specs:
            spec["true_answer"] = compute_true_answer(spec)
        return specs

    prompt = """
Generate exactly 30 structured math problem specs as valid JSON.
Return only JSON. Do not include markdown, explanations, or solutions.

Output schema:
{"problems": [...]}

Generate exactly:
- 10 modular_exponentiation
- 10 linear_equation
- 10 counting_cases

Schemas:
modular_exponentiation:
{"problem_id": int, "problem_type": "modular_exponentiation", "a": int, "b": int, "n": int, "m": int, "problem": "Find the remainder when a^n + b^n is divided by m."}
Constraints: a,b in 2..30, n in 20..800, m in [5,7,11,13,17,19,23]

linear_equation:
{"problem_id": int, "problem_type": "linear_equation", "a": int, "b": int, "x_true": int, "c": int, "problem": "Solve for x: ax + b = c."}
Constraints: a in 2..25, x_true in -30..30, b in -80..80, c = a*x_true + b

counting_cases:
{"problem_id": int, "problem_type": "counting_cases", "N": int, "p": int, "q": int, "problem": "How many integers from 1 to N are divisible by p or q?"}
Constraints: N in 40..300, p,q from [2,3,4,5,6,7,8,9,10,11,12], p != q
"""
    parsed = json_llm_call(prompt, kind="problem_generator")
    raw_specs = parsed.get("problems", []) if parsed else []
    return validate_problem_specs(raw_specs)


def fallback_strategies(problem_type: str) -> List[StrategySpec]:
    defaults = {
        "modular_exponentiation": [
            ("modular_pow", "modular_arithmetic", "Compute residues with modular exponentiation."),
            ("cycle_residues", "cycle_detection", "Find residue cycles and use exponent position."),
            ("direct_bruteforce", "brute_force", "Try direct computation or enumeration."),
        ],
        "linear_equation": [
            ("isolate_variable", "equation_transform", "Rearrange ax + b = c to solve for x."),
            ("search_integer_x", "brute_force", "Search plausible integer values for x."),
            ("sign_cases", "case_analysis", "Use simple sign/case reasoning."),
        ],
        "counting_cases": [
            ("include_exclude", "inclusion_exclusion", "Use N//p + N//q - N//lcm(p,q)."),
            ("enumerate_numbers", "brute_force", "Count all integers from 1 to N."),
            ("split_cases", "case_analysis", "Split into p-divisible and q-divisible cases."),
        ],
    }
    return [
        StrategySpec(strategy_id=sid, strategy_family=family, description=desc)
        for sid, family, desc in defaults[problem_type]
    ]


def propose_strategies(spec: dict) -> List[StrategySpec]:
    if DRY_RUN:
        return fallback_strategies(spec["problem_type"])

    prompt = f"""
Return only valid JSON. Propose exactly 3 reasoning strategies for this problem.

Problem type: {spec['problem_type']}
Problem: {spec['problem']}

Allowed strategy_family values:
{ALLOWED_STRATEGY_FAMILIES}

Output schema:
{{
  "strategies": [
    {{
      "strategy_id": "short_snake_case_name",
      "strategy_family": "one allowed family",
      "description": "short natural language strategy"
    }}
  ]
}}
"""
    parsed = json_llm_call(prompt, kind="strategy_proposer")
    raw_strategies = parsed.get("strategies", []) if parsed else []
    strategies = []
    for item in raw_strategies[:3]:
        if not isinstance(item, dict):
            continue
        strategy_id = re.sub(r"[^a-z0-9_]+", "_", str(item.get("strategy_id", "strategy")).lower()).strip("_")
        family = normalize_family(item.get("strategy_family", "other"))
        desc = str(item.get("description", family))[:300]
        strategies.append(StrategySpec(strategy_id=strategy_id or family, strategy_family=family, description=desc))
    if len(strategies) != 3:
        return fallback_strategies(spec["problem_type"])
    return strategies


def dry_run_solve(spec: dict, strategy: StrategySpec) -> str:
    family = strategy.strategy_family
    true_answer = compute_true_answer(spec)
    wrong_answer = true_answer + 1

    if spec["problem_type"] == "modular_exponentiation":
        correct_families = {"modular_arithmetic", "cycle_detection"}
    elif spec["problem_type"] == "linear_equation":
        correct_families = {"equation_transform", "brute_force"}
    else:
        correct_families = {"inclusion_exclusion", "brute_force"}

    answer = true_answer if family in correct_families else wrong_answer
    return (
        f"Using {strategy.strategy_id}: {strategy.description}\n"
        f"I follow the assigned strategy family {family}.\n"
        f"FINAL_ANSWER: {answer}"
    )


def solve_with_strategy(spec: dict, strategy: StrategySpec) -> str:
    if DRY_RUN:
        return dry_run_solve(spec, strategy)

    prompt = f"""
Problem: {spec['problem']}
Problem type: {spec['problem_type']}
Assigned strategy_id: {strategy.strategy_id}
Assigned strategy_family: {strategy.strategy_family}
Strategy description: {strategy.description}

You must follow the assigned strategy as much as possible.
If the strategy is inefficient or inappropriate, explicitly say so, but still attempt it.
Keep the reasoning concise, no more than 8 steps.
End with exactly:
FINAL_ANSWER: <integer>
"""
    return call_llm(prompt, kind="solver", temperature=0.2)


def direct_baseline_solve(spec: dict) -> str:
    if DRY_RUN:
        return f"Directly solving.\nFINAL_ANSWER: {compute_true_answer(spec)}"
    prompt = f"""
Solve this problem directly and concisely.
Problem type: {spec['problem_type']}
Problem: {spec['problem']}
End with exactly:
FINAL_ANSWER: <integer>
"""
    return call_llm(prompt, kind="direct_baseline", temperature=0.2)


def extract_final_answer(text: str) -> Optional[int]:
    match = re.search(r"FINAL_ANSWER\s*:\s*(-?\d+)", text or "", re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def normalize_weights(weights: dict, default_family: str) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in (weights or {}).items():
        family = normalize_family(key)
        if family not in ALLOWED_STRATEGY_FAMILIES:
            family = "other"
        cleaned[family] = cleaned.get(family, 0.0) + max(0.0, float(value))
    total = sum(cleaned.values())
    if total <= 0:
        return {default_family: 1.0}
    return {key: value / total for key, value in cleaned.items()}


def fallback_analysis(rollout: dict) -> dict:
    correct = bool(rollout["correct"])
    family = rollout["assigned_strategy_family"]
    return {
        "strategy_id": rollout["assigned_strategy_id"],
        "strategy_family": family,
        "strategy_compliance_score": 1.0,
        "actual_strategy_weights": {family: 1.0},
        "strategy_validity_score": 1.0 if correct else 0.3,
        "process_quality_score": 1.0 if correct else 0.3,
        "failure_type": "none" if correct else "other",
        "analysis_notes": "Heuristic fallback analysis.",
    }


def dry_run_analyze(spec: dict, rollouts: List[dict]) -> List[dict]:
    analyses = []
    for rollout in rollouts:
        family = rollout["assigned_strategy_family"]
        weights = {family: 1.0}
        if spec["problem_type"] == "modular_exponentiation" and family == "cycle_detection":
            weights = {"cycle_detection": 0.5, "modular_arithmetic": 0.5}
        analyses.append({
            "strategy_id": rollout["assigned_strategy_id"],
            "strategy_family": family,
            "strategy_compliance_score": 1.0,
            "actual_strategy_weights": weights,
            "strategy_validity_score": 1.0 if rollout["correct"] else 0.25,
            "process_quality_score": 1.0 if rollout["correct"] else 0.25,
            "failure_type": "none" if rollout["correct"] else "other",
            "analysis_notes": "Dry-run deterministic analyzer.",
        })
    return analyses


def analyze_rollouts(spec: dict, rollouts: List[dict]) -> List[dict]:
    if DRY_RUN:
        return dry_run_analyze(spec, rollouts)

    compact_rollouts = [
        {
            "strategy_id": item["assigned_strategy_id"],
            "strategy_family": item["assigned_strategy_family"],
            "strategy_description": item["strategy_description"],
            "rollout_text": item["raw_response"],
            "predicted_answer": item["predicted_answer"],
            "correct": item["correct"],
        }
        for item in rollouts
    ]
    prompt = f"""
Return only valid JSON.

Analyze all rollout responses for one problem.
Problem: {spec['problem']}
Problem type: {spec['problem_type']}
True answer: {spec['true_answer']}

Allowed strategy_family values: {ALLOWED_STRATEGY_FAMILIES}
Allowed failure_type values: {FAILURE_TYPES}

Rollouts:
{json.dumps(compact_rollouts, ensure_ascii=False)}

Output schema:
{{
  "analyses": [
    {{
      "strategy_id": "...",
      "strategy_family": "...",
      "strategy_compliance_score": float between 0 and 1,
      "actual_strategy_weights": {{"strategy_family_name": float}},
      "strategy_validity_score": float between 0 and 1,
      "process_quality_score": float between 0 and 1,
      "failure_type": "none | calculation_error | invalid_strategy | strategy_drift | final_answer_extraction_error | incomplete_reasoning | other",
      "analysis_notes": "short explanation"
    }}
  ]
}}

Rules:
- actual_strategy_weights must sum to 1.0.
- actual_strategy_weights keys must be allowed strategy_family values.
- Lower compliance if the solver drifted away from the assigned strategy.
- A correct answer with poor or irrelevant reasoning should not receive high process_quality_score.
- A wrong answer can still have high process_quality_score if only final arithmetic was minorly wrong.
"""
    parsed = json_llm_call(prompt, kind="analyzer")
    raw_analyses = parsed.get("analyses", []) if parsed else []
    by_id = {str(item.get("strategy_id")): item for item in raw_analyses if isinstance(item, dict)}

    analyses = []
    for rollout in rollouts:
        raw = by_id.get(rollout["assigned_strategy_id"])
        if raw is None:
            analyses.append(fallback_analysis(rollout))
            continue
        family = normalize_family(raw.get("strategy_family", rollout["assigned_strategy_family"]))
        failure_type = str(raw.get("failure_type", "other"))
        if failure_type not in FAILURE_TYPES:
            failure_type = "other"
        analyses.append({
            "strategy_id": rollout["assigned_strategy_id"],
            "strategy_family": family,
            "strategy_compliance_score": clamp01(raw.get("strategy_compliance_score", 0.5)),
            "actual_strategy_weights": normalize_weights(
                raw.get("actual_strategy_weights", {}),
                rollout["assigned_strategy_family"],
            ),
            "strategy_validity_score": clamp01(raw.get("strategy_validity_score", 0.3)),
            "process_quality_score": clamp01(raw.get("process_quality_score", 0.3)),
            "failure_type": failure_type,
            "analysis_notes": str(raw.get("analysis_notes", ""))[:500],
        })
    return analyses


def make_empty_belief_table() -> Dict[str, Dict[str, Belief]]:
    return {
        problem_type: {family: Belief() for family in ALLOWED_STRATEGY_FAMILIES}
        for problem_type in PROBLEM_TYPES
    }


def make_failure_counts() -> Dict[str, Dict[str, Counter]]:
    return {
        problem_type: {family: Counter() for family in ALLOWED_STRATEGY_FAMILIES}
        for problem_type in PROBLEM_TYPES
    }


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
        problem_type: {family: belief_snapshot(belief) for family, belief in family_table.items()}
        for problem_type, family_table in table.items()
    }


def nested_failure_snapshot(failure_counts: Dict[str, Dict[str, Counter]]) -> dict:
    return {
        problem_type: {family: dict(counter) for family, counter in family_table.items()}
        for problem_type, family_table in failure_counts.items()
    }


def update_beliefs(
    problem_type: str,
    rollout: dict,
    analysis: dict,
    assigned_success_beliefs: Dict[str, Dict[str, Belief]],
    actual_success_beliefs: Dict[str, Dict[str, Belief]],
    validity_beliefs: Dict[str, Dict[str, Belief]],
    failure_counts: Dict[str, Dict[str, Counter]],
) -> None:
    assigned_family = rollout["assigned_strategy_family"]
    compliance = clamp01(analysis["strategy_compliance_score"])
    correct = bool(rollout["correct"])

    assigned_success_beliefs[problem_type][assigned_family].update_assigned_success(correct, compliance)

    weights = normalize_weights(analysis["actual_strategy_weights"], assigned_family)
    for family, weight in weights.items():
        actual_success_beliefs[problem_type][family].update_actual_success(correct, weight)

    validity_signal = clamp01(analysis["strategy_validity_score"]) * clamp01(analysis["process_quality_score"])
    assigned_validity = validity_beliefs[problem_type][assigned_family]
    assigned_validity.update_validity(
        validity_signal * compliance,
        (1.0 - validity_signal) * compliance,
        1.0,
    )
    for family, weight in weights.items():
        validity_beliefs[problem_type][family].update_validity(
            weight * validity_signal,
            weight * (1.0 - validity_signal),
            weight,
        )

    failure_type = analysis.get("failure_type", "other")
    if failure_type != "none":
        failure_counts[problem_type][assigned_family][failure_type] += 1


def rollout_with_strategy(spec: dict, strategy: StrategySpec) -> dict:
    raw_response = solve_with_strategy(spec, strategy)
    predicted_answer = extract_final_answer(raw_response)
    correct = predicted_answer == spec["true_answer"] if predicted_answer is not None else False
    return {
        "problem_id": spec["problem_id"],
        "problem_type": spec["problem_type"],
        "problem": spec["problem"],
        "true_answer": spec["true_answer"],
        "assigned_strategy_id": strategy.strategy_id,
        "assigned_strategy_family": strategy.strategy_family,
        "strategy_description": strategy.description,
        "raw_response": raw_response,
        "predicted_answer": predicted_answer,
        "correct": correct,
        "analysis": None,
    }


def attach_analyses(rollouts: List[dict], analyses: List[dict]) -> List[dict]:
    by_id = {analysis["strategy_id"]: analysis for analysis in analyses}
    attached = []
    for rollout in rollouts:
        item = dict(rollout)
        item["analysis"] = by_id.get(item["assigned_strategy_id"], fallback_analysis(item))
        attached.append(item)
    return attached


def choose_belief_guided_strategy(
    strategies: List[StrategySpec],
    problem_type: str,
    actual_success_beliefs: Dict[str, Dict[str, Belief]],
    validity_beliefs: Dict[str, Dict[str, Belief]],
) -> StrategySpec:
    evidence = [
        actual_success_beliefs[problem_type][strategy.strategy_family].total_trials
        for strategy in strategies
    ]
    if max(evidence) == 0:
        return RNG.choice(strategies)
    return max(
        strategies,
        key=lambda strategy: (
            actual_success_beliefs[problem_type][strategy.strategy_family].usability_belief,
            validity_beliefs[problem_type][strategy.strategy_family].success_belief,
            strategy.strategy_family,
        ),
    )


def accuracy(logs: List[dict], key: str) -> float:
    if not logs:
        return 0.0
    return sum(1 for item in logs if item[key]["correct"]) / len(logs)


def per_type_accuracy(logs: List[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for item in logs:
        grouped[item["problem_type"]].append(item)
    return {problem_type: accuracy(items, key) for problem_type, items in grouped.items()}


def rank_beliefs(table: Dict[str, Dict[str, Belief]]) -> dict:
    rankings = {}
    for problem_type, family_table in table.items():
        ordered = sorted(
            family_table.items(),
            key=lambda item: item[1].usability_belief,
            reverse=True,
        )
        rankings[problem_type] = [
            {"strategy_family": family, **belief_snapshot(belief)}
            for family, belief in ordered
        ]
    return rankings


def run_experiment() -> dict:
    problem_specs = generate_problem_specs()
    phase_a_specs = problem_specs[:PHASE_A_SIZE]
    phase_b_specs = problem_specs[PHASE_A_SIZE:]

    assigned_success_beliefs = make_empty_belief_table()
    actual_success_beliefs = make_empty_belief_table()
    validity_beliefs = make_empty_belief_table()
    failure_counts = make_failure_counts()

    phase_a_logs = []
    phase_b_logs = []
    selected_counts = {
        "random": Counter(),
        "belief_guided": Counter(),
    }
    direct_logs = []

    for spec in phase_a_specs:
        strategies = propose_strategies(spec)
        rollouts = [rollout_with_strategy(spec, strategy) for strategy in strategies]
        analyses = analyze_rollouts(spec, rollouts)
        rollouts = attach_analyses(rollouts, analyses)
        for rollout in rollouts:
            update_beliefs(
                spec["problem_type"],
                rollout,
                rollout["analysis"],
                assigned_success_beliefs,
                actual_success_beliefs,
                validity_beliefs,
                failure_counts,
            )
        phase_a_logs.append({
            "problem_id": spec["problem_id"],
            "problem_type": spec["problem_type"],
            "problem": spec["problem"],
            "true_answer": spec["true_answer"],
            "strategies": [asdict(strategy) for strategy in strategies],
            "rollouts": rollouts,
        })

    for spec in phase_b_specs:
        strategies = propose_strategies(spec)
        random_strategy = RNG.choice(strategies)
        guided_strategy = choose_belief_guided_strategy(
            strategies,
            spec["problem_type"],
            actual_success_beliefs,
            validity_beliefs,
        )
        random_rollout = rollout_with_strategy(spec, random_strategy)
        guided_rollout = rollout_with_strategy(spec, guided_strategy)
        analyses = analyze_rollouts(spec, [random_rollout, guided_rollout])
        random_rollout, guided_rollout = attach_analyses([random_rollout, guided_rollout], analyses)

        for rollout in [random_rollout, guided_rollout]:
            update_beliefs(
                spec["problem_type"],
                rollout,
                rollout["analysis"],
                assigned_success_beliefs,
                actual_success_beliefs,
                validity_beliefs,
                failure_counts,
            )

        selected_counts["random"][(spec["problem_type"], random_strategy.strategy_family)] += 1
        selected_counts["belief_guided"][(spec["problem_type"], guided_strategy.strategy_family)] += 1

        item = {
            "problem_id": spec["problem_id"],
            "problem_type": spec["problem_type"],
            "problem": spec["problem"],
            "true_answer": spec["true_answer"],
            "strategies": [asdict(strategy) for strategy in strategies],
            "random_strategy": asdict(random_strategy),
            "belief_guided_strategy": asdict(guided_strategy),
            "random_rollout": random_rollout,
            "belief_guided_rollout": guided_rollout,
        }

        if RUN_DIRECT_BASELINE:
            raw = direct_baseline_solve(spec)
            predicted = extract_final_answer(raw)
            direct_item = {
                "raw_response": raw,
                "predicted_answer": predicted,
                "correct": predicted == spec["true_answer"] if predicted is not None else False,
            }
            item["direct_baseline_rollout"] = direct_item
            direct_logs.append({"problem_type": spec["problem_type"], "direct": direct_item})

        phase_b_logs.append(item)

    result = {
        "model": MODEL,
        "dry_run": DRY_RUN,
        "num_problems": len(problem_specs),
        "phase_a_size": len(phase_a_specs),
        "phase_b_size": len(phase_b_specs),
        "api_calls": api_calls.snapshot(),
        "phase_b_random_accuracy": accuracy(phase_b_logs, "random_rollout"),
        "phase_b_belief_guided_accuracy": accuracy(phase_b_logs, "belief_guided_rollout"),
        "phase_b_accuracy_by_problem_type": {
            "random": per_type_accuracy(phase_b_logs, "random_rollout"),
            "belief_guided": per_type_accuracy(phase_b_logs, "belief_guided_rollout"),
        },
        "direct_baseline_accuracy": accuracy(direct_logs, "direct") if RUN_DIRECT_BASELINE else None,
        "final_assigned_success_beliefs": nested_belief_snapshot(assigned_success_beliefs),
        "final_actual_success_beliefs": nested_belief_snapshot(actual_success_beliefs),
        "final_validity_beliefs": nested_belief_snapshot(validity_beliefs),
        "final_assigned_success_rankings": rank_beliefs(assigned_success_beliefs),
        "final_actual_success_rankings": rank_beliefs(actual_success_beliefs),
        "final_validity_rankings": rank_beliefs(validity_beliefs),
        "failure_counts": nested_failure_snapshot(failure_counts),
        "selected_strategy_families": {
            "random": counter_to_nested_dict(selected_counts["random"]),
            "belief_guided": counter_to_nested_dict(selected_counts["belief_guided"]),
        },
        "problem_specs": problem_specs,
        "phase_a_logs": phase_a_logs,
        "phase_b_logs": phase_b_logs,
    }
    return result


def counter_to_nested_dict(counter: Counter) -> dict:
    result = {problem_type: {} for problem_type in PROBLEM_TYPES}
    for (problem_type, family), count in counter.items():
        result[problem_type][family] = count
    return result


def print_ranking(title: str, rankings: dict, top_k: int = 5) -> None:
    print(title)
    for problem_type in PROBLEM_TYPES:
        print(f"[{problem_type}]")
        for item in rankings[problem_type][:top_k]:
            print(
                f"  {item['strategy_family']:<22} "
                f"success={item['success_belief']:.3f} "
                f"usability={item['usability_belief']:.3f}"
            )
    print()


def print_summary(result: dict) -> None:
    print(f"Model: {result['model']} | DRY_RUN={result['dry_run']}")
    print(f"API calls: {result['api_calls']}")
    print(f"Phase B random accuracy:        {result['phase_b_random_accuracy']:.3f}")
    print(f"Phase B belief-guided accuracy: {result['phase_b_belief_guided_accuracy']:.3f}")
    print(f"Phase B accuracy by problem_type: {json.dumps(result['phase_b_accuracy_by_problem_type'], indent=2)}")
    if RUN_DIRECT_BASELINE:
        print(f"Direct baseline accuracy: {result['direct_baseline_accuracy']:.3f}")
    print()
    print_ranking("Final actual_success_belief ranking", result["final_actual_success_rankings"])
    print_ranking("Final assigned_success_belief ranking", result["final_assigned_success_rankings"])
    print_ranking("Final validity_belief ranking", result["final_validity_rankings"])
    print("Failure counts")
    print(json.dumps(result["failure_counts"], indent=2))
    print()
    print("Selected strategy families in Phase B")
    print(json.dumps(result["selected_strategy_families"], indent=2))
    print()
    print("Sample logs")
    print("-" * 60)
    for item in result["phase_b_logs"][:6]:
        print(
            f"{item['problem_type']:<24} id={item['problem_id']:<2} "
            f"true={item['true_answer']:<4} "
            f"random={item['random_strategy']['strategy_family']}:{item['random_rollout']['correct']} "
            f"guided={item['belief_guided_strategy']['strategy_family']}:{item['belief_guided_rollout']['correct']} | "
            f"{item['problem']}"
        )


def main() -> None:
    result = run_experiment()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result)
    print()
    print(f"Saved JSON result to: {RESULT_PATH}")
    if DRY_RUN:
        print()
        print("Real run command:")
        print("python llm_self_data_bayesian_experiment.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
