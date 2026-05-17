#!/usr/bin/env python3
"""Minimal strategy-belief self-data reasoning demo.

Default mode uses fixed mock rollouts so the core belief update can be tested
without any API dependency. Use --backend hf or --backend openai for real
model rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


PROBLEM = "Find the remainder when 3^2025 + 5^2025 is divided by 7."
PROBLEM_TYPE = "modular_exponentiation"
STRATEGIES = [
    "brute_force",
    "modular_arithmetic",
    "cycle_detection",
    "case_analysis",
]
STRATEGY_INSTRUCTIONS = {
    "brute_force": (
        "Solve using ONLY the brute_force strategy. Do not use Fermat's Little "
        "Theorem, Euler's theorem, modular cycles, cycle detection, or any "
        "number-theoretic shortcut. If brute force is inefficient, still attempt "
        "direct computation/enumeration and explain the limitation."
    ),
    "modular_arithmetic": (
        "Solve using ONLY modular_arithmetic: reduce expressions modulo 7 and "
        "use congruence manipulations. Do not present this as brute force or "
        "case analysis."
    ),
    "cycle_detection": (
        "Solve using ONLY cycle_detection: explicitly list residues until the "
        "cycle repeats, then use the cycle position. Do not invoke Fermat's "
        "Little Theorem or Euler's theorem as a shortcut."
    ),
    "case_analysis": (
        "Solve using ONLY case_analysis: split the problem into explicit cases "
        "and reason from those cases. Do not switch to Fermat's Little Theorem, "
        "Euler's theorem, modular cycles, or cycle detection."
    ),
}
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Use .env or the shell for local testing. Do not hardcode secrets here.
OPENAI_API_KEY = ""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Belief:
    alpha: float = 1.0
    beta: float = 1.0
    strategy_drift: float = 0.0
    total_trials: int = 0

    @property
    def success_belief(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def value(self) -> float:
        return self.success_belief

    @property
    def avg_strategy_drift(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.strategy_drift / self.total_trials

    @property
    def usability_belief(self) -> float:
        return self.success_belief * (1.0 - self.avg_strategy_drift)

    def update(self, correct: bool, strategy_compliance_score: float) -> None:
        score = max(0.0, min(1.0, strategy_compliance_score))
        drift_score = 1.0 - score
        self.total_trials += 1

        if correct:
            self.alpha += score
        else:
            self.beta += score

        self.strategy_drift += drift_score

    def update_actual_credit(self, correct: bool, weight: float) -> None:
        credit = max(0.0, min(1.0, weight))
        if correct:
            self.alpha += credit
        else:
            self.beta += credit


@dataclass
class RolloutResult:
    assigned_strategy: str
    response: str
    extracted_answer: Optional[int]
    correct: bool
    strategy_compliance_score: float
    actual_strategy: str
    actual_strategy_weights: Dict[str, float]
    strategy_drift_score: float


def compute_true_answer() -> int:
    return (pow(3, 2025, 7) + pow(5, 2025, 7)) % 7


def extract_final_answer(text: str) -> Optional[int]:
    final_line = re.search(r"FINAL_ANSWER\s*:\s*(-?\d+)", text, re.IGNORECASE)
    if final_line:
        return int(final_line.group(1))

    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def extract_used_strategy(text: str) -> Optional[str]:
    used_line = re.search(r"USED_STRATEGY\s*:\s*(.+)", text, re.IGNORECASE)
    if not used_line:
        return None
    return used_line.group(1).strip().splitlines()[0].strip()


def infer_actual_strategy(text: str) -> str:
    lower = text.lower()
    uses_fermat = "fermat" in lower
    uses_euler = "euler" in lower
    uses_cycle = any(term in lower for term in ["cycle", "repeat every", "period"])
    uses_modular = any(term in lower for term in ["modulo", " mod ", "\\mod", "congru"])
    uses_brute = any(term in lower for term in ["brute force", "direct computation", "enumerat"])
    uses_case = any(term in lower for term in ["case analysis", "case-by-case", "split into cases"])

    labels = []
    if uses_fermat:
        labels.append("fermat_little_theorem")
    if uses_euler:
        labels.append("euler_theorem")
    if uses_cycle:
        labels.append("cycle_detection")
    if uses_modular:
        labels.append("modular_arithmetic")
    if uses_brute:
        labels.append("brute_force")
    if uses_case:
        labels.append("case_analysis")

    return " / ".join(dict.fromkeys(labels)) if labels else "unknown"


def analyze_strategy_compliance(assigned_strategy: str, response: str) -> tuple[float, str]:
    reported_strategy = extract_used_strategy(response)
    inferred_strategy = infer_actual_strategy(response)
    lower = response.lower()

    uses_fermat = "fermat" in lower
    uses_euler = "euler" in lower
    uses_cycle = any(term in lower for term in ["cycle", "repeat every", "period"])
    uses_modular = any(term in lower for term in ["modulo", " mod ", "\\mod", "congru"])
    uses_brute = any(term in lower for term in ["brute force", "direct computation", "enumerat"])
    uses_case = any(term in lower for term in ["case analysis", "case-by-case", "split into cases"])

    actual_strategy = inferred_strategy
    if reported_strategy and inferred_strategy == "unknown":
        actual_strategy = reported_strategy

    score = 0.0
    if assigned_strategy == "brute_force":
        if uses_brute:
            score += 0.7
        if not (uses_fermat or uses_euler or uses_cycle or uses_modular):
            score += 0.3
    elif assigned_strategy == "modular_arithmetic":
        if uses_modular:
            score += 0.7
        if not (uses_cycle or uses_brute):
            score += 0.3
    elif assigned_strategy == "cycle_detection":
        if uses_cycle:
            score += 0.7
        if not (uses_fermat or uses_euler or uses_brute):
            score += 0.3
    elif assigned_strategy == "case_analysis":
        if uses_case:
            score += 0.7
        if not (uses_fermat or uses_euler or uses_cycle or uses_modular):
            score += 0.3

    if reported_strategy:
        normalized_report = reported_strategy.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_report == assigned_strategy and inferred_strategy == "unknown":
            score = 1.0
            actual_strategy = assigned_strategy

    return max(0.0, min(1.0, score)), actual_strategy


def actual_strategy_weights(
    assigned_strategy: str,
    actual_strategy: str,
) -> Dict[str, float]:
    weights = {strategy: 0.0 for strategy in STRATEGIES}

    labels = [
        label.strip()
        for label in actual_strategy.split("/")
        if label.strip() in weights
    ]
    if not labels:
        labels = [assigned_strategy]

    unique_labels = list(dict.fromkeys(labels))
    share = 1.0 / len(unique_labels)
    for label in unique_labels:
        weights[label] += share

    return {
        strategy: round(weight, 4)
        for strategy, weight in weights.items()
        if weight > 0.0
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
        "belief": belief.success_belief,
    }


def mock_rollout(strategy: str) -> str:
    responses = {
        "brute_force": (
            "Using brute force directly is impractical for exponent 2025. "
            "I will attempt direct computation but stop due to the large exponent.\n"
            "USED_STRATEGY: brute_force\nFINAL_ANSWER: 1"
        ),
        "modular_arithmetic": (
            "Modulo 7, 3^6 = 1 and 5^6 = 1. Since 2025 mod 6 = 3, "
            "3^2025 = 3^3 = 27 = 6 mod 7 and 5^2025 = 5^3 = 125 = 6 mod 7. "
            "The sum is 12 = 5 mod 7.\nUSED_STRATEGY: modular_arithmetic\n"
            "FINAL_ANSWER: 5"
        ),
        "cycle_detection": (
            "The powers of 3 mod 7 repeat every 6, and the powers of 5 mod 7 "
            "also repeat every 6. 2025 leaves remainder 3 after division by 6. "
            "Thus 3^2025 = 6 and 5^2025 = 6 modulo 7, so the result is 5.\n"
            "USED_STRATEGY: cycle_detection\nFINAL_ANSWER: 5"
        ),
        "case_analysis": (
            "Consider odd and even exponents separately. Since 2025 is odd, "
            "both terms should preserve their bases in a simple way, giving "
            "3 + 5 = 8 = 1.\nUSED_STRATEGY: case_analysis\nFINAL_ANSWER: 1"
        ),
    }
    return responses[strategy]


def build_strategy_prompt(strategy: str) -> str:
    return (
        f"Problem: {PROBLEM}\n"
        f"Strategy: {strategy}\n\n"
        f"{STRATEGY_INSTRUCTIONS[strategy]}\n\n"
        "Do not silently switch strategies. If you drift from the requested "
        "strategy, report the actual strategy you used in USED_STRATEGY.\n\n"
        "End with exactly these two lines:\n"
        "USED_STRATEGY: <actual strategy used>\n"
        "FINAL_ANSWER: <integer>"
    )


def openai_rollout(strategy: str, model: str) -> str:
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is required. Fill OPENAI_API_KEY in this file "
            "or set OPENAI_API_KEY in .env/the shell."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = build_strategy_prompt(strategy)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful mathematical reasoning agent.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


class HFRolloutEngine:
    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def rollout(self, strategy: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a careful mathematical reasoning agent.",
            },
            {"role": "user", "content": build_strategy_prompt(strategy)},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def run(backend: str, model: str, max_new_tokens: int, output_path: Path) -> dict:
    true_answer = compute_true_answer()
    assigned_beliefs: Dict[str, Belief] = {strategy: Belief() for strategy in STRATEGIES}
    actual_beliefs: Dict[str, Belief] = {strategy: Belief() for strategy in STRATEGIES}
    initial_assigned_beliefs = {
        strategy: belief_snapshot(belief)
        for strategy, belief in assigned_beliefs.items()
    }
    initial_actual_beliefs = {
        strategy: belief_snapshot(belief)
        for strategy, belief in actual_beliefs.items()
    }

    hf_engine = HFRolloutEngine(model, max_new_tokens) if backend == "hf" else None
    rollouts = []
    for strategy in STRATEGIES:
        if backend == "mock":
            response = mock_rollout(strategy)
        elif backend == "openai":
            response = openai_rollout(strategy, model)
        elif backend == "hf":
            assert hf_engine is not None
            response = hf_engine.rollout(strategy)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        extracted_answer = extract_final_answer(response)
        correct = extracted_answer == true_answer
        strategy_compliance_score, actual_strategy = analyze_strategy_compliance(
            strategy, response
        )
        weights = actual_strategy_weights(
            strategy,
            actual_strategy,
        )
        assigned_beliefs[strategy].update(correct, strategy_compliance_score)
        for actual_strategy_name, weight in weights.items():
            actual_beliefs[actual_strategy_name].update_actual_credit(correct, weight)

        rollouts.append(
            RolloutResult(
                assigned_strategy=strategy,
                response=response,
                extracted_answer=extracted_answer,
                correct=correct,
                strategy_compliance_score=strategy_compliance_score,
                actual_strategy=actual_strategy,
                actual_strategy_weights=weights,
                strategy_drift_score=1.0 - strategy_compliance_score,
            )
        )

    final_assigned_beliefs = {
        strategy: belief_snapshot(belief)
        for strategy, belief in assigned_beliefs.items()
    }
    final_actual_beliefs = {
        strategy: belief_snapshot(belief)
        for strategy, belief in actual_beliefs.items()
    }

    result = {
        "problem": PROBLEM,
        "problem_type": PROBLEM_TYPE,
        "backend": backend,
        "model": model if backend != "mock" else None,
        "true_answer": true_answer,
        "strategies": STRATEGIES,
        "initial_beliefs": initial_assigned_beliefs,
        "initial_assigned_beliefs": initial_assigned_beliefs,
        "initial_actual_beliefs": initial_actual_beliefs,
        "rollouts": [asdict(rollout) for rollout in rollouts],
        "final_beliefs": final_assigned_beliefs,
        "final_assigned_beliefs": final_assigned_beliefs,
        "final_actual_beliefs": final_actual_beliefs,
        "total_strategy_drift": sum(
            rollout.strategy_drift_score for rollout in rollouts
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def print_summary(result: dict) -> None:
    print(f"Problem: {result['problem']}")
    print(f"True answer: {result['true_answer']}")
    print()
    print(
        "assigned_strategy     answer  correct  score  drift  "
        "success  usability  actual"
    )
    print("-" * 104)
    for rollout in result["rollouts"]:
        strategy = rollout["assigned_strategy"]
        belief = result["final_assigned_beliefs"][strategy]
        actual_strategy = rollout["actual_strategy"]
        if len(actual_strategy) > 28:
            actual_strategy = actual_strategy[:25] + "..."
        print(
            f"{strategy:<21} "
            f"{str(rollout['extracted_answer']):<7} "
            f"{str(rollout['correct']):<8} "
            f"{rollout['strategy_compliance_score']:<5.2f} "
            f"{belief['strategy_drift']:<5.2f} "
            f"{belief['success_belief']:<8.2f} "
            f"{belief['usability_belief']:<9.2f} "
            f"{actual_strategy}"
        )

    print()
    print("actual_strategy       alpha  beta  success  usability")
    print("-" * 58)
    for strategy, belief in result["final_actual_beliefs"].items():
        print(
            f"{strategy:<21} "
            f"{belief['alpha']:<5.2f} "
            f"{belief['beta']:<4.2f} "
            f"{belief['success_belief']:<8.2f} "
            f"{belief['usability_belief']:<9.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["mock", "hf", "openai"],
        default="mock",
        help="Rollout backend.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Backward-compatible alias for --backend openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name for hf/openai backends. Defaults to "
            f"{DEFAULT_HF_MODEL} for hf and {DEFAULT_OPENAI_MODEL} for openai."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        default="results/minimal_belief_demo.json",
        help="Path for JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(Path(".env"))
    backend = "openai" if args.use_openai else args.backend
    if args.model is not None:
        model = args.model
    elif backend == "openai":
        model = DEFAULT_OPENAI_MODEL
    else:
        model = DEFAULT_HF_MODEL

    result = run(
        backend=backend,
        model=model,
        max_new_tokens=args.max_new_tokens,
        output_path=Path(args.output),
    )
    print_summary(result)


if __name__ == "__main__":
    main()
