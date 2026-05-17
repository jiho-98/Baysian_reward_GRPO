#!/usr/bin/env python3
"""Benchmark-based Qwen Bayesian strategy belief experiment on MATH-500."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import get_close_matches
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL = os.getenv("QWEN_MODEL", os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
DEFAULT_BENCHMARK = "HuggingFaceH4/MATH-500"
RESULT_PATH = Path("results/math500_qwen_belief_experiment_low_process.json")
SAMPLE_PATH = Path("results/math500_50_sample.json")
RAW_JSONL_CACHE = Path("results/math500_test_raw.jsonl")
PROGRESS_LOG = os.getenv("PROGRESS_LOG", "1") == "1"

ALLOWED_STRATEGY_FAMILIES = [
    "algebraic_transformation",
    "modular_reasoning",
    "counting_formula",
    "case_analysis",
    "direct_computation",
]
LIKERT_TO_SCORE = {
    1: 0.00,
    2: 0.25,
    3: 0.50,
    4: 0.75,
    5: 1.00,
}


@dataclass
class Strategy:
    strategy_id: str
    strategy_family: str
    description: str


@dataclass
class BeliefState:
    alpha: float = 1.0
    beta: float = 1.0
    total_trials: int = 0

    @property
    def belief(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ModelCallCounter:
    def __init__(self) -> None:
        self.counts = Counter()

    def add(self, kind: str) -> None:
        self.counts[kind] += 1

    def snapshot(self) -> dict:
        return {
            "total": sum(self.counts.values()),
            "strategy_proposer": self.counts["strategy_proposer"],
            "solver": self.counts["solver"],
            "judge": self.counts["judge"],
        }


class BayesianController:
    def __init__(self, name: str, problem_types: List[str]) -> None:
        self.name = name
        self.problem_types = list(problem_types)
        self.strategy_belief = {
            problem_type: {
                family: BeliefState() for family in ALLOWED_STRATEGY_FAMILIES
            }
            for problem_type in self.problem_types
        }

    def snapshot(self, problem_type: str, strategy_family: str) -> dict:
        state = self.strategy_belief[problem_type][strategy_family]
        return {
            "alpha": state.alpha,
            "beta": state.beta,
            "total_trials": state.total_trials,
            "belief": state.belief,
        }

    def update(self, problem_type: str, strategy_family: str, reward: float) -> None:
        reward = clamp01(reward)
        state = self.strategy_belief[problem_type][strategy_family]
        state.alpha += reward
        state.beta += 1.0 - reward
        state.total_trials += 1

    def select_strategy(
        self,
        problem_type: str,
        strategies: List[Strategy],
        rng: random.Random,
    ) -> Strategy:
        scores = []
        for strategy in strategies:
            belief = self.strategy_belief[problem_type][strategy.strategy_family].belief
            scores.append((belief, strategy))
        best_belief = max(score for score, _ in scores)
        best = [strategy for score, strategy in scores if score == best_belief]
        if len(best) == 1:
            return best[0]
        return rng.choice(best)

    def ranking(self) -> dict:
        output = {}
        for problem_type, rows in self.strategy_belief.items():
            output[problem_type] = [
                {
                    "strategy_family": family,
                    "alpha": state.alpha,
                    "beta": state.beta,
                    "total_trials": state.total_trials,
                    "belief": state.belief,
                }
                for family, state in sorted(
                    rows.items(),
                    key=lambda item: (
                        item[1].belief,
                        item[1].total_trials,
                        item[0],
                    ),
                    reverse=True,
                )
            ]
        return output


class QwenEngine:
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing model dependencies. Install at least: pip install torch transformers"
            ) from exc

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("torch device =", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = {"": 0}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.model_device = next(self.model.parameters()).device
        print("model parameter device =", self.model_device)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer([rendered], return_tensors="pt").to(self.device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = max(0.01, temperature)
            generate_kwargs["top_p"] = 0.95

        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **generate_kwargs)
        output_ids = generated[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


_ENGINE: Optional[QwenEngine] = None
MODEL_CALLS = ModelCallCounter()


def progress(message: str) -> None:
    if PROGRESS_LOG:
        print(f"[progress] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen benchmark-based Bayesian strategy belief experiment on MATH-500."
    )
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--phase_a_size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens_solver", type=int, default=768)
    parser.add_argument("--max_new_tokens_judge", type=int, default=256)
    parser.add_argument("--max_new_tokens_strategy", type=int, default=256)
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK)
    args = parser.parse_args()
    if args.num_problems <= 0:
        raise SystemExit("--num_problems must be positive.")
    if args.phase_a_size <= 0:
        raise SystemExit("--phase_a_size must be positive.")
    if args.phase_a_size >= args.num_problems:
        raise SystemExit("--phase_a_size must be smaller than --num_problems.")
    return args


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def safe_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.find("{") : cleaned.rfind("}") + 1])
    if "[" in cleaned and "]" in cleaned:
        candidates.append(cleaned[cleaned.find("[") : cleaned.rfind("]") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except json.JSONDecodeError:
            continue
    return None


def get_engine() -> QwenEngine:
    global _ENGINE
    if _ENGINE is None:
        progress(f"Loading Qwen model with backend=hf_qwen: {MODEL}")
        _ENGINE = QwenEngine(MODEL)
        progress("Qwen model loaded.")
    return _ENGINE


def ask_model(prompt: str, kind: str, max_new_tokens: int, temperature: float) -> str:
    MODEL_CALLS.add(kind)
    return get_engine().generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def ask_json(
    prompt: str,
    kind: str,
    max_new_tokens: int,
    temperature: float,
    retries: int = 2,
) -> Optional[dict]:
    for attempt in range(retries + 1):
        parsed = safe_json(ask_model(prompt, kind, max_new_tokens, temperature))
        if parsed is not None:
            return parsed
        time.sleep(0.5 * (attempt + 1))
    return None


def standardize_problem_item(row: dict, problem_id: int, benchmark: str) -> dict:
    benchmark_name = benchmark.split("/")[-1]
    return {
        "problem_id": problem_id,
        "benchmark": benchmark_name,
        "unique_id": row.get("unique_id"),
        "problem_type": row.get("subject", "math"),
        "level": row.get("level"),
        "problem": row["problem"],
        "gold_answer": row["answer"],
        "solution": row.get("solution"),
    }


def raw_jsonl_cache_path(benchmark: str) -> Path:
    if benchmark == DEFAULT_BENCHMARK:
        return RAW_JSONL_CACHE
    safe_name = benchmark.lower().replace("/", "__")
    return Path(f"results/{safe_name}_test_raw.jsonl")


def load_with_datasets(benchmark: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        progress("datasets library is not installed. Trying raw jsonl fallback.")
        return None, None

    try:
        ds = load_dataset(benchmark, split="test")
        return [dict(row) for row in ds], "datasets"
    except Exception as exc:
        progress(f"datasets load failed: {exc}")
        return None, None


def load_raw_jsonl_rows(benchmark: str) -> Tuple[List[dict], str]:
    cache_path = raw_jsonl_cache_path(benchmark)
    if cache_path.exists():
        progress(f"Loading cached raw jsonl from {cache_path}")
        rows = [
            json.loads(line)
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, f"cached_raw_jsonl:{cache_path}"

    urls = [
        f"https://huggingface.co/datasets/{benchmark}/resolve/main/test.jsonl",
        f"https://huggingface.co/datasets/{benchmark}/resolve/main/test.jsonl?download=true",
    ]
    last_error = None
    for url in urls:
        try:
            progress(f"Trying raw jsonl fallback: {url}")
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read().decode("utf-8")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(payload, encoding="utf-8")
            rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
            return rows, f"raw_jsonl_download:{url}"
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to load benchmark from datasets and raw jsonl fallback. "
        "Install the datasets library with: pip install datasets"
    ) from last_error


def load_benchmark_problems(num_problems: int, seed: int, benchmark: str) -> Tuple[List[dict], str]:
    rows, source = load_with_datasets(benchmark)
    if rows is None:
        rows, source = load_raw_jsonl_rows(benchmark)

    if len(rows) < num_problems:
        raise RuntimeError(
            f"Benchmark split only has {len(rows)} rows, but {num_problems} were requested."
        )

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(rows)), num_problems)
    sampled = [
        standardize_problem_item(dict(rows[row_index]), problem_id=problem_id, benchmark=benchmark)
        for problem_id, row_index in enumerate(sampled_indices)
    ]
    return sampled, source


def sanitize_strategy_id(value: Any, fallback: str) -> str:
    strategy_id = re.sub(r"[^a-z0-9_]+", "_", str(value or fallback).lower()).strip("_")
    return strategy_id or fallback


def map_to_allowed_family(raw_family: Any, description: str = "") -> str:
    family = str(raw_family or "").strip().lower().replace("-", "_").replace(" ", "_")
    if family in ALLOWED_STRATEGY_FAMILIES:
        return family

    alias_groups = {
        "algebraic_transformation": [
            "algebra",
            "equation",
            "transform",
            "substitution",
            "rewrite",
            "simplify",
            "factor",
            "manipulation",
        ],
        "modular_reasoning": [
            "mod",
            "modular",
            "number_theory",
            "residue",
            "divisibility",
            "remainder",
            "parity",
            "congruence",
            "cycle",
        ],
        "counting_formula": [
            "count",
            "combinatorics",
            "probability",
            "inclusion",
            "exclusion",
            "binomial",
            "formula",
        ],
        "case_analysis": [
            "case",
            "casework",
            "split",
            "partition",
            "scenario",
        ],
        "direct_computation": [
            "direct",
            "compute",
            "calculation",
            "brute",
            "enumerate",
            "numeric",
        ],
    }
    haystack = f"{family} {description}".lower()
    for allowed_family, keywords in alias_groups.items():
        if any(keyword in haystack for keyword in keywords):
            return allowed_family

    close = get_close_matches(family, ALLOWED_STRATEGY_FAMILIES, n=1, cutoff=0.0)
    return close[0] if close else "direct_computation"


def fallback_strategies(problem_item: dict) -> List[Strategy]:
    problem_type = problem_item["problem_type"].lower()
    if "counting" in problem_type or "probability" in problem_type:
        return [
            Strategy(
                "counting_formula_plan",
                "counting_formula",
                "Set up the relevant counting formula or combinatorial expression directly.",
            ),
            Strategy(
                "case_split_count",
                "case_analysis",
                "Split the problem into disjoint cases and count each case carefully.",
            ),
            Strategy(
                "direct_small_enumeration",
                "direct_computation",
                "Compute the required quantity directly, checking concrete values when feasible.",
            ),
        ]
    if "number theory" in problem_type:
        return [
            Strategy(
                "modular_residue_method",
                "modular_reasoning",
                "Use modular residues or divisibility structure to simplify the problem.",
            ),
            Strategy(
                "algebraic_rewrite",
                "algebraic_transformation",
                "Rewrite the expression into a more tractable algebraic form before solving.",
            ),
            Strategy(
                "direct_number_check",
                "direct_computation",
                "Evaluate the needed quantities directly and simplify the final result.",
            ),
        ]
    return [
        Strategy(
            "algebraic_setup",
            "algebraic_transformation",
            "Translate the problem into equations or identities and manipulate them cleanly.",
        ),
        Strategy(
            "case_split_reasoning",
            "case_analysis",
            "Break the problem into manageable cases and solve each one separately.",
        ),
        Strategy(
            "direct_compute",
            "direct_computation",
            "Compute the target quantity directly using the given definitions and formulas.",
        ),
    ]


def deduplicate_and_fill_strategies(problem_item: dict, raw_strategies: List[dict]) -> List[Strategy]:
    strategies: List[Strategy] = []
    seen_ids = set()
    for item in raw_strategies:
        if not isinstance(item, dict):
            continue
        strategy_id = sanitize_strategy_id(item.get("strategy_id"), "strategy")
        if strategy_id in seen_ids:
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        strategies.append(
            Strategy(
                strategy_id=strategy_id,
                strategy_family=map_to_allowed_family(item.get("strategy_family"), description),
                description=description[:400],
            )
        )
        seen_ids.add(strategy_id)
        if len(strategies) == 3:
            break

    for fallback in fallback_strategies(problem_item):
        if len(strategies) == 3:
            break
        if fallback.strategy_id in seen_ids:
            continue
        strategies.append(fallback)
        seen_ids.add(fallback.strategy_id)

    return strategies[:3]


def propose_strategies(problem_item: dict, args: argparse.Namespace) -> List[Strategy]:
    prompt = f"""
You are a strategy proposer for math reasoning problems.

Given a benchmark math problem, propose exactly 3 distinct reasoning strategies that could solve it.

Each strategy must include:
- strategy_id: short snake_case
- strategy_family: one of the allowed strategy families
- description: concise explanation of how to solve using this strategy

Allowed strategy families:
- algebraic_transformation
- modular_reasoning
- counting_formula
- case_analysis
- direct_computation

Do not use any other family.
Return only valid JSON:
{{
  "strategies": [
    {{
      "strategy_id": "...",
      "strategy_family": "...",
      "description": "..."
    }}
  ]
}}

Problem:
{problem_item["problem"]}
""".strip()
    parsed = ask_json(
        prompt,
        kind="strategy_proposer",
        max_new_tokens=args.max_new_tokens_strategy,
        temperature=0.1,
    )
    raw_strategies = []
    if parsed is not None:
        raw_strategies = parsed.get("strategies", [])
    strategies = deduplicate_and_fill_strategies(problem_item, raw_strategies)
    if len(strategies) != 3:
        strategies = fallback_strategies(problem_item)
    return strategies


def solve_with_strategy(problem_item: dict, strategy: Strategy, args: argparse.Namespace) -> str:
    prompt = f"""
You are solving a benchmark math problem using a specific assigned reasoning strategy.

Problem:
{problem_item["problem"]}

Assigned strategy:
- strategy_id: {strategy.strategy_id}
- strategy_family: {strategy.strategy_family}
- description: {strategy.description}

Instructions:
- Follow the assigned strategy as much as possible.
- Keep reasoning concise but sufficient.
- Use at most 8 reasoning steps.
- Put the final answer FIRST in the exact format:
FINAL_ANSWER: <answer>
- Then provide reasoning.

Output format:
FINAL_ANSWER: <answer>
REASONING:
1. ...
2. ...

Important:
The final answer should be in the same mathematical form as the expected benchmark answer when possible.
Do not include markdown fences.
""".strip()
    return ask_model(
        prompt,
        kind="solver",
        max_new_tokens=args.max_new_tokens_solver,
        temperature=0.2,
    )


def extract_final_answer(solver_response: str) -> Tuple[Optional[str], bool]:
    lines = [line.strip() for line in (solver_response or "").strip().splitlines() if line.strip()]
    if not lines:
        return None, True
    match = re.match(r"FINAL_ANSWER:\s*(.+)", lines[0])
    if match is None:
        return None, True
    predicted_answer = match.group(1).strip()
    if not predicted_answer:
        return None, True
    return predicted_answer, False


def unwrap_boxed(text: str) -> str:
    cleaned = text.strip()
    prefix = "\\boxed{"
    while cleaned.startswith(prefix) and cleaned.endswith("}"):
        cleaned = cleaned[len(prefix) : -1].strip()
    return cleaned


def strip_outer_text_wrapper(text: str) -> str:
    match = re.fullmatch(r"\\text\{(.+)\}", text)
    return match.group(1) if match else text


def normalize_answer(value: Optional[str]) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    normalized = normalized.rstrip(".")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\;", "")
    normalized = normalized.replace("\\:", "")
    normalized = normalized.replace("\\tfrac", "\\frac")
    normalized = normalized.replace("\\dfrac", "\\frac")
    normalized = normalized.replace("−", "-")
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    normalized = unwrap_boxed(normalized)
    normalized = strip_outer_text_wrapper(normalized)
    normalized = normalized.strip().rstrip(".")
    normalized = normalized.replace(" ", "")
    return normalized


def try_parse_fraction(text: str) -> Optional[float]:
    if re.fullmatch(r"[-+]?\d+/\d+", text):
        return float(Fraction(text))
    match = re.fullmatch(r"([-+]?)\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", text)
    if match is None:
        return None
    sign, numerator, denominator = match.groups()
    value = Fraction(int(numerator), int(denominator))
    if sign == "-":
        value = -value
    return float(value)


def try_parse_number(value: Optional[str]) -> Optional[float]:
    normalized = normalize_answer(value)
    if not normalized:
        return None
    fraction_value = try_parse_fraction(normalized)
    if fraction_value is not None:
        return fraction_value
    if re.fullmatch(r"[-+]?\d+", normalized):
        return float(int(normalized))
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", normalized):
        return float(normalized)
    return None


def verify_answer(predicted_answer: Optional[str], gold_answer: Optional[str]) -> dict:
    normalized_predicted = normalize_answer(predicted_answer)
    normalized_gold = normalize_answer(gold_answer)
    exact_match = normalized_predicted == normalized_gold

    predicted_number = try_parse_number(predicted_answer)
    gold_number = try_parse_number(gold_answer)
    numeric_match = False
    if predicted_number is not None and gold_number is not None:
        numeric_match = abs(predicted_number - gold_number) <= 1e-6

    # TODO: add a stronger symbolic equivalence checker, e.g. via sympy, when needed.
    return {
        "normalized_predicted_answer": normalized_predicted,
        "normalized_gold_answer": normalized_gold,
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "correct": exact_match or numeric_match,
    }


def heuristic_validity_likert(problem_type: str, strategy_family: str) -> int:
    lowered = problem_type.lower()
    if "counting" in lowered or "probability" in lowered:
        mapping = {
            "counting_formula": 5,
            "case_analysis": 4,
            "direct_computation": 3,
            "algebraic_transformation": 2,
            "modular_reasoning": 1,
        }
        return mapping[strategy_family]
    if "number theory" in lowered:
        mapping = {
            "modular_reasoning": 5,
            "direct_computation": 4,
            "algebraic_transformation": 3,
            "case_analysis": 2,
            "counting_formula": 1,
        }
        return mapping[strategy_family]
    if "geometry" in lowered:
        mapping = {
            "algebraic_transformation": 4,
            "case_analysis": 4,
            "direct_computation": 4,
            "modular_reasoning": 1,
            "counting_formula": 1,
        }
        return mapping[strategy_family]
    mapping = {
        "algebraic_transformation": 5,
        "direct_computation": 4,
        "case_analysis": 3,
        "modular_reasoning": 2,
        "counting_formula": 2,
    }
    return mapping[strategy_family]


def fallback_judge(problem_item: dict, strategy: Strategy, correct: bool, extraction_failed: bool) -> dict:
    strategy_compliance_likert = 1 if extraction_failed else 3
    strategy_validity_likert = heuristic_validity_likert(
        problem_item["problem_type"], strategy.strategy_family
    )
    if correct:
        process_quality_likert = 4
    elif extraction_failed:
        process_quality_likert = 1
    else:
        process_quality_likert = 2
    return {
        "strategy_compliance_likert": strategy_compliance_likert,
        "strategy_validity_likert": strategy_validity_likert,
        "process_quality_likert": process_quality_likert,
        "brief_reason": "Fallback judge used because the LLM judge response was invalid.",
    }


def normalize_likert(value: Any, default: int = 3) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return min(5, max(1, score))


def judge_rollout(
    problem_item: dict,
    strategy: Strategy,
    solver_response: str,
    predicted_answer: Optional[str],
    gold_answer: str,
    correct: bool,
    extraction_failed: bool,
    args: argparse.Namespace,
) -> dict:
    prompt = f"""
You are a strict reasoning-quality judge.

Your task is to evaluate a solver's rollout for a benchmark math problem.

You will be given:
- problem
- benchmark gold answer
- predicted answer
- whether the predicted answer matches the gold answer
- assigned strategy_id
- assigned strategy_family
- assigned strategy_description
- solver_response

Evaluate the rollout using a 1-5 Likert scale.

Scale:
1 = Strongly disagree / very poor
2 = Disagree / poor
3 = Neutral / partially acceptable
4 = Agree / good
5 = Strongly agree / excellent

Criteria:

1. strategy_compliance:
Does the solver actually follow the assigned strategy?
- Give 5 if the rollout clearly follows the assigned strategy.
- Give 3 if it partially follows the strategy but mixes other approaches.
- Give 1 if it ignores the assigned strategy.

2. strategy_validity:
Is the assigned strategy appropriate for this problem?
- Give 5 if the strategy is highly suitable.
- Give 3 if it is somewhat relevant but not ideal.
- Give 1 if it is inappropriate or inefficient for the problem.

3. process_quality:
Is the intermediate reasoning logically valid and computationally reliable?
- Give 5 if the reasoning is clear, complete, and correct.
- Give 3 if the approach is reasonable but has minor gaps or unclear steps.
- Give 1 if the reasoning is incomplete, contradictory, or mathematically wrong.

Important:
- A wrong final answer can still receive a moderate process_quality score if the reasoning path is mostly valid but has a small arithmetic or finalization error.
- A correct final answer should not automatically receive high process_quality if the reasoning is lucky, unsupported, or inconsistent.
- Focus on the reasoning process, not only final correctness.

Return only valid JSON:
{{
  "strategy_compliance_likert": 1,
  "strategy_validity_likert": 1,
  "process_quality_likert": 1,
  "brief_reason": "short explanation"
}}

No markdown.
No extra text.

problem: {problem_item["problem"]}
benchmark gold answer: {gold_answer}
predicted answer: {predicted_answer}
matches gold answer: {str(correct).lower()}
assigned strategy_id: {strategy.strategy_id}
assigned strategy_family: {strategy.strategy_family}
assigned strategy_description: {strategy.description}
solver_response: {solver_response}
""".strip()

    parsed = ask_json(
        prompt,
        kind="judge",
        max_new_tokens=args.max_new_tokens_judge,
        temperature=0.0,
    )
    if parsed is None:
        parsed = fallback_judge(problem_item, strategy, correct, extraction_failed)

    strategy_compliance_likert = normalize_likert(
        parsed.get("strategy_compliance_likert"), default=3
    )
    strategy_validity_likert = normalize_likert(
        parsed.get("strategy_validity_likert"),
        default=heuristic_validity_likert(problem_item["problem_type"], strategy.strategy_family),
    )
    process_quality_likert = normalize_likert(
        parsed.get("process_quality_likert"),
        default=4 if correct else 2,
    )
    judge_scores = {
        "strategy_compliance_likert": strategy_compliance_likert,
        "strategy_validity_likert": strategy_validity_likert,
        "process_quality_likert": process_quality_likert,
        "strategy_compliance_score": LIKERT_TO_SCORE[strategy_compliance_likert],
        "strategy_validity_score": LIKERT_TO_SCORE[strategy_validity_likert],
        "process_quality_score": LIKERT_TO_SCORE[process_quality_likert],
        "brief_reason": str(parsed.get("brief_reason", ""))[:500],
    }
    judge_scores["process_score"] = (
        judge_scores["strategy_compliance_score"]
        + judge_scores["strategy_validity_score"]
        + judge_scores["process_quality_score"]
    ) / 3.0
    return judge_scores


def answer_only_reward(correct: bool) -> float:
    return 1.0 if correct else 0.0


def answer_process_reward(correct: bool, process_score: float) -> float:
    if correct:
        return 0.7 + 0.3 * process_score
    return 0.1 * process_score


def run_single_rollout(
    problem_item: dict,
    strategy: Strategy,
    condition: str,
    answer_only_controller: BayesianController,
    answer_process_controller: BayesianController,
    args: argparse.Namespace,
) -> dict:
    problem_type = problem_item["problem_type"]
    belief_before = {
        "answer_only": answer_only_controller.snapshot(problem_type, strategy.strategy_family),
        "answer_process": answer_process_controller.snapshot(problem_type, strategy.strategy_family),
    }

    raw_response = solve_with_strategy(problem_item, strategy, args)
    predicted_answer, extraction_failed = extract_final_answer(raw_response)
    verification = verify_answer(predicted_answer, problem_item["gold_answer"])
    correct = verification["correct"] and not extraction_failed
    judge_scores = judge_rollout(
        problem_item=problem_item,
        strategy=strategy,
        solver_response=raw_response,
        predicted_answer=predicted_answer,
        gold_answer=problem_item["gold_answer"],
        correct=correct,
        extraction_failed=extraction_failed,
        args=args,
    )
    process_score = judge_scores["process_score"]
    rollout_reward = answer_process_reward(correct, process_score)
    rollout_answer_only_reward = answer_only_reward(correct)

    answer_only_controller.update(problem_type, strategy.strategy_family, rollout_answer_only_reward)
    answer_process_controller.update(problem_type, strategy.strategy_family, rollout_reward)

    belief_after = {
        "answer_only": answer_only_controller.snapshot(problem_type, strategy.strategy_family),
        "answer_process": answer_process_controller.snapshot(problem_type, strategy.strategy_family),
    }

    return {
        "problem_id": problem_item["problem_id"],
        "benchmark": problem_item["benchmark"],
        "unique_id": problem_item["unique_id"],
        "problem_type": problem_type,
        "level": problem_item["level"],
        "problem": problem_item["problem"],
        "gold_answer": problem_item["gold_answer"],
        "strategy_id": strategy.strategy_id,
        "strategy_family": strategy.strategy_family,
        "strategy_description": strategy.description,
        "condition": condition,
        "raw_response": raw_response,
        "predicted_answer": predicted_answer,
        "normalized_predicted_answer": verification["normalized_predicted_answer"],
        "normalized_gold_answer": verification["normalized_gold_answer"],
        "correct": correct,
        "extraction_failed": extraction_failed,
        "judge_scores": {
            "strategy_compliance_likert": judge_scores["strategy_compliance_likert"],
            "strategy_validity_likert": judge_scores["strategy_validity_likert"],
            "process_quality_likert": judge_scores["process_quality_likert"],
            "strategy_compliance_score": judge_scores["strategy_compliance_score"],
            "strategy_validity_score": judge_scores["strategy_validity_score"],
            "process_quality_score": judge_scores["process_quality_score"],
            "brief_reason": judge_scores["brief_reason"],
        },
        "process_score": process_score,
        "reward": rollout_reward,
        "controller_rewards": {
            "answer_only": rollout_answer_only_reward,
            "answer_process": rollout_reward,
        },
        "belief_before": belief_before,
        "belief_after": belief_after,
    }


def accuracy(logs: List[dict]) -> float:
    return sum(1 for item in logs if item["correct"]) / len(logs) if logs else 0.0


def average_reward(logs: List[dict]) -> float:
    return sum(item["reward"] for item in logs) / len(logs) if logs else 0.0


def group_by_problem_type(logs: List[dict]) -> dict:
    grouped = defaultdict(list)
    for item in logs:
        grouped[item["problem_type"]].append(item)
    return grouped


def metrics_by_problem_type(logs: List[dict]) -> Tuple[dict, dict]:
    grouped = group_by_problem_type(logs)
    accuracy_output = {}
    reward_output = {}
    for problem_type, rows in grouped.items():
        accuracy_output[problem_type] = accuracy(rows)
        reward_output[problem_type] = average_reward(rows)
    return accuracy_output, reward_output


def filter_condition(logs: List[dict], condition: str) -> List[dict]:
    return [item for item in logs if item["condition"] == condition]


def get_global_top_belief(controller: BayesianController) -> dict:
    best = None
    for problem_type, rankings in controller.ranking().items():
        for item in rankings:
            candidate = {
                "problem_type": problem_type,
                "strategy_family": item["strategy_family"],
                "belief": item["belief"],
                "alpha": item["alpha"],
                "beta": item["beta"],
                "total_trials": item["total_trials"],
            }
            if best is None or (
                candidate["belief"],
                candidate["total_trials"],
                candidate["strategy_family"],
            ) > (
                best["belief"],
                best["total_trials"],
                best["strategy_family"],
            ):
                best = candidate
    return best or {}


def compute_analysis_summary(
    phase_b_rollouts: List[dict],
    all_rollouts: List[dict],
    answer_only_controller: BayesianController,
    answer_process_controller: BayesianController,
) -> dict:
    random_logs = filter_condition(phase_b_rollouts, "random")
    answer_only_logs = filter_condition(phase_b_rollouts, "answer_only_belief")
    answer_process_logs = filter_condition(phase_b_rollouts, "answer_process_belief")

    random_accuracy = accuracy(random_logs)
    answer_only_accuracy = accuracy(answer_only_logs)
    answer_process_accuracy = accuracy(answer_process_logs)

    random_reward = average_reward(random_logs)
    answer_only_reward_value = average_reward(answer_only_logs)
    answer_process_reward_value = average_reward(answer_process_logs)

    effect_by_problem_type = {}
    random_by_type = group_by_problem_type(random_logs)
    answer_process_by_type = group_by_problem_type(answer_process_logs)
    for problem_type in sorted(set(random_by_type) | set(answer_process_by_type)):
        random_acc = accuracy(random_by_type.get(problem_type, []))
        answer_process_acc = accuracy(answer_process_by_type.get(problem_type, []))
        effect_by_problem_type[problem_type] = {
            "random_accuracy": random_acc,
            "answer_process_accuracy": answer_process_acc,
            "delta_accuracy": answer_process_acc - random_acc,
        }

    biggest_effect = None
    for problem_type, item in effect_by_problem_type.items():
        candidate = {"problem_type": problem_type, **item}
        if biggest_effect is None or (
            candidate["delta_accuracy"],
            candidate["answer_process_accuracy"],
            candidate["problem_type"],
        ) > (
            biggest_effect["delta_accuracy"],
            biggest_effect["answer_process_accuracy"],
            biggest_effect["problem_type"],
        ):
            biggest_effect = candidate

    partial_reward_rollouts = [
        item
        for item in all_rollouts
        if (not item["correct"]) and item["process_score"] >= 0.5 and item["reward"] > 0.0
    ]

    return {
        "question_1_answer_process_vs_random_accuracy_higher": {
            "result": answer_process_accuracy > random_accuracy,
            "answer_process_accuracy": answer_process_accuracy,
            "random_accuracy": random_accuracy,
        },
        "question_2_answer_process_vs_answer_only_accuracy_higher": {
            "result": answer_process_accuracy > answer_only_accuracy,
            "answer_process_accuracy": answer_process_accuracy,
            "answer_only_accuracy": answer_only_accuracy,
        },
        "question_3_answer_process_average_reward_better": {
            "vs_random": answer_process_reward_value > random_reward,
            "vs_answer_only": answer_process_reward_value > answer_only_reward_value,
            "answer_process_average_reward": answer_process_reward_value,
            "random_average_reward": random_reward,
            "answer_only_average_reward": answer_only_reward_value,
        },
        "question_4_incorrect_but_high_process_received_partial_reward": {
            "result": len(partial_reward_rollouts) > 0,
            "count": len(partial_reward_rollouts),
            "examples": partial_reward_rollouts[:5],
        },
        "question_5_biggest_effect_problem_type": biggest_effect or {},
        "question_6_highest_belief_strategy_family": {
            "answer_only_controller": get_global_top_belief(answer_only_controller),
            "answer_process_controller": get_global_top_belief(answer_process_controller),
        },
    }


def print_question_summary(summary: dict) -> None:
    q1 = summary["question_1_answer_process_vs_random_accuracy_higher"]
    q2 = summary["question_2_answer_process_vs_answer_only_accuracy_higher"]
    q3 = summary["question_3_answer_process_average_reward_better"]
    q4 = summary["question_4_incorrect_but_high_process_received_partial_reward"]
    q5 = summary["question_5_biggest_effect_problem_type"]
    q6 = summary["question_6_highest_belief_strategy_family"]

    print("\nAnalysis summary")
    print(
        "1. Answer+Process belief > Random accuracy:",
        q1["result"],
        f"({q1['answer_process_accuracy']:.3f} vs {q1['random_accuracy']:.3f})",
    )
    print(
        "2. Answer+Process belief > Answer-only belief accuracy:",
        q2["result"],
        f"({q2['answer_process_accuracy']:.3f} vs {q2['answer_only_accuracy']:.3f})",
    )
    print(
        "3. Answer+Process average reward better:",
        f"vs Random={q3['vs_random']}",
        f"vs Answer-only={q3['vs_answer_only']}",
        f"({q3['answer_process_average_reward']:.3f}, {q3['random_average_reward']:.3f}, {q3['answer_only_average_reward']:.3f})",
    )
    print(
        "4. Incorrect but high-process rollouts received partial reward:",
        q4["result"],
        f"(count={q4['count']})",
    )
    print("5. Biggest effect problem_type:", json.dumps(q5, ensure_ascii=False))
    print(
        "6. Highest belief strategy family:",
        json.dumps(q6, ensure_ascii=False),
    )


def run_experiment(args: argparse.Namespace) -> dict:
    sampled_problems, benchmark_source = load_benchmark_problems(
        num_problems=args.num_problems,
        seed=args.seed,
        benchmark=args.benchmark,
    )
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(sampled_problems, indent=2, ensure_ascii=False), encoding="utf-8")

    phase_a = sampled_problems[: args.phase_a_size]
    phase_b = sampled_problems[args.phase_a_size :]
    problem_types = sorted({item["problem_type"] for item in sampled_problems})
    answer_only_controller = BayesianController("answer_only_belief", problem_types)
    answer_process_controller = BayesianController("answer_process_belief", problem_types)
    rng = random.Random(args.seed)

    phase_a_rollouts: List[dict] = []
    phase_b_rollouts: List[dict] = []
    phase_b_problem_summaries: List[dict] = []

    progress(
        f"Loaded {len(sampled_problems)} benchmark problems from {args.benchmark} via {benchmark_source}. "
        f"Phase A={len(phase_a)}, Phase B={len(phase_b)}"
    )

    for index, problem_item in enumerate(phase_a, start=1):
        progress(
            f"Phase A {index}/{len(phase_a)} | problem_id={problem_item['problem_id']} "
            f"| problem_type={problem_item['problem_type']}"
        )
        strategies = propose_strategies(problem_item, args)
        for strategy in strategies:
            phase_a_rollouts.append(
                run_single_rollout(
                    problem_item=problem_item,
                    strategy=strategy,
                    condition="phase_a_all",
                    answer_only_controller=answer_only_controller,
                    answer_process_controller=answer_process_controller,
                    args=args,
                )
            )

    for index, problem_item in enumerate(phase_b, start=1):
        progress(
            f"Phase B {index}/{len(phase_b)} | problem_id={problem_item['problem_id']} "
            f"| problem_type={problem_item['problem_type']}"
        )
        strategies = propose_strategies(problem_item, args)
        selected = {
            "random": rng.choice(strategies),
            "answer_only_belief": answer_only_controller.select_strategy(
                problem_item["problem_type"], strategies, rng
            ),
            "answer_process_belief": answer_process_controller.select_strategy(
                problem_item["problem_type"], strategies, rng
            ),
        }
        phase_b_problem_summaries.append(
            {
                "problem_id": problem_item["problem_id"],
                "unique_id": problem_item["unique_id"],
                "problem_type": problem_item["problem_type"],
                "level": problem_item["level"],
                "problem": problem_item["problem"],
                "strategies": [asdict(strategy) for strategy in strategies],
                "selected_strategies": {
                    condition: asdict(strategy) for condition, strategy in selected.items()
                },
            }
        )
        for condition in ["random", "answer_only_belief", "answer_process_belief"]:
            phase_b_rollouts.append(
                run_single_rollout(
                    problem_item=problem_item,
                    strategy=selected[condition],
                    condition=condition,
                    answer_only_controller=answer_only_controller,
                    answer_process_controller=answer_process_controller,
                    args=args,
                )
            )

    random_logs = filter_condition(phase_b_rollouts, "random")
    answer_only_logs = filter_condition(phase_b_rollouts, "answer_only_belief")
    answer_process_logs = filter_condition(phase_b_rollouts, "answer_process_belief")

    random_accuracy_by_problem_type, random_reward_by_problem_type = metrics_by_problem_type(random_logs)
    answer_only_accuracy_by_problem_type, answer_only_reward_by_problem_type = metrics_by_problem_type(
        answer_only_logs
    )
    answer_process_accuracy_by_problem_type, answer_process_reward_by_problem_type = metrics_by_problem_type(
        answer_process_logs
    )

    all_rollouts = phase_a_rollouts + phase_b_rollouts
    analysis_summary = compute_analysis_summary(
        phase_b_rollouts=phase_b_rollouts,
        all_rollouts=all_rollouts,
        answer_only_controller=answer_only_controller,
        answer_process_controller=answer_process_controller,
    )

    return {
        "model": MODEL,
        "backend": "hf_qwen",
        "benchmark": args.benchmark,
        "benchmark_source": benchmark_source,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_problems": args.num_problems,
        "phase_a_size": args.phase_a_size,
        "phase_b_size": len(phase_b),
        "seed": args.seed,
        "max_new_tokens_solver": args.max_new_tokens_solver,
        "max_new_tokens_judge": args.max_new_tokens_judge,
        "max_new_tokens_strategy": args.max_new_tokens_strategy,
        "model_calls": MODEL_CALLS.snapshot(),
        "sampled_problem_count": len(sampled_problems),
        "sampled_problems_path": str(SAMPLE_PATH),
        "result_path": str(RESULT_PATH),
        "sampled_problems": sampled_problems,
        "phase_a_rollouts": phase_a_rollouts,
        "phase_b_rollouts": phase_b_rollouts,
        "phase_b_problem_summaries": phase_b_problem_summaries,
        "metrics": {
            "overall": {
                "random_accuracy": accuracy(random_logs),
                "answer_only_belief_accuracy": accuracy(answer_only_logs),
                "answer_process_belief_accuracy": accuracy(answer_process_logs),
                "random_average_reward": average_reward(random_logs),
                "answer_only_belief_average_reward": average_reward(answer_only_logs),
                "answer_process_belief_average_reward": average_reward(answer_process_logs),
            },
            "by_problem_type": {
                "accuracy_by_problem_type": {
                    "random": random_accuracy_by_problem_type,
                    "answer_only_belief": answer_only_accuracy_by_problem_type,
                    "answer_process_belief": answer_process_accuracy_by_problem_type,
                },
                "average_reward_by_problem_type": {
                    "random": random_reward_by_problem_type,
                    "answer_only_belief": answer_only_reward_by_problem_type,
                    "answer_process_belief": answer_process_reward_by_problem_type,
                },
            },
        },
        "beliefs": {
            "final_answer_only_strategy_belief_ranking_by_problem_type": answer_only_controller.ranking(),
            "final_answer_process_strategy_belief_ranking_by_problem_type": answer_process_controller.ranking(),
        },
        "analysis_summary": analysis_summary,
    }


def print_summary(result: dict) -> None:
    overall = result["metrics"]["overall"]
    print(f"Model: {result['model']} | backend={result['backend']}")
    print(f"Benchmark: {result['benchmark']} | source={result['benchmark_source']}")
    print(f"Model calls: {result['model_calls']}")
    print(f"Random accuracy: {overall['random_accuracy']:.3f}")
    print(f"Answer-only belief accuracy: {overall['answer_only_belief_accuracy']:.3f}")
    print(f"Answer+Process belief accuracy: {overall['answer_process_belief_accuracy']:.3f}")
    print(f"Random average reward: {overall['random_average_reward']:.3f}")
    print(f"Answer-only belief average reward: {overall['answer_only_belief_average_reward']:.3f}")
    print(f"Answer+Process belief average reward: {overall['answer_process_belief_average_reward']:.3f}")
    print("\nAccuracy by problem_type")
    print(
        json.dumps(
            result["metrics"]["by_problem_type"]["accuracy_by_problem_type"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\nAverage reward by problem_type")
    print(
        json.dumps(
            result["metrics"]["by_problem_type"]["average_reward_by_problem_type"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\nFinal answer-only strategy belief ranking by problem_type")
    print(
        json.dumps(
            result["beliefs"]["final_answer_only_strategy_belief_ranking_by_problem_type"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\nFinal answer+process strategy belief ranking by problem_type")
    print(
        json.dumps(
            result["beliefs"]["final_answer_process_strategy_belief_ranking_by_problem_type"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print_question_summary(result["analysis_summary"])


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(result)
    print(f"\nSaved sampled benchmark problems to: {SAMPLE_PATH}")
    print(f"Saved experiment result to: {RESULT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
