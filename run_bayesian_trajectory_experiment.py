#!/usr/bin/env python3
"""Unified Bayesian posterior reward analysis for Qwen math rollouts.

This script is reward-analysis code, not RL training code.

Research idea:
- The solver is not given a predefined strategy list.
- The solver generates its own strategy, reasoning, and final answer.
- A prior judge estimates how promising each generated strategy is before
  seeing the reasoning.
- An evidence judge then evaluates the semantic reliability of each rollout.
- Final answer correctness remains deterministic and rule-based.
- Likelihood is answer-heavy by default.
- Posterior reward combines LLM-estimated strategy prior and
  LLM-estimated evidence likelihood to produce a trajectory-level reward
  signal that can later be used for GRPO.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import re
import statistics
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_BENCHMARK = "HuggingFaceH4/MATH-500"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SAMPLE_PATH = Path("results/math500_100_sample.json")
LEGACY_SAMPLE_PATH = Path("results/math500_50_sample.json")
RAW_JSONL_CACHE = Path("results/math500_test_raw.jsonl")

DEFAULT_EXPERIMENT_NAME = "bayesian_qwen_unified_strategy_llm_judge_answer_heavy"
DEFAULT_OUTPUT_ROOT = "outputs"

DEFAULT_SOLVER_MODEL = os.getenv("QWEN_SOLVER_MODEL", os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
DEFAULT_JUDGE_MODEL = os.getenv("QWEN_JUDGE_MODEL", DEFAULT_SOLVER_MODEL)
DEFAULT_PRIOR_JUDGE_MODEL = os.getenv("QWEN_PRIOR_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
DEFAULT_EVIDENCE_JUDGE_MODEL = os.getenv("QWEN_EVIDENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

JUDGE_JSON_SYSTEM_PROMPT = """You are a JSON-only evaluation API.
Return exactly one valid JSON object and nothing else.
Do not include markdown.
Do not include ```json.
Do not include explanations before or after the JSON.
Use double quotes for all keys and string values.
Do not use LaTeX notation inside JSON string values.
Do not use backslashes inside JSON string values.
Do not write \\( ... \\), \\[ ... \\], \\frac, \\boxed, or any LaTeX command.
Write all reason, risk_flag, key_strength, key_weakness, and critical_failure_step fields in plain English text only.
The first character of your response must be {.
The last character of your response must be }."""

PROGRESS_LOG = os.getenv("PROGRESS_LOG", "1") == "1"
NUMERIC_TOLERANCE = 1e-6
HIGH_POSTERIOR_THRESHOLD = 0.20
VERIFIER_CLOSE_SIMILARITY_THRESHOLD = 0.82
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉₋", "0123456789-")
ORDER_INSENSITIVE_HINTS = (
    "find all",
    "all solutions",
    "solutions",
    "roots",
    "possible values",
    "values of",
    "integers",
    "numbers",
    "real values",
    "what are the",
)

DEFAULT_FULL_WEIGHTS = {
    "answer_correctness": 0.40,
    "step_validity": 0.25,
    "proof_completeness": 0.20,
    "strategy_compliance": 0.10,
    "consistency": 0.05,
}

DEFAULT_ANSWER_HEAVY_WEIGHTS = {
    "answer_correctness": 0.70,
    "step_validity": 0.10,
    "proof_completeness": 0.10,
    "strategy_compliance": 0.05,
    "consistency": 0.05,
}

LEGACY_STATIC_STRATEGIES = [
    "Compute directly from the problem statement.",
    "Rewrite the expressions or equations into a simpler algebraic form.",
    "Use divisibility, congruences, gcd structure, or invariants if relevant.",
    "Check a small set of plausible cases or candidates carefully.",
    "Solve once, then verify the result with an independent consistency check.",
]

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


@dataclass
class AnswerNormalization:
    raw: str
    normalized: str
    stripped_units: bool
    had_pi: bool
    had_fraction: bool
    had_base_notation: bool
    comma_items: Optional[list[str]]
    numeric_value: Optional[float]


@dataclass
class PriorAssessment:
    rollout_id: int
    suitability: int
    prior_probability: float
    reason: str
    risk_flag: str
    probability_source: str
    missing_from_judge: bool


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


@dataclass
class RolloutAnalysisRecord:
    problem_id: str
    unique_id: Optional[str]
    benchmark: str
    problem_type: str
    level: Any
    rollout_id: int
    problem: str
    gold_answer: str
    strategy: str
    reasoning: str
    final_answer: str
    normalized_final_answer: str
    normalized_ground_truth: str
    verification_method: str
    answer_correctness: float
    prior_suitability: int
    prior_probability: float
    prior_reason: str
    prior_risk_flag: str
    prior_probability_source: str
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
    likelihood: float
    unnormalized_posterior: float
    posterior_reward: float
    outcome_only_reward: float
    posterior_advantage: float
    outcome_advantage: float
    solver_format_failure: bool
    empty_strategy: bool
    empty_reasoning: bool
    empty_final_answer: bool
    prior_judge_fallback_used: bool
    evidence_judge_fallback_used: bool
    prior_missing_from_judge: bool
    judge_label_inconsistency: bool
    strategy_source: str
    evidence_source: str
    generation_backend: str
    solver_temperature: float


class RunLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        line = f"[{datetime.utcnow().isoformat(timespec='seconds')}] {message}"
        self.lines.append(line)
        if PROGRESS_LOG:
            print(line, flush=True)


class QwenEngine:
    """Thin HF Qwen wrapper used for both solver and judges."""

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
        self.model_name = model_name

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

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str = "Follow the requested output format exactly.",
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
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


_ENGINE_CACHE: dict[str, QwenEngine] = {}
_ENGINE_LOAD_ERRORS: dict[str, Exception] = {}


def can_attempt_qwen() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False in this runtime"
    return True, ""


def get_engine(model_name: str, logger: RunLogger, *, strict: bool = False) -> Optional[QwenEngine]:
    if model_name in _ENGINE_CACHE:
        return _ENGINE_CACHE[model_name]
    if model_name in _ENGINE_LOAD_ERRORS:
        if strict:
            raise RuntimeError(f"Qwen engine is unavailable for model={model_name}.") from _ENGINE_LOAD_ERRORS[model_name]
        return None

    allowed, reason = can_attempt_qwen()
    if not allowed:
        exc = RuntimeError(reason)
        _ENGINE_LOAD_ERRORS[model_name] = exc
        if strict:
            raise exc
        logger.log(f"Qwen backend unavailable for model={model_name}: {reason}")
        return None

    try:
        logger.log(f"Loading model: {model_name}")
        engine = QwenEngine(model_name)
        _ENGINE_CACHE[model_name] = engine
        logger.log(f"Loaded model: {model_name}")
        return engine
    except Exception as exc:  # pragma: no cover - depends on local model setup
        _ENGINE_LOAD_ERRORS[model_name] = exc
        if strict:
            raise
        logger.log(f"Failed to load model={model_name}: {exc}")
        return None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def repair_json_text(text: str) -> str:
    cleaned = text.strip()
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
    cleaned = repair_json_text(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def diagnose_json_output_failure(text: str) -> list[str]:
    if not text or not text.strip():
        return ["empty_output"]

    stripped = text.strip()
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


def clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def clamp_score_0_to_4(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = 0
    return max(0, min(4, numeric))


def unwrap_boxed(text: str) -> str:
    cleaned = text.strip()
    prefix = "\\boxed{"
    while cleaned.startswith(prefix) and cleaned.endswith("}"):
        cleaned = cleaned[len(prefix) : -1].strip()
    return cleaned


def strip_outer_text_wrapper(text: str) -> str:
    match = re.fullmatch(r"\\text\{(.+)\}", text)
    return match.group(1) if match else text


def cleanup_extracted_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = re.split(r"(?:\n|\r)", answer, maxsplit=1)[0].strip()
    answer = re.sub(r"^(?:the answer is|answer is|is)\s+", "", answer, flags=re.IGNORECASE).strip()
    return answer.strip(" \t\r\n$`")


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"\[Final Answer\]\s*([^\n\r]+)",
        r"FINAL_ANSWER:\s*([^\n\r]+)",
        r"Final answer\s*[:：]\s*([^\n\r]+)",
        r"final answer\s*[:：]\s*([^\n\r]+)",
        r"answer\s*[:：]\s*([^\n\r]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return cleanup_extracted_answer(matches[-1])

    boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed_matches:
        return cleanup_extracted_answer(boxed_matches[-1])
    return cleanup_extracted_answer(text[-120:])


def strip_trailing_punctuation(text: str) -> str:
    return text.rstrip(" \t\r\n.,;:!?")


def strip_outer_braces(text: str) -> str:
    cleaned = text.strip()
    while cleaned.startswith("{") and cleaned.endswith("}"):
        depth = 0
        balanced = True
        for index, char in enumerate(cleaned):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0 and index != len(cleaned) - 1:
                balanced = False
                break
        if not balanced:
            break
        cleaned = cleaned[1:-1].strip()
    return cleaned


def replace_latex_fractions(text: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    cleaned = text
    while cleaned != previous:
        previous = cleaned
        cleaned = pattern.sub(lambda match: f"({match.group(1)})/({match.group(2)})", cleaned)
    return cleaned


def normalize_base_notation_text(text: str) -> tuple[str, bool]:
    original = text
    cleaned = re.sub(
        r"([A-Za-z0-9+-]+)([₀₁₂₃₄₅₆₇₈₉]+)\b",
        lambda match: f"{match.group(1)}_{match.group(2).translate(SUBSCRIPT_TRANSLATION)}",
        text,
    )
    cleaned = cleaned.translate(SUBSCRIPT_TRANSLATION)
    cleaned = re.sub(r"\b([A-Za-z0-9+-]+)\s*_\s*(\d+)\b", r"\1_\2", cleaned)
    cleaned = re.sub(
        r"\b([A-Za-z0-9+-]+)\s+(?:in\s+)?base\s+(\d+)\b",
        r"\1_\2",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned, cleaned != original


def normalize_pi_text(text: str) -> tuple[str, bool]:
    original = text
    cleaned = text.replace("\\pi", "pi").replace("π", "pi")
    cleaned = re.sub(r"(?<=\d)\s*pi\b", "*pi", cleaned)
    cleaned = re.sub(r"(?<=\))\s*pi\b", "*pi", cleaned)
    cleaned = re.sub(r"\bpi(?=\()", "pi*", cleaned)
    cleaned = re.sub(r"(?<=\d)pi\b", "*pi", cleaned)
    cleaned = re.sub(r"\bpi(?=\d)", "pi*", cleaned)
    return cleaned, cleaned != original


def expression_looks_numeric_like(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    if re.fullmatch(r"[-+*/().0-9pi_,\s]+", candidate):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9+-]+_\d+", candidate))


def strip_trailing_units(text: str) -> tuple[str, bool]:
    cleaned = text.strip()
    match = re.fullmatch(r"(.+?)\s+([A-Za-z%]+(?:\s+[A-Za-z%]+)*)", cleaned)
    if match is None:
        return cleaned, False
    expr, unit_phrase = match.groups()
    if not expression_looks_numeric_like(expr):
        return cleaned, False
    if "base" in unit_phrase.lower():
        return cleaned, False
    return expr.strip(), True


def safe_eval_numeric_expression(text: str) -> Optional[float]:
    candidate = text.strip()
    if not candidate or "_" in candidate or "," in candidate:
        return None
    candidate = candidate.replace("^", "**")
    if not re.fullmatch(r"[0-9pi+\-*/(). ]+", candidate):
        return None
    try:
        value = eval(candidate, {"__builtins__": {}}, {"pi": math.pi})
    except Exception:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def split_comma_collection(text: str) -> Optional[list[str]]:
    if "," not in text:
        return None
    if any(char in text for char in "()[]"):
        return None
    items = [item.strip() for item in text.split(",")]
    if len(items) < 2 or any(not item for item in items):
        return None
    return items


def normalize_answer_details(value: Optional[str], allow_collection: bool = True) -> AnswerNormalization:
    raw = "" if value is None else str(value).strip()
    cleaned = cleanup_extracted_answer(raw)
    cleaned = cleaned.strip().strip("$").strip("`")
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace("\\!", "")
    cleaned = cleaned.replace("\\;", "")
    cleaned = cleaned.replace("\\:", "")
    cleaned = cleaned.replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\dfrac", "\\frac")
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    cleaned = re.sub(r"\\text\{([^{}]+)\}", r" \1 ", cleaned)
    cleaned = re.sub(r"\\(?:mathrm|operatorname)\{([^{}]+)\}", r"\1", cleaned)
    cleaned = unwrap_boxed(cleaned)
    cleaned = strip_outer_text_wrapper(cleaned)
    cleaned = strip_outer_braces(cleaned)
    cleaned = strip_trailing_punctuation(cleaned)

    had_fraction = "\\frac" in cleaned or bool(re.fullmatch(r"[-+]?\d+\s*/\s*\d+", cleaned))
    cleaned = replace_latex_fractions(cleaned)

    cleaned, had_base_notation = normalize_base_notation_text(cleaned)
    cleaned, had_pi = normalize_pi_text(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned, stripped_units = strip_trailing_units(cleaned)
    cleaned = strip_trailing_punctuation(cleaned)
    cleaned = strip_outer_braces(cleaned)
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)

    comma_items: Optional[list[str]] = None
    if allow_collection:
        raw_items = split_comma_collection(cleaned)
        if raw_items is not None:
            comma_items = [
                normalize_answer_details(item, allow_collection=False).normalized for item in raw_items
            ]

    numeric_value = safe_eval_numeric_expression(cleaned)
    return AnswerNormalization(
        raw=raw,
        normalized=cleaned,
        stripped_units=stripped_units,
        had_pi=had_pi,
        had_fraction=had_fraction,
        had_base_notation=had_base_notation,
        comma_items=comma_items,
        numeric_value=numeric_value,
    )


def normalize_answer(value: Optional[str]) -> str:
    return normalize_answer_details(value).normalized


def collection_key(items: list[str]) -> tuple[str, ...]:
    keys: list[str] = []
    for item in items:
        numeric = safe_eval_numeric_expression(item)
        if numeric is not None:
            keys.append(f"num:{numeric:.12g}")
        else:
            keys.append(f"str:{item}")
    return tuple(sorted(keys))


def should_treat_as_unordered_collection(
    problem_text: str,
    predicted: AnswerNormalization,
    gold: AnswerNormalization,
) -> bool:
    if predicted.comma_items is None or gold.comma_items is None:
        return False
    if any(char in predicted.raw + gold.raw for char in "()[]"):
        return False
    if any(hint in problem_text.lower() for hint in ORDER_INSENSITIVE_HINTS):
        return True
    return "{" in predicted.raw or "{" in gold.raw or "}" in predicted.raw or "}" in gold.raw


def detect_possible_verifier_miss(
    predicted: AnswerNormalization,
    gold: AnswerNormalization,
    problem_text: str,
) -> list[str]:
    reasons: list[str] = []
    if not predicted.normalized or not gold.normalized:
        return reasons

    similarity = difflib.SequenceMatcher(None, predicted.normalized, gold.normalized).ratio()
    if similarity >= VERIFIER_CLOSE_SIMILARITY_THRESHOLD:
        reasons.append("high_string_similarity")

    if predicted.comma_items and gold.comma_items and collection_key(predicted.comma_items) == collection_key(
        gold.comma_items
    ):
        reasons.append("same_collection_items")
        if not should_treat_as_unordered_collection(problem_text, predicted, gold):
            reasons.append("order_may_matter")

    if predicted.numeric_value is not None and gold.numeric_value is not None:
        delta = abs(predicted.numeric_value - gold.numeric_value)
        if NUMERIC_TOLERANCE < delta <= max(1e-3, 1e-4 * max(1.0, abs(gold.numeric_value))):
            reasons.append("numerically_very_close")
    return sorted(set(reasons))


def verify_answer(predicted_answer: Optional[str], gold_answer: Optional[str], problem_text: str = "") -> dict[str, Any]:
    predicted = normalize_answer_details(predicted_answer)
    gold = normalize_answer_details(gold_answer)

    verification_method = "no_match"
    correct = False

    if predicted.normalized and predicted.normalized == gold.normalized:
        correct = True
        if predicted.stripped_units or gold.stripped_units:
            verification_method = "unit_stripped_match"
        elif predicted.had_pi or gold.had_pi:
            verification_method = "latex_pi_match"
        else:
            verification_method = "exact_string"
    elif should_treat_as_unordered_collection(problem_text, predicted, gold):
        if predicted.comma_items is not None and gold.comma_items is not None:
            if collection_key(predicted.comma_items) == collection_key(gold.comma_items):
                correct = True
                verification_method = "comma_set_match"
    elif predicted.numeric_value is not None and gold.numeric_value is not None:
        if abs(predicted.numeric_value - gold.numeric_value) <= NUMERIC_TOLERANCE:
            correct = True
            if predicted.had_pi or gold.had_pi:
                verification_method = "latex_pi_match"
            elif predicted.had_fraction or gold.had_fraction:
                verification_method = "fraction_decimal_match"
            else:
                verification_method = "numeric_match"

    return {
        "normalized_predicted_answer": predicted.normalized,
        "normalized_gold_answer": gold.normalized,
        "correct": correct,
        "verification_method": verification_method,
        "possible_miss_reasons": [] if correct else detect_possible_verifier_miss(predicted, gold, problem_text),
    }


def standardize_problem_item(row: dict[str, Any], problem_id: int, benchmark: str) -> dict[str, Any]:
    benchmark_name = benchmark.split("/")[-1]
    return {
        "problem_id": problem_id,
        "benchmark": benchmark_name,
        "unique_id": row.get("unique_id"),
        "problem_type": row.get("subject", row.get("problem_type", "math")),
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


def load_with_datasets(benchmark: str, logger: RunLogger) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.log("datasets library is not installed. Trying raw jsonl fallback.")
        return None, None

    try:
        ds = load_dataset(benchmark, split="test")
        return [dict(row) for row in ds], "datasets"
    except Exception as exc:
        logger.log(f"datasets load failed: {exc}")
        return None, None


def load_raw_jsonl_rows(benchmark: str, logger: RunLogger) -> tuple[list[dict[str, Any]], str]:
    cache_path = raw_jsonl_cache_path(benchmark)
    if cache_path.exists():
        logger.log(f"Loading cached raw jsonl from {cache_path}")
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
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            logger.log(f"Trying raw jsonl fallback: {url}")
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


def load_benchmark_problems(
    num_problems: int,
    seed: int,
    benchmark: str,
    logger: RunLogger,
) -> tuple[list[dict[str, Any]], str]:
    rows, source = load_with_datasets(benchmark, logger)
    if rows is None:
        rows, source = load_raw_jsonl_rows(benchmark, logger)

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


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append(parsed)
        return rows

    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if isinstance(parsed, list):
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: item {idx} is not a JSON object")
            rows.append(item)
        return rows
    if isinstance(parsed, dict):
        for key in ("problems", "data", "items", "examples"):
            if isinstance(parsed.get(key), list):
                for idx, item in enumerate(parsed[key]):
                    if not isinstance(item, dict):
                        raise ValueError(f"{path}: {key}[{idx}] is not a JSON object")
                    rows.append(item)
                return rows
        return [parsed]
    raise ValueError(f"Unsupported JSON shape in {path}")


def standardize_loaded_row(row: dict[str, Any], idx: int, benchmark: str) -> Optional[dict[str, Any]]:
    if "problem" in row and "gold_answer" in row:
        return {
            "problem_id": row.get("problem_id", idx),
            "benchmark": row.get("benchmark", benchmark.split("/")[-1]),
            "unique_id": row.get("unique_id"),
            "problem_type": row.get("problem_type", "math"),
            "level": row.get("level"),
            "problem": row["problem"],
            "gold_answer": row["gold_answer"],
            "solution": row.get("solution"),
        }

    problem_text = (
        row.get("problem")
        or row.get("question")
        or row.get("problem_text")
        or row.get("prompt")
        or row.get("input")
    )
    gold_answer = (
        row.get("answer")
        or row.get("gold_answer")
        or row.get("ground_truth")
        or row.get("final_answer")
        or row.get("target")
        or row.get("label")
    )
    if problem_text is None or gold_answer is None:
        return None

    return {
        "problem_id": row.get("problem_id", row.get("id", row.get("uid", idx))),
        "benchmark": benchmark.split("/")[-1],
        "unique_id": row.get("unique_id"),
        "problem_type": row.get("subject", row.get("problem_type", "math")),
        "level": row.get("level"),
        "problem": str(problem_text),
        "gold_answer": str(gold_answer),
        "solution": row.get("solution"),
    }


def load_existing_sample(sample_path: Path, num_problems: int) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    if not sample_path.exists():
        return None, None
    rows = read_json_or_jsonl(sample_path)
    standardized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = standardize_loaded_row(row, idx, DEFAULT_BENCHMARK)
        if item is not None:
            standardized.append(item)
    if len(standardized) < num_problems:
        raise RuntimeError(
            f"{sample_path} only has {len(standardized)} problems, but {num_problems} were requested."
        )
    return standardized[:num_problems], f"existing_sample:{sample_path}"


def load_experiment_problems(args: argparse.Namespace, logger: RunLogger) -> tuple[list[dict[str, Any]], str]:
    if args.input_path:
        rows = read_json_or_jsonl(Path(args.input_path))
        problems: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            item = standardize_loaded_row(row, idx, args.benchmark)
            if item is not None:
                problems.append(item)
            if len(problems) >= args.num_problems:
                break
        if not problems:
            raise RuntimeError(f"No usable problems found in {args.input_path}")
        return problems, f"input_path:{args.input_path}"

    sample_path = Path(args.sample_path)
    sample_rows, source = load_existing_sample(sample_path, args.num_problems)
    if sample_rows is not None:
        return sample_rows, source or "existing_sample"

    target_sample_size = args.num_problems
    if args.benchmark == DEFAULT_BENCHMARK and sample_path == DEFAULT_SAMPLE_PATH:
        target_sample_size = max(args.num_problems, DEFAULT_SAMPLE_SIZE)

    sampled, source = load_benchmark_problems(target_sample_size, args.seed, args.benchmark, logger)
    if args.benchmark == DEFAULT_BENCHMARK:
        write_json(sample_path, sampled)
        source = f"{source};saved_sample:{sample_path}"
    return sampled[: args.num_problems], source


def build_unified_solver_prompt(problem_text: str) -> str:
    return f"""
You are a mathematical reasoning solver.

Solve the given problem independently.

First, write a concise strategy that you believe is appropriate for solving the problem.
Then, solve the problem by following that strategy.
Finally, provide the final answer.

Do not force an unusual strategy.
Do not choose a strategy from a predefined list.
Use whatever strategy naturally fits the problem.

You MUST include all three exact section headers:
[Strategy]
[Reasoning]
[Final Answer]

The final answer must be written under the exact header [Final Answer].
Do not omit [Final Answer].
Do not end the response inside [Reasoning].
Keep the reasoning concise enough to always include [Final Answer].

Return your response in the following format:

[Strategy]
...

[Reasoning]
...

[Final Answer]
...

Problem:
{problem_text}
""".strip()


def build_legacy_static_solver_prompt(problem_text: str, assigned_strategy: str) -> str:
    return f"""
You are a mathematical reasoning solver.

Solve the given problem independently by following the assigned strategy.

Assigned strategy:
{assigned_strategy}

Return your response in the following format:

[Strategy]
Repeat the assigned strategy in concise form.

[Reasoning]
...

[Final Answer]
...

Problem:
{problem_text}
""".strip()


def extract_section(text: str, heading: str, following_headings: list[str]) -> str:
    if following_headings:
        next_pattern = "|".join(re.escape(item) for item in following_headings)
        pattern = re.compile(
            rf"\[{re.escape(heading)}\]\s*(.*?)(?=(?:\n\s*\[(?:{next_pattern})\])|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
    else:
        pattern = re.compile(
            rf"\[{re.escape(heading)}\]\s*(.*)$",
            flags=re.IGNORECASE | re.DOTALL,
        )
    match = pattern.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def parse_solver_output(raw_output: str) -> dict[str, Any]:
    strategy = extract_section(raw_output, "Strategy", ["Reasoning", "Final Answer"])
    reasoning = extract_section(raw_output, "Reasoning", ["Final Answer"])
    final_answer_section = extract_section(raw_output, "Final Answer", [])
    final_answer = final_answer_section

    if not final_answer:
        final_answer = extract_final_answer(raw_output)

    return {
        "strategy": strategy.strip(),
        "reasoning": reasoning.strip(),
        "final_answer": cleanup_extracted_answer(final_answer),
        "strategy_section_present": bool(strategy.strip()),
        "reasoning_section_present": bool(reasoning.strip()),
        "final_answer_section_present": bool(final_answer_section.strip()),
        "used_final_answer_fallback": bool(not final_answer_section.strip() and final_answer.strip()),
    }


def validate_solver_output(parsed: dict[str, Any]) -> dict[str, Any]:
    strategy = str(parsed.get("strategy", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    final_answer = str(parsed.get("final_answer", "")).strip()

    empty_strategy = not strategy
    empty_reasoning = not reasoning
    empty_final_answer = not final_answer

    validation_errors: list[str] = []
    if empty_strategy:
        validation_errors.append("empty_strategy")
    if empty_reasoning:
        validation_errors.append("empty_reasoning")
    if empty_final_answer:
        validation_errors.append("empty_final_answer")
    if not parsed.get("strategy_section_present", False):
        validation_errors.append("missing_strategy_section")
    if not parsed.get("reasoning_section_present", False):
        validation_errors.append("missing_reasoning_section")
    if not parsed.get("final_answer_section_present", False):
        validation_errors.append("missing_final_answer_section")

    return {
        "is_valid": len(validation_errors) == 0,
        "empty_strategy": empty_strategy,
        "empty_reasoning": empty_reasoning,
        "empty_final_answer": empty_final_answer,
        "validation_errors": validation_errors,
    }


def run_solver_with_format_retry(
    *,
    problem: dict[str, Any],
    args: argparse.Namespace,
    logger: RunLogger,
    raw_solver_outputs: list[dict[str, Any]],
    rollout_id: int,
    prompt: str,
    rollout_temperature: float,
    strategy_source: str,
    generation_backend: str,
) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    final_parsed: Optional[dict[str, Any]] = None
    final_validation: Optional[dict[str, Any]] = None
    final_raw_output = ""

    for attempt in range(1, 3):
        raw_output = call_model(
            args.solver_model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=rollout_temperature,
            logger=logger,
            strict=True,
        )
        parsed = parse_solver_output(raw_output)
        validation = validate_solver_output(parsed)
        raw_solver_outputs.append(
            {
                "problem_id": str(problem["problem_id"]),
                "unique_id": problem.get("unique_id"),
                "rollout_id": rollout_id,
                "attempt": attempt,
                "model_name": args.solver_model,
                "solver_prompt": prompt,
                "raw_output": raw_output,
                "parsed_strategy": parsed["strategy"],
                "parsed_reasoning": parsed["reasoning"],
                "parsed_final_answer": parsed["final_answer"],
                "solver_temperature": rollout_temperature,
                "strategy_source": strategy_source,
                "generation_backend": generation_backend,
                "strategy_section_present": parsed.get("strategy_section_present", False),
                "reasoning_section_present": parsed.get("reasoning_section_present", False),
                "final_answer_section_present": parsed.get("final_answer_section_present", False),
                "used_final_answer_fallback": parsed.get("used_final_answer_fallback", False),
                "solver_format_valid": validation["is_valid"],
                "validation_errors": validation["validation_errors"],
            }
        )
        final_parsed = parsed
        final_validation = validation
        final_raw_output = raw_output
        if validation["is_valid"]:
            return parsed, validation, raw_output, attempt
        logger.log(
            f"Solver format validation failed for problem_id={problem['problem_id']}, "
            f"rollout_id={rollout_id}, attempt={attempt}: {validation['validation_errors']}"
        )

    assert final_parsed is not None and final_validation is not None
    logger.log(
        f"Solver output remained invalid after retry for problem_id={problem['problem_id']}, "
        f"rollout_id={rollout_id}; marking rollout as format_error."
    )
    return final_parsed, final_validation, final_raw_output, 2


def call_model(
    model_name: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    logger: RunLogger,
    system_prompt: str = "Follow the requested output format exactly.",
    strict: bool = True,
) -> str:
    engine = get_engine(model_name, logger, strict=strict)
    if engine is None:
        raise RuntimeError(f"Model backend unavailable for model={model_name}")
    return engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def call_json_judge(
    *,
    judge_type: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    logger: RunLogger,
    raw_judge_outputs: list[dict[str, Any]],
    problem_id: str,
    rollout_id: Optional[int],
) -> tuple[Optional[Any], Optional[str]]:
    for attempt in range(1, 3):
        try:
            raw_output = call_model(
                model_name,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                logger=logger,
                strict=True,
                system_prompt=JUDGE_JSON_SYSTEM_PROMPT,
            )
        except Exception as exc:
            raw_judge_outputs.append(
                {
                    "judge_type": judge_type,
                    "problem_id": problem_id,
                    "rollout_id": rollout_id,
                    "attempt": attempt,
                    "model_name": model_name,
                    "system_prompt": JUDGE_JSON_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "raw_output": "",
                    "parsed_success": False,
                    "parsed_json": None,
                    "call_error": str(exc),
                }
            )
            logger.log(
                f"{judge_type} call failed for problem_id={problem_id}, rollout_id={rollout_id}, "
                f"attempt={attempt}: {exc}"
            )
            continue
        parsed = safe_json_parse(raw_output)
        parse_failure_reasons = [] if parsed is not None else diagnose_json_output_failure(raw_output)
        raw_judge_outputs.append(
            {
                "judge_type": judge_type,
                "problem_id": problem_id,
                "rollout_id": rollout_id,
                "attempt": attempt,
                "model_name": model_name,
                "system_prompt": JUDGE_JSON_SYSTEM_PROMPT,
                "prompt": prompt,
                "raw_output": raw_output,
                "repaired_json_candidate": repair_json_text(raw_output),
                "parsed_success": parsed is not None,
                "parsed_json": parsed,
                "parse_failure_reasons": parse_failure_reasons,
            }
        )
        if parsed is not None:
            return parsed, raw_output
        logger.log(
            f"{judge_type} JSON parse failed for problem_id={problem_id}, rollout_id={rollout_id}, attempt={attempt}"
        )
    return None, None


def build_rollout_temperatures(num_rollouts: int, base_temperature: float) -> list[float]:
    return [base_temperature for _ in range(num_rollouts)]


def generate_qwen_unified_rollouts(
    problem: dict[str, Any],
    args: argparse.Namespace,
    logger: RunLogger,
    raw_solver_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt = build_unified_solver_prompt(str(problem["problem"]))
    temperatures = build_rollout_temperatures(args.num_rollouts, args.temperature)
    rollouts: list[dict[str, Any]] = []

    for rollout_id, rollout_temperature in enumerate(temperatures):
        parsed, validation, raw_output, parse_attempts = run_solver_with_format_retry(
            problem=problem,
            args=args,
            logger=logger,
            raw_solver_outputs=raw_solver_outputs,
            rollout_id=rollout_id,
            prompt=prompt,
            rollout_temperature=rollout_temperature,
            strategy_source="qwen_unified",
            generation_backend="qwen",
        )
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "strategy": parsed["strategy"],
                "reasoning": parsed["reasoning"],
                "final_answer": parsed["final_answer"],
                "raw_output": raw_output,
                "solver_format_failure": not validation["is_valid"],
                "empty_strategy": validation["empty_strategy"],
                "empty_reasoning": validation["empty_reasoning"],
                "empty_final_answer": validation["empty_final_answer"],
                "solver_parse_attempts": parse_attempts,
                "validation_errors": list(validation["validation_errors"]),
                "strategy_source": "qwen_unified",
                "generation_backend": "qwen",
                "solver_temperature": rollout_temperature,
            }
        )

    if len(rollouts) != args.num_rollouts:
        raise RuntimeError(
            f"Expected exactly {args.num_rollouts} rollouts for problem_id={problem['problem_id']}, "
            f"got {len(rollouts)}"
        )
    return rollouts


def generate_legacy_static_rollouts(
    problem: dict[str, Any],
    args: argparse.Namespace,
    logger: RunLogger,
    raw_solver_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rollouts: list[dict[str, Any]] = []
    temperatures = build_rollout_temperatures(args.num_rollouts, args.temperature)

    for rollout_id, rollout_temperature in enumerate(temperatures):
        strategy = LEGACY_STATIC_STRATEGIES[rollout_id % len(LEGACY_STATIC_STRATEGIES)]
        prompt = build_legacy_static_solver_prompt(str(problem["problem"]), strategy)
        parsed, validation, raw_output, parse_attempts = run_solver_with_format_retry(
            problem=problem,
            args=args,
            logger=logger,
            raw_solver_outputs=raw_solver_outputs,
            rollout_id=rollout_id,
            prompt=prompt,
            rollout_temperature=rollout_temperature,
            strategy_source="static",
            generation_backend="qwen",
        )
        strategy_text = parsed["strategy"] or strategy
        rollouts.append(
            {
                "rollout_id": rollout_id,
                "strategy": strategy_text,
                "reasoning": parsed["reasoning"],
                "final_answer": parsed["final_answer"],
                "raw_output": raw_output,
                "solver_format_failure": not validation["is_valid"],
                "empty_strategy": validation["empty_strategy"],
                "empty_reasoning": validation["empty_reasoning"],
                "empty_final_answer": validation["empty_final_answer"],
                "solver_parse_attempts": parse_attempts,
                "validation_errors": list(validation["validation_errors"]),
                "strategy_source": "static",
                "generation_backend": "qwen",
                "solver_temperature": rollout_temperature,
            }
        )

    return rollouts


def generate_rollouts(
    problem: dict[str, Any],
    args: argparse.Namespace,
    logger: RunLogger,
    raw_solver_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.strategy_source == "qwen_unified":
        return generate_qwen_unified_rollouts(problem, args, logger, raw_solver_outputs)
    if args.strategy_source == "static":
        if not args.allow_legacy_static:
            raise ValueError("Static strategy mode is disabled for the main experiment.")
        logger.log(
            f"Using legacy static strategy mode for problem_id={problem['problem_id']}. "
            "This is disabled by default and only exists for explicit comparison."
        )
        return generate_legacy_static_rollouts(problem, args, logger, raw_solver_outputs)
    raise ValueError(f"Unknown strategy_source: {args.strategy_source}")


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def format_required_rollout_ids(num_rollouts: int) -> str:
    return ",".join(str(index) for index in range(num_rollouts))


def build_prior_judge_prompt(problem_text: str, rollouts: list[dict[str, Any]], num_rollouts: int) -> str:
    strategies_block = "\n\n".join(
        f"Rollout {rollout['rollout_id']}:\n{rollout['strategy'] or '(empty strategy)'}"
        for rollout in rollouts
    )
    required_rollout_ids = format_required_rollout_ids(num_rollouts)
    return f"""
You are a mathematical strategy evaluator.

You will be given a math problem and several candidate strategies generated by a solver.
Your task is to evaluate how promising each strategy is before seeing the actual solution.

Important rules:
- Do not solve the full problem.
- Do not look at reasoning or final answers.
- Evaluate only whether the strategy is appropriate for the problem type.
- Similar strategies may receive similar scores.
- The strategies do not need to be unique.
- Use suitability as an integer from 0 to 4.
- 0 = unsuitable or vague
- 1 = weak
- 2 = partially useful
- 3 = good
- 4 = very strong/directly appropriate
- Do not output prior_probability.
- The code will compute prior probabilities from suitability scores.
- Do not use LaTeX notation in any JSON string field.
- Do not use backslashes.
- Write reason and risk_flag in plain English only.
- You must evaluate every rollout_id from 0 to {num_rollouts - 1}.
- Return exactly {num_rollouts} items in the "priors" list.
- For num_rollouts = {num_rollouts}, return exactly rollout_id {required_rollout_ids}.
- Do not omit any rollout_id.
- Do not merge similar strategies.
- Even if two strategies are identical or very similar, evaluate both separately.
- If a strategy is weak, still include it with low suitability.
- Each rollout_id must appear exactly once.
- Do not invent rollout IDs.
- Do not return fewer than {num_rollouts} rows.
- Do not return more than {num_rollouts} rows.
- suitability must be an integer from 0 to 4.

Return only valid JSON.
The first character must be {{ and the last character must be }}.

JSON schema:
{{
  "priors": [
    {{
      "rollout_id": 0,
      "suitability": 3,
      "reason": "Uses the polynomial remainder structure directly.",
      "risk_flag": "none"
    }}
  ]
}}

Problem:
{problem_text}

Candidate strategies:
{strategies_block}
""".strip()


def build_prior_repair_prompt(problem_text: str, rollouts: list[dict[str, Any]], num_rollouts: int) -> str:
    required_rollout_ids = format_required_rollout_ids(num_rollouts)
    strategies_block = "\n\n".join(
        f"Rollout {rollout['rollout_id']}:\n{rollout['strategy'] or '(empty strategy)'}"
        for rollout in rollouts
    )
    return f"""
Your previous response was invalid because it did not include every rollout_id.
You must return exactly {num_rollouts} rows.
Required rollout_ids: {required_rollout_ids}.
Each required rollout_id must appear exactly once.
Do not omit any rollout.
Do not merge similar strategies.
Return only valid JSON.

You are a mathematical strategy evaluator.

Evaluate every rollout_id separately before seeing the actual solution.
Use suitability as an integer from 0 to 4.
- 0 = unsuitable or very vague
- 1 = weak
- 2 = partially useful
- 3 = good
- 4 = very strong/directly appropriate
- Do not use LaTeX notation in any JSON string value.
- Do not use backslashes.
- Write reasons in plain English only.
- The first character must be {{ and the last character must be }}.

JSON schema:
{{
  "priors": [
    {{
      "rollout_id": 0,
      "suitability": 3,
      "reason": "Plain English reason only.",
      "risk_flag": "none"
    }},
    {{
      "rollout_id": 1,
      "suitability": 2,
      "reason": "Plain English reason only.",
      "risk_flag": "vague"
    }}
  ]
}}

Problem:
{problem_text}

Candidate strategies:
{strategies_block}
""".strip()


def validate_prior_coverage(raw_rows: Any, num_rollouts: int) -> dict[str, Any]:
    if not isinstance(raw_rows, list):
        return {
            "valid": False,
            "missing_ids": list(range(num_rollouts)),
            "duplicate_ids": [],
            "invalid_ids": ["priors_not_list"],
            "num_rows": 0,
        }

    seen_counts: Counter[int] = Counter()
    invalid_ids: list[Any] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            invalid_ids.append("non_dict_row")
            continue
        try:
            rollout_id = int(row.get("rollout_id"))
        except (TypeError, ValueError):
            invalid_ids.append(row.get("rollout_id"))
            continue
        if rollout_id < 0 or rollout_id >= num_rollouts:
            invalid_ids.append(rollout_id)
            continue
        seen_counts[rollout_id] += 1

    missing_ids = [rollout_id for rollout_id in range(num_rollouts) if seen_counts[rollout_id] == 0]
    duplicate_ids = [rollout_id for rollout_id, count in seen_counts.items() if count > 1]
    valid = len(raw_rows) == num_rollouts and not missing_ids and not duplicate_ids and not invalid_ids
    return {
        "valid": valid,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "invalid_ids": invalid_ids,
        "num_rows": len(raw_rows),
    }


def normalize_prior_rows(
    raw_rows: list[dict[str, Any]],
    num_rollouts: int,
    prior_softmax_temperature: float,
) -> list[PriorAssessment]:
    by_rollout_id: dict[int, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        try:
            rollout_id = int(row.get("rollout_id"))
        except (TypeError, ValueError):
            continue
        if 0 <= rollout_id < num_rollouts:
            by_rollout_id[rollout_id] = row

    if not by_rollout_id:
        raise ValueError("No valid prior rows were returned by the judge.")

    if prior_softmax_temperature <= 0:
        raise ValueError("--prior_softmax_temperature must be positive.")

    suitabilities = [
        clamp_score_0_to_4(by_rollout_id[rollout_id].get("suitability", 0))
        for rollout_id in range(num_rollouts)
    ]

    probabilities = softmax([value / prior_softmax_temperature for value in suitabilities])
    probability_source = "softmax_from_llm_suitability"

    return [
        PriorAssessment(
            rollout_id=rollout_id,
            suitability=suitabilities[rollout_id],
            prior_probability=probabilities[rollout_id],
            reason=str(by_rollout_id.get(rollout_id, {}).get("reason", "")),
            risk_flag=str(by_rollout_id.get(rollout_id, {}).get("risk_flag", "")),
            probability_source=probability_source,
            missing_from_judge=False,
        )
        for rollout_id in range(num_rollouts)
    ]


def uniform_prior_assessments(num_rollouts: int, probability_source: str, reason: str) -> list[PriorAssessment]:
    if num_rollouts <= 0:
        return []
    probability = 1.0 / num_rollouts
    return [
        PriorAssessment(
            rollout_id=rollout_id,
            suitability=0,
            prior_probability=probability,
            reason=reason,
            risk_flag="",
            probability_source=probability_source,
            missing_from_judge=False,
        )
        for rollout_id in range(num_rollouts)
    ]


def assess_strategy_priors(
    problem: dict[str, Any],
    rollouts: list[dict[str, Any]],
    args: argparse.Namespace,
    logger: RunLogger,
    raw_judge_outputs: list[dict[str, Any]],
    prior_debug_stats: dict[str, int],
) -> tuple[list[PriorAssessment], bool]:
    if args.prior_mode == "uniform":
        return uniform_prior_assessments(len(rollouts), "uniform", "Uniform prior baseline."), False
    if args.prior_mode != "llm_strategy_prior":
        raise ValueError(f"Unknown prior_mode: {args.prior_mode}")

    prompt = build_prior_judge_prompt(str(problem["problem"]), rollouts, len(rollouts))
    parsed, raw_output = call_json_judge(
        judge_type="prior_judge",
        model_name=args.prior_judge_model,
        prompt=prompt,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.prior_judge_temperature,
        logger=logger,
        raw_judge_outputs=raw_judge_outputs,
        problem_id=str(problem["problem_id"]),
        rollout_id=None,
    )
    if not isinstance(parsed, dict) or not isinstance(parsed.get("priors"), list):
        prior_debug_stats["prior_judge_parse_failure_count"] += 1
        logger.log(
            f"Prior judge failed for problem_id={problem['problem_id']}; using uniform prior fallback."
        )
        return (
            uniform_prior_assessments(
                len(rollouts),
                "uniform_fallback_after_judge_failure",
                "Prior judge parse failure.",
            ),
            True,
        )

    logger.log(
        f"Prior judge first raw output for problem_id={problem['problem_id']}: "
        f"{(raw_output or '')[:2000]}"
    )
    coverage = validate_prior_coverage(parsed["priors"], len(rollouts))
    if coverage["valid"]:
        try:
            return (
                normalize_prior_rows(
                    parsed["priors"],
                    len(rollouts),
                    args.prior_softmax_temperature,
                ),
                False,
            )
        except Exception as exc:
            prior_debug_stats["prior_judge_parse_failure_count"] += 1
            logger.log(
                f"Prior judge normalization failed for problem_id={problem['problem_id']}: {exc}. "
                "Using uniform prior fallback."
            )
            return (
                uniform_prior_assessments(
                    len(rollouts),
                    "uniform_fallback_after_judge_failure",
                    "Prior judge normalization failure.",
                ),
                True,
            )
    prior_debug_stats["prior_judge_coverage_failure_count"] += 1
    prior_debug_stats["prior_missing_rollout_count"] += len(coverage["missing_ids"])
    prior_debug_stats["prior_duplicate_rollout_count"] += len(coverage["duplicate_ids"])
    prior_debug_stats["prior_invalid_rollout_count"] += len(coverage["invalid_ids"])
    logger.log(
        f"Prior judge coverage validation failed for problem_id={problem['problem_id']}: "
        f"num_rows={coverage['num_rows']}, missing_ids={coverage['missing_ids']}, "
        f"duplicate_ids={coverage['duplicate_ids']}, invalid_ids={coverage['invalid_ids']}"
    )

    repair_prompt = build_prior_repair_prompt(str(problem["problem"]), rollouts, len(rollouts))
    parsed_repair, raw_output_repair = call_json_judge(
        judge_type="prior_judge_repair",
        model_name=args.prior_judge_model,
        prompt=repair_prompt,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.prior_judge_temperature,
        logger=logger,
        raw_judge_outputs=raw_judge_outputs,
        problem_id=str(problem["problem_id"]),
        rollout_id=None,
    )
    if not isinstance(parsed_repair, dict) or not isinstance(parsed_repair.get("priors"), list):
        prior_debug_stats["prior_judge_parse_failure_count"] += 1
        logger.log(
            f"Prior judge repair failed to produce parseable priors for problem_id={problem['problem_id']}; "
            "using uniform prior fallback."
        )
        return (
            uniform_prior_assessments(
                len(rollouts),
                "uniform_fallback_after_judge_failure",
                "Prior judge repair parse failure.",
            ),
            True,
        )

    logger.log(
        f"Prior judge retry raw output for problem_id={problem['problem_id']}: "
        f"{(raw_output_repair or '')[:2000]}"
    )
    repair_coverage = validate_prior_coverage(parsed_repair["priors"], len(rollouts))
    if not repair_coverage["valid"]:
        prior_debug_stats["prior_judge_coverage_failure_count"] += 1
        prior_debug_stats["prior_missing_rollout_count"] += len(repair_coverage["missing_ids"])
        prior_debug_stats["prior_duplicate_rollout_count"] += len(repair_coverage["duplicate_ids"])
        prior_debug_stats["prior_invalid_rollout_count"] += len(repair_coverage["invalid_ids"])
        logger.log(
            f"Prior judge retry coverage still invalid for problem_id={problem['problem_id']}: "
            f"num_rows={repair_coverage['num_rows']}, missing_ids={repair_coverage['missing_ids']}, "
            f"duplicate_ids={repair_coverage['duplicate_ids']}, invalid_ids={repair_coverage['invalid_ids']}. "
            "Using uniform prior fallback."
        )
        return (
            uniform_prior_assessments(
                len(rollouts),
                "uniform_fallback_after_judge_failure",
                "Prior judge repair coverage failure.",
            ),
            True,
        )

    prior_debug_stats["prior_judge_retry_success_count"] += 1
    logger.log(
        f"Prior judge retry fixed coverage for problem_id={problem['problem_id']}."
    )
    try:
        return (
            normalize_prior_rows(
                parsed_repair["priors"],
                len(rollouts),
                args.prior_softmax_temperature,
            ),
            False,
        )
    except Exception as exc:
        prior_debug_stats["prior_judge_parse_failure_count"] += 1
        logger.log(
            f"Prior judge normalization failed after successful coverage repair for problem_id={problem['problem_id']}: "
            f"{exc}. Using uniform prior fallback."
        )
        return (
            uniform_prior_assessments(
                len(rollouts),
                "uniform_fallback_after_judge_failure",
                "Prior judge normalization failure after repair.",
            ),
            True,
        )


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
- Penalize hidden gaps, invalid assumptions, contradictions, unsupported jumps, and strategy mismatch.
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
{reasoning}

Final Answer:
{final_answer}

Deterministic correctness flag:
{int(answer_correctness)}

Final output rules:
Return ONLY one valid JSON object.
Do not include markdown.
Do not include a code block.
Do not include any explanation outside the JSON.
Do not use LaTeX notation in any JSON string value.
Do not use backslashes.
Write mathematical expressions in plain English text.
Your response must start with {{ and end with }}.
Use exactly the keys shown in the JSON schema.
""".strip()


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
            key_weakness="Heuristic evidence mode does not inspect reasoning semantics.",
            critical_failure_step="",
            evidence_source="heuristic",
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
        evidence_source="heuristic",
        judge_label_inconsistency=False,
    )


def parse_evidence_assessment(parsed: Any) -> Optional[EvidenceAssessment]:
    if not isinstance(parsed, dict):
        return None
    required_score_keys = (
        "step_validity",
        "proof_completeness",
        "strategy_compliance",
        "consistency",
        "error_type",
    )
    if any(key not in parsed for key in required_score_keys):
        return None

    error_type = str(parsed.get("error_type", "")).strip()
    if error_type not in ALLOWED_ERROR_TYPES:
        return None

    step_validity = clamp_score_0_to_4(parsed.get("step_validity"))
    proof_completeness = clamp_score_0_to_4(parsed.get("proof_completeness"))
    strategy_compliance = clamp_score_0_to_4(parsed.get("strategy_compliance"))
    consistency = clamp_score_0_to_4(parsed.get("consistency"))

    return EvidenceAssessment(
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
        key_strength=str(parsed.get("key_strength", "")),
        key_weakness=str(parsed.get("key_weakness", "")),
        critical_failure_step=str(parsed.get("critical_failure_step", "")),
        evidence_source="llm_judge",
        judge_label_inconsistency=False,
    )


def enforce_evidence_label_consistency(
    evidence: EvidenceAssessment,
    *,
    answer_correctness: float,
    logger: RunLogger,
    problem_id: str,
    rollout_id: int,
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
        remapped_error_type = "finalization_error" if evidence.step_validity >= 2 else "wrong_direction"
        logger.log(
            f"Evidence judge label inconsistency for problem_id={problem_id}, rollout_id={rollout_id}: "
            f"answer_correctness=0 but error_type={evidence.error_type}. Remapping to {remapped_error_type}."
        )
        return replace(
            evidence,
            error_type=remapped_error_type,
            judge_label_inconsistency=True,
        )

    if answer_correctness == 1.0 and evidence.error_type in correct_disallowed:
        logger.log(
            f"Evidence judge label inconsistency for problem_id={problem_id}, rollout_id={rollout_id}: "
            f"answer_correctness=1 but error_type={evidence.error_type}. Remapping to correct_weak_proof."
        )
        return replace(
            evidence,
            error_type="correct_weak_proof",
            judge_label_inconsistency=True,
        )

    return evidence


def assess_evidence(
    problem: dict[str, Any],
    rollout: dict[str, Any],
    answer_correctness: float,
    args: argparse.Namespace,
    logger: RunLogger,
    raw_judge_outputs: list[dict[str, Any]],
) -> tuple[EvidenceAssessment, bool]:
    if args.evidence_source == "heuristic":
        if not args.allow_heuristic_evidence:
            raise ValueError("Heuristic evidence mode is disabled for the main experiment.")
        logger.log(
            f"Using heuristic evidence mode for problem_id={problem['problem_id']}, rollout_id={rollout['rollout_id']}. "
            "This is disabled by default and only exists for explicit comparison."
        )
        return heuristic_evidence_assessment(answer_correctness), False

    if args.evidence_source != "llm_judge":
        raise ValueError(f"Unknown evidence_source: {args.evidence_source}")

    prompt = build_evidence_judge_prompt(
        str(problem["problem"]),
        str(rollout["strategy"]),
        str(rollout["reasoning"]),
        str(rollout["final_answer"]),
        answer_correctness,
    )
    parsed, _ = call_json_judge(
        judge_type="evidence_judge",
        model_name=args.evidence_judge_model,
        prompt=prompt,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.evidence_judge_temperature,
        logger=logger,
        raw_judge_outputs=raw_judge_outputs,
        problem_id=str(problem["problem_id"]),
        rollout_id=int(rollout["rollout_id"]),
    )
    evidence = parse_evidence_assessment(parsed)
    if evidence is not None:
        return (
            enforce_evidence_label_consistency(
                evidence,
                answer_correctness=answer_correctness,
                logger=logger,
                problem_id=str(problem["problem_id"]),
                rollout_id=int(rollout["rollout_id"]),
            ),
            False,
        )

    logger.log(
        f"Evidence judge failed for problem_id={problem['problem_id']}, rollout_id={rollout['rollout_id']}; "
        "using conservative format_error fallback."
    )
    return conservative_evidence_fallback("llm_judge_fallback", error_type="format_error"), True


def resolve_likelihood_weights(args: argparse.Namespace) -> dict[str, float]:
    if args.reward_ablation == "full":
        weights = dict(DEFAULT_FULL_WEIGHTS)
    elif args.reward_ablation == "answer_heavy":
        weights = dict(DEFAULT_ANSWER_HEAVY_WEIGHTS)
    else:
        raise ValueError(f"Unknown reward_ablation: {args.reward_ablation}")

    overrides = {
        "answer_correctness": args.answer_correctness_weight,
        "step_validity": args.step_validity_weight,
        "proof_completeness": args.proof_completeness_weight,
        "strategy_compliance": args.strategy_compliance_weight,
        "consistency": args.consistency_weight,
    }
    if any(value is not None for value in overrides.values()):
        for key, value in overrides.items():
            if value is not None:
                weights[key] = float(value)

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Likelihood weights must sum to a positive value.")
    if abs(total - 1.0) > 1e-9:
        weights = {key: value / total for key, value in weights.items()}
    return weights


def compute_likelihood(answer_correctness: float, evidence: EvidenceAssessment, weights: dict[str, float]) -> float:
    likelihood = (
        weights["answer_correctness"] * answer_correctness
        + weights["step_validity"] * evidence.step_validity_norm
        + weights["proof_completeness"] * evidence.proof_completeness_norm
        + weights["strategy_compliance"] * evidence.strategy_compliance_norm
        + weights["consistency"] * evidence.consistency_norm
    )
    return max(0.0, min(1.0, likelihood))


def compute_posteriors(
    priors: list[float],
    likelihoods: list[float],
    *,
    prior_lambda: float,
    logger: RunLogger,
    problem_id: str,
) -> tuple[list[float], list[float], bool]:
    if len(priors) != len(likelihoods):
        raise ValueError("priors and likelihoods must have the same length.")
    if not priors:
        return [], [], False

    unnormalized: list[float] = []
    for prior, likelihood in zip(priors, likelihoods):
        prior_term = max(0.0, float(prior)) ** prior_lambda
        likelihood_term = max(0.0, float(likelihood))
        unnormalized.append(prior_term * likelihood_term)

    total = sum(unnormalized)
    if not math.isfinite(total) or total <= 0:
        logger.log(
            f"Posterior normalization failed for problem_id={problem_id}; "
            "falling back to uniform posterior."
        )
        uniform = [1.0 / len(priors)] * len(priors)
        return uniform, unnormalized, True
    return [value / total for value in unnormalized], unnormalized, False


def majority_vote_correct(records: list[RolloutAnalysisRecord]) -> float:
    if not records:
        return 0.0
    normalized_answers = [record.normalized_final_answer for record in records]
    majority_answer, _ = Counter(normalized_answers).most_common(1)[0]
    verification = verify_answer(majority_answer, records[0].gold_answer, problem_text=records[0].problem)
    return 1.0 if verification["correct"] else 0.0


def entropy(values: list[float]) -> float:
    result = 0.0
    for value in values:
        if value > 0:
            result -= value * math.log(value)
    return result


def top_two_gap(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    if len(ordered) == 1:
        return ordered[0]
    return ordered[0] - ordered[1]


def select_first_max(records: list[RolloutAnalysisRecord], key: str) -> RolloutAnalysisRecord:
    return max(records, key=lambda record: getattr(record, key))


def analyze_records(
    records: list[RolloutAnalysisRecord],
    *,
    num_rollouts: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[RolloutAnalysisRecord]] = defaultdict(list)
    for record in records:
        grouped[record.problem_id].append(record)

    rng = random.Random(seed)
    posterior_top1_accuracy: list[float] = []
    likelihood_top1_accuracy: list[float] = []
    outcome_only_top1_accuracy: list[float] = []
    oracle_best_of_n: list[float] = []
    random_accuracy: list[float] = []
    self_consistency_accuracy: list[float] = []
    posterior_mass_on_correct: list[float] = []
    posterior_entropy: list[float] = []
    posterior_gap: list[float] = []
    posterior_advantage_unique_values: list[float] = []
    per_problem_summary: list[dict[str, Any]] = []

    error_type_distribution = Counter(record.error_type for record in records)
    incorrect_high_posterior_count = 0
    wrong_direction_high_posterior_count = 0
    posterior_uniform_fallback_count = 0
    prior_judge_fallback_count = len({record.problem_id for record in records if record.prior_judge_fallback_used})
    evidence_judge_fallback_count = sum(1 for record in records if record.evidence_judge_fallback_used)
    solver_format_failure_count = sum(1 for record in records if record.solver_format_failure)
    empty_strategy_count = sum(1 for record in records if record.empty_strategy)
    empty_reasoning_count = sum(1 for record in records if record.empty_reasoning)
    empty_final_answer_count = sum(1 for record in records if record.empty_final_answer)
    judge_label_inconsistency_count = sum(1 for record in records if record.judge_label_inconsistency)
    prior_missing_rollout_count = sum(1 for record in records if record.prior_missing_from_judge)
    prior_probability_source_distribution = dict(Counter(record.prior_probability_source for record in records))
    evidence_source_distribution = dict(Counter(record.evidence_source for record in records))

    for problem_id, items in grouped.items():
        items = sorted(items, key=lambda record: record.rollout_id)
        posterior_choice = select_first_max(items, "posterior_reward")
        likelihood_choice = select_first_max(items, "likelihood")
        outcome_choice = select_first_max(items, "answer_correctness")
        random_choice = rng.choice(items)
        oracle_value = 1.0 if any(item.answer_correctness == 1.0 for item in items) else 0.0

        posterior_top1_accuracy.append(posterior_choice.answer_correctness)
        likelihood_top1_accuracy.append(likelihood_choice.answer_correctness)
        outcome_only_top1_accuracy.append(outcome_choice.answer_correctness)
        oracle_best_of_n.append(oracle_value)
        random_accuracy.append(random_choice.answer_correctness)
        self_consistency_accuracy.append(majority_vote_correct(items))
        posterior_mass_on_correct.append(sum(item.posterior_reward for item in items if item.answer_correctness == 1.0))
        posterior_values = [item.posterior_reward for item in items]
        posterior_entropy.append(entropy(posterior_values))
        posterior_gap.append(top_two_gap(posterior_values))
        posterior_advantage_unique_values.append(
            float(len({round(item.posterior_advantage, 8) for item in items}))
        )

        if all(
            abs(item.posterior_reward - (1.0 / len(items))) <= 1e-12
            for item in items
        ):
            posterior_uniform_fallback_count += 1

        for item in items:
            if item.answer_correctness == 0.0 and item.posterior_reward > HIGH_POSTERIOR_THRESHOLD:
                incorrect_high_posterior_count += 1
                if item.error_type == "wrong_direction":
                    wrong_direction_high_posterior_count += 1

        per_problem_summary.append(
            {
                "problem_id": problem_id,
                "unique_id": items[0].unique_id,
                "problem_type": items[0].problem_type,
                "num_rollouts": len(items),
                "posterior_top1_rollout_id": posterior_choice.rollout_id,
                "likelihood_top1_rollout_id": likelihood_choice.rollout_id,
                "outcome_only_top1_rollout_id": outcome_choice.rollout_id,
                "random_rollout_id": random_choice.rollout_id,
                "oracle_best_of_n": oracle_value,
                "self_consistency_accuracy": self_consistency_accuracy[-1],
                "posterior_mass_on_correct": posterior_mass_on_correct[-1],
                "posterior_entropy": posterior_entropy[-1],
                "posterior_top1_top2_gap": posterior_gap[-1],
                "posterior_advantage_unique_values": posterior_advantage_unique_values[-1],
                "error_type_distribution": dict(Counter(item.error_type for item in items)),
                "rollouts": [
                    {
                        "rollout_id": item.rollout_id,
                        "strategy": item.strategy,
                        "final_answer": item.final_answer,
                        "answer_correctness": item.answer_correctness,
                        "prior_probability": item.prior_probability,
                        "likelihood": item.likelihood,
                        "posterior_reward": item.posterior_reward,
                        "error_type": item.error_type,
                        "judge_confidence": item.judge_confidence,
                        "solver_format_failure": item.solver_format_failure,
                        "prior_judge_fallback_used": item.prior_judge_fallback_used,
                        "evidence_judge_fallback_used": item.evidence_judge_fallback_used,
                        "prior_missing_from_judge": item.prior_missing_from_judge,
                        "judge_label_inconsistency": item.judge_label_inconsistency,
                    }
                    for item in items
                ],
            }
        )

    metrics = {
        "num_problems": len(grouped),
        "num_rollouts_total": len(records),
        "expected_num_rollouts_total": len(grouped) * num_rollouts,
        "exact_rollout_count_match": len(records) == len(grouped) * num_rollouts,
        "posterior_top1_accuracy": statistics.fmean(posterior_top1_accuracy) if posterior_top1_accuracy else 0.0,
        "likelihood_top1_accuracy": statistics.fmean(likelihood_top1_accuracy) if likelihood_top1_accuracy else 0.0,
        "outcome_only_top1_accuracy": statistics.fmean(outcome_only_top1_accuracy)
        if outcome_only_top1_accuracy
        else 0.0,
        "oracle_best_of_n": statistics.fmean(oracle_best_of_n) if oracle_best_of_n else 0.0,
        "random_accuracy": statistics.fmean(random_accuracy) if random_accuracy else 0.0,
        "self_consistency_accuracy": statistics.fmean(self_consistency_accuracy)
        if self_consistency_accuracy
        else 0.0,
        "posterior_mass_on_correct": statistics.fmean(posterior_mass_on_correct)
        if posterior_mass_on_correct
        else 0.0,
        "posterior_entropy": statistics.fmean(posterior_entropy) if posterior_entropy else 0.0,
        "posterior_top1_top2_gap": statistics.fmean(posterior_gap) if posterior_gap else 0.0,
        "posterior_advantage_unique_values_mean": statistics.fmean(posterior_advantage_unique_values)
        if posterior_advantage_unique_values
        else 0.0,
        "incorrect_high_posterior_count": incorrect_high_posterior_count,
        "wrong_direction_high_posterior_count": wrong_direction_high_posterior_count,
        "error_type_distribution": dict(error_type_distribution),
        "posterior_uniform_fallback_count": posterior_uniform_fallback_count,
        "prior_judge_fallback_count": prior_judge_fallback_count,
        "evidence_judge_fallback_count": evidence_judge_fallback_count,
        "solver_format_failure_count": solver_format_failure_count,
        "empty_strategy_count": empty_strategy_count,
        "empty_reasoning_count": empty_reasoning_count,
        "empty_final_answer_count": empty_final_answer_count,
        "judge_label_inconsistency_count": judge_label_inconsistency_count,
        "prior_missing_rollout_count": prior_missing_rollout_count,
        "prior_probability_source_distribution": prior_probability_source_distribution,
        "evidence_source_distribution": evidence_source_distribution,
        "high_posterior_threshold": HIGH_POSTERIOR_THRESHOLD,
    }
    return metrics, per_problem_summary


def save_run_outputs(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    rollout_records: list[RolloutAnalysisRecord],
    per_problem_summary: list[dict[str, Any]],
    metrics: dict[str, Any],
    raw_solver_outputs: list[dict[str, Any]],
    raw_judge_outputs: list[dict[str, Any]],
    logger: RunLogger,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / args.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(run_dir / "config.json", config)
    write_jsonl(run_dir / "per_rollout_results.jsonl", rollout_records)
    write_jsonl(run_dir / "per_problem_summary.jsonl", per_problem_summary)
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "raw_solver_outputs.jsonl", raw_solver_outputs)
    write_jsonl(run_dir / "raw_judge_outputs.jsonl", raw_judge_outputs)
    (run_dir / "logs.txt").write_text("\n".join(logger.lines) + "\n", encoding="utf-8")
    return run_dir


def run_experiment(args: argparse.Namespace) -> tuple[list[RolloutAnalysisRecord], dict[str, Any], Path]:
    if args.num_problems <= 0:
        raise SystemExit("--num_problems must be positive.")
    if args.num_rollouts <= 0:
        raise SystemExit("--num_rollouts must be positive.")
    if args.strategy_source == "static" and not args.allow_legacy_static:
        raise ValueError("Static strategy mode is disabled for the main experiment.")
    if args.evidence_source == "heuristic" and not args.allow_heuristic_evidence:
        raise ValueError("Heuristic evidence mode is disabled for the main experiment.")

    logger = RunLogger()
    random.seed(args.seed)
    problems, problem_source = load_experiment_problems(args, logger)
    logger.log(
        f"Loaded {len(problems)} problems from {problem_source}. "
        f"experiment_name={args.experiment_name}, strategy_source={args.strategy_source}, "
        f"prior_mode={args.prior_mode}, evidence_source={args.evidence_source}"
    )

    likelihood_weights = resolve_likelihood_weights(args)
    logger.log(f"Likelihood weights: {likelihood_weights}")
    logger.log(f"prior_lambda={args.prior_lambda}")

    raw_solver_outputs: list[dict[str, Any]] = []
    raw_judge_outputs: list[dict[str, Any]] = []
    rollout_records: list[RolloutAnalysisRecord] = []
    prior_debug_stats = {
        "prior_judge_parse_failure_count": 0,
        "prior_judge_coverage_failure_count": 0,
        "prior_judge_retry_success_count": 0,
        "prior_missing_rollout_count": 0,
        "prior_duplicate_rollout_count": 0,
        "prior_invalid_rollout_count": 0,
    }

    for index, problem in enumerate(problems, start=1):
        logger.log(
            f"Problem {index}/{len(problems)} | problem_id={problem['problem_id']} "
            f"| unique_id={problem.get('unique_id')}"
        )
        rollouts = generate_rollouts(problem, args, logger, raw_solver_outputs)
        assert len(rollouts) == args.num_rollouts

        prior_assessments, prior_judge_fallback_used = assess_strategy_priors(
            problem,
            rollouts,
            args,
            logger,
            raw_judge_outputs,
            prior_debug_stats,
        )
        if len(prior_assessments) != args.num_rollouts:
            raise RuntimeError(
                f"Expected {args.num_rollouts} prior rows for problem_id={problem['problem_id']}, "
                f"got {len(prior_assessments)}"
            )

        answer_correctness_values: list[float] = []
        evidence_assessments: list[EvidenceAssessment] = []
        evidence_fallback_used_flags: list[bool] = []
        verification_rows: list[dict[str, Any]] = []

        for rollout in rollouts:
            verification = verify_answer(
                rollout["final_answer"],
                problem["gold_answer"],
                problem_text=str(problem["problem"]),
            )
            verification_rows.append(verification)
            answer_correctness = 1.0 if verification["correct"] else 0.0
            answer_correctness_values.append(answer_correctness)
            if rollout.get("solver_format_failure", False):
                logger.log(
                    f"Skipping evidence judge for problem_id={problem['problem_id']}, "
                    f"rollout_id={rollout['rollout_id']} because solver format validation failed."
                )
                evidence_assessment = conservative_evidence_fallback(
                    "solver_format_guard",
                    error_type="format_error",
                )
                evidence_fallback_used = False
            else:
                evidence_assessment, evidence_fallback_used = assess_evidence(
                    problem,
                    rollout,
                    answer_correctness,
                    args,
                    logger,
                    raw_judge_outputs,
                )
            evidence_assessments.append(evidence_assessment)
            evidence_fallback_used_flags.append(evidence_fallback_used)

        likelihoods = [
            compute_likelihood(answer_correctness, evidence_assessment, likelihood_weights)
            for answer_correctness, evidence_assessment in zip(answer_correctness_values, evidence_assessments)
        ]
        priors = [assessment.prior_probability for assessment in prior_assessments]
        posteriors, unnormalized_posteriors, _ = compute_posteriors(
            priors,
            likelihoods,
            prior_lambda=args.prior_lambda,
            logger=logger,
            problem_id=str(problem["problem_id"]),
        )

        mean_posterior = statistics.fmean(posteriors) if posteriors else 0.0
        mean_outcome = statistics.fmean(answer_correctness_values) if answer_correctness_values else 0.0

        for rollout, prior_assessment, evidence_assessment, evidence_fallback_used, verification, answer_correctness, likelihood, posterior, unnormalized in zip(
            rollouts,
            prior_assessments,
            evidence_assessments,
            evidence_fallback_used_flags,
            verification_rows,
            answer_correctness_values,
            likelihoods,
            posteriors,
            unnormalized_posteriors,
        ):
            rollout_records.append(
                RolloutAnalysisRecord(
                    problem_id=str(problem["problem_id"]),
                    unique_id=problem.get("unique_id"),
                    benchmark=str(problem["benchmark"]),
                    problem_type=str(problem.get("problem_type", "math")),
                    level=problem.get("level"),
                    rollout_id=int(rollout["rollout_id"]),
                    problem=str(problem["problem"]),
                    gold_answer=str(problem["gold_answer"]),
                    strategy=str(rollout["strategy"]),
                    reasoning=str(rollout["reasoning"]),
                    final_answer=str(rollout["final_answer"]),
                    normalized_final_answer=verification["normalized_predicted_answer"],
                    normalized_ground_truth=verification["normalized_gold_answer"],
                    verification_method=str(verification["verification_method"]),
                    answer_correctness=answer_correctness,
                    prior_suitability=prior_assessment.suitability,
                    prior_probability=prior_assessment.prior_probability,
                    prior_reason=prior_assessment.reason,
                    prior_risk_flag=prior_assessment.risk_flag,
                    prior_probability_source=prior_assessment.probability_source,
                    step_validity=evidence_assessment.step_validity,
                    proof_completeness=evidence_assessment.proof_completeness,
                    strategy_compliance=evidence_assessment.strategy_compliance,
                    consistency=evidence_assessment.consistency,
                    step_validity_norm=evidence_assessment.step_validity_norm,
                    proof_completeness_norm=evidence_assessment.proof_completeness_norm,
                    strategy_compliance_norm=evidence_assessment.strategy_compliance_norm,
                    consistency_norm=evidence_assessment.consistency_norm,
                    error_type=evidence_assessment.error_type,
                    judge_confidence=evidence_assessment.judge_confidence,
                    key_strength=evidence_assessment.key_strength,
                    key_weakness=evidence_assessment.key_weakness,
                    critical_failure_step=evidence_assessment.critical_failure_step,
                    likelihood=likelihood,
                    unnormalized_posterior=unnormalized,
                    posterior_reward=posterior,
                    outcome_only_reward=answer_correctness,
                    posterior_advantage=posterior - mean_posterior,
                    outcome_advantage=answer_correctness - mean_outcome,
                    solver_format_failure=bool(rollout.get("solver_format_failure", False)),
                    empty_strategy=bool(rollout.get("empty_strategy", False)),
                    empty_reasoning=bool(rollout.get("empty_reasoning", False)),
                    empty_final_answer=bool(rollout.get("empty_final_answer", False)),
                    prior_judge_fallback_used=prior_judge_fallback_used,
                    evidence_judge_fallback_used=evidence_fallback_used,
                    prior_missing_from_judge=prior_assessment.missing_from_judge,
                    judge_label_inconsistency=evidence_assessment.judge_label_inconsistency,
                    strategy_source=str(rollout["strategy_source"]),
                    evidence_source=evidence_assessment.evidence_source,
                    generation_backend=str(rollout["generation_backend"]),
                    solver_temperature=float(rollout["solver_temperature"]),
                )
            )

    metrics, per_problem_summary = analyze_records(
        rollout_records,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
    )
    metrics.update(prior_debug_stats)
    config = {
        "experiment_name": args.experiment_name,
        "benchmark": args.benchmark,
        "input_path": args.input_path,
        "sample_path": args.sample_path,
        "problem_source": problem_source,
        "num_problems": args.num_problems,
        "num_rollouts": args.num_rollouts,
        "strategy_source": args.strategy_source,
        "prior_mode": args.prior_mode,
        "evidence_source": args.evidence_source,
        "reward_ablation": args.reward_ablation,
        "likelihood_weights": likelihood_weights,
        "prior_lambda": args.prior_lambda,
        "prior_softmax_temperature": args.prior_softmax_temperature,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "prior_judge_temperature": args.prior_judge_temperature,
        "evidence_judge_temperature": args.evidence_judge_temperature,
        "solver_model": args.solver_model,
        "prior_judge_model": args.prior_judge_model,
        "evidence_judge_model": args.evidence_judge_model,
        "allow_legacy_static": args.allow_legacy_static,
        "allow_heuristic_evidence": args.allow_heuristic_evidence,
        "output_dir": args.output_dir,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "method_note": (
            "Unified solver rollouts with LLM-estimated strategy priors, "
            "LLM-estimated evidence likelihood components, deterministic final-answer correctness, "
            "and answer-heavy posterior reward analysis."
        ),
    }
    run_dir = save_run_outputs(
        args=args,
        config=config,
        rollout_records=rollout_records,
        per_problem_summary=per_problem_summary,
        metrics=metrics,
        raw_solver_outputs=raw_solver_outputs,
        raw_judge_outputs=raw_judge_outputs,
        logger=logger,
    )
    return rollout_records, metrics, run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified Bayesian posterior reward analysis with Qwen solver and LLM judges."
    )
    parser.add_argument("--input_path", default=None, help="Optional JSON/JSONL dataset path.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark dataset name.")
    parser.add_argument(
        "--sample_path",
        default=str(DEFAULT_SAMPLE_PATH),
        help=(
            "Reproducible MATH sample path. Default uses the 100-problem sample; "
            f"pass {LEGACY_SAMPLE_PATH} to reuse the old 50-problem subset."
        ),
    )
    parser.add_argument("--num_problems", type=int, default=100, help="Number of problems to load.")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of rollouts per problem.")
    parser.add_argument(
        "--strategy_source",
        choices=["qwen_unified", "static"],
        default="qwen_unified",
        help="Unified solver-generated strategies by default. 'static' is legacy comparison mode only.",
    )
    parser.add_argument(
        "--prior_mode",
        choices=["llm_strategy_prior", "uniform"],
        default="llm_strategy_prior",
        help="Strategy prior mode.",
    )
    parser.add_argument(
        "--evidence_source",
        choices=["llm_judge", "heuristic"],
        default="llm_judge",
        help="Evidence scoring mode.",
    )
    parser.add_argument(
        "--allow_legacy_static",
        action="store_true",
        help="Explicitly allow legacy static strategy mode for comparison runs.",
    )
    parser.add_argument(
        "--allow_heuristic_evidence",
        action="store_true",
        help="Explicitly allow heuristic evidence mode for comparison runs.",
    )
    parser.add_argument(
        "--reward_ablation",
        choices=["answer_heavy", "full"],
        default="answer_heavy",
        help="Likelihood weight preset.",
    )
    parser.add_argument("--prior_lambda", type=float, default=1.0, help="Exponent applied to prior before posterior normalization.")
    parser.add_argument(
        "--prior_softmax_temperature",
        type=float,
        default=1.0,
        help="Temperature used when converting LLM suitability scores into prior probabilities.",
    )
    parser.add_argument("--answer_correctness_weight", type=float, default=None, help="Optional override.")
    parser.add_argument("--step_validity_weight", type=float, default=None, help="Optional override.")
    parser.add_argument("--proof_completeness_weight", type=float, default=None, help="Optional override.")
    parser.add_argument("--strategy_compliance_weight", type=float, default=None, help="Optional override.")
    parser.add_argument("--consistency_weight", type=float, default=None, help="Optional override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for solver rollouts.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens per solver rollout.")
    parser.add_argument("--judge_max_new_tokens", type=int, default=768, help="Max new tokens per judge call.")
    parser.add_argument("--prior_judge_temperature", type=float, default=0.0, help="Temperature for prior judge.")
    parser.add_argument("--evidence_judge_temperature", type=float, default=0.0, help="Temperature for evidence judge.")
    parser.add_argument("--solver_model", default=DEFAULT_SOLVER_MODEL, help="Solver model name.")
    parser.add_argument("--prior_judge_model", default=DEFAULT_PRIOR_JUDGE_MODEL, help="Prior judge model name.")
    parser.add_argument("--evidence_judge_model", default=DEFAULT_EVIDENCE_JUDGE_MODEL, help="Evidence judge model name.")
    parser.add_argument("--strict_qwen", action="store_true", help="Accepted for compatibility; unified Qwen generation is required either way.")
    parser.add_argument("--experiment_name", default=DEFAULT_EXPERIMENT_NAME, help="Experiment name used in the output path.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_ROOT, help="Base output directory.")
    return parser


def print_console_summary(metrics: dict[str, Any], run_dir: Path) -> None:
    print(f"output_dir: {run_dir}")
    print(f"num_problems: {metrics['num_problems']}")
    print(f"num_rollouts_total: {metrics['num_rollouts_total']}")
    print(f"expected_num_rollouts_total: {metrics['expected_num_rollouts_total']}")
    print(f"exact_rollout_count_match: {metrics['exact_rollout_count_match']}")
    print(f"posterior_top1_accuracy: {metrics['posterior_top1_accuracy']:.6f}")
    print(f"likelihood_top1_accuracy: {metrics['likelihood_top1_accuracy']:.6f}")
    print(f"outcome_only_top1_accuracy: {metrics['outcome_only_top1_accuracy']:.6f}")
    print(f"oracle_best_of_n: {metrics['oracle_best_of_n']:.6f}")
    print(f"random_accuracy: {metrics['random_accuracy']:.6f}")
    print(f"self_consistency_accuracy: {metrics['self_consistency_accuracy']:.6f}")
    print(f"posterior_mass_on_correct: {metrics['posterior_mass_on_correct']:.6f}")
    print(f"posterior_entropy: {metrics['posterior_entropy']:.6f}")
    print(f"posterior_top1_top2_gap: {metrics['posterior_top1_top2_gap']:.6f}")
    print(
        "posterior_advantage_unique_values_mean: "
        f"{metrics['posterior_advantage_unique_values_mean']:.6f}"
    )
    print(f"incorrect_high_posterior_count: {metrics['incorrect_high_posterior_count']}")
    print(f"wrong_direction_high_posterior_count: {metrics['wrong_direction_high_posterior_count']}")
    print(f"prior_judge_fallback_count: {metrics['prior_judge_fallback_count']}")
    print(f"evidence_judge_fallback_count: {metrics['evidence_judge_fallback_count']}")
    print(f"solver_format_failure_count: {metrics['solver_format_failure_count']}")
    print(f"empty_strategy_count: {metrics['empty_strategy_count']}")
    print(f"empty_reasoning_count: {metrics['empty_reasoning_count']}")
    print(f"empty_final_answer_count: {metrics['empty_final_answer_count']}")
    print(f"judge_label_inconsistency_count: {metrics['judge_label_inconsistency_count']}")
    print(f"prior_judge_parse_failure_count: {metrics['prior_judge_parse_failure_count']}")
    print(f"prior_judge_coverage_failure_count: {metrics['prior_judge_coverage_failure_count']}")
    print(f"prior_judge_retry_success_count: {metrics['prior_judge_retry_success_count']}")
    print(f"prior_missing_rollout_count: {metrics['prior_missing_rollout_count']}")
    print(f"prior_duplicate_rollout_count: {metrics['prior_duplicate_rollout_count']}")
    print(f"prior_invalid_rollout_count: {metrics['prior_invalid_rollout_count']}")
    print(f"prior_probability_source_distribution: {metrics['prior_probability_source_distribution']}")
    print(f"evidence_source_distribution: {metrics['evidence_source_distribution']}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _, metrics, run_dir = run_experiment(args)
    print_console_summary(metrics, run_dir)


if __name__ == "__main__":
    main()
