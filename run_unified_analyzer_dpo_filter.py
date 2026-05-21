#!/usr/bin/env python3
"""Run the cheap filter for analyzer DPO checkpoints.

This wrapper runs:
1. schema / parse validation on held-out analyzer labels
2. posterior recomputation over one or more prior lambdas
3. optional comparison against a baseline recompute summary

It is designed to gate Base Qwen -> DPO and v0 SFT -> DPO before any solver
GRPO run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_EVIDENCE_VAL_PATH = "outputs/unified_analyzer_sft_v0/evidence_clean_val_marked.jsonl"
DEFAULT_PRIOR_VAL_PATH = "outputs/unified_analyzer_sft_v0/prior_clean_val_marked.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run analyzer DPO schema/posterior cheap filter."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_debug_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--evidence_val_path", default=DEFAULT_EVIDENCE_VAL_PATH)
    parser.add_argument("--prior_val_path", default=DEFAULT_PRIOR_VAL_PATH)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 0.7, 1.0])
    parser.add_argument("--baseline_summary_json", default=None)

    parser.add_argument("--max_examples_per_task", type=int, default=0)
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--answer_weight", type=float, default=0.8)
    parser.add_argument("--evidence_weight", type=float, default=0.2)
    parser.add_argument("--prior_temperature", type=float, default=1.0)
    parser.add_argument("--fallback_mode", choices=["neutral", "teacher"], default="neutral")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_task_prefix", action="store_true")

    parser.add_argument("--min_prior_parse_rate", type=float, default=0.99)
    parser.add_argument("--min_evidence_parse_rate", type=float, default=0.99)
    parser.add_argument("--min_top1_all", type=float, default=0.822)
    parser.add_argument("--min_top1_when_correct_exists", type=float, default=1.0)
    parser.add_argument("--max_wrong_top_count", type=int, default=0)
    parser.add_argument(
        "--mass_on_correct_slack",
        type=float,
        default=0.0,
        help="Allowed drop versus baseline mass_on_correct_mean.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return parsed


def format_lambda_dir_name(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def run_command(command: list[str]) -> None:
    print(f"[INFO] running: {' '.join(command)}")
    subprocess.run(command, check=True)


def extract_learned_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if "results" in summary and "recommendation" in summary:
        primary_candidate = float(summary["recommendation"]["primary_candidate"])
        for row in summary["results"]:
            if abs(float(row["prior_lambda"]) - primary_candidate) <= 1e-9:
                return {
                    "prior_parse_rate": 1.0,
                    "evidence_parse_rate": 1.0,
                    "top1_all": float(row["top1_acc_all"]),
                    "top1_when_correct_exists": float(row["top1_acc_when_correct_exists"]),
                    "wrong_top_count": int(row["wrong_top_when_correct_exists"]),
                    "mass_on_correct_mean": float(row["mass_on_correct_mean"]),
                }
        raise RuntimeError("Primary candidate lambda was not found in baseline results.")
    if "counts" not in summary and "best_lambda" in summary:
        best_metrics = summary["best_lambda"]["metrics"]
        return {
            "prior_parse_rate": float(best_metrics["prior_parse_rate"]),
            "evidence_parse_rate": float(best_metrics["evidence_parse_rate"]),
            "top1_all": float(best_metrics["top1_all"]),
            "top1_when_correct_exists": float(best_metrics["top1_when_correct_exists"]),
            "wrong_top_count": int(best_metrics["wrong_top_count"]),
            "mass_on_correct_mean": float(best_metrics["mass_on_correct_mean"]),
        }
    return {
        "prior_parse_rate": float(summary["counts"]["prior_parse_rate"]),
        "evidence_parse_rate": float(summary["counts"]["evidence_parse_rate"]),
        "top1_all": float(summary["learned"]["posterior_top1_accuracy_all_groups"]),
        "top1_when_correct_exists": float(
            summary["learned"]["posterior_top1_accuracy_when_correct_exists"]
        ),
        "wrong_top_count": int(summary["learned"]["wrong_top_count_when_correct_exists"]),
        "mass_on_correct_mean": float(summary["learned"]["mass_on_correct_mean"]),
    }


def best_lambda_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("No lambda results were produced.")
    return max(
        results,
        key=lambda item: (
            float(item["metrics"]["top1_when_correct_exists"]),
            float(item["metrics"]["top1_all"]),
            -int(item["metrics"]["wrong_top_count"]),
            float(item["metrics"]["mass_on_correct_mean"]),
            float(item["metrics"]["prior_parse_rate"]),
            float(item["metrics"]["evidence_parse_rate"]),
            -float(item["lambda"]),
        ),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_dir = output_dir / "schema_eval"
    eval_command = [
        sys.executable,
        "eval_unified_analyzer.py",
        "--model_name",
        args.model_name,
        "--adapter_path",
        args.adapter_path,
        "--evidence_val_path",
        args.evidence_val_path,
        "--prior_val_path",
        args.prior_val_path,
        "--output_dir",
        str(schema_dir),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    if args.max_examples_per_task > 0:
        eval_command.extend(["--max_examples_per_task", str(args.max_examples_per_task)])
    if args.bf16:
        eval_command.append("--bf16")
    run_command(eval_command)
    schema_summary = read_json(schema_dir / "summary.json")

    lambda_results: list[dict[str, Any]] = []
    for prior_lambda in args.lambdas:
        lambda_dir = output_dir / f"lambda_{format_lambda_dir_name(prior_lambda)}"
        recompute_command = [
            sys.executable,
            "recompute_posterior_with_learned_analyzer.py",
            "--input_debug_jsonl",
            args.input_debug_jsonl,
            "--output_dir",
            str(lambda_dir),
            "--model_name",
            args.model_name,
            "--adapter_path",
            args.adapter_path,
            "--batch_size",
            str(args.batch_size),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--max_input_tokens",
            str(args.max_input_tokens),
            "--answer_weight",
            str(args.answer_weight),
            "--evidence_weight",
            str(args.evidence_weight),
            "--prior_lambda",
            str(prior_lambda),
            "--prior_temperature",
            str(args.prior_temperature),
            "--fallback_mode",
            args.fallback_mode,
        ]
        if args.max_groups > 0:
            recompute_command.extend(["--max_groups", str(args.max_groups)])
        if args.bf16:
            recompute_command.append("--bf16")
        if args.no_task_prefix:
            recompute_command.append("--no_task_prefix")
        run_command(recompute_command)
        summary = read_json(lambda_dir / "summary.json")
        lambda_results.append(
            {
                "lambda": prior_lambda,
                "summary_path": str(lambda_dir / "summary.json"),
                "metrics": extract_learned_metrics(summary),
                "summary": summary,
            }
        )

    best_result = best_lambda_result(lambda_results)
    baseline_summary = (
        read_json(Path(args.baseline_summary_json))
        if args.baseline_summary_json
        else None
    )
    baseline_metrics = extract_learned_metrics(baseline_summary) if baseline_summary else None

    best_metrics = best_result["metrics"]
    gate_checks = {
        "prior_parse_rate": best_metrics["prior_parse_rate"] >= args.min_prior_parse_rate,
        "evidence_parse_rate": best_metrics["evidence_parse_rate"] >= args.min_evidence_parse_rate,
        "top1_all": best_metrics["top1_all"] >= args.min_top1_all,
        "top1_when_correct_exists": (
            best_metrics["top1_when_correct_exists"] >= args.min_top1_when_correct_exists
        ),
        "wrong_top_count": best_metrics["wrong_top_count"] <= args.max_wrong_top_count,
    }

    mass_gate = None
    if baseline_metrics is not None:
        mass_gate = (
            best_metrics["mass_on_correct_mean"]
            >= baseline_metrics["mass_on_correct_mean"] - args.mass_on_correct_slack
        )
        gate_checks["mass_on_correct_vs_baseline"] = mass_gate

    gate_pass = all(gate_checks.values())
    delta_vs_baseline = None
    if baseline_metrics is not None:
        delta_vs_baseline = {
            key: best_metrics[key] - baseline_metrics[key]
            for key in (
                "prior_parse_rate",
                "evidence_parse_rate",
                "top1_all",
                "top1_when_correct_exists",
                "mass_on_correct_mean",
            )
        }
        delta_vs_baseline["wrong_top_count"] = (
            int(best_metrics["wrong_top_count"]) - int(baseline_metrics["wrong_top_count"])
        )

    filter_summary = {
        "config": {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "input_debug_jsonl": args.input_debug_jsonl,
            "evidence_val_path": args.evidence_val_path,
            "prior_val_path": args.prior_val_path,
            "lambdas": args.lambdas,
            "fallback_mode": args.fallback_mode,
            "task_prefix_enabled": not args.no_task_prefix,
            "max_examples_per_task": args.max_examples_per_task,
            "max_groups": args.max_groups,
            "gate_thresholds": {
                "min_prior_parse_rate": args.min_prior_parse_rate,
                "min_evidence_parse_rate": args.min_evidence_parse_rate,
                "min_top1_all": args.min_top1_all,
                "min_top1_when_correct_exists": args.min_top1_when_correct_exists,
                "max_wrong_top_count": args.max_wrong_top_count,
                "mass_on_correct_slack": args.mass_on_correct_slack,
            },
            "baseline_summary_json": args.baseline_summary_json,
        },
        "schema_eval": schema_summary,
        "lambda_results": [
            {
                "lambda": result["lambda"],
                "summary_path": result["summary_path"],
                "metrics": result["metrics"],
            }
            for result in lambda_results
        ],
        "best_lambda": {
            "lambda": best_result["lambda"],
            "summary_path": best_result["summary_path"],
            "metrics": best_metrics,
        },
        "baseline_metrics": baseline_metrics,
        "delta_vs_baseline": delta_vs_baseline,
        "gate_checks": gate_checks,
        "gate_pass": gate_pass,
    }
    write_json(output_dir / "filter_summary.json", filter_summary)

    print(f"[INFO] wrote cheap filter summary to {output_dir / 'filter_summary.json'}")
    print(json.dumps(filter_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
