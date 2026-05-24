#!/usr/bin/env python3
"""Run unified max-new-tokens=4096 evals with repeat aggregation.

This script orchestrates the existing evaluation entrypoints:

- eval_pure_base_model.py for pure-base chat-template problem-only evals
- eval_solver_checkpoint.py for structured-prompt base and LoRA checkpoint evals

It is intentionally manifest-driven because the current result table can use
different adapters for different benchmark columns of the same model/method
row, e.g. GSM8K-trained adapters for GSM8K, MATH-trained adapters for MATH-500,
and BigMath-trained adapters for Minerva/OlympiadBench.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS: dict[str, dict[str, str]] = {
    "gsm8k": {
        "label": "GSM8K",
        "metadata_path": "outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl",
    },
    "math500": {
        "label": "MATH 500",
        "metadata_path": "outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl",
    },
    "minervamath": {
        "label": "Minerva",
        "metadata_path": "outputs/eval_benchmarks/minervamath_metadata.jsonl",
    },
    "olympiadbench": {
        "label": "OlympiadBench",
        "metadata_path": "outputs/eval_benchmarks/olympiadbench_metadata.jsonl",
    },
}

MODEL_NAMES = {
    "qwen3_1p7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}

INCOMING_ROOT = Path("outputs/incoming/unified_4096_eval_adapters")


def incoming(name: str) -> str:
    return str(INCOMING_ROOT / name)


# Canonical table manifest. Adapter paths may be local existing checkpoints or
# incoming placeholders created for files transferred from other servers.
RUN_SPECS: list[dict[str, Any]] = [
    # Pure base: chat-template only, no system prompt, no user instruction.
    {"model_key": "qwen3_1p7b", "method": "Pure-based", "eval_type": "pure"},
    {"model_key": "qwen3_4b", "method": "Pure-based", "eval_type": "pure"},
    {"model_key": "qwen3_8b", "method": "Pure-based", "eval_type": "pure"},
    # Base structured prompt: solver prompt, no adapter.
    {"model_key": "qwen3_1p7b", "method": "Base (Structured Prompt)", "eval_type": "solver_base"},
    {"model_key": "qwen3_4b", "method": "Base (Structured Prompt)", "eval_type": "solver_base"},
    {"model_key": "qwen3_8b", "method": "Base (Structured Prompt)", "eval_type": "solver_base"},
    # Answer-only GRPO.
    {
        "model_key": "qwen3_1p7b",
        "method": "Answer-only GRPO",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_1p7b_answer_only_gsm8k"),
            "math500": "outputs/math500_experiments/grpo_answer_only_qwen3_1p7b_fulltrain12k_n8_steps1500/checkpoint-1500",
            "minervamath": "outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500",
            "olympiadbench": "outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_1p7b_n8_steps1500/checkpoint-1500",
        },
    },
    {
        "model_key": "qwen3_4b",
        "method": "Answer-only GRPO",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_4b_answer_only_gsm8k"),
            "math500": "outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/checkpoint-1500",
            "minervamath": "outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/checkpoint-1536",
            "olympiadbench": "outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_4b_n8_steps1536/checkpoint-1536",
        },
    },
    {
        "model_key": "qwen3_8b",
        "method": "Answer-only GRPO",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_8b_answer_only_gsm8k"),
            "math500": incoming("qwen3_8b_answer_only_math500"),
            "minervamath": "outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536/checkpoint-1536",
            "olympiadbench": "outputs/bigmath_barl_style_12x1024_seed42/grpo_answer_only_qwen3_8b_n8_steps1536/checkpoint-1536",
        },
    },
    # BPR-GRPO prompted analyzer.
    {
        "model_key": "qwen3_1p7b",
        "method": "BPR-GRPO (Prompted Analyzer)",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_1p7b_bpr_prompted_gsm8k"),
            "math500": incoming("qwen3_1p7b_bpr_prompted_math500"),
            "minervamath": incoming("qwen3_1p7b_bpr_prompted_bigmath"),
            "olympiadbench": incoming("qwen3_1p7b_bpr_prompted_bigmath"),
        },
    },
    {
        "model_key": "qwen3_4b",
        "method": "BPR-GRPO (Prompted Analyzer)",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_4b_bpr_prompted_gsm8k"),
            "math500": incoming("qwen3_4b_bpr_prompted_math500"),
            "minervamath": incoming("qwen3_4b_bpr_prompted_bigmath"),
            "olympiadbench": incoming("qwen3_4b_bpr_prompted_bigmath"),
        },
    },
    {
        "model_key": "qwen3_8b",
        "method": "BPR-GRPO (Prompted Analyzer)",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": "outputs/gsm8k_experiments/bpr_grpo_prompted_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz8_acc1_judge768_vllm/checkpoint-1000",
            "math500": "outputs/math500_experiments/grpo_bayesian_prompted_qwen3_8b_fulltrain12k_n8_steps1500_bsz1_acc8_lambda1_vllm/checkpoint-1500",
            "minervamath": incoming("qwen3_8b_bpr_prompted_bigmath"),
            "olympiadbench": incoming("qwen3_8b_bpr_prompted_bigmath"),
        },
    },
    # BPR-GRPO learned analyzer.
    {
        "model_key": "qwen3_1p7b",
        "method": "BPR-GRPO (Learned Analyzer)",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_1p7b_bpr_learned_gsm8k"),
            "math500": "outputs/math500_experiments/bayesian_prompted_qwen3_1p7b_sft_dpo_grpo_fulltrain12k_n8_steps1500_lambda1_originaldebug_dpo_vllm_pro6000_r5_native_eager/checkpoint-1500",
            "minervamath": incoming("qwen3_1p7b_bpr_learned_bigmath"),
            "olympiadbench": incoming("qwen3_1p7b_bpr_learned_bigmath"),
        },
    },
    {
        "model_key": "qwen3_4b",
        "method": "BPR-GRPO (Learned Analyzer)",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_4b_bpr_learned_gsm8k"),
            "math500": incoming("qwen3_4b_bpr_learned_math500"),
            "minervamath": "outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500",
            "olympiadbench": "outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_learned_qwen3_4b_n8_steps1500_bsz8_acc1_lambda1/checkpoint-1500",
        },
    },
    {
        "model_key": "qwen3_8b",
        "method": "BPR-GRPO (Learned Analyzer)",
        "eval_type": "solver_adapter",
        "adapters": {
            "gsm8k": incoming("qwen3_8b_bpr_learned_gsm8k"),
            "math500": incoming("qwen3_8b_bpr_learned_math500"),
            "minervamath": incoming("qwen3_8b_bpr_learned_bigmath"),
            "olympiadbench": incoming("qwen3_8b_bpr_learned_bigmath"),
        },
    },
]


@dataclass(frozen=True)
class Job:
    repeat_index: int
    seed: int
    model_key: str
    model_name: str
    method: str
    eval_type: str
    dataset_key: str
    dataset_label: str
    metadata_path: str
    output_dir: Path
    adapter_path: str | None

    @property
    def run_slug(self) -> str:
        return f"{self.model_key}_{slugify(self.method)}"


def job_to_dict(job: Job) -> dict[str, Any]:
    payload = dict(job.__dict__)
    payload["output_dir"] = str(job.output_dir)
    return payload


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "item"


def parse_csv_filter(value: str) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def adapter_exists(path: str | None) -> bool:
    if not path:
        return True
    p = Path(path)
    return (p / "adapter_config.json").exists() and (
        (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()
    )


def build_jobs(args: argparse.Namespace) -> list[Job]:
    model_filter = parse_csv_filter(args.models)
    method_filter = parse_csv_filter(args.methods)
    method_filter_slugs = {slugify(item) for item in method_filter} if method_filter else None
    dataset_filter = parse_csv_filter(args.benchmarks)
    if dataset_filter:
        normalized_dataset_filter = set()
        aliases = {
            "gsm": "gsm8k",
            "gsm8k": "gsm8k",
            "math": "math500",
            "math500": "math500",
            "math-500": "math500",
            "minerva": "minervamath",
            "minervamath": "minervamath",
            "olympiad": "olympiadbench",
            "olympiadbench": "olympiadbench",
        }
        for item in dataset_filter:
            normalized_dataset_filter.add(aliases.get(item.lower(), item.lower()))
        dataset_filter = normalized_dataset_filter

    jobs: list[Job] = []
    for repeat_index in range(args.repeats):
        seed = args.seed_start + repeat_index
        repeat_dir = Path(args.output_root) / f"repeat_{repeat_index:02d}"
        for spec in RUN_SPECS:
            model_key = spec["model_key"]
            method = spec["method"]
            if model_filter and model_key not in model_filter:
                continue
            if method_filter and slugify(method) not in method_filter_slugs and method not in method_filter:
                continue
            eval_type = spec["eval_type"]
            model_name = MODEL_NAMES[model_key]
            for dataset_key, dataset in DATASETS.items():
                if dataset_filter and dataset_key not in dataset_filter:
                    continue
                adapter_path = None
                if eval_type == "solver_adapter":
                    adapter_path = spec["adapters"][dataset_key]
                output_dir = repeat_dir / f"{model_key}_{slugify(method)}" / dataset_key
                jobs.append(
                    Job(
                        repeat_index=repeat_index,
                        seed=seed,
                        model_key=model_key,
                        model_name=model_name,
                        method=method,
                        eval_type=eval_type,
                        dataset_key=dataset_key,
                        dataset_label=dataset["label"],
                        metadata_path=dataset["metadata_path"],
                        output_dir=output_dir,
                        adapter_path=adapter_path,
                    )
                )
    return jobs


def command_for_job(job: Job, args: argparse.Namespace) -> list[str]:
    python = args.python
    if job.eval_type == "pure":
        return [
            python,
            "eval_pure_base_model.py",
            "--model_key",
            job.model_key,
            "--dataset_key",
            job.dataset_key,
            "--output_dir",
            str(job.output_dir),
            "--batch_size",
            str(args.batch_size),
            "--max_examples",
            "0",
            "--max_prompt_length",
            str(args.max_prompt_length),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--seed",
            str(job.seed),
            "--no_do_sample",
            "--bf16",
            "--use_chat_template",
            "--use_vllm",
            "--vllm_gpu_memory_utilization",
            str(args.vllm_gpu_memory_utilization),
            "--vllm_tensor_parallel_size",
            str(args.vllm_tensor_parallel_size),
            "--vllm_max_model_len",
            str(args.vllm_max_model_length),
        ]

    cmd = [
        python,
        "eval_solver_checkpoint.py",
        "--model_name",
        job.model_name,
        "--eval_metadata_path",
        job.metadata_path,
        "--output_dir",
        str(job.output_dir),
        "--batch_size",
        str(args.batch_size),
        "--max_examples",
        "0",
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--seed",
        str(job.seed),
        "--no_do_sample",
        "--bf16",
        "--use_vllm",
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_tensor_parallel_size",
        str(args.vllm_tensor_parallel_size),
        "--vllm_max_model_length",
        str(args.vllm_max_model_length),
    ]
    if job.eval_type == "solver_adapter":
        assert job.adapter_path is not None
        cmd.extend(["--adapter_path", job.adapter_path, "--load_adapter"])
    else:
        cmd.append("--no_load_adapter")
    return cmd


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_missing(jobs: list[Job]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.eval_type != "solver_adapter":
            continue
        if adapter_exists(job.adapter_path):
            continue
        key = (job.model_key, job.method, job.dataset_key, str(job.adapter_path))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "model_key": job.model_key,
                "model_name": job.model_name,
                "method": job.method,
                "benchmark": job.dataset_key,
                "expected_adapter_dir": job.adapter_path,
                "required_files": ["adapter_config.json", "adapter_model.safetensors"],
            }
        )
    return rows


def prepare_incoming_dirs(missing_rows: list[dict[str, Any]]) -> None:
    for row in missing_rows:
        adapter_dir = Path(row["expected_adapter_dir"])
        adapter_dir.mkdir(parents=True, exist_ok=True)
        readme = adapter_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Incoming LoRA adapter placeholder\n\n"
                "Place `adapter_config.json` and `adapter_model.safetensors` here.\n",
                encoding="utf-8",
            )


def run_job(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = job.output_dir / "summary.json"
    log_path = job.output_dir / "eval.log"
    command_path = job.output_dir / "command.json"
    if summary_path.exists() and not args.rerun_existing:
        return {"status": "skipped_existing", "summary_path": str(summary_path)}

    if job.eval_type == "solver_adapter" and not adapter_exists(job.adapter_path):
        message = f"missing adapter for {job.model_key} {job.method} {job.dataset_key}: {job.adapter_path}"
        if args.strict:
            raise RuntimeError(message)
        return {"status": "skipped_missing_adapter", "message": message}

    job.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = command_for_job(job, args)
    write_json(
        command_path,
        {
            "command": cmd,
            "job": job_to_dict(job),
        },
    )
    if args.dry_run:
        return {"status": "dry_run", "command": cmd}

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"[RUN] repeat={job.repeat_index} {job.model_key} | {job.method} | {job.dataset_key}", flush=True)
    print("[CMD] " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("[CMD] " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(cmd, cwd=args.repo_root, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        if args.keep_going:
            return {"status": "failed", "returncode": completed.returncode, "log_path": str(log_path)}
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return {"status": "completed", "summary_path": str(summary_path), "log_path": str(log_path)}


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def aggregate(jobs: list[Job], output_root: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for job in jobs:
        summary = load_summary(job.output_dir / "summary.json")
        if not summary:
            continue
        grouped.setdefault((job.model_key, job.method, job.dataset_key), []).append(
            {
                "repeat_index": job.repeat_index,
                "seed": job.seed,
                "accuracy": summary.get("accuracy"),
                "correct": summary.get("correct_count", summary.get("correct")),
                "num_examples": summary.get("num_examples"),
                "generated_length_mean": summary.get("generated_length_mean"),
                "summary_path": str(job.output_dir / "summary.json"),
            }
        )

    cells: dict[str, Any] = {}
    for (model_key, method, dataset_key), rows in sorted(grouped.items()):
        accuracies = [float(row["accuracy"]) for row in rows if row.get("accuracy") is not None]
        gen_lengths = [
            float(row["generated_length_mean"])
            for row in rows
            if row.get("generated_length_mean") is not None
        ]
        key = f"{model_key}|{method}|{dataset_key}"
        cells[key] = {
            "model_key": model_key,
            "model_name": MODEL_NAMES[model_key],
            "method": method,
            "benchmark": dataset_key,
            "num_repeats_completed": len(accuracies),
            "accuracy_mean": statistics.fmean(accuracies) if accuracies else None,
            "accuracy_std": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0 if accuracies else None,
            "generated_length_mean": statistics.fmean(gen_lengths) if gen_lengths else None,
            "repeats": rows,
        }

    table_rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        model_key = spec["model_key"]
        method = spec["method"]
        row = {
            "model_key": model_key,
            "model_name": MODEL_NAMES[model_key],
            "method": method,
            "benchmarks": {},
            "average_accuracy_mean": None,
        }
        means: list[float] = []
        for dataset_key in DATASETS:
            cell = cells.get(f"{model_key}|{method}|{dataset_key}")
            row["benchmarks"][dataset_key] = cell
            if cell and cell.get("accuracy_mean") is not None:
                means.append(float(cell["accuracy_mean"]))
        if len(means) == len(DATASETS):
            row["average_accuracy_mean"] = statistics.fmean(means)
        table_rows.append(row)

    payload = {
        "output_root": str(output_root),
        "num_cells_completed": len(cells),
        "cells": cells,
        "table_rows": table_rows,
    }
    write_json(output_root / "aggregate_summary.json", payload)
    write_aggregate_markdown(output_root / "aggregate_table.md", table_rows)
    return payload


def format_cell(cell: dict[str, Any] | None) -> str:
    if not cell or cell.get("accuracy_mean") is None:
        return "missing"
    mean = float(cell["accuracy_mean"]) * 100
    std = float(cell.get("accuracy_std") or 0.0) * 100
    n = int(cell.get("num_repeats_completed") or 0)
    if n > 1:
        return f"{mean:.2f}% +/- {std:.2f} (n={n})"
    return f"{mean:.2f}% (n={n})"


def write_aggregate_markdown(path: Path, table_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Unified 4096 Eval Aggregate",
        "",
        "| Model | Method | GSM8K | MATH 500 | Minerva | OlympiadBench | Average |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        benchmark_cells = row["benchmarks"]
        avg = row.get("average_accuracy_mean")
        avg_text = f"{avg * 100:.2f}%" if avg is not None else "missing"
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_NAMES[row["model_key"]].replace("Qwen/Qwen3-", "Qwen3-"),
                    row["method"],
                    format_cell(benchmark_cells.get("gsm8k")),
                    format_cell(benchmark_cells.get("math500")),
                    format_cell(benchmark_cells.get("minervamath")),
                    format_cell(benchmark_cells.get("olympiadbench")),
                    avg_text,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_required_adapters(output_root: Path, missing_rows: list[dict[str, Any]]) -> None:
    write_json(output_root / "required_adapters.json", missing_rows)
    lines = [
        "# Required Incoming Adapters",
        "",
        "Place each LoRA adapter directory at the path below. Each directory should contain `adapter_config.json` and `adapter_model.safetensors`.",
        "",
        "| Model | Method | Benchmark | Expected Dir |",
        "|---|---|---|---|",
    ]
    for row in missing_rows:
        lines.append(
            f"| {row['model_name']} | {row['method']} | {row['benchmark']} | `{row['expected_adapter_dir']}` |"
        )
    (output_root / "required_adapters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--output_root", default="outputs/unified_4096_eval_5x")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_max_model_length", type=int, default=6144)
    parser.add_argument("--cuda_visible_devices", default="")
    parser.add_argument("--models", default="", help="Comma-separated model keys, e.g. qwen3_1p7b,qwen3_4b")
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method names or slugs, e.g. Pure-based,answer_only_grpo",
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark keys: gsm8k,math500,minervamath,olympiadbench",
    )
    parser.add_argument("--check_only", action="store_true", help="Only write required adapter lists and aggregate.")
    parser.add_argument("--dry_run", action="store_true", help="Write commands without executing.")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if an expected adapter is missing.")
    parser.add_argument("--keep_going", action="store_true", help="Continue after a failed eval command.")
    parser.add_argument(
        "--prepare_incoming_dirs",
        action="store_true",
        help="Create missing incoming adapter directories with README placeholders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    missing_rows = collect_missing(jobs)
    if args.prepare_incoming_dirs:
        prepare_incoming_dirs(missing_rows)
    write_required_adapters(output_root, missing_rows)

    manifest = {
        "repeats": args.repeats,
        "seed_start": args.seed_start,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_length": args.max_prompt_length,
        "vllm_max_model_length": args.vllm_max_model_length,
        "num_jobs": len(jobs),
        "num_missing_adapter_cells": len(missing_rows),
        "jobs": [
            {
                **job_to_dict(job),
            }
            for job in jobs
        ],
    }
    write_json(output_root / "run_manifest.json", manifest)

    if args.check_only:
        aggregate(jobs, output_root)
        print(f"[CHECK] jobs={len(jobs)} missing_adapters={len(missing_rows)}")
        print(f"[CHECK] required adapters: {output_root / 'required_adapters.md'}")
        print(f"[CHECK] aggregate: {output_root / 'aggregate_table.md'}")
        return

    results: list[dict[str, Any]] = []
    for job in jobs:
        result = run_job(job, args)
        results.append({"job": job_to_dict(job), **result})
        write_json(output_root / "run_status.json", results)
        aggregate(jobs, output_root)

    aggregate(jobs, output_root)
    print(f"[DONE] aggregate table: {output_root / 'aggregate_table.md'}")


if __name__ == "__main__":
    main()
