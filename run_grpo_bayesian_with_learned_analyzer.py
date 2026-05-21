#!/usr/bin/env python3
"""Run GSM8K Bayesian GRPO with a learned analyzer and collect eval summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gsm8k_learned_analyzer_utils import write_json


DEFAULT_METADATA_DIR = "outputs/gsm8k_3000_500_seed42"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_PROMPTED_LOG_DIR = (
    "outputs/gsm8k_experiments/"
    "grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10"
)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dashed_name = name.replace("_", "-")
    positive_options = [f"--{name}"]
    if dashed_name != name:
        positive_options.append(f"--{dashed_name}")

    negative_options = [f"--no-{name}", f"--no_{name}"]
    if dashed_name != name:
        negative_options.append(f"--no-{dashed_name}")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(*positive_options, dest=name, action="store_true", help=help_text)
    group.add_argument(*negative_options, dest=name, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate GSM8K Bayesian GRPO with a learned analyzer adapter."
    )
    parser.add_argument("--analyzer_adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default="fixed_metadata")
    parser.add_argument("--analyzer_model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--metadata_dir", default=DEFAULT_METADATA_DIR)
    parser.add_argument("--train_size", default="3000")
    parser.add_argument("--eval_size", default="500")
    parser.add_argument("--min_solve_rate", type=float, default=0.0)
    parser.add_argument("--max_solve_rate", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    add_bool_arg(
        parser,
        "use_lora",
        True,
        "Enable LoRA for the solver GRPO run. Disable for full fine-tuning.",
    )
    add_bool_arg(parser, "gradient_checkpointing", True, "Enable gradient checkpointing for solver GRPO.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior_lambda", type=float, default=1.0)
    parser.add_argument("--prior_softmax_temperature", type=float, default=1.0)
    parser.add_argument("--judge_max_new_tokens", type=int, default=768)
    parser.add_argument("--progress_interval_percent", type=int, default=10)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)

    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_max_new_tokens", type=int, default=1024)
    parser.add_argument("--eval_max_prompt_length", type=int, default=2048)
    parser.add_argument("--preflight_recompute_log_dir", default=DEFAULT_PROMPTED_LOG_DIR)
    parser.add_argument("--preflight_recompute_output_dir", default="")
    parser.add_argument("--preflight_recompute_batch_size", type=int, default=8)
    parser.add_argument("--preflight_recompute_max_new_tokens", type=int, default=512)
    parser.add_argument("--skip_preflight_recompute", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_valid_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--step_logs_dir", default="")
    parser.add_argument("--method_name", default="")
    parser.add_argument("--train_data_label", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--analyzer_type", default="learned_sft")
    return parser.parse_args()


def load_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resolve_requested_size(requested: str, available_count: int, *, label: str, allow_zero: bool) -> int:
    normalized = str(requested or "").strip().lower()
    if normalized in {"full", "-1"}:
        return available_count
    if normalized in {"", "none", "null"}:
        if allow_zero:
            return 0
        raise SystemExit(f"--{label}_size cannot be empty/none/null")
    try:
        value = int(normalized)
    except ValueError as exc:
        raise SystemExit(
            f"--{label}_size must be an integer, or {'full/-1/none' if allow_zero else 'full/-1'}."
        ) from exc
    if value < 0:
        raise SystemExit(f"--{label}_size={value} is invalid. Use -1/full for all rows.")
    if value == 0 and not allow_zero:
        raise SystemExit(f"--{label}_size must be positive.")
    return value


def print_command(name: str, command: list[str], log_path: Path | None = None) -> None:
    payload = {"name": name, "command": command}
    if log_path is not None:
        payload["log_path"] = str(log_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_and_log(name: str, command: list[str], log_path: Path, *, dry_run: bool) -> None:
    print_command(name, command, log_path)
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def default_method_name(analyzer_type: str) -> str:
    if analyzer_type == "learned_sft_dpo":
        return "GRPO Bayesian Learned SFT+DPO Analyzer"
    if analyzer_type == "learned_sft":
        return "GRPO Bayesian Learned SFT Analyzer"
    return "GRPO Bayesian Learned Analyzer"


def default_train_data_label(train_size: int, eval_size: int) -> str:
    if train_size > 3000 and eval_size == 0:
        return "GSM8K official train full"
    return "GSM8K-3K"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    train_metadata_path = metadata_dir / "selected_train_metadata.jsonl"
    valid_metadata_path = metadata_dir / "selected_valid_metadata.jsonl"
    test_metadata_path = metadata_dir / "selected_test_metadata.jsonl"

    if not train_metadata_path.exists():
        raise RuntimeError(f"Missing train metadata: {train_metadata_path}")
    if not test_metadata_path.exists():
        raise RuntimeError(f"Missing test metadata: {test_metadata_path}")

    train_available = load_jsonl_count(train_metadata_path)
    valid_available = load_jsonl_count(valid_metadata_path) if valid_metadata_path.exists() else 0
    resolved_train_size = resolve_requested_size(
        str(args.train_size),
        train_available,
        label="train",
        allow_zero=False,
    )
    resolved_eval_size = resolve_requested_size(
        str(args.eval_size),
        valid_available,
        label="eval",
        allow_zero=True,
    )
    if args.skip_valid_eval:
        resolved_eval_size = 0

    resolved_valid_metadata_path = valid_metadata_path
    if resolved_eval_size == 0 and not resolved_valid_metadata_path.exists():
        resolved_valid_metadata_path = output_dir / "empty_valid_metadata.jsonl"

    run_valid_eval = resolved_eval_size > 0 and not args.skip_valid_eval
    step_logs_dir = Path(args.step_logs_dir) if args.step_logs_dir else output_dir / "step_logs"
    method_name = args.method_name or default_method_name(args.analyzer_type)
    train_data_label = args.train_data_label or default_train_data_label(
        resolved_train_size,
        resolved_eval_size,
    )
    notes = args.notes or "deterministic full-test evaluation"
    if not run_valid_eval:
        notes = f"{notes}; valid evaluation skipped"
    if resolved_train_size > 3000 and resolved_eval_size == 0:
        normalized_output_dir = str(output_dir).lower()
        if (
            "full" not in normalized_output_dir
        ):
            raise RuntimeError(
                "Full-train runs must use an output path containing 'full'."
            )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "metadata_dir": str(metadata_dir),
                "train_metadata_path": str(train_metadata_path),
                "valid_metadata_path": str(resolved_valid_metadata_path),
                "test_metadata_path": str(test_metadata_path),
                "resolved_train_size": resolved_train_size,
                "resolved_eval_size": resolved_eval_size,
                "run_valid_eval": run_valid_eval,
                "dry_run": args.dry_run,
                "analyzer_type": args.analyzer_type,
                "analyzer_adapter_path": args.analyzer_adapter_path,
                "dataset_name": args.dataset_name,
                "use_lora": args.use_lora,
                "gradient_checkpointing": args.gradient_checkpointing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        step_logs_dir.mkdir(parents=True, exist_ok=True)
        if resolved_eval_size == 0 and not resolved_valid_metadata_path.exists():
            resolved_valid_metadata_path.write_text("", encoding="utf-8")

    train_cmd = [
        "python3",
        "Bayesian_Full_GRPO_learned.py",
        "--model_name",
        args.model_name,
        "--dataset_name",
        args.dataset_name,
        "--prior_mode",
        "learned_unified_analyzer",
        "--analyzer_model_name",
        args.analyzer_model_name,
        "--analyzer_adapter_path",
        args.analyzer_adapter_path,
        "--use_fixed_metadata",
        "--train_metadata_path",
        str(train_metadata_path),
        "--eval_metadata_path",
        str(resolved_valid_metadata_path),
        "--train_size",
        str(resolved_train_size),
        "--eval_size",
        str(resolved_eval_size),
        "--min_solve_rate",
        str(args.min_solve_rate),
        "--max_solve_rate",
        str(args.max_solve_rate),
        "--num_generations",
        str(args.num_generations),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_steps",
        str(args.max_steps),
        "--max_completion_length",
        str(args.max_completion_length),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--seed",
        str(args.seed),
        "--prior_lambda",
        str(args.prior_lambda),
        "--prior_softmax_temperature",
        str(args.prior_softmax_temperature),
        "--judge_max_new_tokens",
        str(args.judge_max_new_tokens),
        "--progress_interval_percent",
        str(args.progress_interval_percent),
        "--reward_debug_jsonl",
        str(output_dir / "bayesian_reward_debug.jsonl"),
        "--output_dir",
        str(output_dir),
    ]
    train_cmd.append("--use_lora" if args.use_lora else "--no-use_lora")
    train_cmd.append(
        "--gradient_checkpointing" if args.gradient_checkpointing else "--no-gradient_checkpointing"
    )
    if args.bf16:
        train_cmd.append("--bf16")

    eval_valid_cmd = [
        "python3",
        "eval_solver_checkpoint.py",
        "--model_name",
        args.model_name,
        "--adapter_path",
        str(output_dir),
        "--eval_metadata_path",
        str(resolved_valid_metadata_path),
        "--output_dir",
        str(output_dir / "valid"),
        "--batch_size",
        str(args.eval_batch_size),
        "--max_new_tokens",
        str(args.eval_max_new_tokens),
        "--max_prompt_length",
        str(args.eval_max_prompt_length),
        "--seed",
        str(args.seed),
        "--no_do_sample",
    ]
    eval_test_cmd = [
        "python3",
        "eval_solver_checkpoint.py",
        "--model_name",
        args.model_name,
        "--adapter_path",
        str(output_dir),
        "--eval_metadata_path",
        str(test_metadata_path),
        "--output_dir",
        str(output_dir / "test"),
        "--batch_size",
        str(args.eval_batch_size),
        "--max_new_tokens",
        str(args.eval_max_new_tokens),
        "--max_prompt_length",
        str(args.eval_max_prompt_length),
        "--seed",
        str(args.seed),
        "--no_do_sample",
    ]
    diagnostics_cmd = [
        "python3",
        "analyze_bayesian_debug_jsonl.py",
        "--input_debug_jsonl",
        str(output_dir / "bayesian_reward_debug.jsonl"),
        "--output_dir",
        str(output_dir / "bayesian_reward_diagnostics"),
    ]
    preflight_output_dir = (
        Path(args.preflight_recompute_output_dir)
        if args.preflight_recompute_output_dir
        else output_dir / "preflight_recompute_on_prompted_pool"
    )
    preflight_cmd = [
        "python3",
        "recompute_posterior_with_learned_analyzer.py",
        "--input_debug_jsonl",
        str(Path(args.preflight_recompute_log_dir) / "bayesian_reward_debug.jsonl"),
        "--output_dir",
        str(preflight_output_dir),
        "--model_name",
        args.analyzer_model_name,
        "--adapter_path",
        args.analyzer_adapter_path,
        "--batch_size",
        str(args.preflight_recompute_batch_size),
        "--max_new_tokens",
        str(args.preflight_recompute_max_new_tokens),
        "--prior_lambda",
        str(args.prior_lambda),
        "--prior_temperature",
        str(args.prior_softmax_temperature),
    ]
    if args.bf16:
        preflight_cmd.append("--bf16")

    summary_cmd = [
        "python3",
        "summarize_gsm8k_experiment.py",
        "--experiment_output_dir",
        str(output_dir),
        "--metadata_dir",
        str(metadata_dir),
        "--method",
        method_name,
        "--train_data",
        train_data_label,
        "--reward",
        "full Bayesian posterior",
        "--analyzer_type",
        args.analyzer_type,
        "--notes",
        notes,
        "--checkpoint_path",
        str(output_dir),
        "--test_summary_path",
        str(output_dir / "test" / "summary.json"),
    ]
    if run_valid_eval:
        summary_cmd.extend(
            [
                "--valid_summary_path",
                str(output_dir / "valid" / "summary.json"),
            ]
        )

    launcher_payload = {
        "preflight_cmd": preflight_cmd,
        "train_cmd": train_cmd,
        "eval_valid_cmd": eval_valid_cmd if run_valid_eval else None,
        "eval_test_cmd": eval_test_cmd,
        "diagnostics_cmd": diagnostics_cmd,
        "summary_cmd": summary_cmd,
        "resolved_train_size": resolved_train_size,
        "resolved_eval_size": resolved_eval_size,
        "run_valid_eval": run_valid_eval,
        "step_logs_dir": str(step_logs_dir),
        "dataset_name": args.dataset_name,
        "use_lora": args.use_lora,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    if args.dry_run:
        print(json.dumps(launcher_payload, ensure_ascii=False, indent=2))
        return

    write_json(output_dir / "launcher_config.json", launcher_payload)

    if not args.skip_preflight_recompute:
        run_and_log(
            "preflight_recompute",
            preflight_cmd,
            step_logs_dir / "00_preflight_recompute.log",
            dry_run=args.dry_run,
        )

    if not args.skip_train:
        run_and_log("train", train_cmd, step_logs_dir / "01_train.log", dry_run=args.dry_run)
        run_and_log(
            "diagnostics",
            diagnostics_cmd,
            step_logs_dir / "02_diagnostics.log",
            dry_run=args.dry_run,
        )

    if not args.skip_eval:
        if run_valid_eval:
            run_and_log(
                "eval_valid",
                eval_valid_cmd,
                step_logs_dir / "03_eval_valid.log",
                dry_run=args.dry_run,
            )
        run_and_log(
            "eval_test",
            eval_test_cmd,
            step_logs_dir / "04_eval_test.log",
            dry_run=args.dry_run,
        )

    test_summary_path = output_dir / "test" / "summary.json"
    if test_summary_path.exists():
        run_and_log(
            "write_summary",
            summary_cmd,
            step_logs_dir / "05_summary.log",
            dry_run=args.dry_run,
        )
    else:
        print(
            f"[WARN] Skipping root summary generation because test summary is missing: {test_summary_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
