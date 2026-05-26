# Experiment Results Summary

Last updated: 2026-05-24 17:14:24 KST

This file stores the current paper/result table and the local eval/test configs found on this H200 server. The master table below is the user-provided working table. The eval/test config catalog is derived from local `outputs/**/summary.json` files that contain actual evaluation accuracy fields.

## Master Results Table

| Model | Method | GSM8K | MATH 500 | Minerva | OlympiadBench | Average |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | Pure-based | 51.48% | 50.00% | 4.04% | 6.23% | 27.94% |
| Qwen3-1.7B | Base (Structured Prompt) | 77.71% | 60.60% | 15.07% | 24.33% | 44.43% |
| Qwen3-1.7B | Answer-only GRPO | 77.26% | 60.20% | 14.71% | 26.41% | 44.65% |
| Qwen3-1.7B | BPR-GRPO (Prompted Analyzer) | 79.61% | 62.60% | 16.54% | 29.67% | 47.11% |
| Qwen3-1.7B | BPR-GRPO (Learned Analyzer) | 79.53% | 60.00% | 13.24% | 27.00% | 44.94% |
| Qwen3-4B | Pure-based | 47.76% | 52.00% | 5.88% | 4.75% | 27.60% |
| Qwen3-4B | Base (Structured Prompt) | 88.25% | 70.40% | 25.74% | 36.35% | 55.19% |
| Qwen3-4B | Answer-only GRPO | 90.14% | 69.40% | 24.26% | 35.16% | 54.74% |
| Qwen3-4B | BPR-GRPO (Prompted Analyzer) | 90.45% | 73.40% | 26.47% | 34.87% | 56.30% |
| Qwen3-4B | BPR-GRPO (Learned Analyzer) | 90.67% | 72.80% | in-progress | in-progress | in-progress |
| Qwen3-8B | Pure-based | 47.23% | 53.40% | 5.88% | 4.45% | 27.74% |
| Qwen3-8B | Base (Structured Prompt) | 92.49% | 73.20% | 26.84% | 39.76% | 58.07% |
| Qwen3-8B | Answer-only GRPO | 92.95% | 72.40% | 27.21% | 40.65% | 58.30% |
| Qwen3-8B | BPR-GRPO (Prompted Analyzer) | 92.57% | 75.00% | 27.21% | 44.96% | 59.94% |
| Qwen3-8B | BPR-GRPO (Learned Analyzer) | 91.89% | in-progress | in-progress | in-progress | in-progress |

## In-Progress Work On This Server

| GPU | Experiment | Current stage at last check | Pipeline root |
|---:|---|---|---|
| 0 | Qwen3-8B BPR-GRPO (Learned Analyzer) / MATH500 | SFT posterior recompute | `outputs/math500_experiments/qwen8b_math500_bpr_grpo_learned_analyzer_pipeline_n8_steps1500` |
| 1 | Qwen3-8B BPR-GRPO (Learned Analyzer) / BigMath | SFT posterior recompute | `outputs/bigmath_barl_style_12x1024_seed42/qwen8b_bigmath_bpr_grpo_learned_analyzer_pipeline_n8_steps1536` |

## Local Eval/Test Config Catalog

Only rows below were found as completed local eval/test `summary.json` outputs on this server. Some master-table cells may have been produced on another server or recorded manually; those remain in the master table but do not appear here unless the local summary exists.

| Local result | Data | Result | Config | Summary path |
|---|---|---:|---|---|
| Qwen3-1.7B BPR-GRPO Prompted Analyzer BigMath / AIME26 | `aime26_metadata.jsonl` | 2/30 = 6.67% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.50, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096/aime26/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer BigMath / MinervaMath | `minervamath_metadata.jsonl` | 42/272 = 15.44% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.50, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096/minervamath/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer BigMath / OlympiadBench | `olympiadbench_metadata.jsonl` | 200/674 = 29.67% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.50, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096/olympiadbench/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer BigMath / AIME26 rerun1 | `aime26_metadata.jsonl` | 0/30 = 0.00% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.50, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096_rerun1/aime26/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer BigMath / MinervaMath rerun1 | `minervamath_metadata.jsonl` | 45/272 = 16.54% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.50, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096_rerun1/minervamath/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer BigMath / GSM8K test | `selected_test_metadata.jsonl` | 1023/1319 = 77.56% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.50, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_gsm8k_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-8B BPR-GRPO Prompted Analyzer BigMath / AIME26 | `aime26_metadata.jsonl` | 4/30 = 13.33% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/aime26/summary.json` |
| Qwen3-8B BPR-GRPO Prompted Analyzer BigMath / MinervaMath | `minervamath_metadata.jsonl` | 67/272 = 24.63% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/minervamath/summary.json` |
| Qwen3-8B BPR-GRPO Prompted Analyzer BigMath / MinervaMath rerun1 | `minervamath_metadata.jsonl` | 74/272 = 27.21% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/minervamath_rerun1/summary.json` |
| Qwen3-8B BPR-GRPO Prompted Analyzer BigMath / OlympiadBench | `olympiadbench_metadata.jsonl` | 303/674 = 44.96% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/olympiadbench/summary.json` |
| Qwen3-4B Dr.GRPO Answer-only BigMath / AIME26 | `aime26_metadata.jsonl` | 2/30 = 6.67% | bs16, deterministic, prompt2048, new4096, seed42, transformers backend | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/aime26/summary.json` |
| Qwen3-4B Dr.GRPO Answer-only BigMath / MinervaMath | `minervamath_metadata.jsonl` | 69/272 = 25.37% | bs16, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/minervamath/summary.json` |
| Qwen3-4B Dr.GRPO Answer-only BigMath / OlympiadBench | `olympiadbench_metadata.jsonl` | 247/674 = 36.65% | bs64, deterministic, prompt2048, new1024, seed42, vLLM, TP1, mem0.90, model_len3072 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_olympiadbench_vllm_bs64_tok1024/summary.json` |
| Qwen3-4B Dr.GRPO Answer-only MATH-500 / MATH-500 | `selected_test_metadata.jsonl` | 357/500 = 71.40% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096/summary.json` |
| Qwen3-8B BPR-GRPO Learned Analyzer / GSM8K test | `selected_test_metadata.jsonl` | 1212/1319 = 91.89% | bs64, deterministic, prompt2048, new1024, seed42, vLLM, TP1, mem0.90, model_len3072 | `outputs/gsm8k_experiments/bpr_grpo_learned_analyzer_qwen3_8b_fulltrain_n8_steps1000_bsz4_acc2_vllm/eval_test_vllm_bs64_tok1024_rerun/summary.json` |
| Qwen3-1.7B BPR-GRPO Learned Analyzer / GSM8K test bs32 rerun | `selected_test_metadata.jsonl` | 1049/1319 = 79.53% | bs32, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/eval_test_bs32_rerun/summary.json` |
| Qwen3-1.7B BPR-GRPO Learned Analyzer / GSM8K test bs32 rerun1 | `selected_test_metadata.jsonl` | 1044/1319 = 79.15% | bs32, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/eval_test_bs32_rerun1/summary.json` |
| Qwen3-1.7B BPR-GRPO Learned Analyzer / GSM8K test bs64 | `selected_test_metadata.jsonl` | 1038/1319 = 78.70% | bs64, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/gsm8k_full_qwen3_1p7b/grpo_bayesian_sft_dpo_analyzer_qwen1p7b_fulltrain_n8_steps1000_bsz8_acc1_lambda1/eval_test_bs64/summary.json` |
| Qwen3-4B Answer-only GRPO / GSM8K test | `selected_test_metadata.jsonl` | 1189/1319 = 90.14% | bs64, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/gsm8k_full_qwen3_4b/grpo_answer_only_qwen4b_fulltrain_n8_steps1500_bsz8_acc1/eval_test_bs64/summary.json` |
| Qwen3-4B BPR-GRPO Prompted Analyzer / GSM8K test | `selected_test_metadata.jsonl` | 1193/1319 = 90.45% | bs64, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/eval_test_bs64/summary.json` |
| Qwen3-8B Answer-only GRPO / GSM8K test | `selected_test_metadata.jsonl` | 1226/1319 = 92.95% | bs64, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/gsm8k_full_qwen3_8b/grpo_answer_only_qwen8b_fulltrain_n8_steps1000_bsz4_acc2/eval_test_bs64/summary.json` |
| Qwen3-1.7B Answer-only GRPO / MATH500 test | `selected_test_metadata.jsonl` | 301/500 = 60.20% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/grpo_answer_only_qwen3_1p7b_fulltrain12k_n8_steps1500/eval_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-4B Answer-only GRPO / MATH500 test | `selected_test_metadata.jsonl` | 347/500 = 69.40% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/eval_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-4B Answer-only GRPO / MATH500 test checkpoint1500 old run | `selected_test_metadata.jsonl` | 344/500 = 68.80% | bs16, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/test_eval_checkpoint1500/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer / MATH500 test old run | `selected_test_metadata.jsonl` | 299/500 = 59.80% | bs64, deterministic, prompt2048, new1024, seed42, transformers backend | `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_bs64/summary.json` |
| Qwen3-1.7B BPR-GRPO Prompted Analyzer / MATH500 test | `selected_test_metadata.jsonl` | 313/500 = 62.60% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-4B BPR-GRPO Prompted Analyzer / MATH500 test old vLLM run | `selected_test_metadata.jsonl` | 317/500 = 63.40% | bs64, deterministic, prompt2048, new1024, seed42, vLLM, TP1, mem0.90, model_len3072 | `outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0/eval_test_vllm_bs64/summary.json` |
| Qwen3-4B BPR-GRPO Prompted Analyzer / MATH500 test | `selected_test_metadata.jsonl` | 367/500 = 73.40% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0/eval_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-4B BPR-GRPO Learned Analyzer / MATH500 test | `selected_test_metadata.jsonl` | 364/500 = 72.80% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-1.7B Base Structured Prompt / MATH500 test | `selected_test_metadata.jsonl` | 303/500 = 60.60% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/qwen3_1p7b_base_structured_prompt/eval_test_vllm_bs64_tok4096/summary.json` |
| Qwen3-4B Base Structured Prompt / MATH500 test | `selected_test_metadata.jsonl` | 352/500 = 70.40% | bs64, deterministic, prompt2048, new4096, seed42, vLLM, TP1, mem0.90, model_len6144 | `outputs/math500_experiments/qwen3_4b_base_structured_prompt/eval_test_vllm_bs64_tok4096/summary.json` |

## Local Training Configs Relevant To Active Comparison

| Experiment | Config path | Notes |
|---|---|---|
| Qwen3-8B BPR-GRPO Prompted Analyzer / BigMath | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/training_config.json` | `train_size=12288`, `max_steps=1536`, `num_generations=8`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=2`, `learning_rate=5e-6`, `max_prompt_length=2048`, `max_completion_length=1024`, `vllm_max_model_length=3072` |
| Qwen3-8B BPR-GRPO Learned Analyzer / BigMath | `run_qwen8b_bigmath_bpr_grpo_learned_analyzer_vllm.sh` | Built to match the prompted BigMath config for final GRPO; currently in progress. |
| Qwen3-8B BPR-GRPO Learned Analyzer / MATH500 | `run_qwen8b_math500_bpr_grpo_learned_analyzer_vllm.sh` | Current H200 pipeline in progress. |
