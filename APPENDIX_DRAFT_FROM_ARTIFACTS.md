# Appendix Draft From Available Artifacts

This draft is written from artifacts currently available on this server under
`/home/kimjh/Baysian_reward_GRPO`. The latest artifact transfer preserved the
original `outputs/...` paths rather than placing everything under
`outputs/incoming/...`; therefore the counts and tables below are based on the
full `outputs` tree.

## A. Artifact Coverage

Local artifact inventory:

| Artifact type | Local count | Use in appendix |
| --- | ---: | --- |
| `summary.json` | 419 | Evaluation accuracy, correct/total, decoding config, backend, average generation length |
| `training_config.json` | 73 | Training budget, LoRA config, reward type, data size, batch and gradient accumulation |
| `launcher_config.json` | 15 | Learned-analyzer pipeline launch/config provenance |
| `metadata_summary.json` | 10 | Dataset/split metadata provenance |
| `bayesian_reward_debug.jsonl` | 49 | BPR posterior diagnostics, analyzer fallback rates, outcome anchoring audit |
| `predictions.jsonl` | 353 | Per-example eval outputs for audit and error analysis |
| `logs/*.log` | 186 | Train/eval command and completion provenance |

The transferred unified-evaluation artifacts now make it possible to regenerate
the main 5-run summary table from `summary.json` files. Some cells have fewer
than five completed repeats; those counts should be disclosed.

## B. Unified 4096 Main-Table Aggregate

The following table aggregates the transferred unified 4096-token evaluation
artifacts in:

```text
outputs/unified_4096_main3_base_grpo_bpr_prompted_5x_gpu0_1p7b_4b
outputs/unified_4096_main3_base_grpo_bpr_prompted_5x_gpu1_8b
```

Values are arithmetic means over available repeats. The `n` column reports the
minimum and maximum number of repeats available among the four benchmark cells
for that row.

| Model | Method | GSM8K | MATH 500 | MinervaMath | OlympiadBench | Avg. | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | Base | 76.99 | 58.60 | 13.97 | 26.41 | 43.99 | 4-4 |
| Qwen3-1.7B | GRPO | 77.56 | 59.60 | 15.62 | 27.23 | 45.00 | 4-4 |
| Qwen3-1.7B | BPR-GRPO (Ours) | 79.87 | 61.20 | 15.81 | 29.34 | 46.55 | 4-4 |
| Qwen3-4B | Base | 88.99 | 70.15 | 24.54 | 40.02 | 55.93 | 4-4 |
| Qwen3-4B | GRPO | 90.01 | 69.45 | 24.91 | 39.21 | 55.89 | 4-4 |
| Qwen3-4B | BPR-GRPO (Ours) | 90.56 | 73.05 | 25.86 | 41.89 | 57.84 | 3-4 |
| Qwen3-8B | Base | 92.25 | 74.95 | 27.21 | 44.81 | 59.80 | 4-5 |
| Qwen3-8B | GRPO | 92.51 | 74.90 | 27.30 | 44.29 | 59.75 | 4-4 |
| Qwen3-8B | BPR-GRPO (Ours) | 92.44 | 75.00 | 26.01 | 45.03 | 59.62 | 4-4 |

This aggregate supports the main claim at the Qwen3-1.7B and Qwen3-4B scales.
At Qwen3-8B, BPR is not better on the macro-average; the 8B row should either be
reported as a scaling observation or moved out of the main positive claim.

## C. Evaluation Artifact Manifest

The following rows are backed by local `summary.json` files. These can be used
directly in an appendix table or as the source manifest for a final result table.

| Scope | Model | Method/checkpoint | Benchmark | Correct/N | Acc. | Backend | Max new | Summary |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| GSM8K eval | Qwen3-4B | Answer-only GRPO | GSM8K | 1189/1319 | 90.14 | transformers | 1024 | `outputs/gsm8k_full_qwen3_4b/grpo_answer_only_qwen4b_fulltrain_n8_steps1500_bsz8_acc1/eval_test_bs64/summary.json` |
| GSM8K eval | Qwen3-4B | BPR-GRPO prompted | GSM8K | 1193/1319 | 90.45 | transformers | 1024 | `outputs/gsm8k_full_qwen3_4b/grpo_bayesian_prompted_qwen4b_fulltrain_n8_steps1500_bsz8_acc1_lambda1/eval_test_bs64/summary.json` |
| GSM8K eval | Qwen3-8B | Answer-only GRPO | GSM8K | 1226/1319 | 92.95 | transformers | 1024 | `outputs/gsm8k_full_qwen3_8b/grpo_answer_only_qwen8b_fulltrain_n8_steps1000_bsz4_acc2/eval_test_bs64/summary.json` |
| MATH-500 eval | Qwen3-1.7B | Base structured prompt | MATH-500 | 303/500 | 60.60 | vLLM | 4096 | `outputs/math500_experiments/qwen3_1p7b_base_structured_prompt/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-1.7B | Answer-only GRPO | MATH-500 | 301/500 | 60.20 | vLLM | 4096 | `outputs/math500_experiments/grpo_answer_only_qwen3_1p7b_fulltrain12k_n8_steps1500/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-1.7B | BPR-GRPO prompted | MATH-500 | 313/500 | 62.60 | vLLM | 4096 | `outputs/math500_experiments/grpo_bayesian_prompted_qwen1p7b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-4B | Base structured prompt | MATH-500 | 352/500 | 70.40 | vLLM | 4096 | `outputs/math500_experiments/qwen3_4b_base_structured_prompt/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-4B | Answer-only GRPO | MATH-500 | 347/500 | 69.40 | vLLM | 4096 | `outputs/math500_experiments/grpo_answer_only_qwen3_4b_fulltrain12k_n8_steps1500/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-4B | BPR-GRPO prompted | MATH-500 | 367/500 | 73.40 | vLLM | 4096 | `outputs/math500_experiments/grpo_bayesian_prompted_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_eval0/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-4B | BPR-GRPO learned | MATH-500 | 364/500 | 72.80 | vLLM | 4096 | `outputs/math500_experiments/grpo_bayesian_sft_dpo_analyzer_qwen4b_fulltrain12k_n8_steps1500_bsz8_acc1_lambda1/eval_test_vllm_bs64_tok4096/summary.json` |
| MATH-500 eval | Qwen3-4B | Dr.GRPO answer-only reproduction | MATH-500 | 357/500 | 71.40 | vLLM | 4096 | `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096/summary.json` |
| BigMath eval | Qwen3-1.7B | BPR-GRPO prompted | GSM8K | 1023/1319 | 77.56 | vLLM | 4096 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_gsm8k_test_vllm_bs64_tok4096/summary.json` |
| BigMath eval | Qwen3-1.7B | BPR-GRPO prompted | MinervaMath | 45/272 | 16.54 | vLLM | 4096 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096_rerun1/minervamath/summary.json` |
| BigMath eval | Qwen3-1.7B | BPR-GRPO prompted | OlympiadBench | 200/674 | 29.67 | vLLM | 4096 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_1p7b_n8_steps1500_bsz8_acc1_lambda1/eval_benchmarks_vllm_bs64_tok4096/olympiadbench/summary.json` |
| BigMath eval | Qwen3-4B | Dr.GRPO answer-only reproduction | AIME26 | 2/30 | 6.67 | transformers | 4096 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/aime26/summary.json` |
| BigMath eval | Qwen3-4B | Dr.GRPO answer-only reproduction | MinervaMath | 69/272 | 25.37 | transformers | 1024 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/minervamath/summary.json` |
| BigMath eval | Qwen3-4B | Dr.GRPO answer-only reproduction | OlympiadBench | 247/674 | 36.65 | vLLM | 1024 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_olympiadbench_vllm_bs64_tok1024/summary.json` |
| BigMath eval | Qwen3-8B | BPR-GRPO prompted | AIME26 | 4/30 | 13.33 | vLLM | 4096 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/aime26/summary.json` |
| BigMath eval | Qwen3-8B | BPR-GRPO prompted | MinervaMath | 74/272 | 27.21 | vLLM | 4096 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/minervamath_rerun1/summary.json` |
| BigMath eval | Qwen3-8B | BPR-GRPO prompted | OlympiadBench | 303/674 | 44.96 | vLLM | 4096 | `outputs/bigmath_barl_style_12x1024_seed42/grpo_bayesian_prompted_qwen3_8b_n8_steps1536_bsz4_acc2_lambda1/eval_benchmarks_vllm_bs64_tok4096/olympiadbench/summary.json` |

## D. Training Configuration Manifest

These are the locally recoverable core training settings for major runs.

| Run | Train size | Steps | n | Per-device batch | Grad acc | Effective prompt batch | LR | Max prompt | Max completion | Reward/analyzer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-1.7B BPR prompted BigMath | 12288 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 2048 | 1024 | posterior BPR, prompted prior/evidence |
| Qwen3-8B BPR prompted BigMath | 12288 | 1536 | 8 | 4 | 2 | 8 | 5e-6 | 2048 | 1024 | posterior BPR, prompted prior/evidence |
| Qwen3-4B Dr.GRPO BigMath | 12288 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 2048 | 1024 | answer-only correctness, `loss_type=dr_grpo`, `scale_rewards=none` |
| Qwen3-1.7B BPR learned GSM8K | 7473 | 1000 | 8 | 8 | 1 | 8 | 5e-6 | 1024 | 1024 | posterior BPR, learned analyzer |
| Qwen3-4B Answer-only GSM8K | 7473 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 1024 | 1024 | answer-only correctness |
| Qwen3-4B BPR prompted GSM8K | 7473 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 1024 | 1024 | posterior BPR, prompted prior/evidence |
| Qwen3-8B Answer-only GSM8K | 7473 | 1000 | 8 | 4 | 2 | 8 | 5e-6 | 1024 | 1024 | answer-only correctness |
| Qwen3-1.7B BPR prompted MATH-500 | 12000 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 1024 | 1024 | posterior BPR, prompted prior/evidence |
| Qwen3-4B BPR prompted MATH-500 | 12000 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 2048 | 1024 | posterior BPR, prompted prior/evidence |
| Qwen3-4B BPR learned MATH-500 | 12000 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 2048 | 1024 | posterior BPR, learned analyzer |
| Qwen3-4B Dr.GRPO MATH-500 | 12000 | 1500 | 8 | 8 | 1 | 8 | 5e-6 | 2048 | 1024 | answer-only correctness, `loss_type=dr_grpo`, `scale_rewards=none` |

All BPR runs above use LoRA with `r=16`, `alpha=32`, `dropout=0.05`, bf16, gradient checkpointing, seed 42, prior strength `lambda=1.0`, prior softmax temperature `T_pi=1.0`, and judge max new tokens 768 unless otherwise stated in the corresponding config.

## E. BPR Formula and Code Alignment

The implemented reward matches the method equation:

```text
L_i = 0.80 c_i
    + 0.07 e_i^val
    + 0.08 e_i^comp
    + 0.02 e_i^str
    + 0.03 e_i^cons

S_i = p_i^lambda L_i

R_i^BPR = S_i / sum_j S_j, if sum_j S_j > 0
        = 1/n, otherwise
```

Code alignment:

| Concept | Implementation field/function |
| --- | --- |
| Evidence weights | `LIKELIHOOD_WEIGHTS` in `Bayesian_Full_GRPO.py` and `Bayesian_Full_GRPO_learned.py` |
| Likelihood surrogate | `compute_likelihood(answer_correctness, evidence)` |
| Posterior support | `prior_probability ** prior_lambda * likelihood` |
| Posterior reward | `compute_posteriors(...)` |
| Prior suitability | `prior_suitability in {0,1,2,3,4}` |
| Prior probability | softmax over suitability divided by `prior_softmax_temperature` |
| Evidence scores | `step_validity`, `proof_completeness`, `strategy_compliance`, `consistency` |
| Zero-support fallback | uniform reward `1/n` |

Appendix prompt/schema correction needed: the actual prior analyzer schema uses
`priors`, `rollout_id`, and `suitability`; the actual evidence analyzer schema
uses `consistency`, not `internal_consistency`.

## F. BPR Reward Diagnostics

The following aggregate is computed over all currently available
`bayesian_reward_debug.jsonl` files in `outputs`. It includes prompted and
learned analyzer runs across GSM8K, MATH-500, and BigMath; therefore it should
be described as a reward-pipeline audit, not as a single-run statistic.

| Diagnostic | Value |
| --- | ---: |
| Debug files audited | 49 |
| Rollouts audited | 287,904 |
| Rollout groups audited | 36,088 |
| Answer correctness over sampled training rollouts | 69.17% |
| Format-valid rollout rate | 94.63% |
| Final-answer section present rate | 94.78% |
| Suspicious final-answer rate | 0.00% |
| Prior judge fallback rate | 0.05% |
| Evidence judge fallback rate | 10.14% |
| Posterior normalization fallback rate | 3.01% |
| Judge label inconsistency remap rate | 9.83% |
| Mean likelihood surrogate | 0.7207 |
| Mean likelihood when correct | 0.9934 |
| Mean likelihood when incorrect | 0.1038 |
| Mean BPR reward | 0.1337 |
| Mean prior probability | 0.1250 |
| Mean prior suitability | 3.5267 |
| Mean reward mass assigned to correct rollouts | 0.8410 |
| Posterior top-1 correctness when any rollout is correct | 99.57% |
| Mean normalized group reward entropy | 0.8853 |
| All-correct groups | 19,324 |
| All-wrong groups | 6,810 |
| Mixed-correctness groups | 9,954 |
| Mean reward variance in all-correct groups | 0.001290 |
| Mean reward variance in all-wrong groups | 0.019075 |

These diagnostics support the claim that BPR changes the within-group credit
allocation while remaining anchored to final-answer correctness.

## G. Outcome Anchoring Audit

BPR is designed so that process evidence cannot dominate final-answer
correctness. The local debug artifacts validate this property:

| Audit check | Violations |
| --- | ---: |
| Incorrect rollout with available `L_i` and `L_i > 0.20` | 0 |
| Correct rollout with available `L_i` and `L_i < 0.80` | 0 |
| Correct rollout missing `L_i` in transferred legacy/debug rows | 2,823 |
| Incorrect rollout missing `L_i` in transferred legacy/debug rows | 1,977 |

This follows directly from the implemented likelihood weights: correctness
contributes 0.80 and all auxiliary process features together contribute 0.20.
Thus an incorrect rollout may receive relative credit within an all-wrong group,
but cannot receive a high answer-centered evidence score.

## H. Offline Prior-Strength Sweep

The following sweep recomputes only posterior allocation from existing debug
rows. It does not retrain a model and should be reported as an offline
credit-allocation diagnostic rather than a downstream accuracy result.

| Lambda | Mean reward mass on correct rollouts | Posterior top-1 correctness | Mean normalized entropy |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.7811 | 1.0000 | 0.9082 |
| 0.5 | 0.7807 | 1.0000 | 0.9012 |
| 0.7 | 0.7803 | 0.9994 | 0.8951 |
| 1.0 | 0.7794 | 0.9953 | 0.8832 |
| 1.5 | 0.7775 | 0.9943 | 0.8580 |
| 2.0 | 0.7752 | 0.9646 | 0.8292 |

Interpretation: increasing `lambda` makes the reward distribution sharper
because the strategy prior has greater influence. In the local aggregate,
`lambda=1.0` preserves high top-1 correctness while reducing entropy relative
to evidence-only normalization.

## I. Error-Type and Prior Suitability Audit

Evidence analyzer error-type distribution over local debug rows:

| Error type | Count |
| --- | ---: |
| `correct_complete` | 187,672 |
| `finalization_error` | 39,114 |
| `format_error` | 30,831 |
| `correct_weak_proof` | 7,189 |
| `strategy_mismatch` | 5,736 |
| `wrong_direction` | 5,451 |
| `no_meaningful_solution` | 4,543 |
| `valid_but_incomplete` | 4,474 |
| `arithmetic_error` | 1,104 |
| `invalid_assumption` | 1,000 |
| `lucky_correct` | 702 |
| `algebraic_error` | 88 |

Prior suitability distribution:

| Suitability | Count |
| ---: | ---: |
| 0 | 181 |
| 1 | 1,611 |
| 2 | 14,969 |
| 3 | 98,511 |
| 4 | 167,832 |
| missing legacy value | 4,800 |

These distributions can be used to show that the analyzer is not only assigning
binary correctness labels; it produces structured process and strategy signals
that feed the posterior allocation rule.

## J. Parser, Verifier, and Evaluation Protocol

All controlled comparisons should state the following protocol:

1. The solver prompt asks for `[Strategy]`, `[Reasoning]`, and `[Final Answer]`.
2. The final answer parser first reads the structured final-answer section.
3. If the structured section is missing or incomplete, deterministic fallback
   extraction is applied to common final-answer markers such as boxed answers.
4. If no answer can be parsed, the parsed answer is empty and the verifier label
   is incorrect.
5. The same parser and deterministic verifier are used for base, answer-only
   GRPO, BPR-GRPO, and reproduced Dr.GRPO evaluations.
6. Evaluation is deterministic pass@1 unless a specific artifact states
   otherwise.

The current local evaluation artifacts record backend, batch size, seed,
`max_prompt_length`, `max_new_tokens`, `generated_length_mean`, and format
success rates in `summary.json`.

## K. External Baseline Notes

The current local Dr.GRPO baseline should be described conservatively:

> We include a Dr.GRPO-style reproduction by using the `dr_grpo` GRPO loss
> formulation with answer-only correctness rewards inside the same local
> training and evaluation pipeline.

Avoid saying "official Dr.GRPO implementation" unless the exact official
repository code and configuration are used end-to-end. Current local artifacts
show `loss_type=dr_grpo` and `scale_rewards=none` in `Answer_only_GRPO.py` runs.

Locally available Dr.GRPO reproduction results:

| Model | Train data | Benchmark | Correct/N | Acc. | Summary |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3-4B | MATH-500 full | MATH-500 | 357/500 | 71.40 | `outputs/drgrpo_answer_only_24h/qwen3_4b_math500_steps1500_probe_n8_bsz8_acc1/eval_math500_vllm_bs64_tok4096/summary.json` |
| Qwen3-4B | BigMath BARL-style | AIME26 | 2/30 | 6.67 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/aime26/summary.json` |
| Qwen3-4B | BigMath BARL-style | MinervaMath | 69/272 | 25.37 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_benchmarks_bs16_mixedtok/minervamath/summary.json` |
| Qwen3-4B | BigMath BARL-style | OlympiadBench | 247/674 | 36.65 | `outputs/drgrpo_answer_only_24h/qwen3_4b_bigmath_steps1500_n8_bsz8_acc1/eval_olympiadbench_vllm_bs64_tok1024/summary.json` |

## L. Remaining Data Needed

To finalize the paper table and appendix without overclaiming, collect the
following from every server that produced a reported Table 1 value:

```text
outputs/**/summary.json
outputs/**/training_config.json
outputs/**/launcher_config.json
outputs/**/metadata_summary.json
outputs/**/bayesian_reward_debug.jsonl
outputs/**/predictions.jsonl        # at least for main-table evals
outputs/**/logs/*.log               # train/eval completion logs
```

For each PDF Table 1 cell, request:

1. The exact `summary.json` that produced the reported accuracy.
2. The checkpoint or adapter path used for evaluation.
3. The `training_config.json` or `launcher_config.json` for that checkpoint.
4. The eval command/log showing deterministic decoding, seed, backend, and max
   token setting.
5. For BPR rows, the matching `bayesian_reward_debug.jsonl`.

Recommended receive structure:

```text
outputs/incoming/paper_table_artifacts/
  qwen3_1p7b/
    base_structured/
    answer_only_grpo/
    bpr_prompted/
    drgrpo/
  qwen3_4b/
    base_structured/
    answer_only_grpo/
    bpr_prompted/
    drgrpo/
  qwen3_8b/
    ...
```

Once these artifacts are received, the main result table should be regenerated
from `summary.json` files rather than manually typed numbers.
