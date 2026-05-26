# Paper Experiment Section Information

Last updated: 2026-05-24

This file collects the information needed to write the experiment section and appendix.
It separates confirmed local facts from items that still need an explicit final decision or additional runs.

## Safe Main Claim

Recommended claim:

> Across Qwen3-1.7B, Qwen3-4B, and Qwen3-8B, BPR-GRPO with a prompted analyzer improves macro-average accuracy over answer-only GRPO on GSM8K, MATH 500, Minerva, and OlympiadBench. The gains are most consistent on MATH 500, where BPR improves all three model scales.

Stronger but still safe:

> BPR improves macro-average accuracy at every model scale and improves 9 out of 12 completed prompted-analyzer benchmark comparisons, with one tie.

Avoid this claim:

> BPR consistently improves all benchmarks.

Reason:
- Qwen3-4B OlympiadBench: prompted BPR is slightly below answer-only GRPO.
- Qwen3-8B GSM8K: prompted BPR is slightly below answer-only GRPO.

## A. Model and Checkpoint Information

Exact Hugging Face model IDs used in the local code/configs:

| Paper label | Exact local model ID | HF stage note |
|---|---|---|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | Qwen model card says `Training Stage: Pretraining & Post-training`; this is not the `Qwen/Qwen3-1.7B-Base` checkpoint. |
| Qwen3-4B | `Qwen/Qwen3-4B` | Qwen model card describes thinking/non-thinking chat behavior; this is not the `Qwen/Qwen3-4B-Base` checkpoint. |
| Qwen3-8B | `Qwen/Qwen3-8B` | Qwen model card says `Training Stage: Pretraining & Post-training`; this is not the `Qwen/Qwen3-8B-Base` checkpoint. |

Important naming recommendation:
- In the paper, do not call these "Qwen3-*-Base" unless the model ID is explicitly changed to the `-Base` checkpoint.
- Use wording like "the released Qwen3 post-trained checkpoints `Qwen/Qwen3-{1.7B,4B,8B}`".
- In this repo, "base" often means "unadapted checkpoint with no GRPO LoRA adapter", not "the HF `-Base` model family".

Definitions:
- `Pure-based`: raw problem-only evaluation. The model input is exactly the metadata `problem` string. No system prompt, no user instruction, no tokenizer chat template. Source: `eval_pure_base_model.py`.
- `Base (Structured Prompt)`: unadapted `Qwen/Qwen3-*` checkpoint evaluated with the structured solver prompt and chat template. No GRPO adapter is loaded.
- `Answer-only GRPO`: LoRA GRPO trained with answer correctness reward only.
- `BPR-GRPO (Prompted Analyzer)`: LoRA GRPO trained with posterior-normalized Bayesian reward using same-size Qwen prompted judges for prior and evidence.
- `BPR-GRPO (Learned Analyzer)`: LoRA GRPO trained with posterior-normalized Bayesian reward using a learned generative analyzer adapter.

Structured prompt template used by solver baselines and GRPO:

```text
System:
You are a careful mathematical reasoning assistant.
Solve the problem independently.
You must follow the required output format exactly.
Do not skip the final answer section.

User:
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
Prefer \boxed{...} when appropriate.
Do not put extra explanation after the final answer.

Return your response in the following format:

[Strategy]
...

[Reasoning]
...

[Final Answer]
...

Problem:
{problem}
```

Implementation:
- `Answer_only_GRPO.py`: `SYSTEM_PROMPT`, `build_user_prompt`, `render_prompt_messages`, `render_prompt`.
- For Qwen3 chat-template calls, the code uses `enable_thinking=False` when supported.

## B. Training Setup

Training datasets used by major experiment families:

| Experiment family | Train metadata | Train size | Eval/test metadata |
|---|---|---:|---|
| GSM8K full-train | `outputs/gsm8k_full_train_seed42/selected_train_metadata.jsonl` | 7,473 | GSM8K official test, 1,319 |
| MATH-500 full-train | `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_train_metadata.jsonl` | 12,000 | MATH-500 official test, 500 |
| BigMath transfer/eval training | `outputs/bigmath_barl_style_12x1024_seed42/selected_train_metadata.jsonl` | 12,288 | MinervaMath, OlympiadBench, AIME26 |

BigMath 12x1024 sampling:
- Source dataset: `SynthLabsAI/Big-Math-RL-Verified`.
- Protocol: 12 rollout batches x 1024 prompts = 12,288 prompts.
- Seed: 42.
- Excluded source: `amc_aime`.
- Deduplicated against `math-ai/olympiadbench`, `math-ai/minervamath`, and `math-ai/aime26`.
- Selected source distribution:
  - `aops_forum`: 280
  - `big_math`: 2300
  - `cn_k12`: 3065
  - `gsm8k`: 82
  - `harp`: 180
  - `math`: 449
  - `olympiads`: 1659
  - `omnimath`: 145
  - `openmath`: 73
  - `orca_math`: 4055

Common GRPO setup from local `training_args.bin`:

| Field | Value |
|---|---|
| GRPO group size | `num_generations=8` |
| Rollout temperature | `0.7` |
| Rollout top-p | `0.95` |
| Top-k | not set in local GRPO configs |
| Learning rate | `5e-6` |
| Optimizer | `adamw_torch_fused` |
| Weight decay | `0.0` |
| Adam betas | `0.9`, `0.999` |
| Adam epsilon | `1e-8` |
| Scheduler | linear |
| Warmup | `warmup_steps=0`; no warmup ratio |
| TRL `beta` | `0.0` |
| TRL `loss_type` | `dapo` |
| TRL `scale_rewards` | `group` |
| Report | `none` |
| Seed | `42` |
| Precision | bf16 |
| Gradient checkpointing | enabled |

Common LoRA setup:
- LoRA enabled for GRPO training, not full fine-tuning.
- `r=16`, `alpha=32`, `dropout=0.05`.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- No QLoRA setting is visible in the local configs.

Representative training configs:

| Run | Steps | Per-device batch | Grad accum | Effective batch | Max prompt | Max completion | vLLM |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B GSM8K Answer-only | 1000 | 1 | 8 | 8 | 1024 | 1024 | no |
| Qwen3-1.7B GSM8K BPR prompted | 1000 | 1 | 8 | 8 | 1024 | 1024 | no |
| Qwen3-4B BigMath BPR prompted | 1536 | 8 | 1 | 8 | 1024 | 1024 | yes, colocate |
| Qwen3-1.7B BigMath BPR learned | 1500 | 1 | 8 | 8 | 2048 | 1024 | yes, colocate |
| Qwen3-8B MATH Answer-only | 1500 | 1 | 8 | 8 | 2048 | 1024 | no |

Learned analyzer training:
- The learned analyzer is a generative causal LM LoRA adapter, not a classifier head.
- It predicts JSON/rubric outputs for prior/evidence tasks.
- Labels are distilled from prompted analyzer reward-debug logs; no human labels are present in the local pipeline.

BigMath Qwen3-1.7B learned analyzer data:
- SFT source: `outputs/bigmath_qwen3_1p7b/grpo_bayesian_prompted_bigmath12x1024_n8_steps1536_bsz8_acc1_lambda1/bayesian_reward_debug.jsonl`.
- SFT unified train/val: 9,765 / 291.
- SFT mixture: evidence 5,859; prior 3,906.
- SFT val mixture: evidence 256; prior 35.
- DPO unified train/val: 1,600 / 167.
- DPO train mixture: evidence 800; prior 800.
- DPO val mixture: evidence 120; prior 47.
- DPO beta: `0.1` in launch commands.

GSM8K Qwen3-4B learned analyzer data:
- SFT train/val unified: 6,795 / 705 in runtime-rebalanced data.
- DPO train/val unified: 5,000 / 500.
- DPO train mixture: evidence 4,000; prior 1,000.
- DPO val mixture: evidence 400; prior 100.
- DPO beta: `0.1`.

## C. Sampling and Decoding Setup

Training rollout generation:
- `temperature=0.7`
- `top_p=0.95`
- `top_k` is not configured.
- `num_generations=8`.

Evaluation decoding:
- Deterministic pass@1.
- `do_sample=False`.
- `temperature=0.0`.
- `top_p=1.0`.
- One completion per example; no best-of or multiple-sample selection.

Evaluation max-token policy:
- GSM8K: normally 1024.
- MATH-500: normally 1024 in the main table; some local retests used 4096.
- MinervaMath: normally 1024 in base/prompted configs; the latest Qwen3-1.7B learned-analyzer retest used 4096.
- OlympiadBench: normally 1024 in base/prompted configs; the latest Qwen3-1.7B learned-analyzer retest used 4096.
- AIME26: usually 4096 in recent evals.

Important fairness warning:
- The current consolidated table excludes AIME26.
- Before final submission, make sure every method in the same table uses the same `max_new_tokens` policy.
- The current local configs contain mixed 1024/4096 retests; the final reported table should pick one policy and stick to it.

Answer extraction:
- Structured outputs are parsed by `[Final Answer]` first.
- The parser then checks the last `\boxed{...}` inside `[Final Answer]`.
- If no boxed answer exists, it uses the first meaningful answer line in `[Final Answer]`.
- Fallbacks include last boxed expression in the full completion and explicit "final answer" patterns.
- Pure-base evaluation uses a format-agnostic parser: it additionally checks GSM8K `####`, boxed answers, explicit answer phrases, and other raw-output patterns.
- Verification uses the shared `verify_answer` routine with numeric/symbolic normalization.

## D. BPR-Specific Setup

Posterior reward:

```text
unnormalized_i = max(prior_i, 0)^lambda * max(likelihood_i, 0)
posterior_i = unnormalized_i / sum_j unnormalized_j
```

Main BPR settings:
- `lambda=1.0`.
- `T_pi=1.0` via `prior_softmax_temperature`.
- Prior scores are LLM suitability scores in `[0, 4]`, softmaxed with `T_pi`.
- Prompted analyzer uses same-size Qwen model as the solver unless explicitly overridden.
  - Qwen3-1.7B prompted: prior/evidence judge `Qwen/Qwen3-1.7B`.
  - Qwen3-4B prompted: prior/evidence judge `Qwen/Qwen3-4B`.
- No API model or larger judge model is used in the main local configs.

Evidence likelihood components:

| Component | Weight |
|---|---:|
| answer correctness | 0.80 |
| step validity | 0.07 |
| proof completeness | 0.08 |
| strategy compliance | 0.02 |
| consistency | 0.03 |

Prompted analyzer appendix material:
- Prior prompt function: `Bayesian_Full_GRPO.py::build_prior_judge_prompt`.
- Evidence prompt function: `Bayesian_Full_GRPO.py::build_evidence_judge_prompt`.
- Learned-analyzer task prefixes: `Bayesian_Full_GRPO_learned.py::TASK_PREFIXES`.

Prior leakage audit status:
- The prior prompt explicitly says: "Do not solve the full problem", "Do not look at reasoning or final answers", and "Evaluate only whether the strategy is appropriate".
- However, no dedicated prior-leakage audit experiment is recorded yet.
- For top-tier submission, add a small audit table showing that the prior analyzer input excludes reasoning/final answer fields and measure behavior on swapped/corrupted answers.

Available analyzer diagnostics:
- GSM8K Qwen3-1.7B prompted BPR:
  - prior parse rate: 1.0
  - evidence parse rate: 0.9075
  - posterior fallback count: 7
  - posterior top-1 accuracy when any correct rollout exists: 0.9958
- GSM8K Qwen3-4B learned analyzer:
  - prior parse rate: 1.0
  - evidence parse rate: 0.9991
  - posterior fallback count: 0
  - posterior top-1 accuracy when any correct rollout exists: 0.9986
- BigMath Qwen3-1.7B SFT analyzer recompute:
  - prior parse rate: 0.978
  - evidence parse rate: 0.98775
  - fallback count: 1
  - learned posterior top-1 accuracy when correct exists: 0.9716
  - teacher posterior top-1 accuracy when correct exists: 0.9912

## E. Evaluation Details

Benchmark sample counts:

| Benchmark | Metadata path | N |
|---|---|---:|
| GSM8K | `outputs/gsm8k_full_train_seed42/selected_test_metadata.jsonl` | 1,319 |
| MATH-500 | `outputs/math500_experiments/metadata_fulltrain12000_test500_seed42/selected_test_metadata.jsonl` | 500 |
| MinervaMath | `outputs/eval_benchmarks/minervamath_metadata.jsonl` | 272 |
| OlympiadBench | `outputs/eval_benchmarks/olympiadbench_metadata.jsonl` | 674 |
| AIME26 | `outputs/eval_benchmarks/aime26_metadata.jsonl` | 30 |

Minerva/Olympiad/AIME source details:
- MinervaMath metadata source label: `minervamath`, domain `math`.
- OlympiadBench metadata source label: `olympiadbench`, domain `math`.
- AIME26 metadata source label: `aime26`, domain `math`.
- The BigMath training builder deduplicates against `math-ai/minervamath`, `math-ai/olympiadbench`, and `math-ai/aime26`.
- The local metadata rows do not store enough fields to claim "English-only" or "text-only" beyond the stored problem text and `domain=math`.

Average:
- Current table uses unweighted macro-average over GSM8K, MATH 500, Minerva, and OlympiadBench.
- AIME26 is excluded from the current main table.
- Recommendation: keep AIME26 out of the main table or make it a separate appendix table because it is small (`N=30`) and currently has different max-token treatment.

## F. Reliability and Statistics

Current reliability status:
- Most recorded results are single seed (`seed=42`).
- No multi-seed mean/std table is currently available.
- No bootstrap confidence intervals are currently stored.

Recommended additions before submission:
- Add bootstrap 95% confidence intervals from `predictions.jsonl` for every final table cell.
- For small AIME26, report binomial confidence intervals if included anywhere.
- Add 2-3 qualitative examples where BPR helps on MATH-500/OlympiadBench.
- Add 1-2 failure examples where BPR hurts or ties, especially Qwen3-4B OlympiadBench and Qwen3-8B GSM8K.

## G. Ablation and Audit Status

Available but not final-scale:
- Older lambda sweeps exist for Qwen2.5-3B style BigMath runs at `lambda=0.5`, `0.7`, and `1.0`.

Missing for a strong top-tier ablation section:
- `lambda=0` evidence-only posterior on the final Qwen3 setup.
- Prior-only reward.
- Evidence-only reward.
- Final-answer-only evidence vs process evidence.
- Prior leakage audit.
- Reward hacking probes:
  - wrong but plausible reasoning
  - correct answer with bad reasoning
  - high-prior wrong strategy
  - low-prior correct strategy

Recommended ablation priority:
1. Add `lambda=0` and prior-only/evidence-only on the smallest final setting first.
2. Add prior-leakage audit using the exact prior prompt input.
3. Add bootstrap CIs for the final result table.
4. Add qualitative examples for BPR gains and failures.

## References for Model Identity

- `Qwen/Qwen3-1.7B`: Hugging Face model card reports `Training Stage: Pretraining & Post-training`.
- `Qwen/Qwen3-1.7B-Base`: Hugging Face model card reports `Training Stage: Pretraining`.
- `Qwen/Qwen3-8B`: Hugging Face model card reports `Training Stage: Pretraining & Post-training`.
