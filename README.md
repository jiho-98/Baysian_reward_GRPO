# Bayesian Self-Data Reasoning

This folder contains the EMNLP research prototype for strategy-belief self-play in LLM reasoning.

## Minimal Experiment

The first milestone is intentionally small:

1. Use one modular arithmetic problem.
2. Run four fixed strategy-conditioned rollouts.
3. Verify final answers with Python.
4. Score how strongly each rollout followed the assigned strategy.
5. Update a Beta-Bernoulli belief table using the compliance score.

Run the API-free demo:

```bash
cd /home/kimjh/EMNLP
python scripts/minimal_belief_demo.py
```

The script writes:

```text
results/minimal_belief_demo.json
```

Expected behavior:

- Correct rollouts increase `alpha` by `strategy_compliance_score`.
- Wrong rollouts increase `beta` by `strategy_compliance_score`.
- Non-compliant portions increment `strategy_drift` by `1 - strategy_compliance_score`.
- Assigned strategy belief and actual strategy credit are tracked separately.
- `success_belief` measures success when a strategy is followed; `usability_belief`
  additionally penalizes average strategy drift.

## Optional Real LLM Rollouts

Create a local `.env` file or export `OPENAI_API_KEY` in your shell. The `.env`
file is ignored by git.

```bash
cd /home/kimjh/EMNLP
printf 'OPENAI_API_KEY=your_key_here\n' > .env
```

Then run:

```bash
cd /home/kimjh/EMNLP
python scripts/minimal_belief_demo.py \
  --backend openai \
  --model gpt-4o-mini \
  --output results/minimal_belief_openai_gpt4omini.json
```

For an open-source Hugging Face model on a GPU node, edit the Slurm template first:

```bash
vim scripts/slurm_qwen_belief_demo.sbatch
```

Confirm and replace these placeholders with the lab/server admin:

```text
GPU_PARTITION_NAME
ACCOUNT_NAME
QOS_NAME
gpu:1
cuda module name, if required
```

Then submit only after confirmation:

```bash
sbatch scripts/slurm_qwen_belief_demo.sbatch
```

Each rollout must end with:

```text
USED_STRATEGY: <actual strategy used>
FINAL_ANSWER: <integer>
```

## Current Scope

Do not add self-generated problems, failure analysis, GRPO, AIME, or MATH500 yet. The current goal is only to validate that strategy-conditioned rollout evidence updates strategy beliefs in code.
