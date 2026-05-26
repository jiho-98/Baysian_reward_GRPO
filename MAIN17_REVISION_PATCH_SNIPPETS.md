# Main(17) Revision Patch Snippets

This file contains LaTeX-ready replacements for the remaining issues visible in
`main (17).pdf`. The repository does not contain the paper `.tex` or `.bib`
source, so these snippets are intended to be pasted into the manuscript source.

## 1. Figure 1 Text Fixes

Replace the figure-internal text as follows:

```text
Are each steps
```

with:

```text
Are the steps
```

Replace:

```text
Dose it follow
```

with:

```text
Does it follow
```

If numeric BPR scores are shown in the figure, either make the visible example
scores sum to one or replace them with symbols. Recommended visible example:

```text
\tau_1: 0.42
\tau_2: 0.31
\tau_n: 0.27
```

Safer symbolic version:

```text
\tau_1: R^{\mathrm{BPR}}_1
\tau_2: R^{\mathrm{BPR}}_2
\tau_n: R^{\mathrm{BPR}}_n
```

Recommended caption:

```latex
\caption{
Overview of Bayesian Posterior Reward. For each problem, the policy samples a
rollout group, decomposes each rollout into strategy, reasoning trace, and final
answer fields, estimates an information-restricted strategy prior and
answer-centered evidence, and normalizes support within the group to obtain a
posterior-style reward distribution.
}
```

Recommended placement: move Figure 1 to the start of Section 3, immediately
after the first Method paragraph. It currently interrupts the Related Work flow.

## 2. Main Results Text Fixes

Replace:

```latex
where Dr.GRPO-style is higher by0.49 points.
```

with:

```latex
where Dr.GRPO-style is higher by 0.49 points.
```

Use this paragraph for the Dr.GRPO baseline description:

```latex
\paragraph{Models and baselines.}
We evaluate the Qwen/Qwen3-1.7B and Qwen/Qwen3-4B scales shown in
Table~\ref{tab:main-controlled-results}. Structured Base evaluates the released
checkpoint with our structured solver prompt and chat template, without a GRPO
LoRA adapter. Answer-only GRPO trains a LoRA adapter using final-answer
correctness as the reward. Dr.GRPO-style keeps the answer-only reward but uses
the Dr.GRPO loss objective in the same local GRPO pipeline, with matched Qwen3
backbone, training split, LoRA configuration, maximum steps, rollout group size,
and evaluation protocol. BPR-GRPO trains with the BPR reward using prompted
analyzers for the strategy prior and answer-centered evidence. The
Dr.GRPO-style row is a controlled local loss-objective baseline, not a comparison
to published Dr.GRPO scores obtained under a different training recipe.
```

Use this main table. It keeps Dr.GRPO in the main comparison but makes the
controlled-local nature explicit.

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llccccc}
\toprule
Model & Method & GSM8K & MATH 500 & MinervaMath & OlympiadBench & Avg. \\
\midrule
Qwen3-1.7B & Structured Base & 76.99 & 58.60 & 13.97 & 26.41 & 43.99 \\
Qwen3-1.7B & Answer-only GRPO & 77.56 & 59.60 & 15.62 & 27.23 & 45.00 \\
Qwen3-1.7B & Dr.GRPO-style & 77.18 & 59.00 & 15.07 & 27.45 & 44.68 \\
Qwen3-1.7B & BPR-GRPO & \textbf{79.87} & \textbf{61.20} & \textbf{15.81} & \textbf{29.34} & \textbf{46.55} \\
\midrule
Qwen3-4B & Structured Base & 88.99 & 70.15 & 24.54 & 40.02 & 55.93 \\
Qwen3-4B & Answer-only GRPO & 90.01 & 69.45 & 24.91 & 39.21 & 55.89 \\
Qwen3-4B & Dr.GRPO-style & 90.07 & 71.40 & 25.37 & 36.65 & 55.87 \\
Qwen3-4B & BPR-GRPO & \textbf{90.56} & \textbf{73.05} & \textbf{25.86} & \textbf{41.89} & \textbf{57.84} \\
\bottomrule
\end{tabular}
\caption{
Controlled local comparison under matched Qwen3 backbones, LoRA adaptation,
training data, rollout group size, and evaluation protocol. Dr.GRPO-style
denotes the Dr.GRPO loss objective implemented in the same local GRPO pipeline
with \texttt{loss\_type=dr\_grpo} and \texttt{scale\_rewards=none}; it is not a
comparison to published Dr.GRPO scores under a different training recipe. Avg.
is the unweighted macro-average over GSM8K, MATH 500, MinervaMath, and
OlympiadBench.
}
\label{tab:main-controlled-results}
\end{table*}
```

## 3. Appendix Language Cleanup

Replace:

```latex
they should be interpreted as a matched local loss-objective baseline
```

with:

```latex
we interpret them as a matched local loss-objective baseline
```

Replace:

```latex
strategy-field answer leakage is therefore treated as a dedicated audit target
```

with:

```latex
we therefore report strategy-field answer leakage as a limitation and include it
in the proposed adversarial audit protocol
```

Replace:

```latex
In addition to the target diagnostics listed in the main appendix
```

with:

```latex
We compute realized reward-pipeline diagnostics from the available
\texttt{bayesian\_reward\_debug.jsonl} files
```

Replace:

```latex
the fulloutputs/tree rather than onlyoutputs/incoming/
```

with:

```latex
the full \texttt{outputs/} tree rather than only \texttt{outputs/incoming/}
```

Replace:

```latex
source summary.json and run configuration
```

with:

```latex
source \texttt{summary.json} and run configuration
```

## 4. Appendix Table Formatting Fixes

The current appendix tables are too dense in the PDF. Use `table*` plus
`\resizebox{\textwidth}{!}{...}` for wide result tables, and use `p{...}` columns
for prose-heavy provenance tables.

### Reproducibility Table

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{p{0.25\textwidth}p{0.68\textwidth}}
\toprule
Field & Recorded value \\
\midrule
Base checkpoints & Qwen/Qwen3-1.7B and Qwen/Qwen3-4B. \\
Comparison scope & Benchmark-family local comparisons. Within each benchmark family, answer-only GRPO, Dr.GRPO-style, and BPR-GRPO are matched by data, split, training budget, parser, verifier, and evaluator in the artifact records. \\
Result provenance & Each accuracy cell is represented in the experiment ledger by method, model, benchmark, training corpus or checkpoint, evaluator backend, verifier, source \texttt{summary.json}, and run configuration. \\
Logged run-config fields & \texttt{train\_size}, \texttt{max\_steps}, \texttt{num\_generations}, per-device batch size, gradient accumulation, learning rate, LoRA rank/alpha/dropout/target modules, seed, \texttt{max\_prompt\_length}, and \texttt{max\_completion\_length}. \\
Dr.GRPO-style baseline & Controlled local loss-objective baseline under the same local GRPO pipeline, using \texttt{loss\_type=dr\_grpo} and \texttt{scale\_rewards=none}; not a comparison to published Dr.GRPO scores under a different training recipe. \\
Group size & $n=8$ rollouts per problem for BPR reward construction. \\
KL coefficient & $\beta=0$ unless otherwise specified in the source run config. \\
BPR prior settings & Prior strength $\lambda=1.0$ and prior softmax temperature $T_\pi=1.0$. \\
BPR evidence weights & 0.80 correctness, 0.07 step validity, 0.08 proof completeness, 0.02 strategy compliance, and 0.03 consistency. \\
Evaluation decoding & Deterministic pass@1 with one completion per example and no Best-of-N or multi-sample selection; source evaluation records store the exact max-token budget and decoding parameters. \\
\bottomrule
\end{tabular}
\caption{Reproducibility and provenance fields recorded for the displayed aggregate comparisons.}
\label{tab:reproducibility-fields}
\end{table*}
```

### Unified 4096-Token Aggregate Table

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.8pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccc}
\toprule
Model & Method & GSM8K & MATH 500 & MinervaMath & OlympiadBench & Avg. & $n$ \\
\midrule
Qwen3-1.7B & Base & 76.99 & 58.60 & 13.97 & 26.41 & 43.99 & 4--4 \\
Qwen3-1.7B & GRPO & 77.56 & 59.60 & 15.62 & 27.23 & 45.00 & 4--4 \\
Qwen3-1.7B & BPR-GRPO & 79.87 & 61.20 & 15.81 & 29.34 & 46.55 & 4--4 \\
Qwen3-4B & Base & 88.99 & 70.15 & 24.54 & 40.02 & 55.93 & 4--4 \\
Qwen3-4B & GRPO & 90.01 & 69.45 & 24.91 & 39.21 & 55.89 & 4--4 \\
Qwen3-4B & BPR-GRPO & 90.56 & 73.05 & 25.86 & 41.89 & 57.84 & 3--4 \\
Qwen3-8B & Base & 92.25 & 74.95 & 27.21 & 44.81 & 59.80 & 4--5 \\
Qwen3-8B & GRPO & 92.51 & 74.90 & 27.30 & 44.29 & 59.75 & 4--4 \\
Qwen3-8B & BPR-GRPO & 92.44 & 75.00 & 26.01 & 45.03 & 59.62 & 4--4 \\
\bottomrule
\end{tabular}
}
\caption{Unified 4096-token deterministic pass@1 evaluation aggregate reconstructed from available \texttt{summary.json} files. Avg. is the unweighted macro-average over GSM8K, MATH 500, MinervaMath, and OlympiadBench.}
\label{tab:unified-4096-aggregate}
\end{table*}
```

### Repeat-Level Variability Table

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.8pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccc}
\toprule
Model & Method & GSM8K & MATH 500 & MinervaMath & OlympiadBench \\
\midrule
Qwen3-1.7B & Base & $76.99{\pm}0.44$ & $58.60{\pm}0.78$ & $13.97{\pm}0.00$ & $26.41{\pm}0.47$ \\
Qwen3-1.7B & GRPO & $77.56{\pm}0.42$ & $59.60{\pm}0.82$ & $15.62{\pm}1.55$ & $27.23{\pm}0.78$ \\
Qwen3-1.7B & BPR-GRPO & $79.87{\pm}0.50$ & $61.20{\pm}0.75$ & $15.81{\pm}1.50$ & $29.34{\pm}0.25$ \\
Qwen3-4B & Base & $88.99{\pm}0.07$ & $70.15{\pm}0.44$ & $24.54{\pm}0.35$ & $40.02{\pm}0.28$ \\
Qwen3-4B & GRPO & $90.01{\pm}0.19$ & $69.45{\pm}0.68$ & $24.91{\pm}1.10$ & $39.21{\pm}0.79$ \\
Qwen3-4B & BPR-GRPO & $90.56{\pm}0.19$ & $73.05{\pm}0.50$ & $25.86{\pm}0.42$ & $41.89{\pm}0.45$ \\
Qwen3-8B & Base & $92.25{\pm}0.19$ & $74.95{\pm}0.25$ & $27.21{\pm}0.30$ & $44.81{\pm}1.51$ \\
Qwen3-8B & GRPO & $92.51{\pm}0.29$ & $74.90{\pm}0.76$ & $27.30{\pm}0.46$ & $44.29{\pm}0.75$ \\
Qwen3-8B & BPR-GRPO & $92.44{\pm}0.14$ & $75.00{\pm}0.43$ & $26.01{\pm}0.46$ & $45.03{\pm}0.56$ \\
\bottomrule
\end{tabular}
}
\caption{Mean and standard deviation over available unified 4096-token evaluation repeats.}
\label{tab:repeat-variability}
\end{table*}
```

## 5. Reference/BibTeX Fixes

The PDF still renders entries such as `Maciej Besta and 1 others` and missing
spaces like `InProceedings`. This is a `.bib`/style hygiene issue. Replace
`author = {Name and others}` entries with either complete author lists or a
safer truncated list ending in `and others` only if the style renders it as
`et al.`. For ACL submissions, complete author lists are safest.

At minimum, replace these entries before submission:

```bibtex
@inproceedings{besta2024graph,
  title = {Graph of Thoughts: Solving Elaborate Problems with Large Language Models},
  author = {Besta, Maciej and Blach, Nils and Kubicek, Ales and Gerstenberger, Robert and Gianinazzi, Lukas and Gajda, Joanna and Lehmann, Tomasz and Podstawski, Micha{\l} and Niewiadomski, Hubert and Nyczyk, Piotr and Hoefler, Torsten},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {38},
  number = {16},
  pages = {17682--17690},
  year = {2024}
}

@inproceedings{zhang2024rest,
  title = {{ReST-MCTS*}: {LLM} Self-Training via Process Reward Guided Tree Search},
  author = {Zhang, Dan and Zhoubian, Sining and Hu, Ziniu and Yue, Yisong and Dong, Yuxiao and Tang, Jie},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```

If the ACL style still prints `Preprint` without spacing, use the standard
ACL/arXiv BibTeX pattern:

```bibtex
@misc{shao2024deepseekmath,
  title = {{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  year = {2024},
  eprint = {2402.03300},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

## 6. Final Checklist for Main(17)

- Move Figure 1 from Related Work into Method.
- Fix Figure 1 grammar: `Are the steps`, `Does it follow`.
- Make visible BPR score examples sum to one or use symbolic rewards.
- Keep Dr.GRPO in the main table, but retain the controlled-local disclaimer.
- Replace appendix wide tables with `table*`/`\resizebox` versions.
- Remove remaining draft-like phrasing: `target diagnostics`, `should be interpreted`.
- Fix spacing artifacts: `sourcesummary.json`, `fulloutputs`, `by0.49`.
- Replace `.bib` entries that render as `and 1 others`.
- Rebuild PDF and visually inspect pages 10--16 for table overflow.
