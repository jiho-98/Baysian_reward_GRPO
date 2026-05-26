# Professor Feedback 6h Action Pack

This file converts the professor feedback into manuscript-ready LaTeX blocks and
separates what can be done with current artifacts from what requires new
training. It assumes the current available paper source is not present in this
repository, so the blocks below are written as paste-ready replacements.

## 0. Six-Hour Feasibility Triage

| Feedback item | 6h status | Action taken here |
|---|---:|---|
| Strategy-prior ablation | Partially possible | Use stored BPR reward logs for an offline prior-strength sensitivity analysis. Do not claim downstream accuracy ablation. |
| BPR hyperparameter analysis | Possible | Add $\lambda$ sweep table over correct reward mass, posterior top-1 correctness, and entropy. |
| RLAIF / LLM-judge reward comparison | Possible in writing | Add Related Work / Discussion paragraph distinguishing BPR from independent weighted-sum LLM-judge rewards. |
| BPR+Dr.GRPO results | Not available in current artifacts | Do not add numeric result. Add a limitation/future-work sentence; only add after new training. |
| Method order aligned with figure | Possible | Replace Section 3 structure with rollout decomposition $\rightarrow$ analyzers $\rightarrow$ posterior-style reward $\rightarrow$ GRPO integration. |
| Reduce Bayesian overclaim risk | Possible | Replace calibrated-posterior language with Bayesian-motivated/posterior-style/finite-candidate support normalization language. |
| Main result feels quantitatively thin | Possible | Keep larger main table and add compact teaser figure / bring repeat variability or reward diagnostics into main or near-main analysis. |
| Structured-format fairness for baselines | Possible | Add explicit Experiment Setup paragraph stating Base, GRPO, Dr.GRPO, and BPR use the same structured solver format, parser, verifier, and evaluator. |
| Strategy prior for exploration | Not a 6h method change | Add future-work paragraph on strategy-level exploration / separate optimization of strategy generation. |

## 1. Main-Text Related Work Addition: LLM-Judge / RLAIF Reward Pipelines

Insert near the RLVR / reward-design paragraph in Related Work.

```latex
\paragraph{LLM-judge rewards and RLAIF-style optimization.}
Another related direction uses a judge model to score generated responses along
one or more rubric dimensions and then optimizes the policy against the resulting
scalar reward. Such RLAIF-style pipelines can provide dense feedback without
human labels, but the reward is often constructed as an independent weighted sum
over each response. BPR differs in two ways. First, final-answer correctness is
the dominant evidence term, so process judgments cannot by themselves give a
high evidence score to an incorrect mathematical solution. Second, BPR normalizes
prior-weighted evidence within the rollout group, making reward assignment depend
on the competing candidate solutions sampled for the same problem. Thus BPR is
closer to group-coupled credit allocation than to independent LLM-judge reward
shaping.
```

## 2. Method Rewrite: Safer Bayesian Wording and Figure-Aligned Order

Use this as the Section 3 opening and subsection structure.

```latex
\section{Method}
\label{sec:method}

Bayesian Posterior Reward (BPR) is a Bayesian-motivated reward-construction
rule for group-relative reinforcement learning in mathematical reasoning. The
method does not maintain a calibrated posterior over policies, environments, or
latent reasoning programs. Instead, it uses the finite group of rollouts sampled
for the same problem as a local candidate set and applies a posterior-style
normalization rule to allocate reward across those candidates. We therefore use
the term posterior-style to refer to the finite-candidate support normalization
in Eq.~\ref{eq:bpr-reward}, not to a full Bayesian generative model.

\subsection{Structured Rollout Generation and Decomposition}

For a problem $q$ with reference answer $a^\star$, the policy samples a rollout
group $G(q)=\{\tau_i\}_{i=1}^{n}$ with $n=8$. Each rollout is generated with the
same structured solver format used by all trained baselines:
\[
\tau_i = (s_i, r_i, a_i),
\]
where $s_i$ is the strategy section, $r_i$ is the completed reasoning trace, and
$a_i$ is the parsed final answer. A deterministic verifier computes
$c_i=\mathbf{1}[\mathrm{verify}(a_i,a^\star,q)]$. The structured format is used
to make the strategy available for prior analysis and is also applied to the
answer-only GRPO and Dr.GRPO-style baselines for fair comparison.

\subsection{Prior and Evidence Analyzers}

BPR uses two analyzer calls with different information access. The prior
analyzer receives only the problem and the candidate strategy sections
$(q,s_{1:n})$. It does not receive completed reasoning traces, final answers, or
verifier labels. It assigns each strategy an integer suitability score
$h_i\in\{0,1,2,3,4\}$, which is converted into a group prior
\[
p_i=\frac{\exp(h_i/T_\pi)}{\sum_{j=1}^{n}\exp(h_j/T_\pi)}.
\]
This prior is ``prior'' in an information-restricted sense: it evaluates the
strategy before seeing post-strategy evidence, rather than before the rollout is
sampled chronologically.

The evidence analyzer receives the completed rollout information
$(q,r_i,a_i,c_i,s_i)$ and assigns auxiliary process scores for step validity,
proof completeness, strategy compliance, and internal consistency. These process
scores are used only as bounded auxiliary evidence. Final-answer correctness is
the dominant evidence term:
\[
L_i =
0.80c_i
+0.07e_i^{\mathrm{val}}
+0.08e_i^{\mathrm{comp}}
+0.02e_i^{\mathrm{str}}
+0.03e_i^{\mathrm{cons}}.
\]
Because all process terms together have weight $0.20$, incorrect rollouts
satisfy $L_i\leq 0.20$ and correct rollouts satisfy $L_i\geq 0.80$ whenever the
evidence fields are available.

\subsection{Prior-Weighted Posterior-Style Reward}

Given the strategy prior $p_i$ and evidence score $L_i$, BPR defines candidate
support as
\[
S_i = p_i^\lambda L_i,
\]
where $\lambda$ controls the strength of the strategy prior. The BPR reward is
the normalized support within the rollout group:
\[
R_i^{\mathrm{BPR}}
=
\begin{cases}
\dfrac{p_i^\lambda L_i}{\sum_{j=1}^{n}p_j^\lambda L_j},
& \text{if } \sum_{j=1}^{n}p_j^\lambda L_j > 0, \\
\dfrac{1}{n}, & \text{otherwise.}
\end{cases}
\label{eq:bpr-reward}
\]
This is the posterior-style credit-allocation step. The reward for a rollout is
coupled to the other candidates sampled for the same problem, rather than scored
as an independent scalar.

\subsection{Integration with GRPO}

BPR changes the reward values supplied to the trainer but does not otherwise
modify the optimizer. In our main experiments, BPR rewards are passed to the same
local GRPO training pipeline used by the answer-only baseline. This makes the
main comparison a reward-construction comparison under matched model, data,
LoRA, rollout-count, parser, verifier, and evaluator settings.
```

## 3. Experiment Setup Paragraph: Structured-Format Fairness

Insert in Section 4.1.

```latex
\paragraph{Prompt and evaluator matching.}
All methods in Table~\ref{tab:main-controlled-results} use the same structured
solver prompt, chat template, final-answer parser, deterministic verifier, and
evaluation script within each benchmark-family comparison. In particular, the
answer-only GRPO and Dr.GRPO-style baselines are not evaluated with a simpler or
less constrained prompt: they also generate the same strategy, reasoning, and
final-answer fields used by BPR. This controls for gains that could otherwise
come from the structured output format rather than from the BPR reward.
```

## 4. Main Results Expansion: Why the Table Is Sufficient

Use after the main results paragraph.

```latex
The comparison is intentionally local and controlled. Rather than comparing to
published scores obtained with different models, data mixtures, and full-training
recipes, we ask how the methods behave when deployed under the same Qwen3
backbone, LoRA budget, rollout count, structured prompt, verifier, and evaluator.
Under this matched setting, BPR-GRPO improves the macro-average over both
answer-only GRPO and the Dr.GRPO-style loss baseline at the Qwen3-1.7B and
Qwen3-4B scales. This supports the claim that group-coupled reward construction
can provide gains beyond both binary outcome rewards and an optimizer-side
Dr.GRPO loss change in the same local training surface.
```

## 5. Prior / Hyperparameter Analysis From Existing Artifacts

This is the strongest immediately available response to the professor's prior
ablation comment. It is not a downstream accuracy ablation. It should be titled
as an offline reward-allocation sensitivity analysis.

```latex
\subsection{Offline Prior-Strength Sensitivity}
\label{app:prior-strength}

To analyze the structural role of the strategy prior without launching new RL
training runs, we recompute BPR reward allocation from stored reward-debug
artifacts while varying the prior strength $\lambda$. This analysis keeps the
same sampled rollout groups, verifier labels, analyzer outputs, and evidence
scores, and changes only the exponent on the strategy prior in
$S_i=p_i^\lambda L_i$. It therefore measures how the prior affects reward
allocation, not downstream policy accuracy.

\begin{table}[t]
\centering
\small
\begin{tabular}{cccc}
\toprule
$\lambda$ & Correct reward mass & Top-1 correct & Entropy \\
\midrule
0.0 & 0.7811 & 1.0000 & 0.9082 \\
0.5 & 0.7807 & 1.0000 & 0.9012 \\
0.7 & 0.7803 & 0.9994 & 0.8951 \\
1.0 & 0.7794 & 0.9953 & 0.8832 \\
1.5 & 0.7775 & 0.9943 & 0.8580 \\
2.0 & 0.7752 & 0.9646 & 0.8292 \\
\bottomrule
\end{tabular}
\caption{
Offline prior-strength sensitivity computed from stored BPR reward logs. Larger
$\lambda$ sharpens the reward distribution, reducing entropy. The reported
$\lambda=1.0$ setting preserves high posterior top-1 correctness while assigning
more concentrated credit than the evidence-only setting $\lambda=0$.
}
\label{tab:prior-strength-sensitivity}
\end{table}
```

Recommended text in the main analysis section:

```latex
Appendix~\ref{app:prior-strength} provides an offline prior-strength sensitivity
analysis over stored BPR reward logs. Increasing $\lambda$ sharpens the
within-group reward distribution, as reflected by lower normalized entropy,
while the reported $\lambda=1.0$ setting preserves high posterior top-1
correctness. This supports the intended structural role of the strategy prior as
a credit-allocation bias rather than an unbounded replacement for verifier
correctness.
```

## 6. BPR+Dr.GRPO: What Can and Cannot Be Claimed Now

Do not add a numeric BPR+Dr.GRPO row unless a run exists. Use this limitation /
future-work wording.

```latex
BPR is optimizer-agnostic at the reward-interface level and can in principle be
combined with GRPO-family objectives such as Dr.GRPO by passing BPR rewards to
the corresponding loss. The present controlled table evaluates BPR with the
standard local GRPO pipeline and compares it to a Dr.GRPO-style answer-only loss
baseline. Running BPR rewards together with Dr.GRPO-style loss normalization is
an important follow-up experiment for isolating reward-construction effects from
optimizer-side improvements.
```

If there is space in Limitations:

```latex
We have not yet reported a full BPR+Dr.GRPO run. Such a run would test whether
BPR's reward-construction gains compose with Dr.GRPO-style loss normalization,
rather than only outperforming it as a separate matched baseline.
```

## 7. Strategy Prior as Exploration: Future Direction

Use this in Discussion or Limitations.

```latex
Finally, BPR currently uses strategy information only at the reward-allocation
stage. A natural extension is to use the strategy prior more actively for
exploration. For example, one could separately optimize the strategy-generation
portion of the rollout, sample additional reasoning continuations from
high-prior diverse strategies, or use the prior to encourage exploration over
underrepresented solution plans before answer generation. We leave this
strategy-level exploration extension to future work.
```

## 8. What Not To Overclaim

Use these guardrails in the final draft.

```text
Safe:
- Bayesian-motivated reward construction
- posterior-style finite-candidate support normalization
- prior-weighted evidence allocation
- information-restricted strategy prior
- outcome-anchored evidence surrogate

Avoid:
- calibrated posterior
- likelihood model
- Bayesian RL algorithm
- posterior over policies/environments
- proof that strategy prior improves downstream accuracy
- BPR+Dr.GRPO result, unless trained and evaluated
```

## 9. Immediate Paper Edits Checklist

- Replace Method section order with Section 2 above.
- Add the RLAIF / LLM-judge related-work paragraph.
- Add the structured-format fairness paragraph in Experiments.
- Keep the compact average teaser figure on page 1 if space allows.
- Move the prior-strength sensitivity table into appendix or one paragraph in main Analysis.
- Add BPR+Dr.GRPO as future-work / not-yet-run, not as a result.
- Avoid saying BPR has a calibrated likelihood or true Bayesian posterior.
