# Reference Audit Table

Scope: this table is based on the LaTeX body pasted in the chat. The local
workspace currently contains only `reference_replacements.bib`, which is a
partial replacement BibTeX file, not the full `latex/main.bib`. Therefore,
`Present in local partial bib` only means the key is present in the local
partial file. Keys marked `No` still need to exist in the final Overleaf
`latex/main.bib`.

Summary:

- Unique citation keys in the pasted paper body: 45
- Cited keys found in local `reference_replacements.bib`: 16
- Cited keys not found in local partial bib: 29
- All 16 entries in local `reference_replacements.bib` are cited in the pasted
  paper body.
- No citation key without a plausible corresponding paper target was found from
  the pasted paper text itself, but exact bibliography-entry verification
  requires the final full `.bib`.

| # | Citation key | First citation location | First citation context | Bibliography target that should appear in References | Present in local partial bib | Audit note |
|---:|---|---|---|---|---|---|
| 1 | `wei2022chain` | Introduction, paragraph 1 | Prior work has improved reasoning through chain-of-thought prompting, self-consistency, decomposition, search, program-aided reasoning, and rationale-based self-training | Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS. | No | Used correctly; final bib must include this key. |
| 2 | `wang2023selfconsistency` | Introduction, paragraph 1 | Same multi-sample/reasoning methods citation group | Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR. | No | Used correctly; final bib must include this key. |
| 3 | `zhou2023least` | Introduction, paragraph 1 | Same multi-sample/reasoning methods citation group | Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. 2023. Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. ICLR. | Yes | Used correctly. |
| 4 | `yao2023tree` | Introduction, paragraph 1 | Same multi-sample/reasoning methods citation group | Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS. | Yes | Used correctly. |
| 5 | `chen2023program` | Introduction, paragraph 1 | Same multi-sample/reasoning methods citation group | Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023. Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. TMLR. | No | Used correctly; final bib must include this key. |
| 6 | `zelikman2022star` | Introduction, paragraph 1 | Same multi-sample/reasoning methods citation group | Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. STaR: Bootstrapping Reasoning with Reasoning. NeurIPS. | No | Used correctly; final bib must include this key. |
| 7 | `cobbe2021training` | Introduction, paragraph 1 | Benchmarks such as GSM8K, MATH, Minerva-style evaluations, and OlympiadBench | Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168. | No | Used correctly; final bib must include this key. |
| 8 | `hendrycks2021measuring` | Introduction, paragraph 1 | Same benchmark citation group | Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring Mathematical Problem Solving with the MATH Dataset. NeurIPS Datasets and Benchmarks. | No | Used correctly; final bib must include this key. |
| 9 | `lewkowycz2022solving` | Introduction, paragraph 1 | Same benchmark citation group | Aitor Lewkowycz et al. 2022. Solving Quantitative Reasoning Problems with Language Models. NeurIPS. | No | Used correctly; final bib must include this key. |
| 10 | `he2024olympiadbench` | Introduction, paragraph 1 | Same benchmark citation group | Chaoqun He et al. 2024. OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems. ACL. | No | Used correctly; final bib must include this key. |
| 11 | `shao2024deepseekmath` | Introduction, paragraph 2 | DeepSeekMath introduced GRPO for mathematical reasoning | Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. | Yes | Used correctly. |
| 12 | `guo2025deepseekr1` | Introduction, paragraph 2 | DeepSeek-R1 showed large-scale rule-based RL can elicit reflection, verification, and extended reasoning | DeepSeek-AI / Daya Guo et al. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. | Yes | Citation currently renders as DeepSeek-AI if author is `{{DeepSeek-AI}}`; acceptable if intentional. |
| 13 | `uesato2022solving` | Introduction, paragraph 3 | Process supervision offers denser feedback by evaluating intermediate reasoning steps | Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. 2022. Solving Math Word Problems with Process- and Outcome-Based Feedback. arXiv:2211.14275. | No | Used correctly; final bib must include this key. |
| 14 | `lightman2024lets` | Introduction, paragraph 3 | Same process-supervision citation group | Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let's Verify Step by Step. ICLR. | Yes | Used correctly. |
| 15 | `wang2024mathshepherd` | Introduction, paragraph 3 | Recent work scaled process supervision through automatic annotation, value estimation, search-based supervision, and generative verification | Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations. ACL. | Yes | Used correctly. |
| 16 | `yu2024ovm` | Introduction, paragraph 3 | Same scaled process-supervision citation group | Fei Yu, Anningzhe Gao, and Benyou Wang. 2024. OVM, Outcome-Supervised Value Models for Planning in Mathematical Reasoning. Findings of ACL: NAACL. | No | Used correctly; final bib must include this key. |
| 17 | `luo2024improve` | Introduction, paragraph 3 | Same scaled process-supervision citation group | Liangchen Luo et al. 2024. Improve Mathematical Reasoning in Language Models by Automated Process Supervision. arXiv:2406.06592. | No | Used correctly; final bib must include this key. |
| 18 | `zhang2024rest` | Introduction, paragraph 3 | Same scaled process-supervision citation group | Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024. ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. NeurIPS. | Yes | Used correctly. |
| 19 | `zhang2025generative` | Introduction, paragraph 3 | Same scaled process-supervision citation group | Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. 2025. Generative Verifiers: Reward Modeling as Next-Token Prediction. ICLR. | No | Used correctly; final bib must include this key. |
| 20 | `setlur2025rewarding` | Introduction, paragraph 3 | Same scaled process-supervision citation group | Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. 2025. Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. ICLR. | Yes | Used correctly. |
| 21 | `kojima2022large` | Related Work: Reasoning with multiple samples and verifiers | Chain-of-thought prompting, zero-shot reasoning, least-to-most prompting, self-consistency, Tree of Thoughts, Graph of Thoughts, Program of Thoughts, PAL, ReAct, and rationale-based self-training | Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large Language Models Are Zero-Shot Reasoners. NeurIPS. | No | Used correctly; final bib must include this key. |
| 22 | `besta2024graph` | Related Work: Reasoning with multiple samples and verifiers | Same inference-time reasoning citation group | Maciej Besta et al. 2024. Graph of Thoughts: Solving Elaborate Problems with Large Language Models. AAAI. | No | Used correctly; final bib must include this key. |
| 23 | `gao2023pal` | Related Work: Reasoning with multiple samples and verifiers | Same inference-time reasoning citation group | Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: Program-Aided Language Models. ICML. | No | Used correctly; final bib must include this key. |
| 24 | `yao2023react` | Related Work: Reasoning with multiple samples and verifiers | Same inference-time reasoning citation group | Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. ICLR. | Yes | Used correctly. |
| 25 | `chen2024step` | Related Work: Process supervision and reasoning credit assignment | Process supervision scaled through automatic annotation, MCTS, process preference optimization, progress-based verification, and process-guided search | Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. 2024. Step-Level Value Preference Optimization for Mathematical Reasoning. Findings of ACL: EMNLP. | No | Used correctly; final bib must include this key. |
| 26 | `kazemnejad2024vineppo` | Related Work: Process supervision and reasoning credit assignment | Reasoning credit assignment through value estimation or Monte Carlo-based intermediate returns | Amirhossein Kazemnejad et al. 2024. VinePPO: Unlocking RL Potential for LLM Reasoning Through Refined Credit Assignment. arXiv:2410.01679. | No | Used correctly; final bib must include this key. |
| 27 | `guan2025rstar` | Related Work: Process supervision and reasoning credit assignment | Same value-estimation/intermediate-return citation group | Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. 2025. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking. ICML. | No | Used correctly; final bib must include this key. |
| 28 | `christiano2017deep` | Related Work: RLVR, GRPO, and reward design | Reward-based optimization for language models builds on RLHF and preference optimization | Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep Reinforcement Learning from Human Preferences. NeurIPS. | No | Used correctly; final bib must include this key. |
| 29 | `ziegler2019finetuning` | Related Work: RLVR, GRPO, and reward design | Same RLHF/preference optimization citation group | Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593. | No | Used correctly; final bib must include this key. |
| 30 | `stiennon2020learning` | Related Work: RLVR, GRPO, and reward design | Same RLHF/preference optimization citation group | Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2020. Learning to Summarize from Human Feedback. NeurIPS. | No | Used correctly; final bib must include this key. Important: missing key would cause `???`. |
| 31 | `ouyang2022training` | Related Work: RLVR, GRPO, and reward design | Same RLHF/preference optimization citation group | Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training Language Models to Follow Instructions with Human Feedback. NeurIPS. | Yes | Used correctly. |
| 32 | `schulman2017proximal` | Related Work: RLVR, GRPO, and reward design | Same RLHF/preference optimization citation group | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347. | No | Used correctly; final bib must include this key. |
| 33 | `rafailov2023direct` | Related Work: RLVR, GRPO, and reward design | Same RLHF/preference optimization citation group | Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. NeurIPS. | No | Used correctly; final bib must include this key. |
| 34 | `liu2025understanding` | Related Work: RLVR, GRPO, and reward design | Dr.GRPO studies GRPO-style training choices | Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025. Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783. | Yes | Used correctly; ensure first author is Zichen Liu. |
| 35 | `bai2022constitutional` | Related Work: LLM-judge rewards and RLAIF | BPR is related to RLAIF and LLM-as-judge reward pipelines | Yuntao Bai et al. 2022. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073. | Yes | Used correctly; local partial bib has likely typo `Lukosuite` -> should be `Lukosiute`. |
| 36 | `lee2023rlaif` | Related Work: LLM-judge rewards and RLAIF | Same RLAIF/LLM-judge citation group | Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. 2023. RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. arXiv:2309.00267. | No | Used correctly; final bib must include this key. |
| 37 | `zheng2023judging` | Related Work: LLM-judge rewards and RLAIF | Same RLAIF/LLM-judge citation group | Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685. | Yes | Used correctly. Title braces should preserve MT-Bench and Chatbot Arena. |
| 38 | `dearden1998bayesian` | Related Work: Bayesian and reward-shaping perspectives | Classical Bayesian RL maintains uncertainty over models, value functions, or policies | Richard Dearden, Nir Friedman, and Stuart Russell. 1998. Bayesian Q-Learning. AAAI/IAAI. | No | Used correctly; final bib must include this key. |
| 39 | `strens2000bayesian` | Related Work: Bayesian and reward-shaping perspectives | Same classical Bayesian RL citation group | Malcolm J. A. Strens. 2000. A Bayesian Framework for Reinforcement Learning. ICML. | Yes | Used correctly. |
| 40 | `ghavamzadeh2015bayesian` | Related Work: Bayesian and reward-shaping perspectives | Same classical Bayesian RL citation group | Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, and Aviv Tamar. 2015. Bayesian Reinforcement Learning: A Survey. Foundations and Trends in Machine Learning. | No | Used correctly; final bib must include this key. |
| 41 | `osband2013more` | Related Work: Bayesian and reward-shaping perspectives | Same classical Bayesian RL citation group | Ian Osband, Daniel Russo, and Benjamin Van Roy. 2013. (More) Efficient Reinforcement Learning via Posterior Sampling. NeurIPS. | No | Used correctly; final bib must include this key. |
| 42 | `russo2018tutorial` | Related Work: Bayesian and reward-shaping perspectives | Same classical Bayesian RL citation group | Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. 2018. A Tutorial on Thompson Sampling. Foundations and Trends in Machine Learning. | Yes | Used correctly. |
| 43 | `zhang2025beyond` | Related Work: Bayesian and reward-shaping perspectives | BARL frames reflective exploration as Bayes-adaptive decision making under a posterior over MDPs | Shenao Zhang, Yaqing Wang, Yinxiao Liu, Tianqi Liu, Peter Grabowski, Eugene Ie, Zhaoran Wang, and Yunxuan Li. 2025. Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning. arXiv:2505.20561. | Yes | Used correctly. |
| 44 | `ng1999policy` | Related Work: Bayesian and reward-shaping perspectives | BPR is also related to reward shaping and reward-hacking concerns | Andrew Y. Ng, Daishi Harada, and Stuart Russell. 1999. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. ICML. | No | Used correctly; final bib must include this key. |
| 45 | `amodei2016concrete` | Related Work: Bayesian and reward-shaping perspectives | Same reward-shaping/reward-hacking citation group | Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mane. 2016. Concrete Problems in AI Safety. arXiv:1606.06565. | No | Used correctly; final bib must include this key. |

## Missing from local partial BibTeX

The following cited keys are not present in the local `reference_replacements.bib`
file and must be present in the final `latex/main.bib` to avoid `???` citations:

`wei2022chain`, `wang2023selfconsistency`, `chen2023program`,
`zelikman2022star`, `cobbe2021training`, `hendrycks2021measuring`,
`lewkowycz2022solving`, `he2024olympiadbench`, `uesato2022solving`,
`yu2024ovm`, `luo2024improve`, `zhang2025generative`, `kojima2022large`,
`besta2024graph`, `gao2023pal`, `chen2024step`, `kazemnejad2024vineppo`,
`guan2025rstar`, `christiano2017deep`, `ziegler2019finetuning`,
`stiennon2020learning`, `schulman2017proximal`, `rafailov2023direct`,
`lee2023rlaif`, `dearden1998bayesian`, `ghavamzadeh2015bayesian`,
`osband2013more`, `ng1999policy`, `amodei2016concrete`.

## Local partial BibTeX entries that are cited

All keys in `reference_replacements.bib` are cited in the pasted paper body:

`bai2022constitutional`, `guo2025deepseekr1`, `shao2024deepseekmath`,
`lightman2024lets`, `setlur2025rewarding`, `ouyang2022training`,
`wang2024mathshepherd`, `zhang2024rest`, `zhou2023least`, `yao2023react`,
`yao2023tree`, `liu2025understanding`, `zheng2023judging`,
`zhang2025beyond`, `strens2000bayesian`, `russo2018tutorial`.

## BibTeX quality notes

- `bai2022constitutional`: local partial BibTeX appears to spell Kamile
  Lukosiute as `Lukosuite`; correct this if the final bib still has that typo.
- `stiennon2020learning`: this key is cited in the paper and must be in the
  final bib. It is not in the local partial file.
- `guo2025deepseekr1`: decide whether the bibliography should render as
  `DeepSeek-AI` or as `Daya Guo et al.`; the current main-text citation
  `DeepSeek-AI, 2025` is consistent with `author={{DeepSeek-AI}}`.
- Title capitalization should be protected for items such as `MT-Bench`,
  `Chatbot Arena`, `Bayesian`, `Thompson`, `Markovian`, `LLM`, `RL`, and method
  names such as `DeepSeek-R1`, `DeepSeekMath`, `ReAct`, `PAL`, `OVM`,
  `ReST-MCTS*`, and `STaR`.
