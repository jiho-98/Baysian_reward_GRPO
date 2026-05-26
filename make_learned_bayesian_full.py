#!/usr/bin/env python3
from pathlib import Path

SRC = Path("Bayesian_Full_GRPO.py")
DST = Path("Bayesian_Full_GRPO_learned.py")

text = SRC.read_text(encoding="utf-8")


def replace_once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"[{label}] expected exactly 1 match, found {count}")
    return src.replace(old, new, 1)


# ---------------------------------------------------------------------
# 1. prior_mode에 learned_unified_analyzer 추가
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '        choices=("uniform", "llm_strategy_prior"),\n        default="llm_strategy_prior",',
    '        choices=("uniform", "llm_strategy_prior", "learned_unified_analyzer"),\n        default="llm_strategy_prior",',
    "add prior_mode choice",
)


# ---------------------------------------------------------------------
# 2. learned analyzer CLI 추가
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '    parser.add_argument("--prior_judge_model", default=None)\n    parser.add_argument("--prior_lambda", type=float, default=1.0)',
    '''    parser.add_argument("--prior_judge_model", default=None)

    # Learned unified analyzer mode.
    # These are used only when --prior_mode learned_unified_analyzer.
    # Existing prompted-analyzer behavior is unchanged for uniform/llm_strategy_prior.
    parser.add_argument(
        "--analyzer_model_name",
        default=None,
        help="Base model name for the learned unified analyzer. Defaults to --model_name.",
    )
    parser.add_argument(
        "--analyzer_adapter_path",
        default=None,
        help="LoRA adapter path for the learned unified analyzer.",
    )
    add_bool_arg(
        parser,
        "learned_analyzer_task_prefix",
        True,
        "Prepend [TASK=prior_judge]/[TASK=evidence_judge] to the existing prompt body for the learned analyzer.",
    )

    parser.add_argument("--prior_lambda", type=float, default=1.0)''',
    "add learned analyzer cli",
)


# ---------------------------------------------------------------------
# 3. TASK_PREFIXES 추가
#    기존 build_prior_judge_prompt / build_evidence_judge_prompt 본문은 건드리지 않음.
# ---------------------------------------------------------------------
text = replace_once(
    text,
    'The last character of your response must be }."""\n\nALLOWED_ERROR_TYPES = {',
    '''The last character of your response must be }."""

TASK_PREFIXES = {
    "evidence_judge": (
        "[TASK=evidence_judge]\\n"
        "Return the evidence-judge JSON schema only.\\n"
        "Do not output prior-judge fields.\\n\\n"
    ),
    "prior_judge": (
        "[TASK=prior_judge]\\n"
        "Return the prior-judge JSON schema only.\\n"
        "Do not output evidence-judge fields.\\n\\n"
    ),
}

ALLOWED_ERROR_TYPES = {''',
    "add task prefixes",
)


# ---------------------------------------------------------------------
# 4. BayesianRewardScorer에 learned analyzer loader/generator 추가
# ---------------------------------------------------------------------
learned_methods = r'''
    def _load_learned_analyzer_bundle(self) -> tuple[Any, Any]:
        model_name = self.args.analyzer_model_name or self.args.model_name
        adapter_path = self.args.analyzer_adapter_path
        if not adapter_path:
            raise RuntimeError(
                "--analyzer_adapter_path is required when --prior_mode learned_unified_analyzer."
            )

        cache_key = f"learned_unified_analyzer::{model_name}::{adapter_path}"
        if cache_key in self.judge_bundles:
            return self.judge_bundles[cache_key]

        torch = import_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers and peft are required for learned_unified_analyzer mode. "
                "Install with `pip install transformers peft`."
            ) from exc

        print(
            "[INFO] loading learned unified analyzer: "
            f"base={model_name} adapter={adapter_path}"
        )

        try:
            judge_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if judge_tokenizer.pad_token is None:
                judge_tokenizer.pad_token = judge_tokenizer.eos_token
            judge_tokenizer.padding_side = "left"

            model_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.bfloat16 if self.args.bf16 else torch.float16

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
            judge_model = PeftModel.from_pretrained(base_model, adapter_path)

            if not torch.cuda.is_available():
                judge_model.to("cpu")

            judge_model.eval()

        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda out of memory" in message:
                raise RuntimeError(
                    "Failed to load the learned unified analyzer due to OOM. "
                    "Try reducing generation/batch settings or use a smaller analyzer."
                ) from exc
            raise

        self.judge_bundles[cache_key] = (judge_model, judge_tokenizer)
        return judge_model, judge_tokenizer

    def _maybe_add_learned_task_prefix(self, *, task: str, prompt: str) -> str:
        if not getattr(self.args, "learned_analyzer_task_prefix", True):
            return prompt
        return TASK_PREFIXES[task] + prompt

    def _generate_learned_analyzer_output(
        self,
        *,
        task: str,
        prompt: str,
        temperature: float,
    ) -> str:
        judge_model, judge_tokenizer = self._load_learned_analyzer_bundle()
        prompt = self._maybe_add_learned_task_prefix(task=task, prompt=prompt)
        messages = [
            {"role": "system", "content": JUDGE_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = render_chat_prompt(messages, judge_tokenizer)
        outputs = generate_smoke_outputs(
            model=judge_model,
            tokenizer=judge_tokenizer,
            prompt=rendered_prompt,
            num_generations=1,
            max_new_tokens=self.args.judge_max_new_tokens,
            temperature=temperature,
            top_p=self.args.top_p,
        )
        return outputs[0] if outputs else ""

'''

text = replace_once(
    text,
    "    def _generate_judge_output(\n",
    learned_methods + "    def _generate_judge_output(\n",
    "insert learned analyzer methods",
)


# ---------------------------------------------------------------------
# 5. _call_prior_judge에서 learned mode 허용
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '''        if self.args.prior_mode != "llm_strategy_prior":
            raise ValueError(f"Unknown prior_mode: {self.args.prior_mode}")

        prompt = build_prior_judge_prompt(problem, rollouts, len(rollouts))''',
    '''        if self.args.prior_mode not in ("llm_strategy_prior", "learned_unified_analyzer"):
            raise ValueError(f"Unknown prior_mode: {self.args.prior_mode}")

        use_learned_analyzer = self.args.prior_mode == "learned_unified_analyzer"

        prompt = build_prior_judge_prompt(problem, rollouts, len(rollouts))''',
    "allow learned prior mode",
)

text = replace_once(
    text,
    '''            raw_output = self._generate_judge_output(
                model_name=self.prior_judge_model_name,
                prompt=prompt,
                temperature=self.args.prior_judge_temperature,
            )''',
    '''            if use_learned_analyzer:
                raw_output = self._generate_learned_analyzer_output(
                    task="prior_judge",
                    prompt=prompt,
                    temperature=self.args.prior_judge_temperature,
                )
            else:
                raw_output = self._generate_judge_output(
                    model_name=self.prior_judge_model_name,
                    prompt=prompt,
                    temperature=self.args.prior_judge_temperature,
                )''',
    "prior learned generation",
)

text = replace_once(
    text,
    '''            repair_raw_output = self._generate_judge_output(
                model_name=self.prior_judge_model_name,
                prompt=repair_prompt,
                temperature=self.args.prior_judge_temperature,
            )''',
    '''            if use_learned_analyzer:
                repair_raw_output = self._generate_learned_analyzer_output(
                    task="prior_judge",
                    prompt=repair_prompt,
                    temperature=self.args.prior_judge_temperature,
                )
            else:
                repair_raw_output = self._generate_judge_output(
                    model_name=self.prior_judge_model_name,
                    prompt=repair_prompt,
                    temperature=self.args.prior_judge_temperature,
                )''',
    "prior repair learned generation",
)


# ---------------------------------------------------------------------
# 6. _call_evidence_judge에서 learned mode 사용
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '''            raw_output = self._generate_judge_output(
                model_name=self.evidence_judge_model_name,
                prompt=prompt,
                temperature=self.args.evidence_judge_temperature,
            )''',
    '''            if self.args.prior_mode == "learned_unified_analyzer":
                raw_output = self._generate_learned_analyzer_output(
                    task="evidence_judge",
                    prompt=prompt,
                    temperature=self.args.evidence_judge_temperature,
                )
            else:
                raw_output = self._generate_judge_output(
                    model_name=self.evidence_judge_model_name,
                    prompt=prompt,
                    temperature=self.args.evidence_judge_temperature,
                )''',
    "evidence learned generation",
)


# ---------------------------------------------------------------------
# 7. debug jsonl에 analyzer 정보 추가
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '''                    "prior_judge_model": self.prior_judge_model_name,
                    "prior_judge_fallback_used": prior_debug["prior_judge_fallback_used"],''',
    '''                    "prior_judge_model": self.prior_judge_model_name,
                    "learned_analyzer_model_name": self.args.analyzer_model_name,
                    "learned_analyzer_adapter_path": self.args.analyzer_adapter_path,
                    "learned_analyzer_task_prefix": getattr(self.args, "learned_analyzer_task_prefix", None),
                    "prior_judge_fallback_used": prior_debug["prior_judge_fallback_used"],''',
    "debug prior analyzer fields",
)

text = replace_once(
    text,
    '''                    "evidence_judge_model": self.evidence_judge_model_name,
                    "evidence_judge_fallback_used": evidence_debug["evidence_judge_fallback_used"],''',
    '''                    "evidence_judge_model": self.evidence_judge_model_name,
                    "learned_evidence_analyzer_model_name": self.args.analyzer_model_name,
                    "learned_evidence_analyzer_adapter_path": self.args.analyzer_adapter_path,
                    "evidence_judge_fallback_used": evidence_debug["evidence_judge_fallback_used"],''',
    "debug evidence analyzer fields",
)


# ---------------------------------------------------------------------
# 8. training_config.json에 learned analyzer 정보 추가
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '''            "prior_judge_model": args.prior_judge_model,
            "prior_lambda": args.prior_lambda,''',
    '''            "prior_judge_model": args.prior_judge_model,
            "analyzer_model_name": args.analyzer_model_name,
            "analyzer_adapter_path": args.analyzer_adapter_path,
            "learned_analyzer_task_prefix": getattr(args, "learned_analyzer_task_prefix", None),
            "prior_lambda": args.prior_lambda,''',
    "config learned analyzer fields",
)


# ---------------------------------------------------------------------
# 9. main validation 추가
# ---------------------------------------------------------------------
text = replace_once(
    text,
    '''    if args.prior_softmax_temperature <= 0:
        raise SystemExit("--prior_softmax_temperature must be positive.")''',
    '''    if args.prior_softmax_temperature <= 0:
        raise SystemExit("--prior_softmax_temperature must be positive.")
    if args.prior_mode == "learned_unified_analyzer":
        if not args.analyzer_adapter_path:
            raise SystemExit("--analyzer_adapter_path is required when --prior_mode learned_unified_analyzer.")
        if args.analyzer_model_name is None:
            args.analyzer_model_name = args.model_name''',
    "main learned analyzer validation",
)


DST.write_text(text, encoding="utf-8")
print(f"[OK] wrote {DST}")
print("[OK] original Bayesian_Full_GRPO.py was not modified.")