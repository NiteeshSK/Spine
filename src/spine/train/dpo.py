"""Phase 2 -- preference tuning with DPO.

    python -m spine.train.dpo --config configs/dpo.yaml
    python -m spine.train.dpo --config configs/dpo.yaml --set train.beta=0.3 output_dir=runs/dpo_b03

Setup notes
-----------
* The policy **continues the SFT adapter** rather than starting a fresh one.
  A fresh adapter over the base weights would make the reference model and the
  policy start from different places, and the KL term would be measuring the
  wrong distance.
* ``ref_model=None`` with PEFT: TRL computes reference log-probs by disabling
  the adapter, so there is no second 8B model in memory. This is what keeps DPO
  inside a single 40 GB A100 (and inside a T4 with a 1.5B base).
* ``beta`` is the KL leash. Low beta with a high learning rate is the classic
  DPO failure: `rewards/chosen` and `rewards/rejected` both collapse, the
  margin still grows (the loss is happy), and the model degenerates into
  repeated tokens. The `CollapseGuard` callback below watches for exactly that
  and stops the run instead of burning the rest of your GPU budget.
* Always re-run the eval harness after DPO. DPO optimises a *relative*
  objective; it is entirely possible to improve the preference loss and lose
  exact match. Only the task metric decides whether phase 2 shipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import apply_overrides, load_yaml, save_metadata
from .sft import CsvLogger


def make_callbacks(out_dir: Path, guard_cfg: dict):
    from transformers import TrainerCallback

    csv_logger = CsvLogger(out_dir / "train_log.csv")

    class CsvCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            csv_logger.on_log(args, state, control, logs=logs, **kwargs)

    class CollapseGuard(TrainerCallback):
        """Stop the run when DPO starts winning the loss by degrading both sides.

        Trip conditions (any two consecutive checks):
          * ``rewards/chosen`` falls below ``min_chosen_reward`` -- the policy is
            assigning less mass to the *good* answer than the reference did.
          * ``logps/chosen`` drops more than ``max_logp_drop`` from its first
            observed value -- the classic degeneration signature.
        """

        def __init__(self, cfg: dict):
            self.min_chosen = cfg.get("min_chosen_reward", -3.0)
            self.max_drop = cfg.get("max_logp_drop", 60.0)
            self.first_logp: float | None = None
            self.strikes = 0
            self.trips: list[dict] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "rewards/chosen" not in logs:
                return
            chosen = logs["rewards/chosen"]
            logp = logs.get("logps/chosen")
            if self.first_logp is None and logp is not None:
                self.first_logp = logp
            bad = chosen < self.min_chosen
            if logp is not None and self.first_logp is not None:
                bad = bad or (self.first_logp - logp) > self.max_drop
            self.strikes = self.strikes + 1 if bad else 0
            if bad:
                self.trips.append({"step": state.global_step, **{k: logs.get(k) for k in ("rewards/chosen", "rewards/margins", "logps/chosen")}})
            if self.strikes >= 2:
                print(f"[dpo] COLLAPSE GUARD tripped at step {state.global_step}: {self.trips[-1]}")
                print("[dpo] lower the learning rate or raise beta, then re-run")
                (out_dir / "collapse_guard.json").write_text(json.dumps(self.trips, indent=2))
                control.should_training_stop = True

    return [CsvCallback(), CollapseGuard(guard_cfg)]


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO preference tuning on top of the SFT adapter.")
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    m = cfg["model"]
    has_cuda = torch.cuda.is_available()
    kwargs = {"trust_remote_code": True}
    if m.get("load_in_4bit", True) and has_cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if has_cuda else torch.float32

    base = AutoModelForCausalLM.from_pretrained(m["base"], **kwargs)
    base.config.use_cache = False
    # Continue the SFT adapter: policy and implicit reference share an origin.
    model = PeftModel.from_pretrained(base, m["sft_adapter"], is_trainable=True)

    tok = AutoTokenizer.from_pretrained(m["base"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = cfg["data"]
    ds = load_dataset("json", data_files={"train": data["pairs"]})["train"]
    if data.get("val_fraction", 0.05) > 0 and len(ds) > 50:
        split = ds.train_test_split(test_size=data.get("val_fraction", 0.05), seed=cfg.get("seed", 42))
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None
    keep = {"prompt", "chosen", "rejected"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    if eval_ds is not None:
        eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep])
    print(f"[dpo] pairs train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")

    t = cfg["train"]
    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        beta=t["beta"],
        loss_type=t.get("loss_type", "sigmoid"),
        label_smoothing=t.get("label_smoothing", 0.0),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t.get("scheduler", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.1),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        logging_steps=t.get("logging_steps", 5),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=t.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=t.get("save_steps", 100),
        save_total_limit=2,
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optim", "paged_adamw_8bit"),
        max_length=t.get("max_length", 1536),
        max_prompt_length=t.get("max_prompt_length", 1024),
        seed=cfg.get("seed", 42),
        report_to=t.get("report_to", ["none"]),
        run_name=cfg.get("run_name", "spine-dpo"),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: reference = adapter disabled, no second model in VRAM
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=make_callbacks(out_dir, cfg.get("collapse_guard", {})),
    )

    save_metadata(cfg, out_dir)
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (out_dir / "train_result.json").write_text(json.dumps(result.metrics, indent=2, default=str))
    print(f"[dpo] adapter -> {adapter_dir}")
    print("[dpo] now re-run the eval harness; preference loss is not the deliverable")


if __name__ == "__main__":
    main()
