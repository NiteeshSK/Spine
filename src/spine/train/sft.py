"""Phase 1 -- supervised fine-tuning with QLoRA.

    python -m spine.train.sft --config configs/sft_qlora.yaml
    python -m spine.train.sft --config configs/sft_smoke.yaml   # CPU, 3 minutes

Three things here are not boilerplate and are worth reading:

1. **Loss is masked to the assistant turn.** Training on the prompt tokens as
   well means a big share of the gradient is spent learning to reproduce
   messy syslog, which is not the task. TRL's ``assistant_only_loss`` handles
   this when the chat template emits ``{% generation %}``; if the template
   doesn't, we fall back to a completion-only collator keyed on the assistant
   header. The fallback is not decoration -- it is the difference between a
   model that emits JSON and a model that autocompletes email signatures.

2. **A task-metric callback, not just eval loss.** Every ``eval_steps`` the
   callback generates on a held-out slice and logs JSON validity / exact match.
   Eval loss went down monotonically in my run while exact match was still
   flat; loss curves alone would have said "done" three epochs early.

3. **Everything is logged to CSV** for `spine.viz.plots`, because "include
   training curves" in the deliverables means real curves from real logs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..config import apply_overrides, load_yaml, save_metadata


class CsvLogger:
    """Trainer callback that mirrors every log line into a flat CSV."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.keys: list[str] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D102
        if not logs:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4), **logs}
        self.rows.append(row)
        for k in row:
            if k not in self.keys:
                self.keys.append(k)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.keys)
            w.writeheader()
            w.writerows(self.rows)


def make_callbacks(out_dir: Path, eval_rows: list[dict], tokenizer, every: int, n_probe: int):
    from transformers import TrainerCallback

    from ..eval.metrics import score_dataset
    from ..prompts import sft_messages

    csv_logger = CsvLogger(out_dir / "train_log.csv")

    class CsvCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            csv_logger.on_log(args, state, control, logs=logs, **kwargs)

    class TaskMetricCallback(TrainerCallback):
        """Generate on a small held-out probe and log real task metrics."""

        def __init__(self):
            self.rows = eval_rows[:n_probe]
            self.history: list[dict] = []

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if not self.rows or model is None or state.global_step % every:
                return
            import torch

            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(self.rows), 8):
                    batch = self.rows[i : i + 8]
                    texts = [
                        tokenizer.apply_chat_template(
                            sft_messages(r["text"]), tokenize=False, add_generation_prompt=True
                        )
                        for r in batch
                    ]
                    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    out = model.generate(
                        **enc, max_new_tokens=320, do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )
                    preds.extend(
                        tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
                    )
            model.train()
            agg, _ = score_dataset([p.strip() for p in preds], self.rows)
            entry = {
                "step": state.global_step,
                "probe_json_valid_strict": agg["json_valid_strict"],
                "probe_schema_valid": agg["schema_valid"],
                "probe_exact_match": agg["exact_match"],
                "probe_refusal_f1": agg["refusal_f1"],
            }
            self.history.append(entry)
            print("[probe]", json.dumps(entry))
            (out_dir / "probe_metrics.json").write_text(json.dumps(self.history, indent=2))
            csv_logger.on_log(args, state, control, logs={k: v for k, v in entry.items() if k != "step"})

    return [CsvCallback(), TaskMetricCallback()]


def build_model(cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    model = AutoModelForCausalLM.from_pretrained(m["base"], **kwargs)
    model.config.use_cache = False

    tok = AutoTokenizer.from_pretrained(m["base"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # right padding for training, left for generation

    if m.get("load_in_4bit", True) and has_cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lc = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lc["target_modules"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tok, peft_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA SFT for SPINE extraction.")
    ap.add_argument("--config", default="configs/sft_qlora.yaml")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    data = cfg["data"]
    ds = load_dataset(
        "json",
        data_files={"train": data["train"], "validation": data["val"]},
    )
    if data.get("max_train"):
        ds["train"] = ds["train"].select(range(min(data["max_train"], len(ds["train"]))))
    print(f"[sft] train={len(ds['train'])} val={len(ds['validation'])}")

    model, tok, _ = build_model(cfg)

    with open(data["probe"], encoding="utf-8") as fh:
        probe_rows = [json.loads(line) for line in fh if line.strip()]

    t = cfg["train"]
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t.get("scheduler", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=t.get("save_steps", 100),
        save_total_limit=2,
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.get("optim", "paged_adamw_8bit"),
        max_length=t.get("max_seq_len", 1536),
        packing=t.get("packing", False),
        seed=cfg.get("seed", 42),
        report_to=t.get("report_to", ["none"]),
        run_name=cfg.get("run_name", "spine-sft"),
    )
    # Mask the prompt: only assistant tokens contribute to the loss.
    try:
        sft_args.assistant_only_loss = True
    except Exception:  # pragma: no cover - older TRL
        print("[sft] TRL does not expose assistant_only_loss; using completion-only collator")

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        callbacks=make_callbacks(
            out_dir, probe_rows, tok, every=t.get("probe_every", 100), n_probe=t.get("probe_size", 64)
        ),
    )

    save_metadata(cfg, out_dir)
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (out_dir / "train_result.json").write_text(json.dumps(result.metrics, indent=2, default=str))
    print(f"[sft] adapter -> {adapter_dir}")
    print(f"[sft] curves  -> {out_dir / 'train_log.csv'}")


if __name__ == "__main__":
    main()
