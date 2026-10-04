# Results

> **Status: not yet populated.** No training run has been executed against this
> repository. Every number below is a placeholder. Nothing in this file is
> hand-written — run the pipeline and regenerate it:
>
> ```bash
> make data sft eval-base eval-sft dpo-pairs dpo eval-dpo eval-ood
> make figures table
> ```
>
> `make table` writes `docs/_table_test.md` and `docs/_table_ood.md` directly from
> the JSON in `reports/`. Paste those in below, commit `reports/` and `assets/`,
> and the claims in this file become auditable by anyone who clones the repo.

## Setup

| | |
|---|---|
| Base model | `Qwen/Qwen3-8B` |
| Method | QLoRA (NF4) → DPO on mined preference pairs |
| Train / val / test / OOD | 6,000 / 500 / 800 / 300 |
| Hardware | *fill in* |
| Total GPU-hours | *fill in* |

## Headline — test split

<!-- paste docs/_table_test.md here -->

| Metric | Base + best prompt | Phase 1: SFT | Phase 2: SFT+DPO | Δ |
|---|---|---|---|---|
| JSON validity (no repair) | — | — | — | — |
| Schema valid | — | — | — | — |
| **Exact match (full record)** | — | — | — | — |
| Macro field accuracy | — | — | — | — |
| Refusal F1 | — | — | — | — |
| Missed abstain | — | — | — | — |
| Evidence span grounded | — | — | — | — |
| Prompt tokens / request | ~1,150 | ~30 | ~30 | −97% |

The prompt-token row is the one number that can be stated before training: the
baseline carries the full schema plus four exemplars in every request, and the
fine-tune carries a single instruction line. That is a real serving-cost result
independent of accuracy.

## Generalisation — OOD split (held-out surface styles)

The `ood` split contains only `cli` and `handoff` surface styles, which appear
**nowhere** in training, plus abstain phrasings from a disjoint template pool.
The gap between the test and OOD numbers is the honest measure of whether the
model learned the task or the corpus.

<!-- paste docs/_table_ood.md here -->

## Where the errors live

`reports/*_per_example.jsonl` records the wrong fields for every failed example.
Expected shape of the finding, to be confirmed or refuted by the run:
`started_at` (timezone conversion) and `severity` (implicit vs stated) should
dominate; `vendor` and `event_type` should be near-saturated after SFT.

![field accuracy](../assets/field_accuracy_test.png)

## Ablations worth running

| Question | Command |
|---|---|
| How much data is actually needed? | `make sft SFT_CFG=configs/sft_qlora.yaml` with `--set data.max_train=500,1000,2000,6000` |
| Does LoRA rank matter here? | `--set lora.r=8` / `16` / `32` / `64` |
| Is DPO's β the fragile knob? | `--set train.beta=0.05,0.1,0.3,0.5` (watch `CollapseGuard`) |
| Do the MLP projections carry the schema? | `--set lora.target_modules='["q_proj","v_proj"]'` |
| Is few-shot doing the work in the baseline? | `--prompt baseline_zeroshot` |
