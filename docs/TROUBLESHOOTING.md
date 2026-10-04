# Troubleshooting narrative

Two parts. **Part A** is a log of defects that were actually found and fixed while
building this pipeline — each one is traceable to a comment in the code and, in most
cases, to the unit test that caught it. **Part B** is the training-run log: the
failure modes to expect on GPU, what each one looks like, and where to record what
happened on your run.

Part A is the more useful half. Most fine-tuning projects fail in the data layer,
silently, and the model gets blamed.

---

## Part A — defects found while building (all fixed)

### #1 — Labels parsed back out of generated text

**Symptom.** The first version of the generator rendered messy text and then
regex-ed the label out of it. Every regex bug became a label bug, and the model
would have learned the bugs faithfully.

**Fix.** Inverted the pipeline: sample the ground-truth facts first, then render
*those facts* into a surface form. The label is never derived from the text.
See the module docstring in `src/spine/data/synth.py`.

**Lesson.** In synthetic data, the label should be the cause of the text, never
its consequence.

---

### #2 — Labels the model could not possibly infer

**Symptom.** `started_at` was populated from the sampled event time, but several
(style × template) combinations never rendered the timestamp into the text. Same
for `device_hostname` under the "Automated NMS notification" email subject, and
for `affected_clients` in styles that omit the impact line.

Roughly 6% of incident rows carried at least one field that no reader could recover
from the input. That is not a hard example. That is an instruction to hallucinate:
the only way to fit those rows is to invent plausible values.

**Fix.** Faithfulness guards immediately before the record is built — if the
hostname, site, ticket, client count or local `HH:MM` is not present in the rendered
text, the label becomes `null`.

```python
low = text.lower()
if host.lower() not in low:      host = None
if site and site.lower() not in low:   site = None
if ticket and ticket.lower() not in low: ticket = None
if clients is not None and str(clients) not in text: clients = None
if has_time and f"{base:%H:%M}" not in text: has_time = False
```

Locked in by `tests/test_synth.py::test_labels_are_faithful_to_the_text`.

**Lesson.** "Is this label *correct*?" is the wrong question. The question is
"is this label *derivable from this input*?"

---

### #3 — Train/eval leakage through the abstain pool

**Symptom.** `tests/test_synth.py::test_splits_do_not_share_examples` failed on the
first run: abstain examples were drawn from one shared template pool, so the exact
same "timesheets are due friday" note appeared in both train and test. Refusal F1
would have been measuring memorisation.

**Fix, in two layers:**
1. The abstain pool is partitioned into disjoint `train` and `eval` groups — the
   test and OOD splits only ever see phrasings the model was never trained on.
2. `build_sft.py` hashes every input and de-duplicates every later split against
   train, then *asserts zero overlap* and exits non-zero if any survives.

The gate earned its keep immediately: after the pool was partitioned, it still
caught 15 chance collisions between train and val and 34 between train and the DPO
prompt pool, and removed them.

**Lesson.** A leakage check that only warns is decoration. Make it exit non-zero.

---

### #4 — The DPO reward double-counted a single signal

**Symptom.** The candidate reward had both a `strict_json` term and a `no_prose`
term. They fire on exactly the same condition. A candidate that was semantically
perfect but wrapped in a markdown fence lost 1.5 of 11.5 points and could rank
*below* a schema-invalid candidate that happened to be bare. Preference pairs built
from that reward would have taught the model that prose formatting matters more
than getting the fields right.

Caught by `tests/test_metrics.py::test_prose_wrapper_costs_strictness_but_not_content`.

**Fix.** One signal, one weight. `no_prose` deleted; see the note above
`REWARD_WEIGHTS` in `src/spine/eval/metrics.py`.

**Lesson.** Write a test that asserts the *ordering* your reward is supposed to
produce, not just that it returns a number in `[0, 1]`.

---

### #5 — The pair miner threw away its best pairs

**Symptom.** The first pair-mining run yielded **0 pairs from 200 prompts**. The
skip condition was `if best_reward >= threshold: skip` — the intent was "don't
train on prompts the model already solves", but as written it discarded every
prompt where *any* of the k samples was perfect. A prompt with one perfect sample
and one badly broken sample is the single most informative pair available.

**Fix.** Skip only when *every* sample is already solved, i.e. when there is no
contrast left:

```python
if worst_r >= args.solved_threshold:      # was: best_r >= ...
    stats["skipped_all_samples_solved"] += 1
    continue
```

Yield went from 0/200 to 127/200 with a mean margin of 0.35.

**Lesson.** "This prompt is already solved" is a statement about the worst sample,
not the best one. Log the yield of every mining run — a yield of zero or of 100%
both mean the gates are wrong.

---

### #6 — Label leakage through the email subject line

**Symptom.** Every generated alert email had a subject of the form
`[P1] link down - sw-core-02`. `event_type` and `severity` were sitting in the
first eight tokens of nearly half the corpus. The task was measurably easier than
the real one, and the reported numbers would not have survived contact with a real
inbox.

**Fix.** Subjects are now sampled from four variants, two of which name neither the
event nor the device (`Automated NMS notification`, `<host> - please check`).

**Lesson.** When a metric looks too good on a synthetic corpus, look for the
shortcut before you look for the insight.

---

## Part B — training-run log (fill in as you run)

### Watch list, with the symptom and the first thing to try

| Symptom | Likely cause | First move |
|---|---|---|
| Loss falls smoothly, generations are still prose | Loss not masked to the assistant turn | Confirm `assistant_only_loss=True` took effect; check the chat template emits `{% generation %}` |
| `eval_loss` flat while train loss falls | Overfitting, or LoRA rank too high for 6k examples | Drop `lora.r` to 16, add dropout, cut to 2 epochs |
| Model emits valid JSON with invented keys | Rank too low / too few MLP modules to encode a closed schema | Add `gate_proj, up_proj, down_proj` to `target_modules` |
| Exact match flat while eval loss improves | Loss is a bad proxy for this task | This is why `TaskMetricCallback` exists — trust the probe, not the loss |
| Timestamps consistently wrong by an offset | Timezone conversion not learned | Check per-field accuracy for `started_at`; consider oversampling non-UTC rows |
| CUDA OOM at step 0 | Sequence length × batch | `max_seq_len: 1024`, `batch_size: 1`, raise `grad_accum` to keep the effective batch |
| CUDA OOM after 100 steps | Eval batch, not train batch | Lower `per_device_eval_batch_size`, raise `eval_steps` |
| `bitsandbytes` import error on T4 | bf16 assumed | `bf16: false` — Turing has no bf16. Use `configs/sft_qlora_t4.yaml` |
| DPO: margins grow, both rewards fall | KL leash too loose for the LR | Raise `beta` to 0.3, or drop LR to 1e-6. The `CollapseGuard` should stop this for you |
| DPO: preference loss improves, exact match drops | Optimising a relative objective away from the task | Fewer steps, higher beta, and re-check the pair mix in `reports/dpo_pairs.json` |
| DPO: pair yield above ~90% | Margin gate too loose; you are training on noise | Raise `--min-margin` |
| Outputs degenerate into repeated tokens | Classic DPO collapse | Roll back to the SFT adapter and restart phase 2 with higher beta |

### Run log

| # | Date | Change | Metric before | Metric after | Kept? |
|---|---|---|---|---|---|
| 1 | | baseline SFT, r=32, lr=2e-4, 3 epochs | — | — | |
| 2 | | | | | |
| 3 | | | | | |

### Compute spent

| Stage | Hardware | Wall-clock | Notes |
|---|---|---|---|
| SFT | | | |
| Candidate sampling (k=6) | | | |
| DPO | | | |
| Evaluation (3 models × 2 splits) | | | |
