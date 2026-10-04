# data/

Generated at build time, not committed.

```bash
make data      # writes data/raw/*.jsonl and data/sft/*.jsonl
```

`sample/` holds 100 committed rows so the corpus can be inspected without running
anything. Regeneration is deterministic — same seed, same corpus.
