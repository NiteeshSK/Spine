"""Figures for the README.

    python -m spine.viz.plots --sft-log runs/sft/train_log.csv \\
        --dpo-log runs/dpo/train_log.csv --reports reports --out assets

Everything is read from artifacts on disk (CSV logs, JSON reports), so the
figures regenerate from a finished run without re-training and without any
hand-entered numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "base": "#9aa5b1",
    "sft": "#2f6fed",
    "dpo": "#12a594",
    "accent": "#e0603a",
    "grid": "#e6e8eb",
}
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)


WATERMARK: str | None = None


def set_watermark(text: str | None) -> None:
    """Stamp every saved figure. Used by the pipeline demo so its plots can
    never be mistaken for measured model results."""
    global WATERMARK
    WATERMARK = text


def _save(fig, out: Path) -> Path:
    if WATERMARK:
        fig.text(0.5, 0.5, WATERMARK, fontsize=26, color="#d0021b", alpha=0.16,
                 ha="center", va="center", rotation=18, fontweight="bold", zorder=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key, "")
        if v not in ("", None):
            try:
                ys.append(float(v))
                xs.append(float(r["step"]))
            except ValueError:
                continue
    return xs, ys


def plot_training_curves(log: Path, out: Path, title: str = "Phase 1 - QLoRA SFT") -> Path:
    rows = read_csv(log)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    ax = axes[0]
    for key, label, color, style in [
        ("loss", "train loss", PALETTE["sft"], "-"),
        ("eval_loss", "val loss", PALETTE["accent"], "--"),
    ]:
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, style, color=color, label=label, linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Loss")
    ax.legend(frameon=False)

    ax = axes[1]
    plotted = False
    for key, label, color in [
        ("probe_json_valid_strict", "JSON validity (strict)", PALETTE["base"]),
        ("probe_schema_valid", "schema validity", PALETTE["dpo"]),
        ("probe_exact_match", "exact match", PALETTE["sft"]),
        ("probe_refusal_f1", "refusal F1", PALETTE["accent"]),
    ]:
        xs, ys = _series(rows, key)
        if xs:
            ax.plot(xs, ys, "-o", color=color, label=label, linewidth=1.6, markersize=3)
            plotted = True
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("step")
    ax.set_ylabel("rate")
    ax.set_title("Held-out task metrics (generation probe)")
    if plotted:
        ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle(title, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_dpo_curves(log: Path, out: Path) -> Path:
    rows = read_csv(log)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))

    xs, ys = _series(rows, "loss")
    axes[0].plot(xs, ys, color=PALETTE["dpo"], linewidth=1.6)
    axes[0].set_title("DPO loss")
    axes[0].set_xlabel("step")

    for key, label, color in [
        ("rewards/chosen", "chosen", PALETTE["sft"]),
        ("rewards/rejected", "rejected", PALETTE["base"]),
    ]:
        xs, ys = _series(rows, key)
        if xs:
            axes[1].plot(xs, ys, color=color, label=label, linewidth=1.6)
    axes[1].axhline(0, color=PALETTE["grid"], linewidth=1)
    axes[1].set_title("Implicit rewards\n(both collapsing = KL leash too loose)")
    axes[1].set_xlabel("step")
    axes[1].legend(frameon=False)

    for key, label, color in [
        ("rewards/margins", "margin", PALETTE["accent"]),
        ("rewards/accuracies", "pref. accuracy", PALETTE["dpo"]),
    ]:
        xs, ys = _series(rows, key)
        if xs:
            axes[2].plot(xs, ys, color=color, label=label, linewidth=1.6)
    axes[2].set_title("Margin & preference accuracy")
    axes[2].set_xlabel("step")
    axes[2].legend(frameon=False)

    fig.suptitle("Phase 2 - DPO", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, out)


HEADLINE = [
    ("json_valid_strict", "JSON validity"),
    ("schema_valid", "Schema valid"),
    ("exact_match", "Exact match"),
    ("refusal_f1", "Refusal F1"),
    ("grounding_rate", "Grounded span"),
    ("interface_f1", "Interface F1"),
]


def plot_comparison(reports: list[tuple[str, dict]], out: Path, title: str = "Test set") -> Path:
    labels = [name for name, _ in reports]
    colors = [PALETTE.get(name.split("_")[0], PALETTE["sft"]) for name in labels]
    n = len(reports)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    xs = range(len(HEADLINE))
    for i, (name, rep) in enumerate(reports):
        vals = [rep.get(k, 0.0) for k, _ in HEADLINE]
        pos = [x + i * width - 0.4 + width / 2 for x in xs]
        bars = ax.bar(pos, vals, width * 0.92, label=name, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl in HEADLINE])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("rate")
    ax.set_title(f"Prompted baseline vs fine-tuned - {title}", fontweight="bold")
    ax.legend(frameon=False, ncol=n)
    fig.tight_layout()
    return _save(fig, out)


def plot_field_accuracy(reports: list[tuple[str, dict]], out: Path) -> Path:
    fields = sorted({f for _, r in reports for f in r.get("field_accuracy", {})})
    fig, ax = plt.subplots(figsize=(9.6, 0.32 * len(fields) + 1.8))
    n = len(reports)
    height = 0.8 / max(n, 1)
    ys = range(len(fields))
    for i, (name, rep) in enumerate(reports):
        vals = [rep.get("field_accuracy", {}).get(f, 0.0) for f in fields]
        pos = [y + i * height - 0.4 + height / 2 for y in ys]
        ax.barh(pos, vals, height * 0.9, label=name, color=PALETTE.get(name.split("_")[0], PALETTE["sft"]))
    ax.set_yticks(list(ys))
    ax.set_yticklabels(fields)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("per-field accuracy")
    ax.set_title("Where the errors actually live", fontweight="bold")
    ax.legend(frameon=False, ncol=n, loc="lower right")
    fig.tight_layout()
    return _save(fig, out)


def load_reports(report_dir: Path, split: str, tags: list[str]) -> list[tuple[str, dict]]:
    found = []
    for tag in tags:
        p = report_dir / f"{tag}_{split}.json"
        if p.exists():
            found.append((tag, json.loads(p.read_text())))
        else:
            print(f"[plots] missing {p}, skipping")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate all README figures from run artifacts.")
    ap.add_argument("--sft-log", type=Path, default=Path("runs/sft/train_log.csv"))
    ap.add_argument("--dpo-log", type=Path, default=Path("runs/dpo/train_log.csv"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--tags", nargs="*", default=["base_prompted", "sft", "dpo"])
    ap.add_argument("--splits", nargs="*", default=["test", "ood"])
    ap.add_argument("--out", type=Path, default=Path("assets"))
    args = ap.parse_args()

    made = []
    if args.sft_log.exists():
        made.append(plot_training_curves(args.sft_log, args.out / "sft_curves.png"))
    if args.dpo_log.exists():
        made.append(plot_dpo_curves(args.dpo_log, args.out / "dpo_curves.png"))
    for split in args.splits:
        reps = load_reports(args.reports, split, args.tags)
        if reps:
            made.append(plot_comparison(reps, args.out / f"comparison_{split}.png", title=f"{split} split"))
            made.append(plot_field_accuracy(reps, args.out / f"field_accuracy_{split}.png"))
    for p in made:
        print(f"[plots] {p}")


if __name__ == "__main__":
    main()
