"""
Plot BSPO Phase-1 training + validation curves.

Reads CSVs produced by analyze_phase1.py (one per combo) and produces:
  - val_acc.png              3-variant val-core/nemogym_math/acc/mean@1 vs step
  - train_signals.png        3×2 grid: pg_loss, grad_norm, entropy, ppo_kl, s_tau_abs, per_traj_abs
  - bspo_diagnostics.png     4-panel: delta_reg/delta_w, clip_pos+neg, penalty_active, s_pos vs s_neg

Usage:  python3 plot_phase1.py
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write("ERROR: install matplotlib in current venv\n")
    sys.exit(2)

BASE = Path("/mnt/public/lichang93/st_verl_dockerfile/bisimpo/analysis/phase1")
COMBOS = {
    "cbsp101": ("simplest (Obj 1)", "tab:blue"),
    "cbsp103": ("w_penalty (Obj 3)", "tab:orange"),
    "cbsp104": ("w_penalty_only (Obj 4)", "tab:green"),
}


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def get_series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r.get(key, "")
        if v in ("", None):
            continue
        try:
            ys.append(float(v))
            xs.append(int(r["_step"]))
        except ValueError:
            continue
    return xs, ys


def plot_val():
    """3 per-source validation panels (beyondaime, aime2025, olympiadbench)."""
    sources = ("beyondaime", "aime2025", "olympiadbench")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, src in zip(axes, sources):
        for combo, (label, color) in COMBOS.items():
            path = BASE / f"{combo}.csv"
            if not path.exists():
                continue
            rows = load_csv(path)
            xs, ys = get_series(rows, f"val-core/{src}/acc/mean@1")
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", color=color, label=f"{combo} — {label}")
        ax.set_xlabel("training step")
        ax.set_title(f"val-core/{src}/acc/mean@1")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("accuracy (mean@1)")
    fig.suptitle("BSPO Phase-1: per-source validation accuracy (eval every 10 steps)", y=1.02)
    fig.tight_layout()
    out = BASE / "val_acc.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)


def plot_train_signals():
    keys = [
        ("actor/pg_loss", "pg_loss"),
        ("actor/grad_norm", "grad_norm"),
        ("actor/entropy", "entropy"),
        ("actor/ppo_kl", "ppo_kl"),
        ("actor/bspo_s_tau_abs_mean", "|s_tau| mean"),
        ("actor/bspo_per_traj_abs_mean", "|per_traj| mean"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, keys):
        for combo, (legend, color) in COMBOS.items():
            path = BASE / f"{combo}.csv"
            if not path.exists():
                continue
            rows = load_csv(path)
            xs, ys = get_series(rows, key)
            if not xs:
                continue
            ax.plot(xs, ys, color=color, alpha=0.8, linewidth=1.2, label=combo)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("BSPO Phase-1: training signals", y=1.01)
    fig.tight_layout()
    out = BASE / "train_signals.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)


def plot_bspo_diagnostics():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 1. Δ_reg (simplest) + Δ_w (w_penalty)
    ax = axes[0, 0]
    for combo, (legend, color) in COMBOS.items():
        path = BASE / f"{combo}.csv"
        if not path.exists():
            continue
        rows = load_csv(path)
        if combo == "cbsp101":
            xs, ys = get_series(rows, "actor/bspo_delta_reg_mean")
            ax.plot(xs, ys, color=color, label=f"{combo} Δ_reg")
        else:
            xs, ys = get_series(rows, "actor/bspo_delta_w_mean")
            ax.plot(xs, ys, color=color, label=f"{combo} Δ_w")
    ax.set_title("Δ_reg / Δ_w mean per batch")
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 2. clip fractions
    ax = axes[0, 1]
    for combo, (legend, color) in COMBOS.items():
        path = BASE / f"{combo}.csv"
        if not path.exists():
            continue
        rows = load_csv(path)
        xs_p, ys_p = get_series(rows, "actor/bspo_clip_pos_frac")
        xs_n, ys_n = get_series(rows, "actor/bspo_clip_neg_frac")
        if xs_p:
            ax.plot(xs_p, ys_p, color=color, linestyle="-", label=f"{combo} pos")
        if xs_n:
            ax.plot(xs_n, ys_n, color=color, linestyle="--", label=f"{combo} neg")
    ax.set_title("δ-clip frequency  (solid=pos, dashed=neg)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

    # 3. penalty_active (Obj 3/4 only)
    ax = axes[1, 0]
    for combo in ("cbsp103", "cbsp104"):
        path = BASE / f"{combo}.csv"
        if not path.exists():
            continue
        rows = load_csv(path)
        xs, ys = get_series(rows, "actor/bspo_penalty_active_frac")
        color = COMBOS[combo][1]
        ax.plot(xs, ys, color=color, label=combo)
    ax.set_title("penalty_active_frac (Obj 3/4 only)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend()

    # 4. s_pos vs s_neg (cbsp101)
    ax = axes[1, 1]
    for combo, (legend, color) in COMBOS.items():
        path = BASE / f"{combo}.csv"
        if not path.exists():
            continue
        rows = load_csv(path)
        xs_p, ys_p = get_series(rows, "actor/bspo_s_pos_mean")
        xs_n, ys_n = get_series(rows, "actor/bspo_s_neg_mean")
        if xs_p:
            ax.plot(xs_p, ys_p, color=color, linestyle="-", label=f"{combo} s+")
        if xs_n:
            ax.plot(xs_n, ys_n, color=color, linestyle="--", label=f"{combo} s-")
    ax.set_title("s⁺ / s⁻ mean per batch  (solid=s+, dashed=s-)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)

    fig.suptitle("BSPO Phase-1: loss-specific diagnostics", y=1.01)
    fig.tight_layout()
    out = BASE / "bspo_diagnostics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"-> {out}")
    plt.close(fig)


def main():
    if not BASE.exists():
        print(f"missing {BASE} — run analyze_phase1.py first")
        sys.exit(1)
    plot_val()
    plot_train_signals()
    plot_bspo_diagnostics()


if __name__ == "__main__":
    main()
