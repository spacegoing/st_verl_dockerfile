"""
BSPO Phase-1 analysis — compare cbsp101/103/104 on val-core and BSPO diagnostics.

For each wandb offline run in WANDB_DIR that was started AFTER CUTOFF_UTC and whose
run_name contains one of the 3 Phase-1 combos, extract:
  - val-core/nemogym_math/acc/mean@1 vs step
  - actor/bspo_* diagnostics vs step
  - actor/pg_loss, actor/grad_norm, actor/entropy vs step

Writes:
  - bisimpo/analysis/phase1/<combo>.csv         (per-run timeseries)
  - bisimpo/analysis/phase1/summary.csv         (per-run summary stats)
  - bisimpo/analysis/phase1/summary.md          (human-readable comparison)

The parsing logic mirrors bisimpo/wandb_parse_offline.py but is filtered to only
keep BSPO-relevant keys, and cross-tabulates the 3 variants.

Usage:  python3 analyze_phase1.py
"""

from __future__ import annotations
import csv
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb
except ImportError:
    sys.stderr.write("ERROR: install wandb in current venv: pip install wandb\n")
    sys.exit(2)

# ── Config ───────────────────────────────────────────────────────────────────
WANDB_DIR = Path("/mnt/public/lichang93/st_verl_dockerfile/verl/wandb_my_dirs/wandb")
OUT_DIR = Path("/mnt/public/lichang93/st_verl_dockerfile/bisimpo/analysis/phase1")
# Keep only runs started after Phase-1 submission (2026-04-18 17:30 UTC)
CUTOFF_UTC = datetime(2026, 4, 18, 17, 30, 0)
PHASE1_COMBOS = ("cbsp101", "cbsp103", "cbsp104")

# BSPO-specific metric keys that the loss emits (core_algos.py)
BSPO_KEYS = [
    "actor/bspo_s_pos_mean",
    "actor/bspo_s_neg_mean",
    "actor/bspo_s_tau_mean",
    "actor/bspo_s_tau_abs_mean",
    "actor/bspo_delta_reg_mean",
    "actor/bspo_delta_w_mean",
    "actor/bspo_clip_pos_frac",
    "actor/bspo_clip_neg_frac",
    "actor/bspo_penalty_active_frac",
    "actor/bspo_penalty_mean",
    "actor/bspo_per_traj_mean",
    "actor/bspo_per_traj_abs_mean",
]
TRAIN_KEYS = [
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/entropy",
    "actor/lr",
    "actor/ppo_kl",
]
VAL_PATTERN = re.compile(r"^val-core/")


# ── Parse one .wandb file ───────────────────────────────────────────────────
def parse_wandb_file(wandb_file: Path):
    ds = DataStore()
    ds.open_for_scan(str(wandb_file))
    run_name = None
    history_rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.HasField("run") and rec.run.display_name:
            run_name = rec.run.display_name
        if rec.HasField("history"):
            row = {}
            for item in rec.history.item:
                k = item.nested_key[0] if len(item.nested_key) else item.key
                if not k:
                    continue
                try:
                    v = json.loads(item.value_json)
                    if isinstance(v, (int, float)):
                        row[k] = v
                except Exception:
                    pass
            if row:
                history_rows.append(row)
    return run_name, history_rows


# ── Discover Phase-1 runs ───────────────────────────────────────────────────
def find_phase1_runs():
    out = {}  # combo -> (run_dir, wandb_file)
    for run_dir in sorted(WANDB_DIR.glob("offline-run-*")):
        m = re.match(r"offline-run-(\d{8})_(\d{6})-", run_dir.name)
        if not m:
            continue
        ts = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        if ts < CUTOFF_UTC:
            continue
        wandb_files = list(run_dir.glob("*.wandb"))
        if not wandb_files:
            continue
        # Peek at display_name to find combo
        run_name, _ = parse_wandb_file(wandb_files[0])
        combo = next((c for c in PHASE1_COMBOS if run_name and c in run_name), None)
        if combo is None:
            continue
        # Keep latest per combo
        if combo not in out or ts > out[combo][2]:
            out[combo] = (run_dir, wandb_files[0], ts)
    return out


# ── Per-run CSV ─────────────────────────────────────────────────────────────
def write_run_csv(history_rows, csv_path: Path):
    by_step = {}
    for row in history_rows:
        step = row.get("_step")
        if step is None:
            continue
        by_step.setdefault(int(step), {}).update(row)
    all_keys = sorted(set(k for r in by_step.values() for k in r.keys()))
    val_keys = sorted(k for k in all_keys if VAL_PATTERN.match(k))
    cols = ["_step"] + TRAIN_KEYS + BSPO_KEYS + val_keys
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for step in sorted(by_step):
            out = {"_step": step}
            for k in cols[1:]:
                v = by_step[step].get(k)
                out[k] = round(v, 8) if isinstance(v, float) else v
            w.writerow(out)
    return cols, by_step


# ── Summary ─────────────────────────────────────────────────────────────────
def summarize(combo, by_step):
    steps = sorted(by_step)
    if not steps:
        return None
    final_step = steps[-1]
    # Validation metric for math
    val_key = "val-core/nemogym_math/acc/mean@1"
    val_points = [(s, by_step[s][val_key]) for s in steps if val_key in by_step[s]]
    last_val = val_points[-1] if val_points else (None, None)
    best_val = max(val_points, key=lambda x: x[1]) if val_points else (None, None)
    # Training signal (mean over last 20% steps)
    cutoff = steps[int(len(steps) * 0.8)] if len(steps) > 5 else steps[0]
    last_quintile = [s for s in steps if s >= cutoff]
    def _mean(key):
        vals = [by_step[s][key] for s in last_quintile if key in by_step[s]]
        return sum(vals) / len(vals) if vals else None
    return {
        "combo": combo,
        "final_step": final_step,
        "n_val_points": len(val_points),
        "last_val_step": last_val[0],
        "last_val_acc": last_val[1],
        "best_val_step": best_val[0],
        "best_val_acc": best_val[1],
        "mean_pg_loss_Q5": _mean("actor/pg_loss"),
        "mean_grad_norm_Q5": _mean("actor/grad_norm"),
        "mean_entropy_Q5": _mean("actor/entropy"),
        "mean_s_tau_abs_Q5": _mean("actor/bspo_s_tau_abs_mean"),
        "mean_delta_reg_Q5": _mean("actor/bspo_delta_reg_mean"),
        "mean_delta_w_Q5": _mean("actor/bspo_delta_w_mean"),
        "mean_clip_pos_frac_Q5": _mean("actor/bspo_clip_pos_frac"),
        "mean_clip_neg_frac_Q5": _mean("actor/bspo_clip_neg_frac"),
        "mean_penalty_active_Q5": _mean("actor/bspo_penalty_active_frac"),
        "mean_per_traj_abs_Q5": _mean("actor/bspo_per_traj_abs_mean"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = find_phase1_runs()
    if not runs:
        print("No Phase-1 runs found (cutoff:", CUTOFF_UTC, ")")
        return
    summaries = []
    for combo, (run_dir, wandb_file, ts) in runs.items():
        print(f"\n=== {combo}  ({run_dir.name})  submitted UTC {ts} ===")
        _, rows = parse_wandb_file(wandb_file)
        csv_path = OUT_DIR / f"{combo}.csv"
        cols, by_step = write_run_csv(rows, csv_path)
        print(f"  -> {csv_path}  ({len(by_step)} unique steps, {len(cols)} cols)")
        s = summarize(combo, by_step)
        if s:
            summaries.append(s)
    if not summaries:
        return

    # summary.csv
    summary_csv = OUT_DIR / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"\n-> {summary_csv}")

    # summary.md
    md_lines = ["# BSPO Phase-1 Summary\n", "\n## Validation accuracy (math)\n\n"]
    md_lines.append("| combo | variant | final_step | last_val_step | last_val_acc | best_val_step | best_val_acc |\n")
    md_lines.append("|---|---|---|---|---|---|---|\n")
    var_map = {"cbsp101": "simplest (Obj 1)", "cbsp103": "w_penalty (Obj 3)", "cbsp104": "w_penalty_only (Obj 4)"}
    for s in sorted(summaries, key=lambda x: x["combo"]):
        md_lines.append(
            f"| {s['combo']} | {var_map.get(s['combo'], '?')} | {s['final_step']} | "
            f"{s['last_val_step']} | {s['last_val_acc']} | {s['best_val_step']} | {s['best_val_acc']} |\n"
        )
    md_lines.append("\n## Training signal (mean over last 20% of steps)\n\n")
    md_lines.append("| combo | pg_loss | grad_norm | entropy | s_tau_abs | Δ_reg | Δ_w | clip_pos | clip_neg | penalty_active | per_traj_abs |\n")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for s in sorted(summaries, key=lambda x: x["combo"]):
        def _f(v):
            return "—" if v is None else f"{v:.4g}"
        md_lines.append(
            f"| {s['combo']} | {_f(s['mean_pg_loss_Q5'])} | {_f(s['mean_grad_norm_Q5'])} | "
            f"{_f(s['mean_entropy_Q5'])} | {_f(s['mean_s_tau_abs_Q5'])} | {_f(s['mean_delta_reg_Q5'])} | "
            f"{_f(s['mean_delta_w_Q5'])} | {_f(s['mean_clip_pos_frac_Q5'])} | "
            f"{_f(s['mean_clip_neg_frac_Q5'])} | {_f(s['mean_penalty_active_Q5'])} | "
            f"{_f(s['mean_per_traj_abs_Q5'])} |\n"
        )
    md_path = OUT_DIR / "summary.md"
    md_path.write_text("".join(md_lines))
    print(f"-> {md_path}")


if __name__ == "__main__":
    main()
