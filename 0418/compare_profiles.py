#!/usr/bin/env python3
"""0418/compare_profiles.py — side-by-side comparison of P0/P1/P2 runs.

Reads perf_log.jsonl files for each profile under 0418/results/ and
prints a decision table showing: mean step time, mean gen time, mean
per-GPU tokens/s during gen, peak GPU memory, response-length mean,
so we can pick the minimum-node profile that stays within acceptable
step-time envelope.

Usage:
  python3 compare_profiles.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PROFILES = [
    ("P0-16node", 128),
    ("P-8node", 64),
    ("P-4node", 32),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(v, w=8, p=1):
    if v is None:
        return " " * (w - 1) + "-"
    return f"{v:{w}.{p}f}"


def main():
    print(f"{'profile':<12} {'gpus':>5} {'n_steps':>8} {'step_s':>8} {'gen_s':>8} "
          f"{'upd_s':>8} {'tok/s/g':>9} {'mem_gb':>8} {'resp_m':>8} {'clip':>6}")
    print("-" * 92)

    summaries = []
    for label, n_gpus in PROFILES:
        rows = load(RESULTS_DIR / f"{label}_perf_log.jsonl")
        if not rows:
            print(f"{label:<12} {n_gpus:>5} {'(no data)':>8}")
            continue
        step_s = mean([r.get("timing_s", {}).get("step") for r in rows])
        gen_s = mean([r.get("timing_s", {}).get("gen") for r in rows])
        upd_s = mean([r.get("timing_s", {}).get("update_actor") for r in rows])
        tokg = mean([r.get("tokens_per_gpu_s_gen") for r in rows])
        mem = mean([r.get("perf", {}).get("max_memory_allocated_gb") for r in rows])
        resp_m = mean([r.get("response_length", {}).get("mean") for r in rows])
        clip = mean([r.get("response_length", {}).get("clip_ratio") for r in rows])
        summaries.append({
            "label": label, "n_gpus": n_gpus, "step_s": step_s, "gen_s": gen_s,
            "upd_s": upd_s, "tokg": tokg, "mem": mem, "resp_m": resp_m, "clip": clip,
        })
        print(f"{label:<12} {n_gpus:>5} {len(rows):>8} "
              f"{fmt(step_s)} {fmt(gen_s)} {fmt(upd_s)} {fmt(tokg, 9, 0)} "
              f"{fmt(mem)} {fmt(resp_m)} {fmt(clip, 6, 2)}")

    # Decision guidance
    print()
    print("Interpretation (per plan_b_node_opt.md):")
    print("  tok/s/g < 1000  → rollout underutilized, reduce nodes is safe")
    print("  1000-1500       → borderline saturation")
    print("  >= 1500         → healthy saturation, rollout is well-fed")
    print("  >= 2000         → approaching lvbc peak (2,436 uniform / ~950 AIME)")
    print()

    if len(summaries) >= 2:
        base = summaries[0]
        for s in summaries[1:]:
            if base["step_s"] and s["step_s"]:
                ratio = s["step_s"] / base["step_s"]
                print(f"  {s['label']} step time = {ratio:.2f}× {base['label']} "
                      f"(target: ≤ 1.3×; acceptable to produce usable training)")


if __name__ == "__main__":
    main()
