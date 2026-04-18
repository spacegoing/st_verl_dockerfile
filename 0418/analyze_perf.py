#!/usr/bin/env python3
"""0418/analyze_perf.py — read perf_log.jsonl files and print a per-step summary.

Usage:
  python3 analyze_perf.py <perf_log.jsonl> [<perf_log.jsonl> ...]

For each file, prints a table: step, step_s, gen_s, update_actor_s,
tok/s/GPU, mem_alloc_gb, resp_len_mean, resp_clip_ratio, curriculum_n_domains.
At the end prints means. Designed to make P0/P1/P2 comparisons easy to read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


FMT = "{:>4}  {:>6.1f}  {:>6.1f}  {:>6.1f}  {:>7.0f}  {:>6.1f}  {:>6.1f}  {:>5.2f}  {:>2}"
HEADER_FMT = "{:>4}  {:>6}  {:>6}  {:>6}  {:>7}  {:>6}  {:>6}  {:>5}  {:>2}"


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"\n[{label}] no rows")
        return
    print(f"\n[{label}]  n_steps={len(rows)}  n_gpus={rows[-1].get('n_gpus')}")
    print(HEADER_FMT.format("step", "step_s", "gen_s", "upd_s", "tok/s/g", "mem_gb", "resp_m", "clip", "nd"))
    sums = {k: 0.0 for k in ("step", "gen", "upd", "tokg", "mem", "resp", "clip")}
    cnt = 0
    for r in rows:
        step = r.get("step", 0)
        timing = r.get("timing_s", {})
        perf = r.get("perf", {})
        resp = r.get("response_length", {})
        curriculum = r.get("curriculum", {})

        step_s = timing.get("step", 0.0) or 0.0
        gen_s = timing.get("gen", 0.0) or 0.0
        upd_s = timing.get("update_actor", 0.0) or 0.0
        tokg = r.get("tokens_per_gpu_s_gen") or 0.0
        mem = perf.get("max_memory_allocated_gb", 0.0) or 0.0
        resp_m = resp.get("mean", 0.0) or 0.0
        clip = resp.get("clip_ratio", 0.0) or 0.0
        nd = curriculum.get("n_domains", 0) or 0

        print(FMT.format(step, step_s, gen_s, upd_s, tokg, mem, resp_m, clip, nd))
        sums["step"] += step_s
        sums["gen"] += gen_s
        sums["upd"] += upd_s
        sums["tokg"] += tokg
        sums["mem"] += mem
        sums["resp"] += resp_m
        sums["clip"] += clip
        cnt += 1

    if cnt:
        print("-" * 72)
        print(FMT.format(
            0,
            sums["step"] / cnt,
            sums["gen"] / cnt,
            sums["upd"] / cnt,
            sums["tokg"] / cnt,
            sums["mem"] / cnt,
            sums["resp"] / cnt,
            sums["clip"] / cnt,
            0,
        ).replace("   0  ", " AVG  ", 1).replace("   0", "   -"))

        gen_frac = sums["gen"] / sums["step"] if sums["step"] else 0.0
        upd_frac = sums["upd"] / sums["step"] if sums["step"] else 0.0
        floor_ratio = (sums["tokg"] / cnt) / 1500.0 if cnt else 0.0
        print(f"\n  gen is {gen_frac:.1%} of step time; update_actor is {upd_frac:.1%}")
        print(f"  gen throughput = {sums['tokg']/cnt:.0f} tok/s/GPU  → {floor_ratio:.2f}× of 1500 tok/s/GPU healthy floor")
        if floor_ratio >= 1.5:
            print("  → HEALTHY: rollout is well-saturated; node count may be over-provisioned")
        elif floor_ratio >= 1.0:
            print("  → OK: rollout is at saturation; any reduction in nodes should be safe")
        else:
            print("  → UNDERUTILIZED: rollout is below floor; check config (TP, batch, memory)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"warning: {p} does not exist", file=sys.stderr)
            continue
        rows = load(path)
        summarize(rows, str(path))


if __name__ == "__main__":
    main()
