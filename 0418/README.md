# 0418/ — Node-count + compute-efficiency work (start here)

> Read this first when you arrive at `0418/` for the first time.
> It tells you what every file is and the order to read them.

This directory holds the plans, scripts, results, and operator docs for
one focused effort: make 40Bra single-domain GRPO training cheaper to run
without affecting training numerics.

---

## 1. File map

### Plans (what we set out to do, with rationale and exit criteria)

| File | Purpose |
|---|---|
| `plan_a_kuberay_volcano.md` | Verify the cluster's KubeRay + Volcano gang-scheduling stack is correctly wired so RayJobs never get stuck in partial-placement deadlocks. |
| `plan_b_node_opt.md` | Node-count ladder (16 → 8 → 4) experiment: does the current 16-node production config over-provision rollout? If so, what is the smallest node count with step time within 1.3× of P0? |
| `plan_c_compute_opt.md` | Compute-efficiency follow-up. Holds the "Category A / B / C" hyperparameter split (training-irrelevant vs training-relevant), the debug-only default overrides, and the 9-experiment matrix (C1–C9). |

### Summaries (what actually happened)

| File | Purpose |
|---|---|
| `summary.md` | Combined overview, production recommendation, open questions. |
| `summary_plan_a.md` | Step-by-step record of Plan A execution, bugs fixed, exit criteria checked. |
| `summary_plan_b.md` | Same for Plan B. Full evidence (P0/P1/P2 numbers) and the 8-node recommendation. |
| `dev_notes.md` | Live log, chronologically ordered. If you need to know "what did we do on 2026-04-17 at 17:46 UTC", read this. |

### Operator tooling (scripts you will actually invoke)

| File | Purpose |
|---|---|
| `submit_debug.sh` | Submit ONE debug RayJob: `./submit_debug.sh <combo> <nnodes> [hydra overrides]` |
| `queue_all.sh` | Submit all 9 Plan C experiments at once. Relies on Volcano gang queueing. |
| `pipeline.sh` | Background monitor that waits for a RayJob to terminate, copies its perf_log + run.log, runs `analyze_perf.py`, and deletes the RayJob to free nodes. Invoked automatically by `submit_debug.sh` / `queue_all.sh`. You should not run it by hand. |
| `verify_gang.sh` | One-shot validator that checks the 6 gang-scheduling invariants for a given RayJob. |
| `analyze_perf.py` | Per-run analyzer. Reads `perf_log.jsonl`, prints a step-by-step table + one-line verdict. |
| `compare_profiles.py` | 3-way comparison (P0/P1/P2) as a decision table. |

### Templates

| File | Purpose |
|---|---|
| `rayjob_debug.yaml` | Parameterized RayJob template for debug runs. envsubst placeholders: `COMBO_ID`, `NNODES`, `WORKER_REPLICAS`, `EXTRA_OVERRIDES`, `DEBUG_LABEL`. |

### Operator reference

| File | Purpose |
|---|---|
| `k8s_cheatsheet.md` | 12-section daily-ops cheatsheet. Read when you need to inspect / cancel / clean up RayJobs and pods. |
| `concepts.md` | Cross-cutting background: apple-to-apple rule, Category A/B/C, hybrid engine, gang scheduling, MoE sharding axes. Read when you want the "why" behind the design choices. |
| `debug_guide.md` | Quick start for running a debug experiment. |

### Data (what the experiments produced)

| Path | Purpose |
|---|---|
| `results/<label>_perf_log.jsonl` | Per-step metrics copy (canonical source is the run's own ckpt dir). |
| `results/<label>_analysis.txt` | Human-readable summary from `analyze_perf.py`. |
| `results/<label>_training.log` | Entrypoint/driver output copy. |
| `results/<label>_head_logs.txt` | Head-pod logs (only when perf_log missing — used for failure diagnosis). |

### Deprecated / archived

| File | Why |
|---|---|
| `queue_tier123.sh` | Superseded by `queue_all.sh` (has a SIGPIPE bug; header note added). |

---

## 2. Reading order for a new reader

Depending on what you need:

### "I just arrived, what is this project doing?"
1. `summary.md`
2. `plan_a_kuberay_volcano.md`, `plan_b_node_opt.md`, `plan_c_compute_opt.md` (skim headers, read Section 1 of each)
3. `concepts.md` for the A/B/C split

### "I want to run a debug experiment"
1. `debug_guide.md`
2. `concepts.md` Section on "Category A vs B" so you know what you are allowed to change
3. `k8s_cheatsheet.md` for monitoring

### "I want to run a production training job"
See `../iter_kuberay_32nodes_verl_training/production_guide.md`. That tree, not this one, is where production submits live.

### "Something failed, I need to debug"
1. `k8s_cheatsheet.md` Section 4 (logs) and Section 10 (troubleshooting patterns)
2. `dev_notes.md` for precedent: has someone hit this failure before?
3. `results/<label>_training.log` for the specific driver output

### "I want to add a new compute-opt experiment"
1. `concepts.md` — confirm your change is Category A
2. `plan_c_compute_opt.md` — extend Section 3 with your new experiment
3. `queue_all.sh` — add a `submit_one "CN" "<overrides>"` line
4. Submit and wait

---

## 3. Relationship to other trees

- **`../iter_kuberay_32nodes_verl_training/`** — the production submit tree. `submit/rayjob.yaml` + `submit/submit.sh` are the production equivalents of `rayjob_debug.yaml` + `submit_debug.sh`. Do not cross-pollinate.
- **`../voltest/`** — the 10×8-node gang-scheduling stress test that verified the Volcano stack. Referenced by `plan_a_kuberay_volcano.md`.
- **`../verl/verl/utils/perf_log.py`** — the per-step telemetry module, active only when `VERL_PERF_LOG=1` is in the pod env. Wired into `verl/verl/trainer/ppo/ray_trainer.py`.
- **`../verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh`** — the shared entrypoint script. One file for both production and debug runs.
- **`../verl/my_scripts/k8s/config/`** — Hydra configs:
  - `40bra_16node_sd.yaml` — base training config
  - `env/` — config group of node-count / offload profiles (16-node optimal default, 16-node legacy, 8-node C10, 4-node C11). Select with `NNODES=N ./submit.sh ...` or `env=<name>` Hydra override.
  - `combo_40Bra.yaml` — combo definitions (FACPO, ppo_epochs, curriculum; includes `cdbg5`).
