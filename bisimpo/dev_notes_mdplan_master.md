# BSPO Multi-Domain Implementation — Master Plan

**Purpose:** implement multi-domain BSPO (variants V1–V5 per
`bspo_algorithm.tex` §2) on the 40Bra k8s training stack, with dataset-driven
δ estimation, richer logging, and a clean debug/formal yaml split.

**Execution rule:** strictly sequential — no stage starts until the previous
stage is committed to git. Each stage has its own `dev_notes_mdplan_s{N}_*.md`
file combining plan + execution log. Keep that file updated *as the stage
progresses*, not retroactively.

**Cluster budget (31 nodes total):**
- Formal runs: up to **7 concurrent × 4 nodes = 28 nodes**
- Debug reserve: **2 nodes** (for cbdg-*-smoke combos on the 2-node env)
- Spare: 1 node buffer

## Naming convention (locked as of 2026-04-19 13:00 UTC)

Combo IDs follow `<type>-v<N>[-<suffix>]`:

| type     | meaning                                                                                     |
|----------|---------------------------------------------------------------------------------------------|
| `cbsp`   | single-domain production (e.g., `cbsp-v5-l1e2`)                                             |
| `cbmd`   | multi-domain production (e.g., `cbmd-v1`, `cbmd-v4`)                                        |
| `cbdg-md`| multi-domain debug smoke (e.g., `cbdg-md-v1-smoke`) — auto-runs on 2 nodes with BSPO_DEBUG=1 |
| `cbdg-sd`| single-domain debug smoke (reserved; not used in this master plan)                          |

`<N>` is the objective variant index (1..5) matching `bspo_algorithm.tex` §2:
V1 simplest, V2 hierarchy, V3 strict_wasserstein, V4 w_penalty, V5 w_penalty_only.

### Legacy combos (cannot rename — rayjobs already running or finishing)

These keep their old names in k8s but are cross-referenced to the new
scheme in this table:

| legacy id | new-scheme equivalent | variant  | domain | status |
|-----------|------------------------|----------|--------|--------|
| cbsp101   | cbsp-v1-base          | V1       | sd     | RUNNING |
| cbsp103   | cbsp-v4-base          | V4       | sd     | RUNNING |
| cbsp104   | cbsp-v5-base          | V5       | sd     | RUNNING |
| cbsp201   | cbsp-v1-d1e3          | V1       | sd     | RUNNING |
| cbsp202   | cbsp-v1-d1e2          | V1       | sd     | RUNNING |
| cbsp301   | cbsp-v4-d1e3          | V4       | sd     | RUNNING |
| cbsp302   | cbsp-v4-l1e2          | V4       | sd     | RUNNING |
| cbsp401   | cbsp-v5-l1e2          | V5       | sd     | **deleted + resubmitted under new name** |
| cdbgmd1   | cbdg-md-v1-smoke      | V1       | md     | **deleted + resubmitted under new name** (2-node) |
| cdbgbspo  | (legacy debug)         | V1       | sd     | SUCCEEDED (earlier smoke) |

## Stage list

| S# | name                                    | summary                                                                                              | md file                                      | git tag      |
|----|-----------------------------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------|--------------|
| S0 | Phase-2 preview fill                    | Fill idle GPUs with single-domain δ/λ variants (cbsp201, 202, 301, 302, 401)                         | in this file, §S0                            | `mdplan-S0`  |
| S1 | Multi-domain infra                      | Switch nemo-gym k8s launch to kuberay + volcano; split debug.yaml and formal.yaml                    | `dev_notes_mdplan_s1_infra.md`               | `mdplan-S1`  |
| S2 | Arg-piercing vs in-place adv decision   | Decide whether to add group/domain index piping through verl or write our own adv estimator         | `dev_notes_mdplan_s2_adv_approach.md`        | `mdplan-S2`  |
| S3 | Multi-domain BSPO loss                  | Add V2 / V3 branches with `scatter_mean`; extend V4 / V5 with hierarchical penalties                 | `dev_notes_mdplan_s3_md_loss.md`             | `mdplan-S3`  |
| S4 | δ estimation from dataset               | Compute `d̂(dom)`, `d̂(group)`, `R_maxres` from `extra_info.pass_rate`, `extra_info.jd_pass_rate`. Rolling on-policy `d̂` tracker per step. | `dev_notes_mdplan_s4_delta_est.md`           | `mdplan-S4`  |
| S5 | Extended logging                        | JSONL per-step / per-minibatch / per-trajectory metrics (`d̂_z`, `s_z`, reward, ratio) to run dir     | `dev_notes_mdplan_s5_logging.md`             | `mdplan-S5`  |
| S6 | Multi-domain debug smoke                | End-to-end on NNODES=2 debug: 1-step, small eval, fast-fail error surface; combo = `cbdg-md-v1-smoke` | `dev_notes_mdplan_s6_smoke.md`               | `mdplan-S6`  |
| S7 | Formal V1–V5 multi-domain sweep         | Launch `cbmd-v1..cbmd-v5` at defaults (7-concurrent × 4-node budget)                                | `dev_notes_mdplan_s7_formal.md`              | `mdplan-S7`  |

## Decisions locked before execution

1. **One-md-per-stage** — plan + dev notes combined in the same file. Single
   rolling log avoids fragmentation.
2. **No parallel stages.** S2 doesn't start until S1 is committed, S3 doesn't
   start until S2 is committed, etc.
3. **Debug/formal yaml split (S1).** Debug yaml: 1 step, no save, 32-sample
   val, tiny batch — used for all smoke iterations in S2–S6. Formal yaml:
   full 120-step config.
4. **δ estimation (S4) feeds back into formal yaml (S7).** Don't launch S7
   until S4's estimator produces sensible `d̂(dom)` ranges logged in S5.
5. **Git commits at each stage boundary.** Commit message format:
   `[mdplan S{N}] {stage title} — {one-line diff summary}`. Tag with
   `mdplan-S{N}` so individual stages are recoverable.

## S0 — Phase-2 preview fill (complete as of 2026-04-19 07:52 UTC)

Submitted 5 preview runs to soak idle GPUs while S1 planning happens:

| combo | variant | δ       | λ     | RayJob                        |
|-------|---------|---------|-------|-------------------------------|
| cbsp201 | V1    | 1.0e-3 | 1e-3 | `bra40-sd-cbsp201-4n-6qf48`  |
| cbsp202 | V1    | 1.0e-2 | 1e-3 | `bra40-sd-cbsp202-4n-x7nb9`  |
| cbsp301 | V4    | 1.0e-3 | 1e-3 | `bra40-sd-cbsp301-4n-r7224`  |
| cbsp302 | V4    | 3.0e-4 | 1e-2 | `bra40-sd-cbsp302-4n-bs4wc`  |
| cbsp401 | V5    | 3.0e-4 | 1e-2 | `bra40-sd-cbsp401-4n-l684q`  |

Rationale: targets the failure modes visible in Phase-1 v2 progress logs
(V1 clip-saturation at δ=3e-4, V4 `s_tau` divergence under light λ).
Will complete roughly 20h after the Phase-1 v2 slot frees (3 run now, 2
queue; queued start around +7h when Phase-1 ends).

Edits for S0:
- `verl/my_scripts/k8s/config/combo_40Bra.yaml` — added 5 combos
- `bisimpo/submit_bspo.sh` — per-combo DELTA/LAMBDA dispatch

Commit after this file is written. Tag `mdplan-S0`.

## Concurrent single-domain sweep (runs alongside multi-domain work)

Phase-1 v2 (cbsp101, 103, 104) + Phase-2 preview (cbsp201, 202, 301, 302, 401)
= 8 runs, all single-domain, all fully handled by the current code path.
These do **not** block any S1–S7 work. Their wandb offlines and run.logs
feed the Phase-1 analysis (`analyze_phase1.py` + `plot_phase1.py`) that
determines the single-domain winner.

## Exit criteria for the master plan

All of:
1. S7 formal runs complete and analysed; winner objective selected.
2. All stages committed with `mdplan-S{N}` tags.
3. `bspo_algorithm.tex` §2 and code are 1:1.
4. Master-plan md updated with "status: complete" and final result summary.
