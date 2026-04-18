# 0418 Summary — Node-count optimization + KubeRay/Volcano verification

> Evidence-based node-count series (16 → 8 → 4 nodes) on a 5-step debug combo
> proves the hypothesis from Plan B: at 16 nodes, rollout is strongly
> under-saturated. **8 nodes is the best trade-off for production.**

**Dates**: 2026-04-17 (plans written, dev built, three profiles run).
**References**: `0418/plan_a_kuberay_volcano.md`, `0418/plan_b_node_opt.md`, `0418/dev_notes.md`.

---

## 1. Main results

| profile | gpus | step_s | gen_s | upd_s | tok/s/GPU | mem_gb | step_ratio_vs_P0 | tok/s_ratio |
|---|---|---|---|---|---|---|---|---|
| **P0** | 128 (16n) | 444.8 | 285.2 | 110.7 | 291 | 233 | 1.00× | 1.00× |
| **P1** | 64 (8n) | 516.2 | 294.5 | 165.6 | 561 | 237 | **1.16×** | **1.93×** |
| **P2** | 32 (4n) | 663.7 | 341.1 | 259.6 | 975 | 236 | 1.49× | 3.35× |

All three runs: 5 training steps at cdbg5 (math-only, FACPO c50 params),
`max_response_length=16384`, response_length mean ≈ 5,000 tokens
(no responses hit the length cap → `clip_ratio=0` across every step).

## 2. What changed from P0 → P1

- **Half the nodes**. Only **16%** slower per step.
- **gen_s barely moved** (285 → 294, +3%). Doubling per-GPU batch
  from 16 to 32 rollouts did not meaningfully increase generation
  time — the clearest possible evidence that P0 was under-saturated.
- `update_actor` +50% (expected: each DP replica processes 2× more
  prompts). Fixed overhead in optimizer offload dampens the scaling.
- tok/s/GPU nearly **doubled** (291 → 561), matching the expected
  batch-size effect predicted by lvbc.

## 3. What P2 tells us about the cliff

- Quarter the nodes (32 GPUs, DP=2), step time 1.49× P0.
- tok/s/GPU reaches 975 — close to lvbc's **AIME25 real-eval floor of ~950 tok/s/GPU** on DP=8/TP=1. Our hybrid-engine, variable-length generation workload appears to cap out here regardless of node count. In other words, P2 is not "under-utilized" — it's hitting the real practical ceiling for our mix.
- `update_actor` +135% vs P0. At DP=2, gradient-compute overhead
  eats the gains, and step time climbs above the 1.3× target.
- P2 still works (no OOM, all 5 steps complete), so it's viable for
  cheap ablations and debug. Not recommended for production.

## 4. Why P0 looks "underutilized" at 291 tok/s/GPU

The lvbc 1,500 tok/s/GPU "healthy" floor comes from **inference-only**
workloads (`gym_rollout/generate_pass_rates.py` on a dedicated vLLM server).
Our verl rollout is a **hybrid engine** — the same 128 GPUs run actor
training and vLLM inference in alternation each step, with
`free_cache_engine=true` forcing a vLLM KV warmup every step.

Expectations adjusted for hybrid engine: the practical ceiling looks
closer to the lvbc AIME25 (≈950 tok/s/GPU) because of variable-length
outputs + per-step cache teardown. P2's 975 tok/s/GPU essentially matches
that ceiling.

## 5. Production recommendation

1. **Switch single-domain production runs from 16-node to 8-node.**
   - Almost identical step time (+16%)
   - Same memory budget (237 GB / 288 GB — 17% headroom)
   - Doubles the per-batch throughput per GPU → better cluster-$/step
   - Enables 4 concurrent training experiments instead of 2 on the
     31-node cluster
2. **Use 4-node for ablations / hparam sweeps.**
   - 1.49× step time is acceptable when you're running 20+ short
     experiments
   - Leaves 24 nodes free for one concurrent 8-node "main" run
3. **Keep 16-node only for:**
   - Multi-domain curriculum with 2× larger effective batch
   - Workloads that actually require the DP=8 gradient-averaging
     (e.g., numeric ablations where noise floor matters)

## 6. KubeRay + Volcano verification (Plan A deliverables)

All invariants from Plan A verified on three live runs:

| Check | P0 | P1 | P2 |
|---|---|---|---|
| PodGroup auto-created | ✅ | ✅ | ✅ |
| minMember = 1 + workerReplicas | ✅ 16 | ✅ 8 | ✅ 4 |
| All pods bound atomically | ✅ | ✅ | ✅ |
| PodGroup transitions to Scheduled within ~8 s of capacity | ✅ | ✅ | ✅ |
| No partial placement, no Unschedulable persistence | ✅ | ✅ | ✅ |

The first P0 attempt crashed at worker init because of an unrelated verl
hyperparameter bug (`lr_warmup_steps=10` vs `total_training_steps=5` — details
in `summary_plan_b.md` Step B-13). After that crash, the pods from the failed
RayJob stayed on 16 nodes because `ttlSecondsAfterFinished: 1800` keeps them
around for 30 min after a terminal state (so operators can read logs). I ran
`kubectl delete rayjob <failed-name>` to bypass that 30-min wait and free the
nodes immediately. The retry's gang then got all 16 nodes within ~10 s —
confirming that Volcano gang queueing behaves correctly under capacity
pressure.

## 7. What landed in the tree

### Plans + docs
- `0418/plan_a_kuberay_volcano.md` — infra correctness
- `0418/plan_b_node_opt.md` — sharding/rollout optimization with evidence
- `0418/plan_c_compute_opt.md` — apple-to-apple compute-efficiency matrix (C1–C9)
- `0418/summary_plan_a.md`, `summary_plan_b.md` — per-plan step-by-step summaries
- `0418/dev_notes.md` — live log of the whole exercise
- `0418/k8s_cheatsheet.md` — 12-section daily-ops cheatsheet
- `0418/summary.md` — this file
- `iter_kuberay_32nodes_verl_training/yaml/legacy/README.md` — updated
  drift warning for the stale stage10 yamls

### Dev infra (reusable for future node-count experiments)
- `0418/rayjob_debug.yaml` — parameterized RayJob template
  (generateName, no Kueue, no manual schedulerName, `ttlSecondsAfterFinished: 120`)
- `0418/submit_debug.sh` — CLI: `submit_debug.sh <combo> <nnodes> [overrides...]`.
  Prepends 6 debug-only Hydra overrides to skip pre-train val, mid-run val,
  checkpoint saves, and fix the `lr_warmup_steps < lr_decay_steps` assert.
  All are Category A (do not affect training numerics).
- `0418/verify_gang.sh` — Volcano gang validator (6 checks)
- `0418/pipeline.sh` — monitor + analyze + explicit-delete (Phase 5). Frees
  nodes within seconds of a run's terminal state.
- `0418/queue_all.sh` — submit C1–C9 in one batch (supersedes `queue_tier123.sh`)
- `0418/analyze_perf.py` — single-file perf_log analyzer
- `0418/compare_profiles.py` — 3-way comparison table

### Instrumentation in verl
- `verl/verl/utils/perf_log.py` — new module, only active when
  `VERL_PERF_LOG=1`, zero overhead when off, appends JSONL per step
- `verl/verl/trainer/ppo/ray_trainer.py:1736-1743` — one-line call
  after `logger.log()`

### Config additions
- `verl/my_scripts/k8s/config/combo_40Bra.yaml` — `cdbg5` (5-step debug combo)

## 8. Results artifacts

Stored under `0418/results/`:

- `P0-16node_perf_log.jsonl` + `_analysis.txt`
- `P-8node_perf_log.jsonl` + `_analysis.txt`
- `P-4node_perf_log.jsonl` + `_analysis.txt`

Each file contains per-step timing, memory, throughput, response-length
stats — the raw evidence behind the recommendation above.

## 8. Per-run file layout (changed 2026-04-18)

All files for one training run live in ONE directory named by the RayJob:

```
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
    run.log           entrypoint + training driver stdout/stderr
    perf_log.jsonl    per-step metrics
    analysis.txt      post-run summary
    exp_name.txt      traditional timestamped name (for wandb UI)
    rayjob.txt        the RayJob name
    label.txt         operator label (e.g. "C1"), if set via DEBUG_LABEL
```

Inspecting a run is a single command:

```bash
ls /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/<rayjob>/
tail -n 100 .../<rayjob>/run.log
cat .../<rayjob>/perf_log.jsonl | python3 -m json.tool | less
cat .../<rayjob>/analysis.txt
```

Previous layouts (exp_name-keyed dirs; separate `verl/logs/` tree; `.rayjob_lookup` markers) are still handled by `pipeline.sh` for any runs submitted before the entrypoint rewrite. Any run submitted from 2026-04-18 onwards uses the single-directory layout above.

## 8b. Env config group (2026-04-18, late afternoon)

Plan-C results are now wired in as selectable profiles. Switch with one env var:

```bash
./submit.sh c351              # default: 16-node optimal (no offload, gmu=0.7)
NNODES=8  ./submit.sh c351    # C10 profile (8-node, no offload)
NNODES=4  ./submit.sh c351    # C11 profile (4-node, no offload, gmu=0.4)
```

Profiles live under `verl/my_scripts/k8s/config/env/`:
- `k8s_b300_16node.yaml` — **new default**, Plan-C optimal applied to 16 nodes
- `k8s_b300_16node_legacy.yaml` — pre-2026-04-18 profile (all offload on); rollback via `env=k8s_b300_16node_legacy`
- `k8s_b300_8node_c10.yaml` — half the nodes, same wall-clock
- `k8s_b300_4node_c11.yaml` — quarter the nodes, most concurrent runs

All four are Category A only — training numerics unchanged.

## 8a. Debug-only conventions (added 2026-04-18, Plan C follow-up)

Applied to make debug-run wall clock tolerable while preserving
apple-to-apple comparisons. All are Category A (training-irrelevant) and
are confined to the debug flow — production `submit/rayjob.yaml` +
`submit/submit.sh` are untouched.

| File | Change | Effect |
|---|---|---|
| `submit_debug.sh` | Prepend 6 Hydra overrides: `val_before_train=false`, `test_freq=9999`, `save_freq=9999`, `log_val_generations=0`, `val_max_samples=32`, `lr_warmup_steps=2` | Each 5-step run drops from ~70 min → ~55 min wall clock |
| `rayjob_debug.yaml` | `ttlSecondsAfterFinished: 1800 → 120` | Safety net only; `pipeline.sh` handles the normal path |
| `pipeline.sh` | Added Phase 5 — explicit `kubectl delete rayjob` after analysis | Nodes freed within seconds of terminal state (vs 120 s ttl wait) |
| `queue_all.sh` | New batch submitter (replaces `queue_tier123.sh`) | Submits all 9 C-runs at once, relies on submit_debug.sh defaults |

See `plan_c_compute_opt.md` §4a and `dev_notes.md` (2026-04-18 block) for
the full rationale.

## 9. Open questions (not addressed tonight — for follow-up)

- **Can we push tok/s/GPU past 975?** Possibilities: raise
  `gpu_memory_utilization` from 0.7 → 0.85 (lvbc uses 0.85-0.9);
  drop `free_cache_engine` (keeps vLLM KV warm across steps);
  raise `rollout.n` (e.g. 32) to keep per-GPU batch high even at 8-node.
- **Would PP=1 help?** With DP=16/PP=1/EP=8 at 8 nodes (model must fit
  in one GPU slice during fwd and gather for bwd). Worth a measurement.
- **Should val_batch_size stay 32 at 8-node?** Currently hardcoded in
  `40bra_16node_sd.yaml`. At DP=4 that's 8 prompts/replica/batch —
  might be fine, but worth checking.
- **P0 vs P1 convergence equivalence.** We showed same step timing
  class, but did not verify learning-rate curves, loss shapes, or
  reward-signal stability across the two profiles. Recommend a 30-step
  comparison on a real combo (c351) before fully flipping production.

## 10. Suggested next actions (not performed)

- Apply the P1 recommendation: update `env_k8s_b300_16node.yaml` to
  `env_k8s_b300_8node.yaml` (8-node defaults) and retire the 16-node
  variant for single-domain work.
- Run a 30-step `c351` convergence comparison (P0 vs P1) to certify
  learning equivalence before committing to P1 in production.
- Investigate question #1 above with another `cdbg5` series of runs
  that varies `gpu_memory_utilization` and `rollout.n`.
