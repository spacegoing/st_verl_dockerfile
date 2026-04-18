# 0418 Dev Notes — KubeRay/Volcano Correctness + Node-count Optimization

Live log of the 0418 work. Kick-off: 2026-04-17 17:30 UTC.

Goal (from the user): trace KubeRay+Volcano stack correctness for our
verl RayJob flow, then hunt for node-count / sharding / rollout
optimizations under evidence (PFS logs). Debug jobs must complete in
≤ ~5 training steps.

---

## 17:30 UTC — Investigation phase

Launched four parallel Explore/general agents:

1. Map RayJob submission flow and kuberay+volcano stack compliance
   - ✅ kuberay-operator has `--batch-scheduler=volcano`
   - ✅ gang plugin enabled in volcano-scheduler-configmap
   - ✅ Kueue fully removed (CRDs gone, `kueue-system` empty)
   - ✅ 31 compute nodes idle, 248 B300 GPUs available
   - Production `submit/rayjob.yaml` already conforms to voltest pattern
   - `yaml/legacy/stage10*.yaml` carry stale Kueue label + manual
     `schedulerName: volcano` — already under `legacy/` and harmless,
     but a copy-paste hazard
2. Map verl k8s→Hydra→training callstack
   - Entry: `submit.sh c351` → envsubst → RayJob → entrypoint shell →
     `ray job submit python3 -m verl.trainer.main_ppo --config-name=40bra_16node_sd`
   - Hydra merge order: `ppo_megatron_trainer` → `env_k8s_b300_16node`
     → `combo_40Bra@combo_bank` → `_self_`
   - Sharding: PP=2, VPP=2, TP=1, EP=8, ETP=1, CP=1
   - 128 GPUs / (2×1×8×1) = **8 DP replicas**, each occupying 16 GPUs
   - Rollout: vLLM TP=1, hybrid engine (colocated with actor)
   - `rollout.n=16`, `train_batch_size=128` → 2048 rollouts per step
     → **16 rollouts per GPU per step**
3. Extract rollout best practices from lvbc/ + gym_rollout/
   - 40Bra fits in 1×B300 (92GB BF16 < 288GB HBM); DP=8/TP=1 is optimal
   - Peak uniform throughput: 2,436–3,951 tok/s/GPU
   - Saturation regime: B=32 per engine (512 token decode window)
   - Below 1,500 tok/s/GPU → investigate (our 16/GPU is in the
     **under-utilized** regime)
4. Probe cluster state
   - All 32 nodes idle; no active RayJobs; no PodGroups
   - Cluster is configured correctly and ready for the experiment

## 17:31 UTC — Plans written

- `0418/plan_a_kuberay_volcano.md` — infra correctness, 9 sections
  including gang invariants, verification checks, legacy yaml drift
  notes, and exit criteria.
- `0418/plan_b_node_opt.md` — 11 sections. Key decisions:
  - Three candidate profiles: **P0=16 nodes** (current), **P1=8**, **P2=4**
  - Reference floor: 1,500 tok/s/GPU "healthy", <1,000 under-utilized
  - Instrumentation: `verl/verl/utils/perf_log.py`, active only when
    `VERL_PERF_LOG=1` is set; appends JSONL to `${trainer.default_local_dir}/perf_log.jsonl`
  - Debug combo `cdbg5`: 5 steps, nemogym_math, curriculum_ts=50,
    paired with `data.val_max_samples=32` and `trainer.test_freq=3`
    to keep wall clock ≤ ~40 min per profile

## 17:32 UTC — Dev infrastructure built

Created under `0418/`:

- `rayjob_debug.yaml` — parameterized RayJob template (envsubst
  placeholders: `COMBO_ID`, `NNODES`, `WORKER_REPLICAS`, `EXTRA_OVERRIDES`)
  - No Kueue label; no manual `schedulerName: volcano` on pods
  - `generateName: bra40-dbg-${COMBO_ID}-${NNODES}n-`
  - `replicas == minReplicas == maxReplicas == WORKER_REPLICAS`
  - Carries `VERL_PERF_LOG=1` env var to enable the new logger
- `submit_debug.sh` — CLI: `./submit_debug.sh <combo> <nnodes> [overrides]`
  - Computes `WORKER_REPLICAS = NNODES - 1`
  - Validates combo format + node count bounds
  - envsubst + `kubectl create` with `generateName`
- `verify_gang.sh` — runs 6 checks on Volcano gang placement
- `analyze_perf.py` — reads `perf_log.jsonl`, prints per-step table +
  summary with per-GPU throughput vs 1,500 tok/s/GPU floor
- `pipeline.sh` — end-to-end orchestration: monitor terminal state →
  copy perf_log from ckpts → run analyze_perf → (optionally) chain
  next profile submission (16 → 8 → 4)

Edited:

- `verl/verl/utils/perf_log.py` — new module (no-op unless
  `VERL_PERF_LOG=1`; append-only to PFS).
- `verl/verl/trainer/ppo/ray_trainer.py:1736-1743` — one call to
  `write_perf_log()` right after `logger.log()`.
- `verl/my_scripts/k8s/config/combo_40Bra.yaml` — added `cdbg5`
  (5-step debug combo).
- `iter_kuberay_32nodes_verl_training/yaml/legacy/README.md` —
  expanded drift warning with three specific divergences from
  `submit/rayjob.yaml`.

## 17:36 UTC — P0 (16-node) run submitted

```
[submit_debug] combo=cdbg5  nnodes=16  worker_replicas=15
[submit_debug] name=bra40-dbg-cdbg5-16n-2mvbt
[submit_debug] overrides: data.val_max_samples=32 trainer.test_freq=3
```

Server-side dry-run passed before real submit. All 16 pods bound
atomically within ~8 s (PodGroup Unschedulable → Scheduled transition
observed in Volcano events).

`verify_gang.sh` — all 6 checks pass:
- RayJob exists
- RayCluster name resolved: `bra40-dbg-cdbg5-16n-2mvbt-vwcpv`
- PodGroup `ray-bra40-dbg-cdbg5-16n-2mvbt-pg` minMember=16 phase=Running
- minMember matches 1+workerReplicas = 16
- All 16 pods bound to nodes (gang placement atomic)
- PodGroup Running + Scheduled=True (gang admitted atomically)

(Initial implementation of `verify_gang.sh` had two false positives:
"mixed pod states" during container init, and "Unschedulable events"
from the transient pre-gang state. Both relaxed to match the actual
gang invariant — pods bound to nodes atomically, current phase is
healthy.)

## 17:38 UTC — Pipeline monitor running

- Background PID 3778843 watching `bra40-dbg-cdbg5-16n-2mvbt`
- On SUCCEEDED: pulls perf_log.jsonl → runs analyze_perf.py → submits
  P1 (8-node) with cdbg5 → chains to P2 (4-node) on P1 success
- Log: `/tmp/0418_pipeline_P0-16node.log`
- Results dir: `0418/results/`

Decision points when P0 completes:
- If `tokens_per_gpu_s_gen < 1000`: rollout is underutilized — P1 is
  safe to try (halves node count, doubles per-GPU batch)
- If `tokens_per_gpu_s_gen ≥ 1500`: rollout is well-saturated — P1
  will push per-GPU batch to 32 (lvbc saturation optimum)
- Memory (`max_memory_allocated_gb`) headroom > 60GB/GPU at P0 is a
  strong signal P1 will fit

## 17:46 UTC — P0 FAILED (lr_warmup_steps assertion)

```
File "/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/optimizer_param_scheduler.py", line 157, in __init__
    assert self.lr_warmup_steps < self.lr_decay_steps
AssertionError
```

Root cause: `env_k8s_b300_16node.yaml` hard-codes `lr_warmup_steps: 10`,
but cdbg5 sets `total_training_steps: 5`. Megatron asserts
`warmup < decay` (decay ≈ total_steps), so 10 < 5 fails.

Same class of issue as the baremetal smoke test noted in MEMORY.md
("total_training_steps=15, lr_warmup_steps=5").

### Fixes

1. `0418/submit_debug.sh` — prepend default Hydra override
   `actor_rollout_ref.actor.optim.lr_warmup_steps=2` to every debug
   submission. Survives being run for any cdbg-style combo.
2. `0418/pipeline.sh` — ckpt-dir discovery was `ls -dt .../40bra_k8s_16node_sd_*`
   which matched an old c353 dir (because the failed run never created
   its own ckpt dir). Fixed to extract combo id from the RayJob's
   entrypoint spec and glob for that specific combo.

## 17:48 UTC — P0 resubmitted with fix

- `bra40-dbg-cdbg5-16n-4pllg` — new P0 baseline
- Pipeline monitor PID 3787386 watching it
- Previous failed run (bra40-dbg-cdbg5-16n-2mvbt) explicitly deleted
  to free capacity (ttl=1800 would have held it 30 min otherwise)
- PodGroup: minMember=16, Inqueue → Scheduled (atomic gang bind) within
  ~10 s after capacity freed
- Pods: 1 head Running + 15 workers Init container = all bound

## 18:52 UTC — P0 SUCCEEDED ✅

All 5 steps ran end-to-end. Raw perf_log (copied to `0418/results/P0-16node_perf_log.jsonl`):

| step | step_s | gen_s | upd_s | tok/s/GPU | mem_gb | resp_m | clip |
|---|---|---|---|---|---|---|---|
| 1 | 531.7 | 266.6 | 155.5 | 300 | 231.8 | 4860 | 0.00 |
| 2 | 458.7 | 323.0 | 101.9 | 251 | 232.2 | 4915 | 0.00 |
| 3 | 420.8 | 283.3 | 103.9 | 306 | 234.0 | 5278 | 0.00 |
| 4 | 420.0 | 289.4 | 95.7 | 287 | 234.0 | 5059 | 0.00 |
| 5 | 392.8 | 263.8 | 96.5 | 310 | 234.0 | 4972 | 0.00 |
| avg | 444.8 | 285.2 | 110.7 | **291** | 233.2 | 5017 | 0.00 |

**Key findings**:
- `gen` = 64.1% of step; `update_actor` = 24.9% → rollout dominates, as expected
- **gen tok/s/GPU = 291 — only 0.19× of lvbc's 1,500 floor, 0.12× of 2,436 peak**
- GPU max_memory_allocated = 233 GB / 288 GB capacity → **55 GB headroom/GPU**
- Response length mean 5,017 / max 16,384 → no responses hit the cap (`clip_ratio=0`), so variable-length shrinkage (not length-cap) is what limits throughput
- First step slower (531s vs ~420s steady state) — NCCL lazy init + vLLM KV warmup — normal

**Interpretation (per plan_b_node_opt.md):**
- 291 tok/s/GPU is 5× below the "healthy" floor. P0 is strongly under-utilizing rollout.
- We expect P1 (8-node, 2× batch per GPU) to roughly **double per-GPU tok/s** (→ ~580) and **roughly double step time** (~850s/step). At 5 steps that's ~70 min.
- We expect P2 (4-node, 4× batch per GPU) to **quadruple per-GPU tok/s** (→ ~1100) and **3-4× step time** (~1500s/step). ~2 h wall clock for 5 steps.
- Memory has plenty of room; neither should OOM.

## 18:52 UTC — P1 (8-node) auto-submitted

- `bra40-dbg-cdbg5-8n-f5d9j` — pipeline-chained from P0
- `verify_gang.sh`: all 6 checks pass — PodGroup minMember=8, all pods bound atomically, Scheduled=True
- Pipeline monitor PID 3839427 watching it; will chain P2 (4-node) on success

## 20:02 UTC — P1 (8-node) SUCCEEDED ✅

5 steps, 66 min wall clock (submit 18:52 → terminal 20:02).

| step | step_s | gen_s | upd_s | tok/s/GPU | mem_gb | resp_m |
|---|---|---|---|---|---|---|
| 1 | 593.3 | 269.4 | 209.0 | 595 | 235.2 | 4870 |
| 2 | 473.8 | 285.6 | 148.8 | 578 | 236.2 | 5017 |
| 3 | 481.8 | 286.1 | 153.3 | 595 | 237.2 | 5184 |
| 4 | 508.4 | 306.1 | 160.9 | 534 | 237.2 | 4975 |
| 5 | 523.8 | 325.6 | 155.9 | 502 | 237.2 | 4969 |
| avg | 516.2 | 294.5 | 165.6 | **561** | 236.6 | 5003 |

**P0 vs P1 side-by-side**:

| metric | P0 (16n) | P1 (8n) | Δ |
|---|---|---|---|
| step_s | 444.8 | 516.2 | **+16%** |
| gen_s | 285.2 | 294.5 | **+3%** ← nearly flat! |
| upd_s | 110.7 | 165.6 | +50% |
| tok/s/GPU | 291 | 561 | **+93%** |
| mem (GB) | 233.2 | 236.6 | +1% (negligible) |
| floor ratio | 0.19× | 0.37× | |

**Interpretation**:
- gen time only +3% despite 2× batch per GPU → **rollout was strongly
  under-saturated at P0, confirming Plan B's hypothesis**.
- update_actor +50% because each DP replica processes 2× more
  prompts/optimizer-step (fixed overhead dampens the linear scaling).
- Overall step +16% for half the hardware = **clear win**.

**Recommendation so far**: P1 (8-node) should become the default for
any 40Bra single-domain GRPO run. Double the experiments for the same
cluster budget.

## 20:02 UTC — P2 (4-node) auto-submitted

- `bra40-dbg-cdbg5-4n-7hzjr` — final profile in the ladder
- `verify_gang.sh`: all 6 checks pass — PodGroup minMember=4, all pods
  bound atomically.
- Pipeline monitor PID 3898552; next_after=none (end of chain)

If P2 succeeds with step_s ≤ ~1000s (< 2.3× P0) and tok/s/GPU ≥ 1,000:
- P2 becomes a viable option for very cheap ablation runs (e.g.
  hyperparameter sweeps or short debug loops)
- P1 stays the recommended production config because
  update_actor scales worse below 8 DP replicas

If P2 OOMs or step times blow up > 2.5× P0:
- Document cliff; keep P1 as the recommended minimum
- Don't regress to P0; the evidence already supports halving

## 21:26 UTC — P2 (4-node) SUCCEEDED ✅

5 steps, 80 min wall clock (submit 20:02 → terminal 21:26).

| step | step_s | gen_s | upd_s | tok/s/GPU | mem_gb | resp_m |
|---|---|---|---|---|---|---|
| 1 | 739.6 | 316.9 | 300.5 | 1021 | 235.5 | 4919 |
| 2 | 600.5 | 330.3 | 222.4 | 992 | 235.9 | 4977 |
| 3 | 651.1 | 333.1 | 269.6 | 1033 | 235.9 | 5236 |
| 4 | 667.7 | 356.7 | 262.6 | 932 | 236.4 | 5058 |
| 5 | 659.8 | 368.7 | 242.8 | 895 | 236.4 | 5024 |
| avg | 663.7 | 341.1 | 259.6 | **975** | 236.0 | 5043 |

**Full 3-way comparison**:

| profile | gpus | step_s | gen_s | upd_s | tok/s/GPU | step_ratio | tok/s_ratio |
|---|---|---|---|---|---|---|---|
| P0 | 128 | 444.8 | 285.2 | 110.7 | 291 | 1.00× | 1.00× |
| P1 | 64 | 516.2 | 294.5 | 165.6 | 561 | **1.16×** | 1.93× |
| P2 | 32 | 663.7 | 341.1 | 259.6 | 975 | 1.49× | 3.35× |

**Key observations**:
1. P2 tok/s/GPU = 975 — essentially equal to lvbc's AIME25 real-eval
   floor of ~950 tok/s/GPU. This is the practical ceiling for
   variable-length hybrid-engine rollout, not the lvbc 1,500
   inference-only floor. P2 is **at the ceiling, not below it**.
2. gen_s scaling: 285 → 294 → 341 (+3%, +16%). Generation scales
   sub-linearly with per-GPU batch — the ceiling is in reach.
3. update_actor scaling: 111 → 166 → 260 (+50%, +57%). Larger per-DP
   batch means more gradient accumulation + optimizer offload dominates.
4. Memory flat at ~236 GB across all three. Plenty of headroom (52 GB).

## 21:30 UTC — Recommendation finalized

**Production winner: P1 (8-node)**.
- Step time only 16% slower than P0
- Doubles cluster utility — 4 concurrent experiments instead of 2
- Identical memory footprint
- Does NOT change gen throughput meaningfully (only +3%) — rollout
  quality/variance should be equivalent to P0

**Ablation/debug tier: P2 (4-node)**.
- 49% slower per step, but at the hybrid-engine ceiling
- Useful for hparam sweeps where 20+ cheap runs beat 4 expensive ones

Full recommendation + open questions in `0418/summary.md`.

## Artifacts

- `0418/plan_a_kuberay_volcano.md`, `plan_b_node_opt.md`, `dev_notes.md`, `summary.md`
- `0418/rayjob_debug.yaml`, `submit_debug.sh`, `verify_gang.sh`, `pipeline.sh`
- `0418/analyze_perf.py`, `compare_profiles.py`
- `0418/results/{P0-16node,P-8node,P-4node}_{perf_log.jsonl,analysis.txt}`
- `verl/verl/utils/perf_log.py` (new, only active when `VERL_PERF_LOG=1` is set)
- `verl/verl/trainer/ppo/ray_trainer.py:1736-1743` (one-line wire-in)
- `verl/my_scripts/k8s/config/combo_40Bra.yaml` (added `cdbg5`)

---

# 2026-04-18 — Plan C (compute-efficiency) + debug-only optimizations

## 09:00–11:00 UTC — Plan C written, apple-to-apple rule

User specified: compute-efficiency optimization must not touch any hyperparameter that can change the training trajectory. Wrote `0418/plan_c_compute_opt.md` with:
- Section 1: Category A (training-irrelevant, safe to change) vs Category B (training-relevant, must hold fixed) vs Category C (gray zone) hyperparameter split.
- Section 3: 9-experiment matrix (C1–C9) varying only Category A knobs.
- Section 5: logger review — existing `perf_log.jsonl` already captures timing, memory, throughput; per-phase memory split is the only gap worth plugging later.

P0/P1/P2 series from Plan B already obeyed the rule (only `trainer.nnodes` varied). P1 (8-node) is the baseline for all Plan C runs.

## 11:30 UTC — first batch submitted (pre-optimization)

Submitted C1–C8 (6 runs) with the OLD ttl=1800s and OLD debug settings (val_before_train=true, test_freq=3, save_freq=5). Volcano gang queueing worked as expected — 3 runs admitted, 3 queued Inqueue.

`queue_tier123.sh` hit SIGPIPE from awk exiting early; C4 got submitted but without a monitor. Manually rescued by starting a pipeline monitor for it.

## 11:50 UTC — nccl-test cleanup

Discovered `vcjob/nccl-test` owned by user `cxu`, running 3 pods × 8 GPUs = 24 GPUs for 2+ days on `sshd && sleep inf` (no active compute). User confirmed this was a defunct admin test. Deleted the parent vcjob; pods terminated; 24 B300 GPUs reclaimed.

## 12:00 UTC — debug-only optimizations landed

User requested shorter test time without changing `total_training_steps`. Per-run overhead breakdown showed ~27 min of startup/val/checkpoint overhead on top of ~43 min real training. All overhead is Category A, so safely removable for debug runs.

Changes (debug-only, production untouched):

1. **`submit_debug.sh` — 6 default Hydra overrides prepended**:
   - `trainer.val_before_train=false` (skip pre-train val)
   - `trainer.test_freq=9999` (skip mid-run val)
   - `trainer.save_freq=9999` (no checkpoint writes)
   - `trainer.log_val_generations=0`
   - `data.val_max_samples=32` (safety net)
   - `actor_rollout_ref.actor.optim.lr_warmup_steps=2` (avoid Megatron `warmup < decay` assert)

2. **`rayjob_debug.yaml` — `ttlSecondsAfterFinished: 1800 → 120`**:
   - Short ttl so Volcano can promote the next gang quickly.
   - 120 s is only a safety net; `pipeline.sh` deletes the RayJob explicitly.

3. **`pipeline.sh` — Phase 5 (explicit RayJob delete)**:
   - After analysis, runs `kubectl delete rayjob $JOB --wait=false` if `perf_log.jsonl` was captured.
   - Nodes released within seconds instead of 120 s.
   - Skipped when `perf_log` is missing so head pod logs stay readable for diagnosis.

4. **`queue_all.sh` — new batch submitter** (supersedes `queue_tier123.sh`):
   - Submits all 9 C-runs at once (C1–C9 including C5/C7 fallbacks).
   - No `awk`/pipeline SIGPIPE issues (uses `grep -oE ... | head -1`).
   - Relies on submit_debug.sh defaults — no repeated overrides per run.
   - Expected total wall clock: ~2.75 h (3 waves of 3 concurrent runs × ~55 min).

## 12:05 UTC — deleted first batch, resubmitted with new defaults

Killed all 6 pipeline monitors, deleted all 6 RayJobs, wiped pipeline logs, restarted fresh. Ran `queue_all.sh`; 9 RayJobs submitted, all with `ttlSecondsAfterFinished: 120` verified. Wave 1 (C1/C2/C3 = Tier 1 memory knobs) immediately admitted. Waves 2 and 3 Inqueue waiting.

## 12:14 UTC — C2 and C3 FAILED (OOM) + pipeline.sh bugs surfaced

Both C2 (`gmu=0.85 + free_cache_engine=false`) and C3 (`gmu=0.90 + free_cache_engine=false`) crashed during the `actor_rollout_compute_log_prob` phase with `torch.OutOfMemoryError`:

- C2: 237.75 GiB allocated by PyTorch on a 267.69 GiB GPU, tried to allocate 10.9 GiB more, failed.
- C3: 248.06 GiB allocated, tried to allocate 1.21 GiB more, only 200-630 MiB free, failed.

Hypothesis confirmed: `free_cache_engine=false` keeps the vLLM KV cache resident through the actor update phase. With `gpu_memory_utilization=0.85`/`0.90`, vLLM alone reserves ~228-244 GiB, leaving too little room for actor activations in the ref/log-prob pass. This combination is incompatible with the hybrid engine unless the actor side also reduces its footprint (e.g. `gpu_memory_utilization ≤ 0.5` AND more offload).

Plan C has C2/C3 recorded as a dead end. Do NOT combine keepkv with high gmu.

### Two bugs in `pipeline.sh` surfaced

During C2/C3 failure collection, the pipeline ran its Phase 2 and produced stale "analysis" using yesterday's P-4node perf_log:

| Bug | Root cause | Impact |
|---|---|---|
| Ckpt-dir picked stale directory | `ls -dt .../40bra_k8s_16node_sd_${COMBO}_*` returned yesterday's ckpt because the failed run created no ckpt dir of its own | C2/C3 analysis files initially showed the P-4node=32 GPU numbers from yesterday as if they were this run's results |
| Phase 5 deleted the failed RayJob | Phase 5 gating was only `if [ -r perf_log.jsonl ]` — it did not check whether the file was actually FROM this run | The OOM pods disappeared before we could inspect them |

### Fixes applied to `pipeline.sh`

1. **Time-matched ckpt-dir search**: read the first record's `wall_clock_utc` from each candidate `perf_log.jsonl`; accept only if it is after `RayJob.creationTimestamp - 300 s`. Otherwise treat as "no fresh perf_log".
2. **Always copy the PFS training log** (`verl/logs/40bra_k8s_single_domain/40bra_k8s_16node_sd_${COMBO}_<timestamp>.log`) to `0418/results/${LABEL}_training.log` when its mtime is after the RayJob was created. Gives us the full Hydra + Ray + Megatron output even when `perf_log.jsonl` was never written.
3. **Phase 5 gating unchanged but now correct**: the check `[ -r "$SUMMARY_DIR/${LABEL}_perf_log.jsonl" ]` is now only true when a FRESH perf_log was copied.

### Restarted all 7 remaining pipeline monitors

Old monitors had the buggy code loaded. Killed old pids; relaunched with the fixed `pipeline.sh` for C1/C4/C5/C6/C7/C8/C9. New monitors will correctly:
- skip the stale-dir trap on failures
- preserve failed RayJobs for 120 s ttl (readable `kubectl logs` window)
- capture PFS training logs as fallback

### Replaced C2/C3 result files

- Deleted `0418/results/C2_perf_log.jsonl`, `C3_perf_log.jsonl` (those were yesterday's P-4node data mis-copied).
- Wrote `0418/results/C2_analysis.txt` and `C3_analysis.txt` with the real OOM diagnosis.
- Copied `0418/results/C2_training.log`, `C3_training.log` from the PFS training-log dir.

## 12:22 UTC — C4 and C5 FAILED (PP=1 kills pod, no Python trace)

Both C4 (PP=1, VPP=1) and C5 (PP=1 + CP=2) failed after step 1.

C4 step 1 succeeded with these numbers:
- `step_s=597`, `gen_s=287`, `update_actor_s=197`
- `max_memory_allocated_gb=275.1`, `max_memory_reserved_gb=290.7`
- `critic/score/mean=0.647`

Step 2 was then killed mid-stream — training log ends abruptly, no Python traceback. Hypothesis: `max_memory_reserved_gb=290.7 > 288 GB HBM capacity`, so any further allocation in step 2 tripped the OS OOM-killer which terminates the container silently.

At PP=2 (our P1 baseline) per-GPU `max_memory_allocated_gb=236.6`. Removing the pipeline split (PP=1) pushed that to 275.1 because each GPU now holds the full layer stack of the 40B model. CP=2 was supposed to rescue by halving the sequence shard per GPU; it did not help in C5 (same failure).

Conclusion for Plan C: **PP=1 is infeasible for 40Bra at `max_prompt=8000`, `max_response=16384`** without additional memory reduction (more offload, or significantly lower `gpu_memory_utilization`). This also means C9 (which combines PP=1) will fail identically — it's not worth the cluster time but we keep it queued for completeness of the experiment matrix.

### Second ckpt-matching bug surfaced

After C4/C5 failed, their `_analysis.txt` files showed the P1 numbers again — but this time from **today's C1 run** (not yesterday's P2). Root cause: my time-based cutoff `CREATE_EPOCH - 300s` was too generous — all 9 Wave-1 RayJobs were created within a few minutes of each other, so their cutoffs all accepted each other's ckpt dirs. The match is ambiguous when concurrent runs share the combo.

### Fix: add a marker file in the k8s entrypoint + pipeline.sh primary lookup

- `verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh`: after computing `exp_name`, extract the RayJob name from the pod's `HOSTNAME` (kuberay pattern: `<rayjob>-<5char>-(head|workers-worker)-<5char>`) and write `${exp_name}` to `${CKPTS_PARENT}/.rayjob_lookup/${RAYJOB_NAME}`. No effect on training numerics; 10-line shell addition.
- `0418/pipeline.sh` Phase 2: try the marker file first (`[ -r "$LOOKUP_DIR/$JOB" ]` → `$CKPT_ROOT/$exp_name`); fall back to the previous time-bounded search only when the marker is missing (backward compatibility for RayJobs submitted before this change).

The currently-running C1/C6/C7/C8/C9 were all submitted BEFORE the entrypoint change so they will still rely on the time-match fallback. Any NEW RayJob submitted from now on gets the marker automatically.

### Restart monitors again

Killed all pipeline monitors for the active cdbg5 runs, restarted each with the updated `pipeline.sh`. Marker lookup is ready for future runs; current runs use fallback.

### Manual C4/C5 result files

- Replaced the wrongly-populated `C4_perf_log.jsonl`, `C5_perf_log.jsonl` (they carried today's C1 step-1 numbers).
- Wrote `0418/results/C4_analysis.txt`, `C5_analysis.txt` with the real "PP=1 OOM-killed" diagnosis.
- `C4_training.log`, `C5_training.log` are still in place but their mtime-matched source may be ambiguous — the actual run-specific PFS logs to inspect are referenced inside the analysis files.

## 12:45 UTC — Per-run directory consolidation

User observation: files for one training run were scattered across four places with three different IDs:
- `verl/logs/40bra_k8s_single_domain/<exp_name>.log` (entrypoint log; ID = exp_name)
- `verl/ckpts/40bra_k8s_single_domain/<exp_name>/perf_log.jsonl` (metrics; ID = exp_name)
- `verl/ckpts/.../.rayjob_lookup/<rayjob>` (marker; ID = RayJob name)
- `0418/results/<LABEL>_*` (analysis + copies; ID = human label like "C1")
- `/tmp/0418_pipeline_<LABEL>.log` (pipeline monitor; ID = human label)

Request: one ID, one place. Different filenames per file type are fine (run.log, perf_log.jsonl, analysis.txt), but the identifying info should be consistent.

### New per-run layout (applies to any RayJob submitted from now on)

```
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
    run.log           entrypoint + Hydra + ray driver stdout/stderr
    perf_log.jsonl    per-step metrics from the verl logger
    analysis.txt      post-run summary from analyze_perf.py
    exp_name.txt      the traditional timestamped exp_name (used for wandb UI)
    rayjob.txt        the RayJob name (self-documenting the lookup)
    label.txt         optional human label like "C1" (present only if DEBUG_LABEL was set)
    <megatron ckpt shards as before>
```

### File changes (all are compatible with existing in-flight runs via fallback logic)

| File | Change | Why |
|---|---|---|
| `verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh` | Derive `RAYJOB_NAME` from pod hostname; `RUN_DIR=verl/ckpts/40bra_k8s_single_domain/$RAYJOB_NAME`; `mkdir`; write exp_name.txt / rayjob.txt / label.txt; `exec &> >(tee -a $RUN_DIR/run.log)`; drop old marker-file write; drop `verl/logs/` path | One directory per run, named unambiguously by RayJob. Old `verl/logs/` target becomes unused; old `.rayjob_lookup/` becomes unused. |
| `0418/rayjob_debug.yaml` | Added `DEBUG_LABEL` env var with `envsubst` placeholder | Propagate the operator's "C1" style label into the pod so the entrypoint can write `label.txt` |
| `0418/submit_debug.sh` | Accepts optional `DEBUG_LABEL` env from caller; whitelists it in `envsubst` | CLI plumbing for the label |
| `0418/queue_all.sh` | Each `submit_one` passes `DEBUG_LABEL=<label>` | Every C-run gets its label written into its run dir |
| `0418/pipeline.sh` | Phase 2: primary lookup `CKPT_ROOT/$JOB` (new layout); fallback 1: old `.rayjob_lookup` marker; fallback 2: time-bounded search. Phase 3 also copies `analysis.txt` into the run dir. | Finds run dirs unambiguously for new runs, still handles in-flight old runs |

### In-flight runs during the transition

C1, C6, C7, C8, C9 were submitted BEFORE the entrypoint rewrite — their pods ran the older script, so their files sit in the OLD layout:
- Entrypoint log under `verl/logs/40bra_k8s_single_domain/<exp_name>.log`
- Checkpoints / perf_log under `verl/ckpts/.../<exp_name>/`
- (Some have a `.rayjob_lookup/<rayjob>` marker; some don't.)

pipeline.sh's fallback chain handles all three layouts, so these runs will still produce correct `0418/results/<LABEL>_*` files when they terminate.

### Deprecated (no new writes after 2026-04-18)

- `verl/logs/40bra_k8s_single_domain/` — used by the old entrypoint log path. Stays readable for historical reference; no new logs go there.
- `verl/ckpts/40bra_k8s_single_domain/.rayjob_lookup/` — no longer written; the dir name itself is now the lookup.

The `0418/results/` directory is kept as a flat index for quick cross-run comparison (`compare_profiles.py` reads from it). Files there are copies; the canonical source is the run dir on PFS.

## 12:30 UTC — C-runs in progress

Wave 1 RUNNING:
- C1 (vmk62): `rollout.gpu_memory_utilization=0.85`
- C2 (n5nhc): `gmu=0.85` + `free_cache_engine=false`
- C3 (7bkkq): `gmu=0.90` + `free_cache_engine=false`

Waiting in queue:
- C4 (6kpkb): PP=1, VPP=1
- C5 (gqg2g): PP=1 + CP=2
- C6 (fb67x): no optimizer offload
- C7 (2p5gl): no offload at all + rollout mem=0.4
- C8 (v6l6f): token budget 2×
- C9 (t8gtb): combined — C1 + C4 + C6

Results expected by ~14:30 UTC.

---

## 2026-04-18 15:30 UTC — `@package _global_` fix for env config group

First smoke test of `NNODES=8` and `NNODES=4` via the new `submit.sh` interface FAILED within 2 minutes with:

```
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or
  'namespace/repo_name': '~/models/deepseek-llm-7b-chat'. Use `repo_type` argument if needed.
```

That path is the generic default in `ppo_megatron_trainer.yaml`. Our env yaml sets
`actor_rollout_ref.model.path: /root/myCodeLab/host/downloads/models/40Bra` but it
was not overriding the default — the merge landed under `cfg.env.actor_rollout_ref.*`
instead of `cfg.actor_rollout_ref.*`.

Root cause: **Hydra config-group behavior**. When you move a file from the config
root (referenced as `- env_k8s_b300_16node`) into a group subdir (referenced as
`- env: k8s_b300_16node`), Hydra by default merges the file's content under the
group name, not at the root. The content inside the file is still treated as a
sub-tree of the composed config.

Fix: add `# @package _global_` as the first line of each env yaml. This directive
tells Hydra to merge the file's content at the root of the composed config — the
same behavior we had with the bare-name include.

Applied to:
- `verl/my_scripts/k8s/config/env/k8s_b300_16node.yaml`
- `verl/my_scripts/k8s/config/env/k8s_b300_16node_legacy.yaml`
- `verl/my_scripts/k8s/config/env/k8s_b300_8node_c10.yaml`
- `verl/my_scripts/k8s/config/env/k8s_b300_4node_c11.yaml`

Resubmitted both test jobs (`bra40-sd-c351-8n-5f9bm`, `bra40-sd-c351-4n-ldmcw`).
Both reached `jobStatus=RUNNING` and passed `validate_config` without the
HFValidationError → fix confirmed.

**Rule for future env yamls in this group:** always start the file with
`# @package _global_`. Any new env profile should use the same pattern.

## 2026-04-18 late afternoon — env config group

After C10/C11 terminated, wired the Plan-C winners into a Hydra config group so operators switch node-count profiles with one env var.

### Changes

- Created `verl/my_scripts/k8s/config/env/` directory.
- Renamed `env_k8s_b300_16node.yaml` → `env/k8s_b300_16node_legacy.yaml` (old all-offload-on P0 profile, kept for rollback).
- New `env/k8s_b300_16node.yaml` — 16 nodes with Plan-C learnings applied (no offload, gmu=0.7). Projected step_s ≈ 430, mem ≈ 220 GiB; measured extrapolation from C10.
- `env/k8s_b300_8node_c10.yaml` — 8-node C10 profile.
- `env/k8s_b300_4node_c11.yaml` — 4-node C11 profile.
- `40bra_16node_sd.yaml` defaults list updated to use the `env:` group.
- `submit/rayjob.yaml` parameterized with `${NNODES}`/`${WORKER_REPLICAS}`.
- `submit/submit.sh` reads `NNODES` (16|8|4) → maps to env name + worker count.

### Operator interface (production)

```bash
./submit.sh c351                  # default: 16-node optimal
NNODES=8  ./submit.sh c351        # C10 profile
NNODES=4  ./submit.sh c351        # C11 profile
NNODES=16 ./submit.sh c351 'env=k8s_b300_16node_legacy'   # rollback
```

Docs updated: `production_guide.md`, `plan_c_report.md`, `summary.md` §8b, `README.md`.

## Also delivered today

- `0418/k8s_cheatsheet.md` — 12-section operator cheatsheet for daily k8s tasks: setup alias+function, active-only queries, blocked-job diagnosis, per-job inspection, logs, cleanup (individual/bulk/time-based/orphan/dry-run), cluster capacity, ownership checks, Volcano/vcjob specifics, watch mode, troubleshooting, one-shot situation report.
- `0418/summary_plan_a.md`, `0418/summary_plan_b.md` — rigorous per-plan executions summaries (file maps, step-by-step logs, status, bugs fixed).
- Language-preference memory: `feedback_language_plain_english.md` in `~/.claude/...` — plain English preferred, CS jargon OK for accuracy, avoid "gate/scrub/squat/sweep/headline/etc."
- Bulk rename across all 0418 docs: "gate/gates/gated" → "check/checks/controlled by"; "headline/massively/sweet spot" → "main/strongly/optimum".
