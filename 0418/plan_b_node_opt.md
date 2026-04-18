# Plan B — Node-count, Sharding, and Rollout Optimization (evidence-based)

> Hypothesis: 16-node × 128-GPU is over-provisioned for our current
> single-domain 40Bra GRPO training. The bottleneck is rollout, and
> vLLM under verl's hybrid-engine layout may not be fully utilized.
> Prove or disprove with on-cluster measurements; then choose the
> minimum node count that keeps step time within a target envelope.

Written: 2026-04-17 (for 0418 work)
References: `lvbc/*`, `gym_rollout/dev_notes.md`, `verl/verl/trainer/ppo/ray_trainer.py`, `verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`, `verl/my_scripts/k8s/config/env_k8s_b300_16node.yaml`

---

## 1. Current sharding (what we're running today)

**Hardware**: 16 × B300 nodes, 128 GPUs.

From `verl/my_scripts/k8s/config/env_k8s_b300_16node.yaml`:

| Axis | Actor | Ref | Rollout (vLLM) |
|---|---|---|---|
| PP (pipeline) | 2 | 2 | — |
| VPP (virtual pipeline) | 2 | 2 | — |
| TP (tensor) | 1 | 1 | **1** |
| EP (expert) | 8 | 8 | — |
| ETP (expert-tp) | 1 | 1 | — |
| CP (context) | 1 | 1 | — |
| **DP replicas implied** | **8** | **8** | up to **128** |

Math: `total_gpus = DP × PP × TP × EP / overlap_factor`. For actor: `128 / (2 · 1 · 8 · 1) = 8` DP replicas. Each DP replica occupies `PP × EP = 16` GPUs (2 nodes).

**Hybrid engine**: `ActorRolloutRefWorker` colocates actor + rollout on the same workers (verl's default, `hybrid_engine=True` per `ray_trainer.py`). During rollout, actor weights stay on GPU but optimizer/grad are offloaded (`param_offload=true, optimizer_offload=true, grad_offload=true` in env yaml). vLLM reuses the actor GPU but runs its own kernels at `rollout.tensor_model_parallel_size=1`.

**Rollout batch shape** (per step):
- `train_batch_size=128` prompts
- `rollout.n=16` responses per prompt → 2048 rollout samples per step
- Distributed across 128 vLLM engines (one per GPU, since TP=1) → **16 rollout samples per GPU per step**
- Max response length 16384 tokens → **≤ 262K tokens per GPU per step** (worst case, 100% of responses hit max)

**Sequence budget**:
- `max_prompt_length=8000`
- `max_response_length=16384`
- `ppo_max_token_len_per_gpu=24384` (= 8000 + 16384; single-GPU workspace)
- `max_num_batched_tokens=24384`
- `free_cache_engine=true` → vLLM KV cache freed before actor update

## 2. Rollout callstack (where the time goes)

```
verl.trainer.main_ppo                              (main_ppo.py)
  └─ RayPPOTrainer.fit()                           (ray_trainer.py:1353+)
      └─ for step in range(total_training_steps):
          ├─ marked_timer("gen"):                   (ray_trainer.py:1444)
          │   └─ actor_rollout_wg.generate_sequences(batch)   (:1446)
          │       └─ ActorRolloutRefWorker.generate_sequences (engine_workers.py:358+)
          │           └─ vLLMAsyncRollout.generate_sequences (vllm_rollout.py:114+)
          │               └─ vLLM WorkerWrapperBase (ZMQ async)  — TP=1, DP=128
          ├─ marked_timer("reward"):                (:1518)
          │   └─ RewardLoopWorker → POST nemogym_server localhost:20001-7
          ├─ marked_timer("old_log_prob"):          (:1553)
          ├─ marked_timer("RefPolicy"):             (:1581)
          ├─ marked_timer("adv"):                   (:1591)
          ├─ marked_timer("update_actor"):          (:1650)
          │   └─ Megatron backward + optimizer step (PP=2, EP=8)
          └─ if step % test_freq == 0:
              marked_timer("testing"):              (:1666)
```

Timing emitted by `compute_timing_metrics` / `compute_throughout_metrics` in `metric_utils.py:230-303`. Known fields: `timing_s/{gen,reward,old_log_prob,adv,update_actor,step,testing}` + `perf/{total_num_tokens,time_per_step,throughput,mfu/actor}`.

**Observed in Stage 9 logs (32-node, max_resp=4096)**: `timing_s/step ≈ 200s`, of which `timing_s/gen ≈ 145-158s` (72-79% of step), `timing_s/update_actor ≈ 34-37s`, `timing_s/old_log_prob ≈ 14s`. **Generation dominates by ~4-5×.** At 16-node with `max_resp=16384` (4× longer responses) we expect gen to be much higher; step budget is likely 8-15 min.

## 3. Reference throughput floor (from lvbc/)

From `lvbc/throughput_benchmark_report.md` (40Bra/Moonlight-16B on 8×B300, DP=8/TP=1, inference-only):

| Workload | Per-GPU tok/s | Aggregate (8 GPU) |
|---|---|---|
| Uniform 4096-token decode, B=32 | **2,436** | 19,490 |
| Uniform 4096-token decode, B=55 | **3,951** | 31,609 |
| AIME25 variable-length (real eval) | ~950 peak, ~706 avg | 5,800 |

**Theoretical peak** for memory-bandwidth-bound decode: 2,783 tok/s/GPU (92 GB weights ÷ 8 TB/s HBM bw).

**Reference floor for "healthy"**: ≥ 1,500 tok/s/GPU on uniform workloads. Below that, investigate: batch too small, config wrong (TP>1 when model fits per-GPU), prefill fraction high, KV cache oversubscribed.

**Implication for our setup**: with 16 samples × 16384 tokens = 262K tokens per GPU per step, a healthy rollout would complete in `262000 / 1500 = ~175 s`. If we measure 300+ s we are ~50% underutilized and either (a) need to push more concurrency through each vLLM engine, or (b) should reduce node count so per-GPU load rises.

## 4. Best-practice concurrency lever (lvbc/)

`lvbc/vllm_throughput_deep_dive.md` — the key knob is **client-side concurrency** (how many async requests are in flight per engine):

| client parallel | per-GPU batch | per-GPU tok/s | regime |
|---|---|---|---|
| 128 | 16 | ~150 | under-utilized |
| 256 | 32 | 953 | near saturation |
| 512 | 64 | 971 | plateau |

Saturation at B≈32/GPU. Beyond that, per-GPU batch grows but throughput flattens.

In our verl setup with 16 rollouts/GPU/step, we are at **B=16 per engine** — exactly the under-utilized regime. Three levers:
1. **Raise n** (more responses per prompt): e.g. n=32 → B=32/GPU, target per-GPU saturation
2. **Lower DP** (fewer nodes): e.g. 8 nodes → 64 GPUs → B=32/GPU at same bsz/n
3. **Raise bsz**: e.g. bsz=256, n=16 → 32/GPU at same 128 GPUs

Lever 2 directly addresses the user's concern ("16 nodes feels wasteful"). Lever 1/3 keep wall-clock similar but may or may not be algorithmically desirable (GRPO gradient variance depends on n).

## 5. gym_rollout async orchestration pattern (applicable to verl?)

From `gym_rollout/generate_pass_rates.py`:
- **Flat asyncio submission**: all N prompts × M samples submitted as independent tasks (not a blocking for-loop)
- **Semaphore(256)** controls concurrency to avoid overwhelming vLLM
- **Single vLLM call with `n=M`**: vLLM amortizes prefill across M responses — much faster than M serial calls with n=1
- **Per-prompt checkpointing**: JSONL, resumable on crash

verl already uses `vLLMAsyncRollout` with ZMQ async — the mechanism is similar. What we should verify:
- Are prompts being submitted to each vLLM engine flat-async, or batched in lockstep?
- Is the reward HTTP call (`localhost:2000x`) blocking any rollout step?
- Does the RewardLoopWorker run concurrently with generation, or sequentially after it?

These are questions for the instrumentation pass (Section 6).

## 6. Instrumentation — evidence-based logs to add

Goal: produce a per-step JSONL file on PFS that makes it *obvious* whether rollout is memory-bound, compute-bound, or stuck on reward calls — without needing wandb.

### 6.1. Per-step summary (every step)
Write one JSON object per step to `${default_local_dir}/perf_log.jsonl`, with fields:

```json
{
  "step": 3,
  "wall_clock_utc": "2026-04-17T18:22:45Z",
  "timing_s": {"gen": 152.4, "reward": 3.1, "old_log_prob": 14.0,
               "ref": 11.2, "adv": 0.01, "update_actor": 35.2,
               "step": 215.9, "testing": null},
  "mem_gb": {
    "gpu_allocated_max_mean": 218.1,
    "gpu_allocated_max_max":  223.8,
    "gpu_reserved_max_mean":  223.6,
    "gpu_reserved_max_max":   228.1,
    "cpu_rss_mean": 474.3
  },
  "batch": {
    "prompts": 128, "responses_per_prompt": 16,
    "rollouts": 2048,
    "response_len_mean": 1453.4, "response_len_max": 16384,
    "response_len_clip_ratio": 0.137
  },
  "tokens_per_gpu_s_gen": 1812,         // compute: total_new_tokens / n_gpus / timing_s.gen
  "util_frac_of_floor": 1.21           // tokens_per_gpu_s_gen / 1500
}
```

### 6.2. Per-rank GPU snapshot (every step, first rank per DP replica)
One extra JSON object per DP replica head to the same file, with fields `rank_global`, `rank_in_dp_group`, `nvidia_smi_mem_used_mb`, `nvidia_smi_util_pct`, `torch_allocated`, `torch_reserved`. Captured via `torch.cuda.max_memory_allocated()` + `pynvml` (pynvml is already available in the image).

### 6.3. Rollout-specific trace (debug combo only)
For the 5-step debug combo, also record per-vLLM-engine counters inside `vLLMAsyncRollout.generate_sequences`:
- num_prompts_processed
- num_output_tokens_generated
- num_engine_seconds (gen call duration)
- peak_kv_cache_used_fraction (if exposed by vllm V1)

Write to `${default_local_dir}/rollout_trace.jsonl` with fields `{step, engine_id, ...}`.

### 6.4. Code touchpoints

| File | Change |
|---|---|
| `verl/verl/trainer/ppo/ray_trainer.py` (around `:1718`, after `compute_timing_metrics`) | Add PFS write of summary record |
| `verl/verl/trainer/ppo/metric_utils.py` | Add helper `_collect_mem_snapshot()` using `torch.cuda` + `psutil` |
| `verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py` (end of `generate_sequences`) | Append per-engine record (guarded by env var `VERL_ROLLOUT_TRACE=1`) |

All writes go to the **experiment checkpoint dir** on PFS, so they survive container death and are easily shipped to b32 for analysis.

All changes are additive and controlled by the env var (so production runs don't carry extra overhead).

## 7. Candidate reduced-node configs to evaluate

| Profile | Nodes | GPUs | Actor sharding | vLLM DP | Per-GPU rollouts | Expected regime |
|---|---|---|---|---|---|---|
| **P0** (current) | 16 | 128 | PP=2, EP=8, DP=8 | 128 | 16 | under-saturated (lvbc says B=32 saturates) |
| **P1** | 8 | 64 | PP=2, EP=8, DP=4 | 64 | 32 | **saturated (B=32)** — optimum per lvbc |
| **P2** | 4 | 32 | PP=2, EP=8, DP=2 | 32 | 64 | over-saturated (plateau regime; step time will rise) |
| **P3** | 8 | 64 | PP=1, EP=8, DP=8 | 64 | 32 | saturated, no pipeline bubbles — model must fit per GPU slice |

Notes:
- P3 assumes the 40Bra shards into EP=8, one layer stack per expert group, no PP. Single forward layer stack must fit in HBM (~92GB weights + activations during gen, ~172GB during backward). With B300 288GB and `param_offload=true` this is feasible for gen but likely tight for update_actor. P3 is a stretch candidate.
- P1 is the obvious first target.

Step-time prediction at P1 (8 nodes, B=32/GPU): if gen hits 2,400 tok/s/GPU, gen = 262000×2 / 2400 = ~218 s; update_actor likely scales linearly with batch per DP replica (2× larger → ~70 s). Total step ≈ 290-320 s — comparable to current 16-node step time but on half the hardware.

## 8. The debug combo: `cdbg5` (max 5 steps)

Add to `verl/my_scripts/k8s/config/combo_40Bra.yaml`:

```yaml
cdbg5:              # DEBUG ONLY — fast iteration for sharding/rollout experiments
  fapo_delta: 0.5
  ppo_epochs: 2
  total_training_steps: 5
  domains: nemogym_math
  curriculum_total_steps: 50   # small curriculum window since only 5 steps
  save_freq: 5                 # save once at end
```

Override at submit to shrink val too:
- `trainer.val_before_train=true` (keep baseline)
- `trainer.test_freq=3` (one mid-run val)
- `data.val_max_samples=32` (fast val; 1 val batch at val_batch_size=32)

With 5 steps at ~300 s/step + 2 val runs at ~5 min each = **~35 min total**. Iteration speed we need.

Use `cdbg5` for all subsequent node-count and rollout experiments. Never touch production combos to change step count.

## 9. Execution order

1. Write Plan A + Plan B (this file) ← we are here
2. Add `cdbg5` combo + debug env override (Section 8)
3. Add PFS instrumentation (Section 6.1-6.2 first; 6.3 last)
4. Run `cdbg5` at **P0** (current 16-node config) to establish baseline perf_log numbers
5. Run `cdbg5` at **P1** (8-node) — compare step time, gen throughput, update_actor; if gen tok/s/GPU ≥ 1,500 and step time within 30% of P0, call P1 viable
6. If P1 passes: run `cdbg5` at **P2** (4-node) to find the cliff
7. Document findings in `0418/dev_notes.md`
8. Propose production profile (likely P1 for most domains, P0 kept for math/code which have longest responses)

Keep each run's perf_log under `ckpts/40bra_k8s_single_domain/<exp_name>/perf_log.jsonl`.

## 10. Exit criteria

Plan B is satisfied when:
- [ ] `cdbg5` combo exists and runs end-to-end at P0 (≤ 60 min wall clock)
- [ ] `perf_log.jsonl` contains per-step timing + memory snapshots readable offline
- [ ] P1 (8-node) `cdbg5` run completes with gen tok/s/GPU ≥ 1,500 AND step time ≤ 1.3 × P0
- [ ] P2 (4-node) `cdbg5` run either completes (document as viable) or OOMs/times-out (document as cliff)
- [ ] A written recommendation in `0418/dev_notes.md`: for each domain (math/code/mcqa/if/structured/qy), which profile to use in production
- [ ] Stage 10 production yamls updated to use recommended profile (or documented to stay at P0 if evidence says so)

## 11. Out of scope (to avoid scope creep tonight)

- Changing PP/EP (we keep 2/8) — would require re-validating numerical correctness
- Switching rollout engine (keep vLLM V1 + CUTLASS_MLA)
- Any Gym-server architecture change — keep localhost:2000x model
- Any Megatron-Bridge / NVSHMEM fiddling — that class of bug is fully handled at Stage 9
- Multi-domain training — single-domain only for this test

---

**Next actions** (start immediately, do not wait):
1. Create `cdbg5` combo
2. Add PFS instrumentation
3. Clean up yaml/legacy README
4. Write verify_gang.sh
5. Submit first debug run at P0 to cluster
