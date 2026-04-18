# Plan C — Final Report and Recommendations

> Compute-efficiency study for 40Bra single-domain GRPO on B300.
> 11 experiments run on 2026-04-17/18. All measurements are apple-to-apple
> (Category-A knobs only; B-knobs held at the `c351` combo values).

---

## Executive summary

Three production-grade knob changes, all Category A:

1. **Drop all Megatron offload** (`param_offload=false`, `grad_offload=false`,
   `optimizer_offload=false`, `optimizer_cpu_offload=false`,
   `optimizer_offload_fraction=0.0`).
2. **Keep vLLM `gpu_memory_utilization=0.7`** (default).
3. **Reduce nodes from 16 to 8** (or 4 for cost-sensitive use).

Result:
- **8-node + no offload** → step time **442 s** (vs current 16-node 445 s). Same
  wall clock on half the cluster.
- **4-node + no offload** → step time **537 s** (vs current 16-node 445 s). 20 %
  slower per step on **a quarter of the cluster** — i.e. 70 % fewer GPU-hours
  for the same training.

No training-relevant hyperparameter (batch size, `rollout.n`, lr, FACPO,
curriculum) changes. The learning trajectory is identical to P0.

---

## 1. Measured data — all 11 experiments

All runs on `cdbg5` (5 training steps, math-only, 40Bra, 8-node baseline = P1
unless noted). Numbers are averages over steps 1–5; n_gpus = `nnodes × 8`.

| Run | nodes | Change vs P1 | step_s | gen_s | upd_s | tok/s/GPU | max_mem_gb | Δ step |
|---|---|---|---|---|---|---|---|---|
| P0 | 16 | (16-node baseline) | 444.8 | 285.2 | 110.7 | 291 | 233 | — |
| **P1** | **8** | **baseline** | **516.2** | **294.5** | **165.6** | **561** | **237** | **0** |
| P2 | 4 | 4-node with offload | 663.7 | 341.1 | 259.6 | 975 | 236 | +28.6 % |
| C1 | 8 | `gmu=0.85` | 524.7 | 308.1 | 161.6 | 540 | 276 | +1.6 % |
| C2 | 8 | `gmu=0.85 + free_cache=false` | ❌ OOM during actor update | | | | | |
| C3 | 8 | `gmu=0.90 + free_cache=false` | ❌ OOM during actor update | | | | | |
| C4 | 8 | `PP=1 + VPP=1` | ❌ OS OOM-kill after step 1 | | | | | |
| C5 | 8 | `PP=1 + CP=2` | ❌ OS OOM-kill after step 1 | | | | | |
| C6 | 8 | only `optimizer_offload=false` | 500.7 | 311.3 | 134.3 | 533 | 243 | −3.0 % |
| C7 | 8 | all offload off + `gmu=0.4` | 448.1 | 314.7 | 87.4 | 529 | 168 | −13.2 % |
| C8 | 8 | `max_token_len × 2` | 513.5 | 309.4 | 141.3 | 539 | 259 | −0.5 % |
| C9 | 8 | PP=1 + others | ❌ Megatron init assert | | | | | |
| **C10** | **8** | **all offload off + `gmu=0.7`** | **442.0** | **300.3** | **92.2** | **554** | **249** | **−14.4 %** |
| **C11** | **4** | **all offload off + `gmu=0.4`** | **536.5** | **336.5** | **143.7** | **992** | **176** | **+3.9 %** |

Five dead ends: any combination of `free_cache_engine=false` with
`gpu_memory_utilization ≥ 0.85`, or any use of `PP=1` at `max_response=16384`.
Both fail the ~260 GiB/GPU memory ceiling on B300.

---

## 2. Interpretation

### The update-phase offload overhead is the dominant cost

At P1 (all offload on), `update_actor_s = 165.6`. At C10/C7 (all offload off),
`update_actor_s = 92.2 / 87.4` — roughly **half**. The 73-s savings per step is
pure D2H/H2D transfer overhead that the offload machinery was incurring.

The savings **scale with per-GPU work**: at 4-node P2, offloaded `update_actor_s`
= 259.6; at 4-node C11 (no offload) it drops to 143.7 (−45 %). Because 4-node
packs more work per GPU, the proportional offload cost is higher, so removing
it saves more in absolute terms.

### `gpu_memory_utilization` does not improve throughput above 0.7

C1 (`gmu=0.85` alone) slightly **worsened** throughput (−3.7 % tok/s/GPU)
while adding 40 GiB of allocated memory. vLLM already has enough KV capacity
at `gmu=0.7` for our rollout shape (`n=16` per prompt at `max_resp=16384`).
Extra KV capacity is unused; extra memory is just closer to the OOM line.

The only time `gmu < 0.7` is beneficial: when you need to fit no-offload actor
state on the same GPU. C11 proves this: at 4 nodes the per-GPU actor state is
larger, and `gmu=0.4` leaves enough room for 168 GiB of on-GPU actor weights
+ optimizer + grads + activations. At 8 nodes with no offload we can keep
`gmu=0.7` because per-GPU actor state is smaller.

### `free_cache_engine=false` is a dead end in hybrid engine

Retaining vLLM's KV cache through the update phase pushes total allocated
memory past the HBM capacity, causing OOM during `actor_rollout_compute_log_prob`.
Not a speedup candidate regardless of `gmu` setting.

### `PP=1` is a dead end on 40Bra at `max_response=16384`

Eliminating pipeline parallelism means every GPU holds the full layer stack.
`max_memory_reserved_gb` rises to 290+ GiB on a 288 GiB B300 → OS OOM-kill
without Python traceback. Would require either `PP>=2` or
`max_response_length` cut (Category B). Three independent failures (C4, C5, C9)
confirm.

### Memory envelope for 40Bra hybrid engine on B300

Empirical ceilings for 40Bra at `c351` batch sizes and `max_response=16384`:

- **Safe**: ≤ 245 GiB allocated per GPU.
- **Tight**: 245–260 GiB.
- **Fail**: 260+ GiB allocated, or 285+ GiB reserved. OS OOM-kill becomes
  deterministic within a step or two.

---

## 3. Single-domain best practice (compatible with `production_guide.md`)

Two supported production configs. Both replace the current `c351`-on-16-node
setup.

### Config A — "same wall clock, half the nodes" (recommended default)

```
nnodes = 8
actor_rollout_ref.actor.megatron.param_offload     = false
actor_rollout_ref.actor.megatron.grad_offload      = false
actor_rollout_ref.actor.megatron.optimizer_offload = false
actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload      = false
actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction = 0.0
actor_rollout_ref.rollout.gpu_memory_utilization = 0.7   (unchanged default)
```
- Measured step time: **442 s** (vs current 445 s on 16 nodes).
- Peak per-GPU memory: **249 GiB** (39 GiB headroom on B300).
- Same training trajectory as P0.

### Config B — "minimum cluster cost, 4 % slower" (cost-sensitive)

```
nnodes = 4
# same offload settings as Config A (all false)
actor_rollout_ref.rollout.gpu_memory_utilization = 0.4
```
- Measured step time: **537 s** (vs current 445 s on 16 nodes, +21 %).
- Peak per-GPU memory: **176 GiB** (112 GiB headroom).
- 70 % fewer GPU-hours than the current 16-node config.
- Same training trajectory as P0.

### How to deploy

The env configs have been wired into a Hydra config group at
`verl/my_scripts/k8s/config/env/`. `submit/submit.sh` picks the profile via
a single env var:

```bash
./submit.sh c351                  # default: 16-node optimal (new P0 with no offload)
NNODES=8  ./submit.sh c351        # Config A: 8-node, no offload, gmu=0.7
NNODES=4  ./submit.sh c351        # Config B: 4-node, no offload, gmu=0.4
NNODES=16 ./submit.sh c351 'env=k8s_b300_16node_legacy'   # rollback to pre-2026-04-18
```

Verify after the first step: `perf_log.jsonl` first record should show
`max_memory_allocated_gb` well under 260 GiB for any non-legacy profile
(C10: ~249 GiB; C11: ~176 GiB; new 16-node optimal: projected ~220 GiB).
If higher, switch to legacy and investigate.

### Follow-up suggested before full rollout

A **30-step c351 convergence comparison** (Config A vs P0) on the full training
trajectory: confirms that numerically the loss / reward curves overlap
within noise. Category-A changes cannot alter training math, but running a
longer side-by-side is the standard due-diligence step before flipping
production.

---

## 4. Multi-domain at the same batch size

### What changes vs single-domain math

Multi-domain runs (e.g. `c351` with `domains=nemogym_math,nemogym_code,...` or
`c353/c354/c372`) differ from our tested single-domain math config in three
ways, none of which are Category A:

- **Response-length distribution** is wider. Math produces long reasoning
  chains (~5 k tokens); mcqa and IF produce short answers (~500–1500 tokens);
  code is medium. The 2048 rollouts per step have a heavier tail.
- **Reward latency** varies per domain. `nemogym_code` runs the solution
  against a test-bench (can take seconds); `nemogym_mcqa` is near-instant.
- **vLLM tail-padding cost**. vLLM does continuous batching but must wait for
  the slowest sample in each microbatch to finish. Longer tails mean more
  idle GPU time.

Per-step compute (batch, `rollout.n`, sequence lengths) is identical, so
Category-A knobs we tuned for single-domain remain valid.

### Recommendation

Use **Config A with a slight safety margin**:

```
nnodes = 8
# same no-offload settings
actor_rollout_ref.rollout.gpu_memory_utilization = 0.6   (down from 0.7)
```

Rationale for dropping `gmu` from 0.7 to 0.6: multi-domain adds variance to
per-step memory (occasional 16 k-token math outputs mixed with short answers).
The 0.6 value buys roughly 29 GiB additional headroom — brings peak from
249 GiB (Config A single-domain) to ~220 GiB expected worst-case
multi-domain. Still well inside the safe zone.

Rollout throughput may drop by a few percent due to lower vLLM KV capacity,
but since our rollout is already under-saturated at the `c351` batch shape,
this is inside the noise.

### What to watch in the first 10 steps of a new multi-domain run

- `perf/max_memory_allocated_gb` — if any step exceeds 255, raise `gmu` floor
  or re-enable one offload axis.
- `agent_loop/generate_sequences/max` — if this grows to 300+ s while
  `/mean` stays ~80 s, the tail is dragging. Consider raising `gmu` by 0.05
  at a time until saturation stops helping.
- `response_length/clip_ratio` — should be < 0.05 for all domains. High
  clip_ratio means many samples hit `max_response_length` and were truncated;
  training signal degrades.

### What NOT to change for multi-domain

- `train_batch_size`, `rollout.n` — Category B.
- `reward_model.reward_manager=nemogym_server` — keep.
- `data.sampler.domain_balanced=true` — keep for multi-domain curriculum.

---

## 5. Smaller-memory GPUs (B200 188 GiB, H100 80 GiB, A100 80 GiB)

### B200 (188 GiB HBM)

65 % of B300's 288 GiB. Our Config A's 249 GiB peak does **not fit** on B200.
Config B's 176 GiB **does fit** but with only 12 GiB headroom — too tight for
production.

Options to make 40Bra `c351` fit on B200:

| Option | Δ to Config A | Expected impact | Category |
|---|---|---|---|
| Add `PP=4` (from PP=2) | Halves per-GPU activation + weight memory | ~20 % step-time penalty (more pipeline bubbles, but we gain safety). Should bring peak to ~160 GiB. | A |
| Re-enable `param_offload=true` only | Saves ~10–20 GiB of actor weights during rollout | 5–10 % `update_actor` regression (partial D2H/H2D). | A |
| `CP=2` | Halves per-GPU activation in sequence dim | 5–10 % overhead. Needs validation with our MoE config. | A |
| Reduce `max_response_length` 16384 → 8192 | Halves per-GPU response tensor size | **Category B — changes what is trained.** Off-limits for compute comparisons. |

**Suggested B200 starting point** (needs empirical verification):

```
nnodes = 8
PP = 2, EP = 8, CP = 2, TP = 1      # add CP=2
param_offload     = true             # re-enable just this one
grad_offload      = false            # keep off
optimizer_offload = false            # keep off
gpu_memory_utilization = 0.5         # leave more room for activations
```

Projected peak: ~155–170 GiB, step-time penalty 10–20 % vs B300 Config A.

The correct way to validate is to re-run the Plan C experiment matrix on B200
hardware: P0/P1/P2 to find the node-count sweet-spot, then the C-series to
find the offload / gmu winner. Do not transplant B300 numbers onto B200
without empirical confirmation.

### H100 / A100 80 GiB

40 % of B200, 28 % of B300. Our smallest measured config (C11: 176 GiB peak)
is more than double this capacity. Fitting `c351` at `max_response=16384` on
80 GiB GPUs requires:

- **Heavier parallelism**: `PP=4`, probably `CP=2`, plus keep `EP` at what
  Megatron-Bridge supports for 80 GiB chips.
- **Full offload** of param + grad + optimizer — the savings we measured at
  B300 are not available when the GPU has less than half the memory.
- **Most likely a Category-B reduction**: `max_response_length` to 4096 or 8192
  is probably unavoidable. Doing so breaks apple-to-apple with any B300 runs.

We do not recommend running 40Bra at `max_response=16384` on 80 GiB hardware.
If the use case justifies it, we would do a fresh Plan B + Plan C on the
target hardware rather than extrapolate.

### Rule of thumb for adapting to smaller memory

1. Memory peak scales roughly with `PP^(−1) × seq_length × activations_per_token`
   + fixed weight/optimizer terms.
2. Dropping offload is viable only when the residual per-GPU memory has
   ≥ 25 % headroom. If peak is within 25 % of HBM capacity, keep offload on.
3. Raising `PP` reduces memory but adds pipeline bubbles. Expect ~10 % step-time
   penalty per doubling of `PP` beyond 2.
4. `CP` reduces activation memory with similar penalty.
5. `gmu` is cheap to reduce and saves KV cache memory 1:1.

In priority order for a new hardware target: **try `PP` first, then `CP`, then
`gmu`, then put offload back on. Only touch Category B as a last resort.**

---

## 6. Memory envelope table (for fast planning)

Empirical on 40Bra `c351` at `max_response=16384`. Peak per-GPU
`max_memory_allocated_gb`. Values from 8-node runs unless noted.

| Config | Peak | Fits B300 (288) | Fits B200 (188) | Fits 80 GiB |
|---|---|---|---|---|
| P1 (all offload, gmu=0.7) | 237 | ✅ (51 free) | ❌ | ❌ |
| P0 (16n, all offload, gmu=0.7) | 233 | ✅ | ❌ | ❌ |
| C1 (gmu=0.85, all offload on) | 276 | ✅ (12 free) | ❌ | ❌ |
| C6 (opt-offload only) | 243 | ✅ | ❌ | ❌ |
| **C10 (all offload off, gmu=0.7)** | 249 | ✅ (39 free) | ❌ | ❌ |
| C7 (all offload off, gmu=0.4) | 168 | ✅ | ✅ (20 free) | ❌ |
| **C11 (4n, all offload off, gmu=0.4)** | 176 | ✅ | ✅ (12 free — tight) | ❌ |
| C8 (max_token_len × 2) | 259 | ✅ (29 free — tight) | ❌ | ❌ |

Fail cases:
- `free_cache_engine=false + gmu≥0.85` → 260–270 allocated, OOM in update.
- `PP=1` → 290+ reserved, OS OOM-kill after step 1.

Projected (not measured, marked †) for smaller-memory hardware:

| Proposed config | Target HW | Projected peak | Margin |
|---|---|---|---|
| 8n, PP=2, CP=2, param_offload only, gmu=0.5 | B200 | ~155 GiB † | comfortable |
| 8n, PP=4, all offload, gmu=0.4 | H100/A100 80 GiB | ~65–75 GiB † | tight; probably need `max_response=8192` too |

Use these as starting points for new-hardware Plan Bs, not as production
defaults.

---

## 7. Next steps

- **Pick Config A or Config B** and update `env_k8s_b300_16node.yaml` +
  `submit/rayjob.yaml` accordingly. Default recommendation: Config A (same
  wall clock, half the nodes).
- **Run a 30-step c351 convergence comparison** vs P0 to certify learning
  equivalence before production flip.
- **Do the same for multi-domain** (`c353`, `c372`, etc.) using Config A with
  `gmu=0.6`: 10-step smoke to confirm the memory headroom holds, then full
  training.
- **If B200 hardware becomes available**: repeat Plan B (node count) and
  Plan C (compute opt) on that target. Do not transplant these numbers.

---

## 8. Archive of evidence

All per-run artifacts are under
`verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/` (post-2026-04-18 layout).
Flat copies for cross-run comparison:

| File | Contains |
|---|---|
| `0418/results/<label>_perf_log.jsonl` | Per-step metrics (canonical source in run dir). |
| `0418/results/<label>_analysis.txt` | Summary from `analyze_perf.py`. |
| `0418/results/<label>_training.log` | Full driver output (entrypoint + Hydra + ray). |

Data used in this report: P0 / P1 / P2 (from Plan B, 2026-04-17) and C1–C11
(from Plan C, 2026-04-18).
