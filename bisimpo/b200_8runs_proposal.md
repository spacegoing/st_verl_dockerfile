# Proposal — 8 additional BSPO single-domain runs for the B200 cluster

This document describes 8 experiments to run on the 64-node B200 cluster,
aligned with and extending the ablation already completed on the 31-node
B300 dev cluster.

## Context — what the B300 ablation proved

All B300 runs: nemogym_math single-domain, 120 steps, **4-node** profile
(PP=2/EP=8/TP=1, no offload, gmu=0.4), `train_batch_size=128`,
`ppo_mini_batch_size=128`, `ppo_epochs=2`, `rollout.n=16`, curriculum
`initial_mean=0.8 → final_mean=0.2` over 300 steps, hard-eval set =
BeyondAIME + AIME2025 + OlympiadBench.

**B200 runs below**: same semantics (same train/val data, same
`train_batch_size=128`, same `ppo_mini_batch_size=128`, same
`rollout.n=16`, same 120 steps, same curriculum), but at **16-node**
compute profile (larger DP, proportionally faster wall-clock). The
per-sample hyperparameters above should transfer — only the parallelism
layout and per-GPU micro batch sizes change.

Completed sweeps (Phase-2 + Phase-3):

```
combo      variant            δ       λ_tj    outcome @ step 120
---------  -----------------  ------  ------  -------------------
cbsp201    V1 simplest        1e-3    1e-3    peak 0.77 @ step 100, crashed to 0.00
cbsp202    V1 simplest        1e-2    1e-3    similar peak-then-crash
cbsp203    V1 simplest        3e-4    1e-2    peak AIME2025=0.83 @ step 80 (+0.16), crashed @ 90
cbsp301    V4 w_penalty       1e-3    1e-3    slow collapse
cbsp302    V4 w_penalty       3e-4    1e-2    slow collapse (BDA 0.33 → 0.09)
cbsp303    V4 w_penalty       3e-4    1e-1    late collapse (BDA 0.53 → 0.15)
cbsp304    V4 w_penalty       3e-4    3e-2    late collapse (BDA 0.53 → 0.20)
cbsp401    V5 w_penalty_only  3e-4    1e-2    STABLE-flat, no peak, no crash
cbsp402    V5 w_penalty_only  1e-3    1e-2    moderate drift
cbsp403    V5 w_penalty_only  3e-4    1e-1    moderate drift
cbsp404    V5 w_penalty_only  3e-4    3e-2    moderate drift
cbsp406    V5 w_penalty_only  3e-4    1e-2    STABLE-flat (reproduces 401)
```

### Headline reading

1. **V1 simplest is the only variant that shows a real UPWARD val signal.**
   cbsp203 @ step 80 hit AIME2025=0.83 (baseline 0.67). V1 learns *fast*
   but collapses *hard* in the last 20–30 steps.
2. **V5 w_penalty_only at δ=3e-4 λ=1e-2 is the only "stable" config**, but
   stable means flat-at-baseline — it never exceeds the starting val.
3. **V4 w_penalty is strictly dominated** — it collapses like V1 but
   without V1's intermediate peak.
4. The real research frontier is **"can we keep V1's peak without the
   late crash?"** — not "can we find a better V5 λ?".

## Proposed 8 runs

All V1 / V5 / V4 flavours below use the same base config as the B300 sd
ablation: same training data, same hard-eval val set, same parallelism
layout, 120 steps (unless explicitly overridden), `save_freq=9999`
(metrics only; no ckpt writes). The 8 runs form **2 calibrations + 3
V1 peak-sustain probes + a matched KL-anchor row across all three
variants**.

All 8 runs use `total_training_steps=120` (uniform with the B300 sweep).

| #  | combo_id | variant            | δ    | λ_tj | extra Hydra override                                                                                      | purpose                                                          |
|----|----------|--------------------|------|------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 1  | cbsp206  | simplest (V1)      | 3e-4 | 1e-2 | —                                                                                                         | **calibration** — reproduce B300 cbsp203 on B200 hardware        |
| 2  | cbsp410  | w_penalty_only(V5) | 3e-4 | 1e-2 | —                                                                                                         | **calibration** — reproduce B300 cbsp401/406 on B200             |
| 3  | cbsp207  | simplest (V1)      | 3e-4 | 1e-2 | `actor_rollout_ref.actor.clip_ratio_low=0.1 actor_rollout_ref.actor.clip_ratio_high=0.15`                 | **V1 + tighter PPO trust region** — smaller per-step policy move, delay collapse |
| 4  | cbsp208  | simplest (V1)      | 3e-4 | 1e-2 | `actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2`                      | **V1 + KL anchor to reference** — does anchoring prevent drift?  |
| 5  | cbsp209  | simplest (V1)      | 3e-4 | 1e-2 | `actor_rollout_ref.actor.entropy_coeff=1e-3`                                                              | **V1 + entropy reg** — prevent deterministic collapse            |
| 6  | cbsp210  | simplest (V1)      | 3e-4 | 1e-2 | `actor_rollout_ref.actor.optim.lr=5e-7`                                                                   | **V1 + half lr** — slower evolution, test if collapse is lr-driven |
| 7  | cbsp411  | w_penalty_only(V5) | 3e-4 | 1e-2 | `actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2`                      | V5 + KL anchor — does KL help V5 actually *gain* instead of stay flat? |
| 8  | cbsp310  | w_penalty (V4)     | 3e-4 | 1e-2 | `actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2`                      | V4 + KL anchor — completes the variant × KL row                  |

### Hypotheses being tested

- **#1, #2** are cross-cluster anchors. Without these, any comparison
  between B200 and B300 numbers is suspect.
- **#3, #4, #5, #6** are four independent knobs against V1's late
  collapse. Orthogonal hypotheses:
  - #3 (tighter PPO clip): the collapse is large single-step policy moves; shrinking the clip ratio enforces smaller safe updates
  - #4 (KL): the collapse is drift from reference; KL anchor pulls it back
  - #5 (entropy): the collapse is policy over-determinism; entropy reg preserves exploration
  - #6 (half lr): the collapse is late-stage lr being too hot
- **#4, #7, #8** form a matched KL-anchor row across V1/V5/V4 — the
  cleanest way to ask "does a reference anchor help, and for whom?".

### What a successful outcome looks like

- **Best case**: one of #3/#4/#5/#6 keeps the V1 curve at AIME2025 ≥ 0.75
  through step 120. That becomes the recommended training recipe.
- **Mid case**: tighter clip / half lr (#3, #6) shift the peak later and
  soften the collapse but the model still degrades by step 120. Then
  the recommendation becomes "run V1 with the best regulariser and take
  the step-80 ckpt" — this is why we need `save_freq=10` when deploying
  (not for these probe runs, which use the B300 `save_freq=9999` to keep
  disk clean, but so we can post-hoc re-train the winner and snapshot it).
- **Null case**: all 4 V1 probes fail → V1's collapse is intrinsic to
  Δ^(reg) saturation dynamics. V5 + KL (#7) becomes the fallback winner.

## Concrete things to edit on the B200 cluster

Assuming the B200 cluster clones `verl` at branch `nemo_bspo_md`, the
changes are:

1. **`verl/my_scripts/k8s/config/combo_40Bra.yaml`** — append 8 combo
   entries. Skeleton (copy cbsp401's shape for each):
   ```yaml
   cbsp206:                # V1 simplest, δ=3e-4 λ=1e-2 — B200 calibration of cbsp203
     fapo_delta: 0.5        # unused
     ppo_epochs: 2
     total_training_steps: 120
     domains: nemogym_math
     curriculum_total_steps: 300
     save_freq: 9999

   cbsp207:                # V1 + tighter PPO clip (clip_ratio 0.1/0.15 via CLI)
     fapo_delta: 0.5
     ppo_epochs: 2
     total_training_steps: 120
     domains: nemogym_math
     curriculum_total_steps: 300
     save_freq: 9999

   cbsp208:                # V1 + KL anchor
     fapo_delta: 0.5
     ppo_epochs: 2
     total_training_steps: 120
     domains: nemogym_math
     curriculum_total_steps: 300
     save_freq: 9999

   cbsp209:                # V1 + entropy reg
     # ... same skeleton as cbsp208

   cbsp210:                # V1 + half lr
     # ... same skeleton

   cbsp310:                # V4 + KL anchor
     # ... same skeleton

   cbsp410:                # V5 calibration of cbsp401/406
     # ... same skeleton

   cbsp411:                # V5 + KL anchor
     # ... same skeleton
   ```

2. **`verl/my_scripts/bspo_scripts/submit_bspo.sh`** — extend the `case`
   statement:
   ```bash
   cbsp201|cbsp202|cbsp203|cbsp204|cbsp205|cbsp206|cbsp207|cbsp208|cbsp209|cbsp210)
       VAR=simplest;       DELTA=3.0e-4; LAMBDA=1.0e-2; LR_WARMUP=10 ;;
   cbsp301|cbsp302|…|cbsp310)
       VAR=w_penalty;      DELTA=3.0e-4; LAMBDA=1.0e-2; LR_WARMUP=10 ;;
   cbsp401|cbsp402|…|cbsp411)
       VAR=w_penalty_only; DELTA=3.0e-4; LAMBDA=1.0e-2; LR_WARMUP=10 ;;
   ```

3. **Cluster-specific changes for B200** — already listed in
   `verl/my_scripts/bspo_scripts/README.md` §"Porting to another cluster".
   Pay special attention to:
   - image + `imagePullSecrets.name` in `rayjob.yaml`
   - network env (NIC list, GID index) in `rayjob.yaml`
   - `rdma-training/roce` resource name (B200 may differ)
   - PVC `claimName` + `subPath`s
   - Add **B200 16-node** env config (e.g. `k8s_b200_16node.yaml`) under
     `verl/my_scripts/k8s/config/env/` and update the `NNODES=16` branch
     of the case statement in `submit.sh` / `submit_md.sh` to point at
     the new filename. The 16-node layout on B200 will differ from B300
     (more GPUs → larger DP → tune per-GPU micro batch accordingly;
     keep `ppo_mini_batch_size=128` fixed so global batch is identical).

## Submit commands (one shell line per run)

All submits assume B200-adapted `submit.sh` + new env `k8s_b200_16node`
(or equivalent) resolves when NNODES=16:

```bash
cd verl/my_scripts/bspo_scripts

# 1. cbsp206 — V1 calibration (no extra overrides; dispatcher default λ=1e-2)
NNODES=16 ./submit_bspo.sh cbsp206

# 2. cbsp410 — V5 calibration
NNODES=16 ./submit_bspo.sh cbsp410

# 3. cbsp207 — V1 early stop (combo yaml already sets total_training_steps=80)
NNODES=16 ./submit_bspo.sh cbsp207

# 4. cbsp208 — V1 + KL anchor
NNODES=16 ./submit_bspo.sh cbsp208 \
    'actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2'

# 5. cbsp209 — V1 + entropy reg
NNODES=16 ./submit_bspo.sh cbsp209 \
    'actor_rollout_ref.actor.entropy_coeff=1e-3'

# 6. cbsp210 — V1 + half lr
NNODES=16 ./submit_bspo.sh cbsp210 \
    'actor_rollout_ref.actor.optim.lr=5e-7'

# 7. cbsp411 — V5 + KL anchor
NNODES=16 ./submit_bspo.sh cbsp411 \
    'actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2'

# 8. cbsp310 — V4 + KL anchor
NNODES=16 ./submit_bspo.sh cbsp310 \
    'actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=1e-2'
```

## Node budget

64 nodes at **16-node** per run = **4 concurrent slots**. The 8 runs
therefore fit as **two back-to-back waves** of 4:

- **Wave 1** (submit first, blocks 4 slots): #1 cbsp206, #2 cbsp410,
  #3 cbsp207, #4 cbsp208 — gives the two calibrations + two V1 probes
  running in parallel. Finish together (same step count, similar
  wall-clock at 16 nodes each).
- **Wave 2** (submit after Wave 1 finishes, reuses the same 4 slots):
  #5 cbsp209, #6 cbsp210, #7 cbsp411, #8 cbsp310 — remaining V1 probes
  + V5 KL + V4 KL.

Volcano will gang-schedule each wave atomically. The other half of the
cluster (32 nodes) is free for any unrelated workload during both waves.

Per-run hyperparameters above don't change with `nnodes`; only the
compute profile file does (global batch `train_batch_size=128` held
fixed; DP grows with nnodes; `ppo_micro_batch_size_per_gpu` and
`ppo_max_token_len_per_gpu` get re-tuned to fit B200 memory).

## Reference — B300 runs for cross-cluster comparison

The calibration runs (#1, #2) should reproduce these exact numbers from
the B300 cluster:

### cbsp203 (V1 simplest, δ=3e-4, λ=1e-2) on B300

```
val @ step   BeyondAIME   AIME2025   OlympiadBench
----------   ----------   --------   -------------
        0        0.52         0.67       0.69
       10        0.51         0.70       0.65
       20        0.55         0.67       0.71
       30        0.50         0.70       0.71
       40        0.53         0.73       0.70
       50        0.52         0.80       0.74
       60        0.57         0.73       0.68
       70        0.58         0.67       0.70
       80        0.50         0.83       0.67   ← AIME peak
       90        0.16         0.33       0.54   ← collapse
```

### cbsp401 / cbsp406 (V5 w_penalty_only, δ=3e-4, λ=1e-2) on B300

```
val @ step   BeyondAIME   AIME2025   OlympiadBench
----------   ----------   --------   -------------
        0        0.47         0.70       0.68
      110        0.41         0.47       0.63
      120        0.36         0.53       0.65
```

If B200 cbsp206 at step 80 ≠ ~0.50 / ~0.83 / ~0.67 and cbsp410 at step
120 ≠ ~0.36 / ~0.53 / ~0.65, something is wrong with the cross-cluster
setup before trusting any of the probe-run (#3–#8) outputs.
