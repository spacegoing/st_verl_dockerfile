# BSPO Dev Notes — Stage: Dev & Debug (1-step smoke)

Chronological record of the smoke-test cycles for the new BSPO loss.

---

## 16:41 UTC — smoke v1 submitted: `bra40-sd-cdbgbspo-4n-4bgpn`

**Result: FAILED at ~2 min.**

Root cause: `omegaconf.errors.InterpolationKeyError: Interpolation key 'combo_bank.cdbgbspo.ppo_epochs' not found`.

The main yaml `40bra_16node_sd.yaml` interpolates `${combo_bank.${combo_id}.ppo_epochs}`. My first edit of `combo_40Bra.yaml` added combos with names containing underscore (`cdbg_bspo`, `cbsp_101`, etc.), but then I attempted to rename them to underscoreless form (`cdbgbspo`, `cbsp101`). A subsequent Edit reported "String to replace not found" because of small whitespace differences from a prior file modification — so the yaml still had `cdbg_bspo` while the submit script looked up `cdbgbspo`.

**Fix:** `Edit` with `replace_all=true` for each combo name individually. Verified via `yaml.safe_load` that all 4 target keys (`cdbgbspo`, `cbsp101`, `cbsp103`, `cbsp104`) now exist.

## 16:47 UTC — smoke v2 submitted: `bra40-sd-cdbgbspo-4n-6r9bs`

**Result: FAILED at ~6 min.**

Root cause: Megatron `assert lr_warmup_steps < lr_decay_steps` in `OptimizerParamScheduler.__init__`. My `submit_bspo.sh` hard-coded `lr_warmup_steps=10` but `cdbgbspo` uses `total_training_steps=1` → `lr_decay_steps=1`, so 10 < 1 fails.

**Fix:** made `LR_WARMUP` per-combo inside `submit_bspo.sh`:
- `cdbgbspo` (1 step) → `LR_WARMUP=0`
- `cbsp101 / cbsp103 / cbsp104` (120 steps) → `LR_WARMUP=10`

## 16:56 UTC — smoke v3 submitted: `bra40-sd-cdbgbspo-4n-jd6nd`

**Result: SUCCEEDED at 17:29:57 UTC (~33 min wall clock).**

- Val step 0: `val-core/nemogym_math/acc/mean@1=0.7` (128 samples).
- Step 1: `actor/pg_loss=0.0`, `actor/grad_norm=0.091`, all `actor/bspo_*` metrics logged (all 0 as expected — see note).
- `timing_s/step=555.86`, `timing_s/save_checkpoint=381.79` (dominant).
- perf_log / ckpt / wandb all produced; offline wandb written to run dir.

**Note on step-1 zeros (expected):** With `ppo_epochs=1`, the first micro-batch
sees `log_prob ≈ old_log_prob` exactly (same weights, same inputs on the first
forward after sampling) → `ratio ≈ 1` → `s+ = s- = 0` → `Δ_reg = Δ_w = 0` →
`bspo_per_traj = 0`. Grad_norm=0.091 comes from entropy bonus, not BSPO. In
Phase-1 runs `ppo_epochs=2`, so the second epoch sees a policy that's absorbed
a mini-batch of updates → non-trivial BSPO signal.

## 17:30 UTC — Phase-1 ablation submitted (3 jobs, concurrent on 12 nodes)

- `bra40-sd-cbsp101-4n-q5xxv` — variant=simplest (Obj 1)
- `bra40-sd-cbsp103-4n-tjxg7` — variant=w_penalty (Obj 3)
- `bra40-sd-cbsp104-4n-x567v` — variant=w_penalty_only (Obj 4)

All 3 at default hparams (δ=3e-4, λ=1e-3), 120 steps, 4-node C11, `ppo_epochs=2`.
Expected wall clock per run: ~18 h. test_freq=10 → 12 val points per run.

## 17:25 UTC — orphan PodGroup cleanup

Observed two Inqueue PodGroups from the earlier NNODES-switch smoke runs
(`ray-bra40-sd-c351-4n-ldmcw-pg`, `ray-bra40-sd-c351-8n-5f9bm-pg`). The
owning RayJobs had already reached SUCCEEDED state (at 16:13 and 16:19
UTC) and all their pods were gone (`shutdownAfterJobFinishes: true`,
`ttlSecondsAfterFinished: 120` already elapsed). But the PodGroups stayed
Inqueue with the event spam:

```
Warning  Unschedulable  2m9s (x2068 over 36m)  volcano
   0/1 tasks in gang unschedulable: pod group is not ready,
   1 Succeeded, 4 minAvailable
```

The Volcano scheduler keeps evaluating the gang (since PodGroup.status.phase
!= terminal) and logs Unschedulable every 2 seconds. Zero capacity impact
(no pods) but lots of scheduler log noise.

**Root cause**: Volcano PodGroup GC has a narrow window tied to the
associated pods' lifetime. When the RayCluster tears down faster than
Volcano's GC loop expects, PodGroups leak. Not new — same as the Apr 17
voltest cleanup.

**Fix**: delete the owning RayJob → PodGroup cascade-deletes through
`ownerReferences`. Also delete orphan submitter pods that were left
behind.

Outcome: cluster state clean. Only `cdbgbspo` PodGroup remains, still
Running normally.

---

## Notes

- The `cdbgbspo` run dir is self-contained on PFS at
  `verl/ckpts/40bra_k8s_single_domain/<rayjob-name>/` (0418 convention).
  If v3 succeeds, its `perf_log.jsonl` should contain a single record
  with `actor/bspo_*` keys in the metrics payload.
- Two older production-test jobs (`bra40-sd-c351-8n-5f9bm` and
  `bra40-sd-c351-4n-ldmcw`) are still running from the NNODES-switch
  validation earlier today. They don't block BSPO runs; they just
  occupy 12 of ~28 compute nodes.
