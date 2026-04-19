# MD-S6 — Multi-domain debug smoke

**Goal.** End-to-end validation of the multi-domain code path landed by
S1 + S2 + S3: one 1-step run with `BSPO_DEBUG=1`, 4-node C11 profile,
tiny eval (`val_max_samples=16`), surfacing any errors quickly.

## Submission

```
bash bisimpo/submit_bspo_md.sh cdbgmd1
→ [submit_md] combo=cdbgmd1 nnodes=4 env=k8s_b300_4node_c11 debug=1
              name=bra40-md-cdbgmd1-4n-dscmv
```

Combo: `cdbgmd1` (V1 simplest, 1 step, `domains: all`, 4-node).
Hyperparameters: default δ=3e-4, λ_Tj=1e-3, λ_Grp=0, λ_D=0.
Image: `myverl:ncr2602_vllm012.dev` with the S2+S3 code baked in via
PFS editable install (no image rebuild needed).

## Cluster state at submission

All 28 nodes occupied by single-domain runs (3 Phase-1 running 13h,
5 Phase-2 preview running ~20min). Volcano will queue `cdbgmd1` until
a single-domain run ends. Phase-1 v2 is ~7h from completion, so
cdbgmd1 is expected to run ~8h after submission.

## Expected behaviour on successful smoke

1. RayJob transitions Initializing → Running → SUCCEEDED.
2. Entrypoint `run_40bra_k8s_multi_domain.sh` starts Gym on head +
   all 3 worker nodes via Ray remote.
3. `BSPO_DEBUG=1` appends the debug override pack at Hydra level:
   `trainer.total_training_steps=1`, `val_max_samples=16`,
   `train_batch_size=8`, `test_freq=1`.
4. Val-before-train runs on the 16-row subset (sampled from the 230-row
   hard parquet). Expect 3 `val-core/<src>/acc/*` keys.
5. 1 training step executes. `compute_policy_loss_bspo` runs with
   `loss_mode=bspo`, `variant=simplest`. No hierarchical penalty active
   (λ_Grp=λ_D=0), so the loss path is identical to single-domain V1.
6. Post-train val runs.
7. `n_groups > 0` logged in `actor/bspo_n_groups`, `n_domains ≥ 1`
   logged in `actor/bspo_n_domains` — confirms the megatron prep step
   delivered `uid` / `data_source` tensors through `rearrange_micro_batches`.
8. Run ends; run_dir populated at
   `verl/ckpts/40bra_k8s_multi_domain/bra40-md-cdbgmd1-4n-dscmv/` with
   `run.log`, `exp_name.txt`, `rayjob.txt`, `bspo_debug.txt`.

## Failure-mode triage (if smoke fails)

| symptom | likely cause | where to look |
|---|---|---|
| RayJob stuck Initializing > 10min | cluster capacity or image pull | `kubectl describe rayjob bra40-md-cdbgmd1-4n-dscmv` |
| Training fails at `assert isinstance(config, ActorConfig)` | OmegaConf vs dataclass mismatch | S2 compute_policy_loss_bspo top |
| `KeyError: 'uid'` in loss_func (megatron) | prep step not entered (loss_mode wrong) | `megatron_actor.py:382-400` |
| `scatter_add_` shape mismatch | uid tensor shape != (B,) | verify `np.unique` inverse shape |
| `n_groups == 1` when batch has > 1 prompt | uid not varying per micro-batch | check `rearrange_micro_batches` index ordering |
| `KeyError: 'data_source'` in reward manager | 40bra_16node_md.yaml server_urls typo | S1 md.yaml server_urls block |
| Very slow Gym startup (>10min) | worker-node gym script fails | `kubectl logs <worker> -c ray-worker` |

## Next action after smoke completes

- **If SUCCEEDED**: proceed to S7 — launch `cbsp501` (V1 multi-domain
  formal). If that launches cleanly, follow with `cbsp502` (V2),
  `cbsp503` (V3), `cbsp504` (V4), `cbsp505` (V5).
- **If FAILED**: root-cause via the triage table; patch with narrowest
  fix; re-submit another `cdbgmd{N}` combo (not re-using the same
  name so k8s history is clear).

## Dev notes / execution log

**2026-04-19 08:38 UTC** — cdbgmd1 submitted; currently queued. Legacy
name per pre-rename scheme.

**2026-04-19 13:00 UTC** — **rename + 2-node debug switch**.
- Legacy `cdbgmd1` rayjob (queued, 4-node) deleted.
- Resubmitted as **`cbdg-md-v1-smoke`** under the new naming scheme,
  on **NNODES=2** (new env `k8s_b300_2node_debug.yaml`,
  minimum-viable-shape for 40Bra-16B: PP=2·EP=8·TP=1 = 16 GPU).
  RayJob: `bra40-md-cbdg-md-v1-smoke-2n-swb9b`.
- submit_bspo_md.sh auto-sets NNODES=2 + BSPO_DEBUG=1 for any combo
  whose id starts with `cbdg-`.
- cbdg-md-v1-smoke is queued behind the 7 single-domain formal runs
  saturating the cluster. Starts when one of those ends (~2h).

Will update here with smoke outcome when it lands.
