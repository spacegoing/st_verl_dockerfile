# Agent guide — BSPO + Qwen3 port on st_verl_dockerfile

For any Claude agent opening this repo fresh. Read this first; then
`CLAUDE.md` (same dir) for Docker / cluster lifecycle; then the per-subdir
plan + dev_notes docs as needed.

## What's in this repo

verl-based RL training for two models, on one 31-node B300 k8s cluster:

1. **40Bra** (Moonlight 16B MoE with DeepSeek-style MLA) — the incumbent.
   Single-domain (math) and multi-domain (nemogym_blend) BSPO training.
2. **Qwen3-30B-A3B-Base** (30B MoE, 3B active, GQA not MLA) — recently
   ported. 1-step smoke green; longer runs untested.

Work organized as two parallel efforts:
- **BSPO** (Bi-Simulation Policy Optimization): a new policy loss. Single-
  domain stable variant found (V5 `w_penalty_only`, δ=3e-4, λ=1e-2 in
  cbsp401). Multi-domain (bspo_md) implemented and smoke-tested on 40Bra
  with 5 variants; formal sweep killed in favour of focusing on sd
  ablation first.
- **Qwen3 port**: reuse the same sd BSPO infra with a non-MLA, non-40Bra
  model as a forward-compatibility check. Smoke green with caveats.

---

## Current cluster snapshot (2026-04-20 evening)

**18-run sd BSPO ablation live, 28/31 nodes occupied.** 7 RUNNING + 11
Volcano-gang-queued. All target 120 steps on nemogym_math. ETA ~30 h from
submission. Check with:

```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob -l training-type=40bra-sd
```

Every job's metrics are under
`verl/ckpts/40bra_k8s_single_domain/<rayjob-name>/run.log` on PFS.

---

## Branch topology

Three git repos, one branch state per concern.

| repo                                      | branch                    | commit   | state                                                                 |
|-------------------------------------------|---------------------------|----------|-----------------------------------------------------------------------|
| main (`st_verl_dockerfile`)               | `nemo_bspo_md_dev`        | per-stage| staged Stage 1/2/3 commits + debug scripts + plan docs                |
| main                                       | `nemo_bspo_md`            | squashed | production: BSPO-md single commit + Qwen3 port stacked on top         |
| verl (submodule, `verl/`)                 | `nemo_bspo_md_dev`        | per-stage| md dev branch                                                         |
| verl                                       | `nemo_bspo_md`            | squashed | prod md                                                               |
| verl                                       | `nemo_bspo_sd_ablation`   | live     | 7dc6b1fa + yaml-only combos for Phase-3 sd + Qwen3 Hydra config       |
| iter_kuberay (`iter_kuberay_.../`)        | `nemo_bspo_md_dev`        | per-stage| md dev                                                                |
| iter_kuberay                               | `nemo_bspo_md`            | squashed | prod md + rayjob_qwen3.yaml                                           |

When you're about to submit a new sd ablation: verl should be on
`nemo_bspo_sd_ablation` (ensures clean 7dc6b1fa bspo loss). For Qwen3
runs: same branch works (the Qwen3 Hydra config was committed there).

When running a new md ablation: verl needs `nemo_bspo_md` (for the
bspo_md loss code).

---

## File map

### Top-level

```
CLAUDE.md                          ← Docker / container / proxy guide (always loaded)
AGENT_GUIDE.md                     ← this file
bisimpo/                           ← BSPO dev docs + outer submit wrappers
qwen3_port/                        ← Qwen3 port dev docs + submit wrapper
iter_kuberay_32nodes_verl_training/  ← kuberay YAML templates + dispatchers
verl/                              ← verl code + Hydra configs + entrypoints (submodule)
```

### `bisimpo/` — BSPO everything

```
bispo_algorithm.tex                ← THE MATH SPEC. §Multi-domain BSPO defines sd + md
bisim_eqs_annotated.tex            ← sd bspo derivation
bisim_eqs_canonical_decomposition.tex  ← MEAN-at-every-level justification
bisim_eqs_multidomain.tex
bisim_eqs_recovery.tex
bisim_eqs_sign_analysis.tex

dev_notes_mdplan_v2_replan.md      ← THE md re-plan doc. Stages / hparam justification / open questions
reading_list_bspo_md_correctness.md  ← tier-ordered correctness review of bspo_md
dev_notes_planning.md              ← earlier bspo planning
dev_notes_implementation.md        ← sd bspo loss implementation notes
dev_notes_dev_debug.md             ← sd bspo debug log
dev_notes_eval_upgrade.md          ← BSPO hard-eval parquet creation
dev_notes_ablation.md              ← Phase-1 ablation results

submit_bspo.sh                     ← sd bspo launcher (combo → δ/λ/var + delegate to kuberay)
submit_bspo_md.sh                  ← md bspo launcher
submit_formal_md.sh                ← one-shot launcher for cbmd-v{1..5}
test_stage1_infra.sh               ← md Stage 1 live smoke (loss_mode=bspo)
test_stage2_plumbing.sh            ← md Stage 2 kwargs plumbing smoke
test_stage3_loss.sh                ← md Stage 3 loss + V1/V2/V3 smokes
squash_to_prod.sh                  ← dev → prod branch automation
```

### `qwen3_port/`

```
plan.md                            ← porting plan (stages + diffs from 40Bra + failure modes)
dev_notes.md                       ← stage-by-stage progress + workarounds + untested items
submit_qwen3.sh                    ← top wrapper (always NNODES=2, 16 GPU)
```

### `iter_kuberay_32nodes_verl_training/submit/`

```
rayjob.yaml            ← sd 40bra RayJob template (uses CUTLASS_MLA)
rayjob_md.yaml         ← md 40bra RayJob template
rayjob_qwen3.yaml      ← Qwen3 template (FLASH_ATTN, no CUTLASS_MLA)
submit.sh              ← sd dispatcher; NNODES → env config map
submit_md.sh           ← md dispatcher
submit_qwen3.sh        ← Qwen3 dispatcher
```

### `verl/my_scripts/k8s/` — entrypoints + Hydra

```
my_deepep_env_k8s.yaml             ← Ray runtime env (NVSHMEM, CUDA vars)

run_40bra_k8s_16node_single_domain.sh       ← sd 40bra entrypoint
run_40bra_k8s_multi_domain.sh               ← md 40bra entrypoint
run_qwen3_k8s_2node_single_domain.sh        ← Qwen3 sd entrypoint

config/
├── 40bra_16node_sd.yaml           ← sd 40bra Hydra base (MLA, attention=fused w/ CUTLASS_MLA)
├── 40bra_16node_md.yaml           ← md 40bra base
├── qwen3_30b_a3b_2node_sd.yaml    ← Qwen3 base (GQA, attention=fused w/ FLASH_ATTN)
├── combo_40Bra.yaml               ← all combo definitions (c351…c37x, cbsp*, cbmd*, cbdg-*)
└── env/
    ├── k8s_b300_16node.yaml        ← 40bra 16-node default
    ├── k8s_b300_8node_c10.yaml     ← 40bra 8-node (no offload, gmu=0.7)
    ├── k8s_b300_4node_c11.yaml     ← 40bra 4-node (no offload, gmu=0.4)
    ├── k8s_b300_8node_debug.yaml   ← md 8-node debug
    └── k8s_b300_2node_qwen3.yaml   ← Qwen3 2-node
```

### `verl/my_scripts/bspo_scripts/` — self-contained submitter (prod only)

For when the verl clone must be self-contained (no outer bisimpo/ or
iter_kuberay_/ repos). Verbatim copies of submit + rayjob YAMLs with the
outer-path references rewritten. See its README for the porting checklist
to the B200 cluster.

### `verl/verl/` — verl source (the actual framework code)

Key files touched by this work:

```
trainer/ppo/core_algos.py
  - compute_policy_loss_bspo       (sd bspo, 3 variants: simplest / w_penalty / w_penalty_only)
  - compute_policy_loss_bspo_md    (md bspo, 5 variants)
  - _scatter_mean                  (group / domain aggregation helper)
  - _bspo_factorize_to_long_tensor (uid / data_source → int64 indices)

workers/actor/megatron_actor.py    ← bspo_md kwargs plumbing (primary for 40Bra)
workers/actor/dp_actor.py          ← bspo_md kwargs plumbing (FSDP mirror)
workers/config/actor.py            ← bspo_variant / bspo_delta / bspo_lambda_{tj,grp,d}
trainer/config/actor/actor.yaml    ← defaults for the above
```

---

## Common tasks

All commands assume you're at `/mnt/public/lichang93/st_verl_dockerfile`.
K8s calls need the proxy-free preamble because the system proxy blocks
the internal k8s API:

```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl …
```

### Inspect cluster

```bash
# all BSPO-related jobs
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob -l 'training-type in (40bra-sd,40bra-md,qwen3-sd)'

# count by state
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get rayjob 2>&1 | awk 'NR>1{print $2}' | sort | uniq -c
```

### Read a specific run's metrics

Rayjob name is `<prefix>-<combo>-<5hash>` (e.g. `bra40-sd-cbsp401-hdxvc`).
The ckpt dir is `<run_dir_root>/<rayjob-name>/`. Root depends on training
type: `40bra_k8s_single_domain`, `40bra_k8s_multi_domain`, or
`qwen3_k8s_single_domain`.

```bash
# from any head pod of any running rayjob:
HEAD=$(... kubectl get pods -l ray.io/cluster=<CLUSTER>,ray.io/node-type=head -o name | head -1 | sed 's|^pod/||')
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl exec $HEAD -- \
    grep -oE 'training/global_step:[0-9]+' \
    /root/myCodeLab/host/verl/ckpts/<sub>/<rayjob>/run.log | tail -1
```

### Submit sd BSPO ablation

```bash
# cbsp401 equivalent: V5 δ=3e-4 λ=1e-2 — the stable config
NNODES=4 bisimpo/submit_bspo.sh cbsp403 \
    'actor_rollout_ref.actor.bspo_delta=3.0e-4 actor_rollout_ref.actor.bspo_lambda_tj=1.0e-1'
```

Only cbsp combos defined in `verl/my_scripts/k8s/config/combo_40Bra.yaml`
can be submitted. Current Phase-3 slots: cbsp{203,204,205,303…309,402…409}.

### Submit md BSPO

Currently disabled. User's sd-first directive means md runs don't launch
until sd ablation lands. To re-enable:

```bash
# on main nemo_bspo_md + verl nemo_bspo_md + iter_kuberay nemo_bspo_md
bisimpo/submit_formal_md.sh   # launches all 5 cbmd-v{1..5}
```

### Submit Qwen3 smoke

```bash
# 1-step, val before + after
qwen3_port/submit_qwen3.sh cbdg-qwen3-smoke 'actor_rollout_ref.actor.optim.lr_warmup_steps=0'
```

For a longer Qwen3 run, either add a new combo with `total_training_steps > 1`
to `combo_40Bra.yaml`, or CLI-override `trainer.total_training_steps=N` and
set `actor_rollout_ref.actor.optim.lr_warmup_steps = min(10, N-1)`.

### Kill jobs

```bash
# all jobs of one combo
... kubectl delete rayjob -l combo_id=cbsp401

# one job by name
... kubectl delete rayjob <rayjob-name>

# all md jobs
... kubectl delete rayjob -l training-type=40bra-md
```

---

## Known workarounds / technical debt

### bspo_md (multi-domain BSPO)

- **Flat thresholds** `δ_τ = δ_g = δ_d = bspo_delta`. Paper's full
  effective-threshold recipe with `d̂(z)`, `R_maxres`, `T_g`-weighted form
  deferred. Under MEAN-at-every-level normalisation all levels live on
  the same O(ε) scale — but this is an approximation. See
  `bisimpo/reading_list_bspo_md_correctness.md` Tier 1 item 2 step 6.
- **`A_g` token-weighting**: `_scatter_mean(A_τ, group_idx)` ignores
  `|τ|` differences within a group. Equals spec under equal response
  lengths; deviates when they vary.
- **Cross-DP group splitting**: `actor.shuffle=false` + `data.shuffle=false`
  + `interleave=True` in rollout repeat means the default DP split lands
  full groups on one rank. If any of those flip, `s_g` / `s_d` become
  biased. Not currently an issue, but fragile.

See `bisimpo/reading_list_bspo_md_correctness.md` for the full tier-ordered
review.

### Qwen3 port

Three real workarounds, all documented in
`qwen3_port/dev_notes.md` §"Compromised workarounds":

1. **`VLLM_ATTENTION_BACKEND=FLASH_ATTN` forced**. Root cause: flashinfer
   wheel vs torch version mismatch in the image. Proper fix: bump
   flashinfer in `Dockerfile.base`.
2. **`attention_backend: fused` instead of `flash`**. Root cause: rayjob
   env globally sets `NVTE_FUSED_ATTN=1`. Proper fix: remove that env
   from `rayjob_qwen3.yaml` and switch to flash.
3. **DeepEP flex dispatcher inherited unverified** for Qwen3's
   128-expert shape. Safer fallback: `moe_enable_deepep: false,
   moe_token_dispatcher_type: alltoall`.

Plus: only 1 training step tested. `bspo_*` metrics all 0 at step 1 because
on-policy log_ratio≈0. Cheapest next check: 5–10 steps at `ppo_epochs=2`.

### Megatron + k8s quirks inherited from 40Bra (see `CLAUDE.md` §10)

- 2-node baremetal DeepEP/NVSHMEM cross-node fails (IB DCT resource
  exhaustion vs NCCL). Workaround: `alltoall` dispatcher. k8s 16-node+
  runs don't hit this.
- `total_training_steps=0` → Megatron scheduler asserts. Use ≥1 step.
- `val_max_samples=null` → `None > 0` TypeError. Use `-1` (verl convention).
- Proxy must be unset for `kubectl` commands (see preamble above).

---

## Where to pick up the work

Sorted by urgency.

### Right now (no-op — running)

- **18 sd ablation jobs** complete autonomously by ~2026-04-21 evening.
  When they finish, analyze val trajectories to find the most stable
  `(variant, δ, λ)` combo. Compare to cbsp401's baseline.

### After sd ablation finishes

1. Pick the winning sd combo; decide if a longer run (> 120 steps) is
   warranted.
2. Re-enable md ablation. Submit 5 formal cbmd-v{1..5}. Already scripted
   in `bisimpo/submit_formal_md.sh`.
3. For Qwen3: extend to a 5–10 step mini-run at `ppo_epochs=2` to
   exercise the non-trivial BSPO paths.

### Pending cleanup

- The 11 Initializing wave-2 sd jobs — Volcano will pick them up as
  current RUNNING ones finish. No action needed.
- Ckpt dirs at `verl/ckpts/40bra_k8s_single_domain/` accumulate. Each run
  ≈ 2 GB (run.log + wandb + tokenizer cache). Cleanup opportunity after
  analysis.

### Tech debt worth fixing before next sprint

- Flashinfer + torch version bump in `Dockerfile.base` (Qwen3 R1).
- `rayjob_qwen3.yaml` env cleanup: drop `NVTE_FUSED_ATTN=1`, switch
  Hydra `attention_backend` to `flash` (Qwen3 R2).
- `val_max_samples: -1` → hardcoded in env files; if someone copy-pastes
  to a new env, `null` will bite.
- Docstring the `_bspo_factorize_to_long_tensor` determinism contract
  (np.unique inverse is sort-determined, so identical across DP ranks iff
  they see the same set of values — currently they do because of
  `interleave=True`).

---

## Anti-patterns — what NOT to do

- **Do not `git checkout 7dc6b1fa` in verl repo without creating a branch
  first.** The editable PFS install means a detached HEAD silently
  affects every NEW worker spawned by currently-running rayjobs. Always
  `git checkout -b <branch> 7dc6b1fa`.
- **Do not submit 8-node md jobs when sd ablation is at 28/31 busy.**
  Volcano gang-schedules → they'll sit in Initializing for hours.
- **Do not forget the `HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201`
  preamble for kubectl.** The system proxy blocks the internal k8s API.
- **Do not use `VLLM_ATTENTION_BACKEND=CUTLASS_MLA` for Qwen3.** That env
  only works for MLA-architecture models (40Bra). Qwen3 must have
  `FLASH_ATTN` (see Qwen3 R1).
- **Do not use `attention_backend: flash` for Qwen3 in the current rayjob
  env.** Will hit the `NVTE_FUSED_ATTN=1` assertion (see Qwen3 R2).
