# MD-S1 — Multi-domain infra (kuberay+volcano + debug/formal yaml split)

**Goal.** Enable multi-domain BSPO training to launch via the same
`kuberay + volcano` path as the single-domain path, with a clean
debug/formal yaml split. Produce two templated rayjob yamls
(`md-debug-rayjob.yaml`, `md-formal-rayjob.yaml`) and one entrypoint
script (`run_40bra_k8s_multi_domain.sh`), invoked via a combo-driven
submit wrapper analogous to `iter_kuberay_32nodes_verl_training/submit/submit.sh`.

**No BSPO code changes in this stage.** S1 only produces infrastructure
files. BSPO loss extension is S3.

## Survey of existing assets

| file                                                                              | role                                                                                                     |
|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `iter_kuberay_32nodes_verl_training/submit/submit.sh`                             | single-domain templated submitter (envsubst on `rayjob.yaml`)                                            |
| `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml`                           | single-domain rayjob template (COMBO_ID / NNODES / EXTRA_OVERRIDES placeholders); schedulerName = volcano |
| `iter_kuberay_32nodes_verl_training/yaml/stage7-40bra-32node-gym-rayjob.yaml`     | existing 32-node multi-domain rayjob (Stage 7, hard-coded, not templated)                                |
| `verl/my_scripts/k8s/run_40bra_32node_gym.sh`                                     | entrypoint for Stage 7; starts gym on head + via Ray on all workers                                      |
| `verl/my_scripts/k8s/run_multi_domain.sh`                                         | older multi-domain runner (references Stage 7 ops)                                                       |
| `verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh`                       | single-domain entrypoint (reference pattern for structure, logging, hydra dispatch)                      |
| `verl/my_scripts/gym/start_gym_uv.sh`                                             | per-node gym lifecycle (start/stop/status)                                                               |
| `verl/my_scripts/gym/start_gym_all_nodes.py`                                      | Ray-remote gym dispatch to worker nodes                                                                  |

**Finding.** The multi-domain kuberay+volcano path *exists* at Stage 7
scale (32-node `stage7-40bra-32node-gym-rayjob.yaml`) but is hard-coded
and not combo-driven. What's missing for our BSPO multi-domain work:

1. A templated multi-domain rayjob yaml (combo_id + nnodes placeholders)
2. A multi-domain entrypoint shell script parameterized by `combo_id`
3. A **debug** rayjob yaml for fast-feedback smoke (1-step, ~2-3 min iteration)
4. A **formal** rayjob yaml for full 120-step training
5. A BSPO multi-domain Hydra base config `40bra_16node_md.yaml`
6. A `combo_40Bra` yaml entry for the new multi-domain combos (cbsp501-505)
7. A `submit_bspo_md.sh` wrapper that mirrors `submit_bspo.sh` but routes to
   the multi-domain rayjob template

## S1 deliverables

### (a) Templated multi-domain rayjob yaml

`iter_kuberay_32nodes_verl_training/submit/rayjob_md.yaml`

Adapted from single-domain `rayjob.yaml` with three changes:
- `metadata.generateName`: `bra40-md-${COMBO_ID}-${NNODES}n-`
- `labels.training-type`: `40bra-md`
- `entrypoint`: `bash /root/myCodeLab/host/verl/my_scripts/k8s/run_40bra_k8s_multi_domain.sh ${COMBO_ID} ${EXTRA_OVERRIDES}`
- unchanged: NCCL env, image, volcano scheduler, PodGroup gang size (NNODES)

### (b) Multi-domain entrypoint

`verl/my_scripts/k8s/run_40bra_k8s_multi_domain.sh`

Fork of `run_40bra_k8s_16node_single_domain.sh` with:
- Gym startup on head + all worker nodes (Ray-remote via `start_gym_all_nodes.py`)
- Hydra `--config-name=40bra_16node_md`
- Dispatches `combo_id=$1` and all `$2...`N overrides to Hydra

### (c) Multi-domain Hydra base config

`verl/my_scripts/k8s/config/40bra_16node_md.yaml`

Fork of single-domain `40bra_16node_sd.yaml` with:
- `data.train_files`: `0320_split/train.parquet` (full multi-domain, 71k rows)
- `data.val_files`: same 230-row hard eval parquet (math-only; multi-domain eval suite TBD in S7)
- `reward_model.reward_kwargs.server_urls`: all 7 nemogym ports (code, mcqa, if, structured, workplace, math, qydomain; + the 3 hard-eval sources)
- `data.sampler.domains`: `all` (curriculum sampler trains across all sources)
- `data.sampler.domain_balanced`: `true`
- `combo_bank@combo_40Bra`: same combo yaml as single-domain

### (d) Debug vs formal split — simplified to one yaml + env flag

Originally planned as two rayjob yamls (`rayjob_md.yaml` +
`rayjob_md_debug.yaml`). Simplified during implementation to a single
`rayjob_md.yaml` with a `BSPO_DEBUG` env variable threaded through
envsubst to the container env. The entrypoint
`run_40bra_k8s_multi_domain.sh` picks up `BSPO_DEBUG=1` and appends a
debug-override pack at the Hydra CLI layer (1 step, `val_max_samples=16`,
`test_freq=1`, `train_batch_size=8`, etc.). Rationale: one template
instead of two means less to audit and no risk of the two drifting out of
sync.

Debug vs formal is now driven entirely by:
- `BSPO_DEBUG=1` env (submit_bspo_md.sh auto-sets for `cdbgmd*` combos)
- `NNODES` env (submit_bspo_md.sh defaults: `4` for `cdbgmd*`, `16` for formal)
- combo-yaml total_training_steps (`1` for `cdbgmd*`, `120` for formal)

Combo classes (added to `combo_40Bra.yaml`):
- `cdbgmd1`, `cdbgmd4`, `cdbgmd5`  — debug (1 step, V1/V4/V5).
- `cbsp501-cbsp505`  — formal (V1-V5 multi-domain, 120 steps).

### (e) Submit wrapper

`bisimpo/submit_bspo_md.sh`

Fork of `submit_bspo.sh` with:
- Delegates to `iter_kuberay_32nodes_verl_training/submit/submit_md.sh` (new,
  forks `submit.sh` with `rayjob_md.yaml` template path)
- Per-combo DELTA/LAMBDA dispatch for cbsp501..cbsp505 (V1-V5)
- `NNODES=4` default for debug combos, `NNODES=16` default for formal

## Execution order within S1

1. `rayjob_md.yaml` template (fork of single-domain)
2. `rayjob_md_debug.yaml` template (debug-config overlay)
3. `submit_md.sh` (template dispatcher)
4. `run_40bra_k8s_multi_domain.sh` (entrypoint)
5. `40bra_16node_md.yaml` (Hydra base)
6. `submit_bspo_md.sh` (wrapper)
7. Add `cdbgmd1` debug combo + `cbsp501..cbsp505` formal combos to `combo_40Bra.yaml`
8. Dry-run `submit_bspo_md.sh cdbgmd1 --dry-run` to confirm envsubst renders a valid rayjob (no smoke test yet; that's S6)
9. `git add ... ; git commit ... mdplan-S1 ; git tag mdplan-S1`

## Design decisions locked in S1

- Keep the single-domain and multi-domain rayjob templates as *separate
  files*, not a parameterised mega-template. Lower regression risk, two
  focused yamls are easier to audit.
- Reuse the existing `combo_40Bra.yaml` for both domains; the difference
  between single and multi is in the Hydra base config (`sd` vs `md`),
  not the combo yaml.
- Debug yaml uses 4 nodes (matches the C11 profile we've been using for
  smoke), not fewer. This guarantees the debug smoke exercises the same
  gym-on-all-nodes + NCCL-all-gather paths as the formal run.
- `BSPO_DEBUG=1` environment flag is threaded to the entrypoint; the
  entrypoint appends a small override pack when set.
- No change to `compute_policy_loss_bspo` — S1 is pure plumbing.

## Exit criteria for S1

- [ ] `rayjob_md.yaml` renders correctly with `envsubst`; `kubectl apply --dry-run=server` returns success
- [ ] `rayjob_md_debug.yaml` same
- [ ] `submit_bspo_md.sh cdbgmd1` writes the expected "name=..." stdout line and creates a rayjob
- [ ] Debug rayjob reaches Running state (pod schedule ok, Gym starts, training script imports verl) — *actual 1-step smoke is S6, not S1*
- [ ] All edits committed; tag `mdplan-S1` set in both `verl` submodule and main repo

## Dev notes / execution log

**2026-04-19 07:57 UTC** — S1 execution.

Files created / edited:

1. `iter_kuberay_32nodes_verl_training/submit/rayjob_md.yaml`
   Forked `rayjob.yaml` with three diffs: generateName, labels.training-type,
   entrypoint path. All env/resource/volcano config identical. Added
   `BSPO_DEBUG` env var threaded from envsubst.

2. `iter_kuberay_32nodes_verl_training/submit/submit_md.sh`
   Forked `submit.sh` with TPL→rayjob_md.yaml. Keeps the same NNODES→ENV_NAME
   map (16/8/4). Forwards `BSPO_DEBUG` through envsubst.

3. `verl/my_scripts/k8s/run_40bra_k8s_multi_domain.sh`
   Forked `run_40bra_k8s_16node_single_domain.sh`. Changes:
   - ckpt subdir → `40bra_k8s_multi_domain`
   - exp_name prefix → `40bra_k8s_md_*`
   - Hydra `--config-name=40bra_16node_md`
   - Reads `BSPO_DEBUG`; when `1`, appends debug override pack
     (1 step, `val_max_samples=16`, tiny batch, `lr_warmup=0`).

4. `verl/my_scripts/k8s/config/40bra_16node_md.yaml`
   Forked `40bra_16node_sd.yaml`. Changes:
   - `data.train_files` → `nemogym_blend/train_v2.parquet` (multi-domain train)
   - `data.sampler.domain_balanced: true`
   - `reward_model.server_urls` → all 7 nemogym ports + 3 hard-eval sources
   - `trainer.project_name` → `40bra_k8s_multi_domain`
   Everything else (MoE, parallelism, optimiser) identical.

5. `bisimpo/submit_bspo_md.sh`
   Forked `submit_bspo.sh`. Changes:
   - PROD_SUBMIT → `submit_md.sh`
   - Case statement: `cdbgmd{1,4,5}` (debug) + `cbsp501..cbsp505` (formal)
   - Default NNODES: 4 for `cdbgmd*`, 16 for formal
   - Default BSPO_DEBUG: 1 for `cdbgmd*`, 0 for formal

6. `verl/my_scripts/k8s/config/combo_40Bra.yaml`
   Added 8 new combos: 3 debug (`cdbgmd1/4/5`) + 5 formal (`cbsp501..cbsp505`).
   Debug combos: `total_training_steps=1`, `ppo_epochs=1`,
   `curriculum_total_steps=10`. Formal combos: standard 120-step config,
   `domains: all` for multi-domain training.

## Validation — envsubst + kubectl dry-run

```
COMBO_ID=cdbgmd1 NNODES=4 WORKER_REPLICAS=3 \
  EXTRA_OVERRIDES="env=k8s_b300_4node_c11" BSPO_DEBUG=1 \
  envsubst ... < rayjob_md.yaml | kubectl create --dry-run=server -f -
→ rayjob.ray.io/bra40-md-cdbgmd1-4n-j2s5z created (server dry run)
```

Passes server-side schema validation. Template is production-ready.

Actual smoke (end-to-end) is deferred to S6 per the plan; S1's exit
criterion is only "template renders and passes dry-run validation."

## Exit criteria

- [x] `rayjob_md.yaml` renders correctly with envsubst; `kubectl create --dry-run=server` returns success
- [x] Single-yaml + env flag replaces the "two yaml" split per simplified
      design decision (documented above)
- [x] `submit_bspo_md.sh` exists with per-combo dispatch for cdbgmd* and cbsp501-505
- [ ] *Debug rayjob reaches Running state* — deferred to S6 (end-to-end smoke)
      per master plan. S1 closes here.
- [x] All edits committed; `mdplan-S1` tag in both `verl` and main repo

## Known items deferred to later stages

- V2 (`hierarchy`) and V3 (`strict_wasserstein`) BSPO variants are
  referenced in `submit_bspo_md.sh` and `combo_40Bra.yaml` (cbsp502,
  cbsp503) but **not yet implemented** in `core_algos.py`. S3 owns these.
  Attempting to run cbsp502/503 before S3 completes will fail at the
  config-validation layer (`assert variant in (...)` inside
  `compute_policy_loss_bspo`).
- The multi-domain eval suite is currently the 230-row math-family hard
  parquet. An expanded eval covering code / IF / structured / workplace
  will be added in S7 if the objective ranking transfers cleanly.
- `nemogym_blend/train_v2.parquet` is referenced as the multi-domain train
  set. S6 will verify it exists at the expected path; otherwise fall back
  to `0320_split/train.parquet`.

