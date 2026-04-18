# Production training guide — 40Bra k8s single-domain

> How to submit, monitor, and recover a production training run on the
> 32-node B300 KubeRay cluster.

Assumes you are on `b32` with the `KP` alias defined (see
`../0418/k8s_cheatsheet.md` §0).

---

## 1. What "production" means here

A production run:
- Runs ~100–500 training steps of real GRPO on 40Bra.
- Uses a real combo (`c351`, `c361`, `c372`, etc.) from
  `verl/my_scripts/k8s/config/combo_40Bra.yaml`, not the debug combo.
- Writes checkpoints at meaningful intervals.
- Validates during training.
- Expected wall clock: 10 h – 40 h depending on combo.

Production runs go through `submit/` in this directory. Debug experiments
go through `../0418/`. Never mix the two.

---

## 2. Submit a run

```bash
cd /mnt/public/lichang93/st_verl_dockerfile/iter_kuberay_32nodes_verl_training/submit
./submit.sh c351                                                  # math FACPO, ppo_epochs=2
./submit.sh c361                                                  # math FACPO, ppo_epochs=1 (on-policy)
./submit.sh c372                                                  # code FACPO, ppo_epochs=2
./submit.sh c361 'data.val_files=/path/to/eval_nemogym_math.parquet'  # with Hydra override
```

Arguments:
- `c<NNN>` — combo id (must match an entry in `combo_40Bra.yaml`).
- `<overrides>` — any number of Hydra overrides as quoted arguments.

`submit.sh` validates the combo id format, unsets proxy vars, `envsubst`s
`submit/rayjob.yaml`, and `kubectl create`s the RayJob. You get back the
RayJob name (unique, via `generateName`).

### What is in `submit/rayjob.yaml`

Conforms to the voltest-proven gang-scheduling pattern (see
`../0418/plan_a_kuberay_volcano.md`):
- `generateName: bra40-sd-${COMBO_ID}-` — unique per submit.
- No Kueue label.
- No manual `schedulerName: volcano` on pods — kuberay-operator sets it.
- `workerGroupSpecs.replicas == minReplicas == maxReplicas == 15`.
- `shutdownAfterJobFinishes: true`, `ttlSecondsAfterFinished: 1800`
  (30 min — gives you time to read head-pod logs after a run ends).

Do not edit `submit/rayjob.yaml` unless you are changing the pod spec in
a way that applies to every future production submit. Per-run variation
goes through Hydra overrides via `submit.sh` arguments.

---

## 3. Monitor progress

### Live status block
```bash
KC() { HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl "$@"; }

echo "=== active rayjobs ==="
KC get rayjob -l training-type=40bra-sd --no-headers \
  | awk '$2!="SUCCEEDED" && $2!="FAILED" {printf "%-40s %-12s %-15s %s\n", $1,$2,$3,$7}'
echo "=== podgroups ==="
KC get podgroup | grep bra40-sd
echo "=== node usage ==="
BUSY=$(KC get pod -l ray.io/cluster --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | sort -u | wc -l)
echo "busy=$BUSY  free=$((31 - BUSY))"
```

### Tail the training driver log
```bash
JOB=bra40-sd-c361-abc12
tail -f /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/run.log
```

### Look at per-step metrics
```bash
tail /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/perf_log.jsonl \
  | python3 -m json.tool
```

### Wandb (offline mode)
Every production run uses `WANDB_MODE=offline`. Logs go to
`verl/wandb_my_dirs/`. To view:
```bash
cd /mnt/public/lichang93/st_verl_dockerfile/verl/wandb_my_dirs
wandb sync <run_dir>     # syncs to wandb.ai (requires online proxy)
# or
wandb local <run_dir>    # serves a local dashboard
```
The wandb run name is the `exp_name` from the entrypoint, which is written
to `<run_dir>/exp_name.txt` for cross-reference.

---

## 4. Where everything lives

Each production RayJob produces exactly one directory on PFS:

```
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
    run.log           entrypoint + training driver stdout/stderr
    perf_log.jsonl    per-step metrics (only if VERL_PERF_LOG=1 is set;
                      optional for production)
    exp_name.txt      traditional timestamped name (for wandb UI)
    rayjob.txt        self-documenting ID
    actor/            Megatron checkpoints at every save_freq
    data/             dataloader state (for resume)
```

Production reward logs go through `nemogym_server` HTTP on each pod; no
PFS reward log artifact.

---

## 5. The Hydra config chain

Three files merged in this order (see Hydra `defaults:` list in
`40bra_16node_sd.yaml`):

1. `verl/verl/trainer/config/ppo_megatron_trainer.yaml` — verl base
   defaults. Do not edit.
2. `verl/my_scripts/k8s/config/env_k8s_b300_16node.yaml` — hardware and
   sharding plan: `PP=2, VPP=2, TP=1, EP=8, ETP=1, CP=1`, `train_batch_size=128`,
   `max_prompt_length=8000`, `max_response_length=16384`. Change only when
   you are genuinely retargeting a different hardware profile.
3. `verl/my_scripts/k8s/config/combo_40Bra.yaml` under `combo_bank` — combo
   definitions: FACPO params, `ppo_epochs`, domain, curriculum steps,
   `save_freq`. Add a new combo here for a new experiment.
4. `verl/my_scripts/k8s/config/40bra_16node_sd.yaml` — the outer file
   that declares defaults + `combo_id` selector. Edit for cross-combo
   knobs (reward URL, val set, mem offload toggles).

### Add a new combo

```yaml
# in combo_40Bra.yaml
c399:                    # my new combo — describe it here
  fapo_delta: 0.5
  ppo_epochs: 2
  total_training_steps: 120
  domains: nemogym_math
  curriculum_total_steps: 300
  save_freq: 20
```

Then `./submit.sh c399`. No other change needed.

---

## 6. Checkpoints and resume

Production uses Megatron distributed checkpoints. `save_freq` in the combo
controls interval (in training steps).

### Find the latest checkpoint for a run
```bash
ls /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/actor/
# expect: global_step_20/  global_step_40/  global_step_60/ ...
```

### Resume a failed run
Add two Hydra overrides to the submit:
```bash
./submit.sh c361 \
  'trainer.resume_mode=auto' \
  'trainer.resume_from_path=/root/myCodeLab/host/verl/ckpts/40bra_k8s_single_domain/<old_rayjob_name>/actor/global_step_100'
```

`resume_mode=auto` makes verl auto-detect the latest step inside the given
path. `resume_mode=resume_path` requires an exact step path.

### Checkpoint permissions
Before submitting a NEW experiment, pre-create the ckpts parent dir with
`o+w`. PFS root-squashes pods to UID 10000, which needs write permission:
```bash
mkdir -p /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain
chmod o+w /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain
```

---

## 7. Best practices

### Treat `submit/rayjob.yaml` as frozen
Any per-experiment variation should be a Hydra CLI override, not a yaml
edit. If you find yourself wanting to edit `rayjob.yaml`, ask whether the
change belongs in:
- `combo_40Bra.yaml` (combo-specific) — most likely yes.
- `env_k8s_b300_16node.yaml` (hardware/env) — usually yes.
- `40bra_16node_sd.yaml` (run-shape) — sometimes.
- `rayjob.yaml` (k8s plumbing) — only for operator changes like new env
  vars, resource requests, pod-spec mutations.

### Apple-to-apple for comparisons
Whenever you want to compare two runs on **learning quality**, hold all
hyperparameters constant; change only the dataset / reward / algorithm
knob you are testing.

Whenever you want to compare two runs on **compute efficiency**, hold all
Category-B hyperparameters constant; change only Category-A knobs
(parallelism, memory fraction, offload, token budget). See
`../0418/concepts.md` for the full A/B split.

### Do NOT bring the debug defaults to production
`../0418/submit_debug.sh` auto-disables validation and checkpointing. That
is fine for 5-step throughput tests, fatal for real training. Production
should always use `submit/submit.sh`, which does NOT inject those
overrides.

### Name-collision safety
`generateName` guarantees each submission gets a unique RayJob name, so
you can resubmit the same combo repeatedly without collision. Historical
record is distinguishable through:
- `rayjob.txt` inside each run dir.
- `exp_name.txt` includes the entrypoint timestamp + git commit short SHA.

### Memory envelope
For 40Bra hybrid engine at `max_response=16384`, the safe per-GPU
`max_memory_allocated_gb` is ≤ 237 GiB. Above 260 GiB, OS OOM-kill risk
becomes real (silent, no Python trace). Any config change that pushes
allocated memory higher than ~240 GiB should be debug-tested first via
`../0418/submit_debug.sh`.

---

## 8. Troubleshooting patterns

| Symptom | Likely cause | What to do |
|---|---|---|
| Job stays `Pending` for > 5 min | No capacity (other runs occupying nodes) | `KC get pod -A -o wide --field-selector=status.phase=Running` to see who is on which node. Wait or contact the tenant. |
| `RayJob` status `FAILED` within 10 min of submit | Hydra config error or optimizer schedule assert | `tail -n 300 <run_dir>/run.log`. The real error is usually the last Python traceback before autoscaler noise. |
| Pod dies mid-training with no Python traceback | OS OOM-killer (silent) | Check `perf/max_memory_allocated_gb` and `max_memory_reserved_gb` in the last `perf_log.jsonl` entry. If approaching 280+ GiB, increase offload or reduce `max_response_length`. |
| Checkpoint save crashes with `PermissionError` | Forgot to `chmod o+w` the ckpts parent dir before submit | See §6 "Checkpoint permissions". Recreate the dir with `o+rwx` and resubmit. |
| Curriculum sampler warns "0 valid samples after filter" | `filter_pass_rate_100=true` combined with a domain where every sample has `jd_pass_rate=1.0` | Pick a different domain or set `filter_pass_rate_100=false` in the sampler config. |
| Gym `reward_manager=nemogym_server` returns 0 for all samples | Gym server not reachable from worker pods | On the head pod: `curl http://localhost:20006/` → should return a response. If not, `start_gym_uv.sh` probably failed; check `/tmp/ng_run_gym.log`. |

---

## 9. Cleanup

After a run ends, the RayJob sticks around for `ttlSecondsAfterFinished:
1800` (30 min) so you can read logs. After that it is gc'd automatically.
The run's PFS directory is NOT deleted by ttl — those files are your
historical record.

To delete a RayJob immediately (e.g., you need the nodes now):
```bash
KC delete rayjob <name>
```
Checkpoint files on PFS stay.

To clean up old terminal-state RayJobs across the tree:
```bash
# see what would be deleted
KC get rayjob --no-headers | awk '$2=="SUCCEEDED" || $2=="FAILED" {print $1}'
# do it
KC get rayjob --no-headers | awk '$2=="SUCCEEDED" || $2=="FAILED" {print $1}' | xargs -r KC delete rayjob
```

---

## 10. Relationship to other trees

- `../voltest/` — gang scheduling verification (one-time). Not part of
  production, but the template in `../voltest/submit/rayjob.yaml` is the
  pattern this production yaml follows.
- `../0418/` — node-count optimization. The 8-node recommendation is
  documented there with evidence. If you adopt it, update
  `env_k8s_b300_16node.yaml` and the submit template together.
- `../k8s_kuberay_kueue_setup/` — the one-time cluster infra fix (remove
  Kueue, enable Volcano gang). Read when you need to understand why the
  cluster is configured the way it is.
