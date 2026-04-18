# Debug run guide

> Run a 5-step `cdbg5` experiment on the cluster, collect per-step metrics,
> compare against the P1 baseline. End-to-end in under 60 minutes per run.

Assumes you are on `b32` with the `KP` alias defined (see
`k8s_cheatsheet.md` Section 0).

---

## 1. When to use a debug run

Debug runs use `combo_id=cdbg5`: 5 training steps, small curriculum window,
math-only. They exist to **test a compute/memory knob without spending
production cluster budget**.

Use them for:
- Checking whether a config change OOMs before it reaches production.
- Measuring the effect of a Category-A knob (parallelism, memory fraction,
  offload, ...) on step time and per-GPU throughput.
- Verifying gang scheduling after an infra change.

Do not use them for:
- Verifying training convergence. 5 steps is too few. For learning-quality
  checks, use 30+ step `c351` runs.
- Varying Category-B hyperparameters (batch size, `rollout.n`, lr, FACPO,
  curriculum). Those change training numerics and break apple-to-apple
  comparisons. See `concepts.md`.

---

## 2. Submit one run

```bash
cd /mnt/public/lichang93/st_verl_dockerfile/0418
./submit_debug.sh cdbg5 8                                        # baseline-ish
./submit_debug.sh cdbg5 8 'actor_rollout_ref.rollout.gpu_memory_utilization=0.85'  # with one override
DEBUG_LABEL=C1 ./submit_debug.sh cdbg5 8 '<overrides>'           # with human label
```

Arguments:
- `cdbg5` — combo id. Must match an entry in `verl/my_scripts/k8s/config/combo_40Bra.yaml`.
- `8` — node count (`[1, 32]`). Workers = nnodes - 1.
- `<overrides>` — any number of Hydra overrides. Quoted as one argument or multiple.
- `DEBUG_LABEL` env — optional human label (e.g. "C1"). Written into the run dir as `label.txt`.

What `submit_debug.sh` does automatically:
- Unsets `HTTPS_PROXY`/`HTTP_PROXY` (k8s API is outside the proxy path).
- Prepends six debug-only overrides that speed up every run (`val_before_train=false`,
  `test_freq=9999`, `save_freq=9999`, `log_val_generations=0`,
  `val_max_samples=32`, `lr_warmup_steps=2`). These are Category A, so they
  do not change training numerics. See `plan_c_compute_opt.md` §4a.
- `envsubst` expands `rayjob_debug.yaml` → `kubectl create` → unique RayJob
  name via `generateName`.
- Starts a `pipeline.sh` background monitor for the submitted RayJob.

Output prints the RayJob name and watch/log/verify commands.

---

## 3. Submit the whole Plan C matrix at once

```bash
cd /mnt/public/lichang93/st_verl_dockerfile/0418
./queue_all.sh
```

Submits all nine C-runs (C1–C9) with Volcano gang queueing. Each gets its
own pipeline monitor. 3 run concurrently (cluster has ~28 free compute
nodes; each job takes 8); the rest stay `Inqueue` until capacity frees.

Expected total wall clock: ~2.75 h if all runs succeed. Shorter if some
OOM-fail early (C2, C3, C4, C5 are known to fail in the current env).

---

## 4. Monitor progress

### Live situation report (copy/paste)
```bash
KC() { HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl "$@"; }

echo "=== rayjob ==="
KC get rayjob -l combo_id=cdbg5 --no-headers | awk '$2!="SUCCEEDED" && $2!="FAILED" {printf "%-35s %-12s %-15s\n", $1,$2,$3}'
echo "=== podgroup ==="
KC get podgroup
echo "=== capacity ==="
BUSY=$(KC get pod -l ray.io/cluster --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | sort -u | wc -l)
echo "busy=$BUSY  free=$((31 - BUSY))"
```

### Tail the driver log of a specific run
```bash
JOB=bra40-dbg-cdbg5-8n-vmk62
tail -f /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/run.log
```

### Verify gang scheduling for a run
```bash
./verify_gang.sh bra40-dbg-cdbg5-8n-vmk62
```
Runs 6 checks (PodGroup exists, minMember matches, pods bound atomically,
PodGroup condition OK, etc.). Exit 0 means all pass.

### Watch for state changes (live)
```bash
KC get rayjob -w
```

---

## 5. Collect results

When a run terminates (SUCCEEDED / FAILED), the pipeline monitor:

1. Waits up to 90 min for terminal state.
2. Copies `<run_dir>/perf_log.jsonl` and `run.log` to `0418/results/<label>_*`.
3. Runs `analyze_perf.py` → writes `<label>_analysis.txt` + puts a copy in the run dir as `analysis.txt`.
4. `kubectl delete rayjob <name>` (if the perf_log was actually fresh) to free nodes.

Read the analysis:
```bash
cat 0418/results/C1_analysis.txt
```

Compare two runs:
```bash
python3 0418/compare_profiles.py          # fixed P0/P1/P2 reference table
```

For ad-hoc comparisons, read `perf_log.jsonl` with your own tool — each line
is a JSON record per training step. Schema in `plan_b_node_opt.md` §6.1.

---

## 6. File layout (where everything lives)

Every run produces exactly one directory on PFS, keyed by the RayJob name:

```
verl/ckpts/40bra_k8s_single_domain/<rayjob_name>/
    run.log           entrypoint + driver stdout/stderr
    perf_log.jsonl    per-step metrics
    analysis.txt      post-run summary from analyze_perf.py
    exp_name.txt      the traditional timestamped name (wandb UI)
    rayjob.txt        the RayJob name (self-documenting)
    label.txt         your DEBUG_LABEL if you passed one
    <megatron ckpt shards if save_freq was set>
```

Flat copies for quick comparison:
```
0418/results/<label>_perf_log.jsonl
0418/results/<label>_analysis.txt
0418/results/<label>_training.log
```

Pipeline monitor logs (disposable):
```
/tmp/0418_pipeline_<label>.log
```

---

## 7. Best practices

### Apple-to-apple or don't bother
If two runs differ in any Category-B hyperparameter, you can no longer
conclude anything about compute efficiency from their delta. Hold batch
size, `rollout.n`, lr, FACPO, curriculum constant. Vary only Category-A
knobs (parallelism, memory, offload). See `concepts.md`.

### Keep `gpu_memory_utilization` × `free_cache_engine` sane
Empirical ceiling for 40Bra hybrid engine is ~260 GiB/GPU allocated. Going
higher OOMs during actor update. Specifically: `free_cache_engine=false`
alone is fine only if `gpu_memory_utilization ≤ 0.5`. With default 0.7 or
higher, keep `free_cache_engine=true`.

### Do not use `PP=1` at `max_response=16384`
Proven dead-end on 40Bra: 275 GiB allocated, 290 GiB reserved (exceeds 288
GiB HBM) → kernel OOM-killer. No Python traceback. If you must try, halve
`max_response_length` first (which is Category B — that breaks apple-to-apple).

### Use meaningful labels
Always pass `DEBUG_LABEL=<short tag>` so the run dir self-documents why it
existed. `label.txt` survives after ttl cleanup.

### Clean up regularly
Terminal-state RayJobs accumulate. Run the one-liner from
`k8s_cheatsheet.md` §11 weekly:
```bash
KC get rayjob --no-headers | awk '$2=="SUCCEEDED" || $2=="FAILED" {print $1}' | xargs -r KP delete rayjob
KP delete pod --field-selector=status.phase=Succeeded
```
Deleting a RayJob does NOT delete its run dir on PFS. Metrics and logs stay.

### Before submitting a big batch
- Check cluster capacity (`KC get pod -l ray.io/cluster --field-selector=status.phase=Running` then count nodes).
- Check no other tenant is running something you would bump (look for vcjob / non-Ray pods).
- If you need many concurrent gangs, submit with `queue_all.sh` or a similar batch — Volcano will queue the ones that cannot fit and admit them atomically as capacity frees.

---

## 8. Troubleshooting patterns

| Symptom | Likely cause | What to do |
|---|---|---|
| `KC get rayjob` shows your job at `FAILED` within ~10 min of submit | Hydra config error, missing file, assert in optimizer scheduler | `tail <run_dir>/run.log` — the real error is usually in the last 200 lines, after filtering out autoscaler noise. |
| `PodGroup` stays `Inqueue` for more than 5 min | Not enough capacity, or another tenant's vcjob is hogging GPUs | `KC get pod -A -o wide` to see who is on which node. Contact the other tenant or wait. |
| RayJob `RUNNING` but head pod log ends abruptly, no traceback | OS OOM-killer silently killed the container | Lower memory pressure: drop `gpu_memory_utilization`, enable more offload, or cut `max_response_length` (Category B — will break apple-to-apple). |
| Pipeline monitor still running but RayJob is gone | Pod terminated after ttl; `pipeline.sh` may have missed the poll | Check `/tmp/0418_pipeline_<label>.log`. If its last entry is `pipeline done`, the monitor finished. Otherwise restart it: `nohup bash pipeline.sh <jobname> <label> &`. |
| Analysis shows numbers identical to a different run | pipeline.sh's time-match fallback picked the wrong ckpt dir — only happens for RayJobs submitted before 2026-04-18 12:45 UTC | For those old runs, verify manually by reading `run.log` of each suspicious dir. All runs from 2026-04-18 onward use direct RayJob-name lookup and are unambiguous. |

---

## 9. Anatomy of one debug run (what happens under the hood)

1. **You call** `./submit_debug.sh cdbg5 8 '<overrides>'` on b32.
2. `submit_debug.sh` renders `rayjob_debug.yaml` via envsubst and pipes to
   `kubectl create`. k8s assigns a unique RayJob name via `generateName`.
3. **kuberay-operator** sees the RayJob; because it runs with
   `--batch-scheduler=volcano`, it creates a PodGroup with
   `minMember = 1 + workerReplicas`.
4. **Volcano scheduler** either admits the gang atomically (if enough nodes
   are free) or places the PodGroup in `Inqueue` until capacity frees.
5. Once admitted, kuberay creates 8 pods; they all land on exclusive nodes.
   The head pod starts Ray first; workers join.
6. **Ray head pod** runs the RayJob entrypoint:
   `run_40bra_k8s_16node_single_domain.sh cdbg5 <all overrides>`.
7. The entrypoint:
   - Derives `RAYJOB_NAME` from its hostname.
   - Creates `verl/ckpts/40bra_k8s_single_domain/<RAYJOB_NAME>/`.
   - Writes `exp_name.txt`, `rayjob.txt`, optional `label.txt`.
   - `exec tee` redirects all stdout/stderr into `<run_dir>/run.log`.
   - Starts Gym servers on all nodes (ports 20001–20007).
   - `ray job submit --runtime-env=my_deepep_env_k8s.yaml -- python3 -m verl.trainer.main_ppo ...`.
8. **Training driver** runs 5 steps. After each step, `ray_trainer.py`
   calls `verl.utils.perf_log.write_perf_log(...)` — only writes to
   `perf_log.jsonl` when `VERL_PERF_LOG=1` is in the pod env
   (`rayjob_debug.yaml` sets that).
9. On success, the driver exits 0 → kuberay sets RayJob status to
   `SUCCEEDED`. On failure, `FAILED`.
10. `pipeline.sh` (running on b32) sees the terminal state within 2 min.
    Copies `perf_log.jsonl` and `run.log` into `0418/results/`; runs
    `analyze_perf.py`; writes `analysis.txt` into both places; then
    `kubectl delete rayjob <name>` (only if perf_log is fresh) to free
    nodes for the next Inqueue gang.
11. kuberay tears down the pods (`shutdownAfterJobFinishes: true`). Any
    run after 2026-04-18 12:45 UTC has ttl=120s so even if pipeline.sh
    crashes, the pods are gone within 2 min.
