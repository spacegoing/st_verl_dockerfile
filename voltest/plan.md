# voltest — Gang-Scheduling Verification Plan

> Post-fix verification: run 10 concurrent 8-GPU RayJobs on the cluster to
> prove that the Phase 4 kuberay-operator upgrade (`batchScheduler.name=volcano`)
> is creating PodGroups correctly and Volcano is gang-scheduling them.
>
> This is deliberately minimal — no training framework, no verl, no Gym.
> Just Ray + torch GEMM on 8 GPUs per job for 5 minutes.

---

## Goal

**Reproduce the deadlock scenario that previously plagued verl training jobs,
but at minimum complexity, to prove Volcano PodGroup gang-scheduling prevents it.**

Previously (Incidents A/B/C documented in `kueue_vs_volcano_deep_analysis.md`),
submitting 2 × 16-node RayJobs concurrently caused partial placement and
deadlock. This test goes much further — 10 × 8-node RayJobs submitted
simultaneously — which is flat impossible on a 31-node cluster without gang
scheduling forcing the jobs to queue atomically.

1. Submit 10 independent RayJobs, each **8 nodes × 8 GPUs = 64 GPUs**.
2. KubeRay auto-creates one `PodGroup` per RayJob with `minMember=8`.
3. Volcano enforces all-or-nothing placement:
   - Only **3 or 4 jobs** can actually run at once (3×8=24, 4×8=32; cluster has 31 usable nodes → 3 running, 1 potentially waiting for the 8th slot).
   - The remaining jobs' PodGroups stay `Pending` with **zero** of their 8 pods bound.
4. As each running job finishes, the next queued PodGroup's 8 pods bind atomically.
5. All 10 jobs eventually complete, zero deadlock, zero partial placement.

If gang scheduling were NOT working, this test would trivially recreate the
exact deadlock we hit in March: 10 jobs × 8 pods = 80 pods fighting for 31
nodes → interleaved partial placement → every job holding a few nodes → none
runnable → permanent deadlock.

## Design choices

| Choice | Value | Reason |
|---|---|---|
| Pods per RayJob | **8 (1 head + 7 workers)** | Reproduce the multi-pod gang scenario from real training jobs |
| GPUs per pod | **8** | Whole-node sizing (matches B300 = 8 GPU/node) |
| GPUs per job | **64** (across 8 nodes) | Simulates scale of actual training jobs (Stage 10 was 16-node × 8 GPU = 128; this is half that) |
| Jobs submitted | **10 concurrently** | 10 × 8 = 80 pods needed vs 31 nodes — worst-case gang scheduling stress |
| Workload per pod | `torch.matmul` loop on 8 GPUs for 5 min | Each pod does local GEMM; no cross-pod NCCL (simpler) |
| Image | `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` | Known-good image on this cluster; has ray + torch + CUDA |
| PFS mount | `pvc-jdwzpnv` subPath `lichang93/st_verl_dockerfile` → `/root/myCodeLab/host` | Script at `/root/myCodeLab/host/voltest/gpu_burn.py` and logs back to PFS |
| No Kueue labels | — | Kueue uninstalled in Phase 1–2 |
| No `schedulerName: volcano` | — | KubeRay v1.3.0+ sets it automatically when `batchScheduler=volcano` |
| Per-job runtime | 5 min of GPU work + ~45s Ray startup | With 10 jobs queued in batches of 3, total wall clock ~15-20 min |

## Expected cluster behavior (if gang scheduling works)

With 31 usable nodes and 10 jobs × 8 nodes = 80 pods needed:

| State | Snapshot |
|---|---|
| Just after `launch.sh` | 10 PodGroups created. 3-4 have `Phase=Running` with 8/8 pods Scheduled; 6-7 have `Phase=Pending`. |
| Mid-run (~7 min in) | First 3-4 jobs nearing completion; next jobs still Pending. |
| After first job finishes | Its 8 nodes freed; next PodGroup transitions Pending → Running with all 8 pods bound in one move. |
| After ~20 min | All 10 jobs `SUCCEEDED`. Zero pods ever got stuck in partial-placement. |

## Expected cluster behavior (if gang scheduling does NOT work) — to be avoided

| State | Snapshot |
|---|---|
| Just after `launch.sh` | ~31 pods land, distributed across all 10 jobs (3 pods per job on average). No PodGroups or PodGroups exist but ignored. |
| 5 minutes in | Every job has 2-4 pods Running + 4-6 pods Pending. No job has 8/8. |
| 10 minutes in | Same state — deadlock. Pending pods wait for nodes currently held by other jobs' Running pods. |

Counting PodGroup state is the signal: with gang, we see clean boolean (`Running` or `Pending`), never "partial". Without gang, we see no PodGroups at all and every job stuck mid-placement.

## Files

```
voltest/
├── plan.md                 # this file
├── dev_notes.md            # live log, updated as work proceeds
├── summary.md              # headline report
├── gpu_burn.py             # the Ray app (main script)
├── yaml/
│   └── rayjob.yaml         # SOLE source-of-truth template (placeholders: __JOBNAME__, __DURATION_S__)
├── run_pipeline.sh         # end-to-end pipeline: preflight → submit → watch → cleanup → verify → report
├── monitor.sh              # (optional) one-shot probe of current state
└── logs/
    ├── voltest-0.log .. voltest-9.log  # per-job PFS logs from gpu_burn
    └── pipeline-<ts>.log               # per-run operational log (submit/watch/cleanup events)
```

**One yaml, no bloat**: `run_pipeline.sh` renders the template via `sed` and
pipes straight into `kubectl apply -f -`; no per-job files are written to
disk. Submitting N=10 jobs produces **zero** intermediate yaml files — only
the single source template persists in `yaml/`.

## Phases

### Phase A — Author code + yamls

1. Write `gpu_burn.py` — connects to local Ray cluster, dispatches **8 parallel** `@ray.remote(num_gpus=8)` tasks. Each task runs on a single node (since each needs 8 GPUs = whole node), does local `torch.matmul` in a loop for 5 min. Since Ray can only fit one such task per node (each node has exactly 8 GPUs), the 8 tasks distribute naturally across the 8 pods of the RayCluster.
2. Write `yaml/voltest-rayjob-template.yaml` following the KubeRay official RayJob shape:
   - `kind: RayJob`
   - `spec.entrypoint: python3 /root/myCodeLab/host/voltest/gpu_burn.py`
   - `spec.shutdownAfterJobFinishes: true`
   - `spec.ttlSecondsAfterFinished: 300`
   - `headGroupSpec` — 1 pod, 8 GPU, whole-node-sized (176 CPU, 1920Gi, 1 RDMA)
   - `workerGroupSpecs` — groupName=workers, **replicas: 7, minReplicas: 7, maxReplicas: 7** (no autoscaling), same pod spec as head
   - Total per job: 1 head + 7 workers = **8 pods**, minMember=8
   - **No** `kueue.x-k8s.io/queue-name` label
   - **No** `schedulerName: volcano` on pods (operator sets it)
3. Write `run_pipeline.sh` — one end-to-end script covering all operational phases (preflight, submit, watch, cleanup, verify, report). Renders template via `sed` and pipes straight to `kubectl apply -f -` so no per-job yaml files ever touch disk.
4. Write `monitor.sh` — optional one-shot status probe for ad-hoc debugging.

### Phase B — Dry test: one job end-to-end

1. Submit a single job (`voltest-0`) of the new 8-pod shape.
2. Verify `PodGroup` is auto-created with **minMember=8**: `kubectl get podgroup -A`.
3. Verify **all 8 pods** reach Running (head + 7 workers).
4. Tail head pod logs; confirm `gpu_burn` tasks are dispatched to 8 distinct nodes.
5. Wait for job to finish (~6 min). Confirm `RayJob` status SUCCEEDED and all 8 pods deleted.
6. Check PFS log file `voltest/logs/voltest-0.log` contains iter counts.

**Gate B**: the 8-pod RayJob must succeed end-to-end before submitting 10.

### Phase C — Full test: 10 concurrent jobs

1. Run `launch.sh` to submit 10 jobs at once.
2. Snapshot immediate state after apply (`kubectl get rayjob,podgroup,pod`).
3. Poll every 30s for 10 min; log state transitions to `dev_notes.md`.
4. Confirm all 10 PodGroups created with minMember=1.
5. Confirm 10 distinct pods running on 10 distinct nodes.
6. Wait for all jobs to complete.
7. Run `collect_logs.sh` to pull pod stdout into `voltest/logs/job-N.log`.

**Gate C** (success criteria — gang scheduling actually working):
- 10 `PodGroup` CRs created immediately after `launch.sh`.
- At any snapshot, each PodGroup is cleanly **either `Running` (8/8 pods scheduled)** or **`Pending` (0/8 pods scheduled)**. Never partial.
- 3-4 jobs run concurrently, cycling through all 10 within ~20 min.
- All 10 RayJobs eventually finish with `SUCCEEDED`.
- Zero pods ever stuck in `Pending` on a per-pod basis while its PodGroup's other pods are `Running` — would indicate gang failure.

### Phase D — Debug loop (if Gate B or C fails)

If anything fails at any step, do NOT stop. Debug in place:
- Read pod events: `kubectl describe pod <name>`
- Read operator logs: `kubectl -n kube-system logs deploy/kuberay-operator --tail=200`
- Read volcano scheduler logs: `kubectl -n kube-system logs deploy/volcano-scheduler --tail=200`
- Check PodGroup status: `kubectl describe podgroup <name>`
- Record diagnosis and fix in `dev_notes.md`
- Iterate until gates pass

### Phase E — Cleanup + report

1. Delete all 10 RayJobs (`kubectl delete rayjob -l voltest=true`).
2. Confirm all pods gone, all PodGroups gone.
3. Summarize pass/fail in `dev_notes.md`.

## Known risks and mitigation

| Risk | Mitigation |
|---|---|
| Image pull is slow on first pull per node | Use `imagePullPolicy: IfNotPresent`; image is already cached on all nodes from prior verl runs |
| Cluster may have < 10 free nodes due to other tenants | If so, some PodGroups stay Pending — that actually *proves* gang scheduling works. Not a failure. |
| Ray head startup may take 30-60s per pod | Wait up to 10 min total before declaring a job stuck |
| Pod can't write to `voltest/logs/` on PFS | Pre-created `logs/` with `o+rwx` (755 + sticky other-write) |
| `shutdownAfterJobFinishes` not quick enough | `ttlSecondsAfterFinished: 300` gives buffer |
| Volcano scheduler logs in `kube-system` not visible without proxy-off | Use `HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl ...` consistently |

## Success definition (explicit)

1. `kubectl get podgroup -A` shows 10 PodGroups immediately after `launch.sh`.
2. `kubectl get rayjob -A -l voltest=true -o wide` shows all 10 eventually with `JOB STATUS=SUCCEEDED`.
3. `voltest/logs/job-{0..9}.log` each contain at least one line like `iter=10 elapsed=XX.Xs` confirming GPU calc ran.

---

## Autonomous operation

User is asleep. Plan is to iterate through Phases A → B → C → (D as needed) → E without asking questions. All decisions made in the spirit of: minimum viable, low blast radius, evidence-based debugging. All progress logged to `voltest/dev_notes.md` for review when user wakes.
