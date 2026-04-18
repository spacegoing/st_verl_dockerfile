# Plan A — KubeRay + Volcano Correctness for verl Training

> Make sure every RayJob we submit for verl training conforms to the
> voltest-proven gang-scheduling pattern, so we never re-enter the
> March-incident partial-placement deadlock class.

Written: 2026-04-17 (for 0418 work)
References: `voltest/summary.md`, `k8s_kuberay_kueue_setup/fix_kuberay_volcano_config_plan.md` (Phase 4), `k8s_kuberay_kueue_setup/kueue_vs_volcano_deep_analysis.md`

---

## 1. Cluster state (verified 2026-04-17)

| Component | Status | Evidence |
|---|---|---|
| kuberay-operator `--batch-scheduler=volcano` | ✅ set | `kubectl -n kube-system get deploy kuberay-operator -o jsonpath='{.spec.template.spec.containers[0].args}'` contains `--batch-scheduler=volcano` |
| kuberay-operator pod | ✅ Running (up 2d2h) | `kuberay-operator-f58bf4ccc-hkb2x  1/1 Running` |
| volcano-scheduler | ✅ Available (deployed 58d, v1.11.2) | |
| volcano-scheduler-configmap `gang` plugin | ✅ enabled in tier 1 alongside `priority`, `conformance` | `kubectl get cm -n kube-system volcano-scheduler-configmap -o jsonpath='{.data.volcano-scheduler\.conf}'` |
| Kueue residuals | ✅ none — CRDs gone, `kueue-system` empty, no ClusterQueue/LocalQueue/ResourceFlavor | |
| Active RayJobs | 0 | 15 historical, all terminal |
| Active PodGroups | 0 | — |
| Free nodes | **31 compute nodes × 8 GPU = 248 B300 GPUs, fully idle** | |

The cluster itself is correctly configured. Any RayJob we submit now gets Volcano gang-scheduling for free.

## 2. The voltest reference template (what "correct" looks like)

File: `voltest/submit/rayjob.yaml` — used in the 10×8-node stress test that **passed**.

Properties we rely on:
1. `kind: RayJob`, `apiVersion: ray.io/v1`
2. `metadata.generateName: voltest-` — k8s appends a unique suffix per submission (no templating of names)
3. **No** `kueue.x-k8s.io/queue-name` label — Kueue is uninstalled, label would be a silent no-op
4. **No** `schedulerName: volcano` on pod specs — kuberay-operator sets it automatically because it was started with `--batch-scheduler=volcano` (voltest/submit/rayjob.yaml:13 comment)
5. **No** `ray.io/scheduler-name` label — not needed on KubeRay ≥ v1.3.0
6. **No** manual `PodGroup` CR — KubeRay creates it from the RayJob spec
7. `rayClusterSpec.workerGroupSpecs[0].{replicas, minReplicas, maxReplicas}` all equal — fixed cluster size, no autoscale, so minMember is deterministic
8. `shutdownAfterJobFinishes: true` + `ttlSecondsAfterFinished` sized for gang promotion speed (voltest used 30s; we use 1800s for post-run log inspection)
9. Head + worker template use YAML anchors (`&full_env`, `&full_mounts`, `&full_volumes`) so the two specs stay in sync

Behavior when all 9 properties hold: kuberay-operator creates exactly one PodGroup per RayJob with `minMember = 1 + workerReplicas`, Volcano holds all pods unscheduled until `minMember` nodes are simultaneously free → proactive gang.

## 3. Current verl-training production submit flow

File: `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml` + `submit/submit.sh`.

Assessment: **fully conforms to voltest pattern.** Specifically:

| Property | Status | Evidence |
|---|---|---|
| `generateName: bra40-sd-${COMBO_ID}-` | ✅ | submit/rayjob.yaml:22 |
| No kueue label | ✅ | submit/rayjob.yaml:24-28 (only submitter/framework/training-type/combo_id labels) |
| No manual `schedulerName: volcano` on pods | ✅ | submit/rayjob.yaml:47-49 ("kuberay-operator sets it automatically") |
| workerGroup `replicas == minReplicas == maxReplicas == 15` | ✅ | submit/rayjob.yaml:110-112 |
| `shutdownAfterJobFinishes: true` | ✅ | submit/rayjob.yaml:31 |
| `ttlSecondsAfterFinished: 1800` | ✅ (intentional, for log inspection) | submit/rayjob.yaml:32 |
| YAML anchors for env/mounts/volumes | ✅ | submit/rayjob.yaml:56, 100, 104 |

`submit/submit.sh`:
- Unsets `HTTPS_PROXY`/`HTTP_PROXY` (required — k8s API is outside proxy path)
- Sets `NO_PROXY=180.184.249.201`
- Validates combo id format `^c[0-9]+$`
- Runs `envsubst '${COMBO_ID} ${EXTRA_OVERRIDES}' < rayjob.yaml | kubectl create -f -`
- Prints rayjob name + watch/log commands

Production submission is correct. No action needed here.

## 4. Legacy yamls — divergences (confined to `yaml/legacy/`)

File prefix: `iter_kuberay_32nodes_verl_training/yaml/legacy/stage10*-rayjob.yaml` (23 files)

Reference: `yaml/legacy/stage10a-40bra-16node-sd-math-rayjob.yaml`

| Divergence | Location | Risk |
|---|---|---|
| Stale Kueue label `kueue.x-k8s.io/queue-name: training-queue` | line 23 | None at runtime (Kueue is uninstalled). Confusion risk: a maintainer reading this might think Kueue is still active. |
| Manual `schedulerName: volcano` on head pod | line 38 | None at runtime (operator would set the same value). Documentation drift from the voltest comment. |
| Manual `schedulerName: volcano` on worker pod | line 124 | Same as line 38. |
| Hard-coded `metadata.name: bra40-16node-sd-math` (no generateName) | line 18 | Cannot submit twice concurrently (name collision). Legacy yamls are single-shot. |

Active production never uses these files — `submit.sh` uses `submit/rayjob.yaml`. The legacy yamls exist only to document past experiments.

**Decision for this work**: leave `yaml/legacy/` untouched (deliberately archived), but add a short note at the top of `yaml/legacy/README.md` flagging the three drift items so no one copy-pastes them into a new yaml. The actual templates we will touch going forward are only `submit/rayjob.yaml`.

## 5. Expected Volcano behavior — what we verify at submit time

For every RayJob submission, the `submit.sh` operator should see, within ~5 s of apply:

```bash
kubectl get rayjob <name>                       # JOB STATUS=PENDING initially, then RUNNING
kubectl get podgroup                            # exactly one PodGroup, minMember=1+N_WORKERS
kubectl get pod -l ray.io/cluster=<cluster>     # all N+1 pods Pending or all N+1 pods Running
```

The critical gang invariant: pods are **never** in a mixed state (some Running, some Pending) for longer than the Ray init window (~30 s). If we observe partial binding for more than 60 s, gang scheduling has failed — stop and debug.

## 6. Verification checks for the 0418 debug run

Before declaring the 5-step debug run passed, all of these must hold:

| Check | How to verify |
|---|---|
| 1 PodGroup created | `kubectl get podgroup \| wc -l` = 2 (header + 1) |
| minMember matches total pods | `kubectl get podgroup -o jsonpath='{.items[0].spec.minMember}'` = 1 + workerReplicas |
| All pods either all Pending or all Running | snapshot 3× over first 2 min; no mixed states |
| No `Unschedulable` PodGroup events | `kubectl describe podgroup <name>` shows no `Unschedulable` |
| Ray dashboard reports correct node count | head pod `ray status` = N nodes |
| All GPUs visible | head pod `ray status` GPUs line = 8 × N |

If any check fails, stop and investigate before running longer jobs.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Someone copy-pastes a legacy yaml into a new run | Add README note to `yaml/legacy/` (Section 4). |
| kuberay-operator rolls back to pre-fix args after a helm upgrade | Before any production submit, re-verify: `kubectl -n kube-system get deploy kuberay-operator -o jsonpath='{.spec.template.spec.containers[0].args}'` contains `--batch-scheduler=volcano`. |
| Node drain / tenant contention causes PodGroup to stay `Inqueue` | Expected — that's Volcano correctly protecting the cluster. Wait for capacity. |
| `ttlSecondsAfterFinished: 1800` blocks next gang for 30 min | Acceptable: logs are valuable, and the cluster has enough idle nodes. If the next job needs to run sooner, `kubectl delete rayjob <name>` releases nodes immediately. |
| kube-system volcano-scheduler pod crashes mid-run | Surviving gang pods continue to run; future PodGroups stay Pending until scheduler recovers. Monitor via `kubectl -n kube-system get pod -l app=volcano-scheduler`. |

## 8. Deliverables under Plan A

1. `yaml/legacy/README.md` — describe the three drift items in legacy yamls so they aren't reused. ✏️
2. Check script: `0418/verify_gang.sh` — one-shot validator for the 6 checks in Section 6. ✏️
3. This plan (`0418/plan_a_kuberay_volcano.md`).

No changes to `submit/rayjob.yaml` or `submit/submit.sh` — they already conform.

## 9. Exit criteria

Plan A is satisfied when:
- [ ] Every active submit path (`submit/submit.sh <combo>`) is confirmed correct (already ✅)
- [ ] Legacy yamls carry a README warning (todo)
- [ ] `0418/verify_gang.sh` exists and passes against the 5-step debug run (todo)
- [ ] A 5-step debug RayJob runs end-to-end with all Section 6 checks passing (done jointly with Plan B)
