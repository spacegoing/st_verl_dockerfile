# Dev Notes — Fix KubeRay + Volcano Config Execution

> Live log of executing `fix_kuberay_volcano_config_plan.md`.
> Started: 2026-04-15

## Phase 0 — Preflight snapshot

**Status: COMPLETE** (2026-04-15 14:28–14:32 UTC)

Backup dir: `/root/k8s_fix_backup/20260415/`

Idle check:
- `kubectl get rayjob,raycluster -A` filtered for non-terminal → no rows ✅
- `kubectl get podgroups.scheduling.volcano.sh -A` → `No resources found` ✅

Files captured:
| File | Size | Content |
|---|---|---|
| `kuberay_helm_v2.yaml` | 464 KB | Helm release secret for kuberay-operator revision 2 (current) |
| `kuberay_deploy.yaml` | 4.0 KB | Current kuberay-operator Deployment manifest |
| `kueue_cluster_resources.yaml` | 3.1 KB | `b300-training-queue` ClusterQueue + `b300-nodes` ResourceFlavor |
| `kueue_localqueue.yaml` | 1.4 KB | `default/training-queue` LocalQueue |
| `kueue_workloads.yaml` | 52 KB | 4 Kueue Workload objects (all FINISHED=True) |
| `kueue_crds.yaml` | 1.8 MB | All 11 Kueue CRDs full definitions (for restore) |

**Deviation**: first attempt at `kueue_crds.yaml` produced 0 bytes because `xargs` subshell didn't inherit `NO_PROXY`; fresh `kubectl` went through the system proxy and got `Forbidden`. Fixed by wrapping the pipeline in `HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 bash -c '...'` so env propagates to the xargs-spawned kubectl. User decision: option A (retry). Retry succeeded.

Gate 0 satisfied: cluster is idle, all snapshots captured.

## Phase 1 — Drain Kueue CRD instances

**Status: COMPLETE** (2026-04-15 14:33 UTC)

Deletions executed in order (LocalQueue → ClusterQueue → ResourceFlavor):

| Resource | Before | Command result | After |
|---|---|---|---|
| `localqueue default/training-queue` | 0 admitted, 0 pending | `"training-queue" deleted from default namespace` | gone |
| `clusterqueue b300-training-queue` | 0 pending workloads | `"b300-training-queue" deleted` | gone |
| `resourceflavor b300-nodes` | 19d old | `"b300-nodes" deleted` | gone (confirmed after 3s delay) |

Minor deviation: first `kubectl get resourceflavor` immediately after delete returned a stale result still showing `b300-nodes`. A follow-up query 3s later correctly returned `No resources found`. Treated as k8s API eventual-consistency artifact, not a real issue.

4 historic Workload objects still present (all `FINISHED=True`); they have ownerReferences to the deleted LocalQueue and will be reaped when the Kueue CRDs are removed in Phase 2.

Gate 1 satisfied.

## Phase 2 — Uninstall Kueue (official command)

**Status: COMPLETE** (2026-04-15 14:34–14:35 UTC)

Proxy reachability pre-check: `curl https://github.com/...` via proxy → HTTP 200 in 2.1s. ✅

Command executed:
```bash
HTTPS_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
HTTP_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
NO_PROXY=180.184.249.201 \
kubectl delete -k "github.com/kubernetes-sigs/kueue/config/default?ref=main"
```

Objects deleted (observed in output):
- namespace `kueue-system` (cascading removes deployment, configmaps, secrets, services, serviceaccounts)
- 39 ClusterRoles (`kueue-*-{editor,viewer}-role`, manager-role, metrics roles, etc.)
- 4 RoleBindings + 2 ClusterRoleBindings
- 2 APIServices (`v1beta1.visibility.kueue.x-k8s.io`, `v1beta2.visibility.kueue.x-k8s.io`)
- `kueue-mutating-webhook-configuration` + `kueue-validating-webhook-configuration`
- All 11 CRDs (admissionchecks, clusterqueues, cohorts, localqueues, multikueueclusters, multikueueconfigs, provisioningrequestconfigs, resourceflavors, topologies, workloadpriorityclasses, workloads) ✅

Minor deviation: 4 `NotFound` errors trailing the command for resources that exist in `ref=main` but were not in our installed v0.16.1:
- `kueue-rayservice-editor-role`, `kueue-rayservice-viewer-role`
- `kueue-sparkapplication-editor-role`, `kueue-sparkapplication-viewer-role`

These are version-skew artifacts (upstream added RayService + SparkApplication integrations after v0.16.1). Treated as benign because the missing objects were already absent, so the delete-everything intent was still achieved. No remediation needed.

Sanity checks (all 8 clean):
```
1. kubectl get ns kueue-system                → Error: NotFound ✅
2. kubectl get crd | grep -i kueue            → empty ✅
3. kubectl get validatingwebhookconfiguration | grep -i kueue → empty ✅
4. kubectl get mutatingwebhookconfiguration   | grep -i kueue → empty ✅
5. kubectl get clusterrole                    | grep -i kueue → empty ✅
6. kubectl get clusterrolebinding             | grep -i kueue → empty ✅
7. kubectl get workload -A                    → "could not find workloads.kueue.x-k8s.io" ✅
8. kubectl get apiservice                     | grep -i kueue → empty ✅
```

Gate 2 satisfied: Kueue fully uninstalled.

## Phase 3 — Install helm CLI

**Status: COMPLETE** (2026-04-15 14:34 UTC)

Option B (aliyun mirror) **FAILED** — `https://mirrors.aliyun.com/helm/helm-v3.14.0-linux-amd64.tar.gz` returned HTTP 404. Aliyun removed this specific path or version.

Option A (official get-helm-3 script via proxy) **SUCCEEDED**:
```bash
HTTPS_PROXY=...(proxy)... curl -fsSL -o /tmp/get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod +x /tmp/get_helm.sh
HTTPS_PROXY=...(proxy)... bash /tmp/get_helm.sh
# Installer fetched: helm-v3.20.2-linux-amd64.tar.gz (latest stable)
# Installed to: /usr/local/bin/helm
```

Verification:
```
$ helm version
version.BuildInfo{Version:"v3.20.2", GitCommit:"8fb76d6...", GoVersion:"go1.25.9"}

$ HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 helm list -n kube-system
NAME              NS           REV  UPDATED                              STATUS    CHART
grafana           kube-system  2    2026-03-19 03:58:10 UTC              deployed  grafana-8.9.3-...
kuberay-operator  kube-system  2    2026-03-19 03:58:12 UTC              deployed  kuberay-operator-1.5.1
volcano           kube-system  2    2026-03-19 03:58:00 UTC              deployed  volcano-1.11.0
```

Deviation: installed helm v3.20.2 (current latest stable) rather than the v3.14.0 in the plan; this is a non-issue because `helm upgrade` only needs the client to be ≥ the API version of the stored release, and v3.20 is fully backward-compatible with v3 releases.

Additional finding: Volcano chart version is **v1.11.0**. Useful data point if we later want to consult the matching docs.

Gate 3 satisfied.

## Phase 4 — Enable Volcano batchScheduler in kuberay-operator

**Status: COMPLETE** (2026-04-15 14:35–14:37 UTC)

### 4a. Prepared helm repo
```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/     # "kuberay" added
helm repo update                                                       # success
helm search repo kuberay/kuberay-operator --version 1.5.1              # chart 1.5.1 found
```

### 4b. Dry-run diff (inspected before committing)
Confirmed three critical diffs against revision 2:
- Container arg added: `--batch-scheduler=volcano` (dry-run line 499)
- ClusterRole `kuberay-operator` gains rule for `podgroups.scheduling.volcano.sh` with verbs `create,delete,get,list,update,watch`
- Image preserved as `registry.cn-sh-02.sensecore.cn/ccr-kuberay/operator:v1.5.1` (`--reuse-values` did what we wanted — no reset to quay.io)

### 4c. Real upgrade
```
helm upgrade kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 --namespace kube-system \
  --reuse-values --set batchScheduler.name=volcano
# → Release "kuberay-operator" has been upgraded. STATUS: deployed, REVISION: 3
kubectl -n kube-system rollout status deploy/kuberay-operator --timeout=120s
# → "kuberay-operator" successfully rolled out
```

### 4d. Post-upgrade verification (all 5 pass)

**Check 1 — Operator args now include `--batch-scheduler=volcano`** ✅
```json
[
  "--feature-gates=RayClusterStatusConditions=true,...",
  "--batch-scheduler=volcano",     ← NEW
  "--log-stdout-encoder","json","--log-file-encoder","json",
  "--enable-leader-election=true","--enable-metrics=true",
  "--reconcile-concurrency=1","--qps=100","--burst=200"
]
```

**Check 2 — Operator pod is Running with fresh rollout** ✅
```
NAME                               READY   STATUS    RESTARTS   AGE
kuberay-operator-f58bf4ccc-hkb2x   1/1     Running   0          25s
```

**Check 3 — Operator logs confirm Volcano init** ✅
```
{"level":"info","logger":"setup","msg":"Feature flag batch-scheduler is enabled","scheduler name":"volcano"}
{"level":"info","logger":"controllers.RayCluster","msg":"Starting EventSource","source":"kind source: *v1beta1.PodGroup"}
```
The second line shows the RayCluster controller is now watching Volcano PodGroup objects — exactly the behavior we want. It will create a PodGroup for every RayCluster/RayJob going forward.

**Check 4 — ClusterRole has PodGroup rule** ✅
```yaml
- apiGroups:
  - scheduling.volcano.sh
  resources:
  - podgroups
  verbs: [create, delete, get, list, update, watch]
```

**Check 5 — Helm release bumped to revision 3** ✅
```
kuberay-operator   kube-system   3   2026-04-15 14:36:15 UTC   deployed   kuberay-operator-1.5.1
```

Gate 4 satisfied — **PLAN COMPLETE.**

## Summary

| Change | Before | After |
|---|---|---|
| Kueue installed | Yes (v0.16.1, raw YAML) | **Removed** |
| Kueue CRs | 1 ClusterQueue, 1 ResourceFlavor, 1 LocalQueue, 4 Workloads | **0** |
| KubeRay helm revision | 2 | **3** |
| KubeRay `--batch-scheduler=volcano` | Absent | **Present** |
| KubeRay watches PodGroups | No (no event source) | **Yes** (`Starting EventSource: *v1beta1.PodGroup`) |
| KubeRay RBAC for PodGroups | None | **create/delete/get/list/update/watch** |
| Volcano scheduler config | gang plugin enabled | **unchanged (already correct)** |
| Image pinning | Sensetime internal registry | **Preserved via `--reuse-values`** |

Total wall clock: ~10 minutes (14:28 – 14:37 UTC).

No rollback performed. If ever needed, backup dir: `/root/k8s_fix_backup/20260415/`.

## Deviations from plan (all benign)

1. Phase 0: initial `kueue_crds.yaml` was empty due to xargs not inheriting NO_PROXY. User chose option A (retry with env at shell level); retry succeeded.
2. Phase 1: `kubectl get resourceflavor` immediately post-delete returned stale data; 3-second delay resolved it. k8s API eventual-consistency artifact.
3. Phase 2: 4 `NotFound` errors for `kueue-rayservice-{editor,viewer}-role` and `kueue-sparkapplication-{editor,viewer}-role` — these existed in `ref=main` but not in our installed v0.16.1 (new upstream integrations); benign.
4. Phase 3: Aliyun helm mirror 404'd; fell back to official get-helm-3 script via proxy. Installed v3.20.2 (current latest) instead of v3.14.0 from plan — fully backward-compatible, no issue.

## What was NOT done (deferred, per user direction)

- No changes to any RayJob / RayCluster yaml
- No smoke test submitting a RayJob to verify PodGroup auto-creation
- No concurrent 2×16-node verification
- No cleanup of `schedulerName: volcano` from pod templates (still harmless, KubeRay now owns the scheduling anyway)
- No node-group queue binding (for me+intern split)

These land in a separate follow-up plan when you're ready.
