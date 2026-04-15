# Fix KubeRay + Volcano k8s Config — Execution Plan

> Scope: infra-side only. Remove Kueue; enable Volcano gang-scheduling in
> kuberay-operator. Training yamls, smoke tests, and operational polish are
> out of scope for this plan.
>
> Rationale (from `kueue_vs_volcano_deep_analysis.md`): single-user cluster +
> our failure mode is physical placement → Volcano PodGroup is the right
> mechanism; Kueue is unnecessary complexity.

---

## Probe results (evidence, already collected)

### Install method per component

| Component | Method | Key evidence |
|---|---|---|
| **Volcano** | Helm (release `volcano`, ns `kube-system`) | `sh.helm.release.v1.volcano.v1/v2` secret in kube-system; deploy annotation `meta.helm.sh/release-name: volcano`; label `app.kubernetes.io/managed-by: Helm` |
| **KubeRay** | Helm (release `kuberay-operator`, ns `kube-system`, chart 1.5.1) | `sh.helm.release.v1.kuberay-operator.v1/v2` secret; label `helm.sh/chart: kuberay-operator-1.5.1`; image `registry.cn-sh-02.sensecore.cn/ccr-kuberay/operator:v1.5.1` |
| **Kueue** | Raw YAML (version v0.16.1) | No helm release/secret; deploy annotation `kubectl.kubernetes.io/last-applied-configuration` present; image `registry.sensetime.com/sensecore-lepton/kueue:v0.16.1` |

Probe commands that established this:

```bash
kubectl get secret -n kube-system -l owner=helm
# Returns: sh.helm.release.v1.kuberay-operator.v{1,2}, sh.helm.release.v1.volcano.v{1,2}
# (No sh.helm.release.* for kueue → Kueue is NOT helm-managed.)

kubectl -n kube-system get deploy kuberay-operator -o jsonpath='{.metadata.labels}'
# {"app.kubernetes.io/managed-by":"Helm","helm.sh/chart":"kuberay-operator-1.5.1", ...}

kubectl -n kueue-system get deploy kueue-controller-manager -o jsonpath='{.metadata.annotations}'
# {"kubectl.kubernetes.io/last-applied-configuration": "{\"apiVersion\":...", ...}
# → raw kubectl apply signature, no helm annotations
```

### Current KubeRay args (the config gap we are fixing)

```bash
kubectl -n kube-system get deploy kuberay-operator \
  -o jsonpath='{.spec.template.spec.containers[0].args}'
```
Result:
```json
["--feature-gates=RayClusterStatusConditions=true,RayJobDeletionPolicy=false,
  RayMultiHostIndexing=false,RayServiceIncrementalUpgrade=false",
 "--log-stdout-encoder","json","--log-file-encoder","json",
 "--enable-leader-election=true","--enable-metrics=true",
 "--reconcile-concurrency=1","--qps=100","--burst=200"]
```
**Missing**: `--batch-scheduler=volcano`. This is what causes KubeRay to skip PodGroup auto-creation.

### Volcano scheduler config (already good — no change needed)

```bash
kubectl get cm -n kube-system volcano-scheduler-configmap -o jsonpath='{.data.volcano-scheduler\.conf}'
```
Result:
```
actions: "allocate, reclaim"
tiers:
- plugins: [priority, gang{enablePreemptable:false}, conformance]
- plugins: [predicates, capacity, nodeorder, binpack]
```
`gang` plugin is already enabled. Volcano is ready to enforce PodGroup gang-scheduling the moment KubeRay starts creating PodGroups.

### Kueue state to drain

```bash
kubectl get clusterqueue,resourceflavor,localqueue -A
```
Result:
```
clusterqueue/b300-training-queue    PENDING=0
resourceflavor/b300-nodes           AGE=19d
localqueue/default/training-queue   PENDING=0  ADMITTED=0

kubectl get workload -A
# 4 workloads, all FINISHED=True (safe to delete)
```

### Cluster idle check (prerequisite for execution)

```bash
kubectl get rayjob,raycluster -A | grep -vE 'SUCCEEDED|Complete|FAILED|Failed'
# Only header row → no RUNNING jobs. Safe to proceed.
```

---

## The fix (3 changes to cluster state)

1. Delete our 3 Kueue CRD instances (`ClusterQueue`, `ResourceFlavor`, `LocalQueue`).
2. Uninstall Kueue operator + CRDs (official kustomize command).
3. `helm upgrade` kuberay-operator with `--set batchScheduler.name=volcano`.

Nothing else is touched: Volcano stays as-is, training yamls untouched, no RayJob submissions, no scheduler-config edits.

---

## Execution plan — phased

### Phase 0 — Preflight snapshot (read-only, ~2 min)

Take backups so everything is reversible.

```bash
mkdir -p /root/k8s_fix_backup/$(date -u +%Y%m%d)
cd /root/k8s_fix_backup/$(date -u +%Y%m%d)

# KubeRay helm state (for rollback)
kubectl get secret -n kube-system sh.helm.release.v1.kuberay-operator.v2 -o yaml > kuberay_helm_v2.yaml
kubectl -n kube-system get deploy kuberay-operator -o yaml > kuberay_deploy.yaml

# Kueue state (for restore if we ever want it back)
kubectl get clusterqueue,resourceflavor -o yaml > kueue_cluster_resources.yaml
kubectl get localqueue -A -o yaml > kueue_localqueue.yaml
kubectl get workload -A -o yaml > kueue_workloads.yaml
kubectl get crd -o name | grep 'kueue\.x-k8s\.io' | \
  xargs -I{} kubectl get {} -o yaml > kueue_crds.yaml

# Confirm no live jobs
kubectl get rayjob,raycluster -A
```

**Gate**: confirm no running RayJobs before proceeding.

---

### Phase 1 — Drain Kueue CRD instances (~1 min)

Kueue has a validating webhook that blocks CR deletion if ClusterQueue has
pending workloads. Our state shows all workloads FINISHED, so this is clean.

```bash
# Delete in this order (LocalQueue is namespaced, references ClusterQueue;
# ClusterQueue references ResourceFlavor; deleting root-first would be blocked
# by finalizers until dependents are gone)
kubectl delete localqueue -n default training-queue
kubectl delete clusterqueue b300-training-queue
kubectl delete resourceflavor b300-nodes

# Verify
kubectl get clusterqueue,resourceflavor 2>&1
# Expect: "No resources found"
kubectl get localqueue -A 2>&1
# Expect: "No resources found in any namespace."
kubectl get workload -A 2>&1
# Workloads have ownerReferences to their parent job — typically already
# reaped. Any stragglers go away when CRDs are deleted in Phase 2.
```

**Gate**: all three kinds must show no resources.

---

### Phase 2 — Uninstall Kueue (official command, ~2 min)

Per Kueue's official uninstall guide (https://kueue.sigs.k8s.io/docs/installation/#uninstall):

```bash
kubectl delete -k "github.com/kubernetes-sigs/kueue/config/default?ref=main"
```

This deletes:
- namespace `kueue-system` (takes the deployment, configmaps, serviceaccounts)
- webhook configurations (`kueue-validating-webhook-configuration`, `kueue-mutating-webhook-configuration`)
- ClusterRole / ClusterRoleBinding objects
- All Kueue CRDs

**Proxy**: `kubectl apply/delete -k github.com/...` uses kustomize to fetch
from github. `github.com` is not reachable without proxy on this host, so:

```bash
HTTPS_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
HTTP_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
NO_PROXY=180.184.249.201 \
kubectl delete -k "github.com/kubernetes-sigs/kueue/config/default?ref=main"
```

(NO_PROXY keeps the k8s API reachable without going through the proxy.)

**Version note**: our install is v0.16.1 but the official command pins `ref=main`. Using `main` vs `v0.16.1` is normally fine for *uninstall* because the deletion targets the stable k8s kinds (namespaces, CRDs, webhook configs) whose names haven't changed. If `ref=main` deletes fewer objects than expected (e.g. a renamed resource), Phase 3 sanity-check will catch it and we fall back to targeted deletes.

**Sanity check post-delete**:

```bash
kubectl get ns kueue-system 2>&1             # Expect: Error from server (NotFound)
kubectl get crd 2>&1 | grep kueue            # Expect: empty
kubectl get validatingwebhookconfiguration 2>&1 | grep kueue   # Expect: empty
kubectl get mutatingwebhookconfiguration 2>&1 | grep kueue     # Expect: empty
kubectl get clusterrole 2>&1 | grep kueue    # Expect: empty
kubectl get clusterrolebinding 2>&1 | grep kueue  # Expect: empty
```

**Fallback** (only if `ref=main` left orphans):

```bash
kubectl delete ns kueue-system --ignore-not-found
kubectl delete validatingwebhookconfiguration kueue-validating-webhook-configuration --ignore-not-found
kubectl delete mutatingwebhookconfiguration kueue-mutating-webhook-configuration --ignore-not-found
kubectl delete clusterrole,clusterrolebinding -l app.kubernetes.io/name=kueue --ignore-not-found
kubectl get crd -o name | grep kueue.x-k8s.io | xargs -r kubectl delete
```

**Gate**: all six sanity checks return empty.

---

### Phase 3 — Install `helm` CLI on this ECS (~1 min)

`helm` is not installed on this host (`helm: command not found`). We need it
to drive the KubeRay upgrade in Phase 4.

Option A — official get-helm-3 script (via proxy, since it hits github releases):

```bash
curl -fsSL -o /tmp/get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod +x /tmp/get_helm.sh
HTTPS_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
HTTP_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
  bash /tmp/get_helm.sh
helm version  # expect v3.x
```

Option B — aliyun mirror (faster, no proxy):

```bash
curl -fsSL -o /tmp/helm.tar.gz https://mirrors.aliyun.com/helm/helm-v3.14.0-linux-amd64.tar.gz
tar -xzf /tmp/helm.tar.gz -C /tmp
install -m 755 /tmp/linux-amd64/helm /usr/local/bin/helm
helm version
```

Verify helm can read the existing release state from the cluster:

```bash
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
  helm list -n kube-system
# Expect rows:
#   kuberay-operator   kube-system   2   deployed   kuberay-operator-1.5.1
#   volcano            kube-system   2   deployed   volcano-...
```

If `helm list` fails, helm can't reach the API — re-check kubeconfig and NO_PROXY.

**Gate**: `helm list -n kube-system` shows the kuberay-operator release as
revision 2, deployed.

---

### Phase 4 — Enable Volcano batchScheduler in kuberay-operator (~3 min)

Per Ray's Volcano doc
(https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html):

> ```
> helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1 \
>   --set batchScheduler.name=volcano
> ```

For our existing install, the equivalent is `helm upgrade --reuse-values`.

```bash
# Add/update the KubeRay chart repo
HTTPS_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
HTTP_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
HTTPS_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
HTTP_PROXY=http://jdtcom:709a64b73eb3@10.119.176.202:3128 \
  helm repo update

# Dry-run first (shows what would change)
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
  helm upgrade kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 \
  --namespace kube-system \
  --reuse-values \
  --set batchScheduler.name=volcano \
  --dry-run

# Review the diff. Expected additions:
#   - container arg: "--batch-scheduler=volcano"
#   - ClusterRole update granting get/list/watch/create/update/patch/delete on
#     podgroups.scheduling.volcano.sh
#   - (possibly) RBAC for queues.scheduling.volcano.sh (read)
# No changes to image, replicas, resource requests.

# Real upgrade
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
  helm upgrade kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 \
  --namespace kube-system \
  --reuse-values \
  --set batchScheduler.name=volcano

# Wait for new pod to become ready
kubectl -n kube-system rollout status deploy/kuberay-operator --timeout=120s
```

**Why `--reuse-values`**: preserves the admin's existing settings, especially
the vendored image reference (`registry.cn-sh-02.sensecore.cn/ccr-kuberay/operator:v1.5.1`).
Without it, helm would reset everything to chart defaults, which may point at
`quay.io/kuberay/operator:v1.5.1` and fail to pull in this cluster.

**If image pull fails after upgrade** (because `--reuse-values` doesn't always
preserve `--set` scalars cleanly):

```bash
# Re-run with explicit image override
helm upgrade kuberay-operator kuberay/kuberay-operator \
  --version 1.5.1 \
  --namespace kube-system \
  --reuse-values \
  --set batchScheduler.name=volcano \
  --set image.repository=registry.cn-sh-02.sensecore.cn/ccr-kuberay/operator \
  --set image.tag=v1.5.1
```

**Post-upgrade verification** (configuration-level only; no RayJob submitted):

```bash
# 1. Operator args must now include --batch-scheduler=volcano
kubectl -n kube-system get deploy kuberay-operator \
  -o jsonpath='{.spec.template.spec.containers[0].args}' | python3 -m json.tool
# Expect the list to contain: "--batch-scheduler=volcano"

# 2. Operator pod must be Running with the new args
kubectl -n kube-system get pod -l app.kubernetes.io/name=kuberay-operator
# Expect: 1/1 Running, AGE < 2m (freshly rolled)

# 3. Operator logs should mention volcano batch scheduler
kubectl -n kube-system logs -l app.kubernetes.io/name=kuberay-operator --tail=100 | grep -i volcano
# Expect at least: "batch scheduler: volcano" or "volcano scheduler initialized"

# 4. RBAC for PodGroups should be in place
kubectl get clusterrole -o name | xargs -I{} sh -c '
  if kubectl get {} -o yaml | grep -q "podgroups.scheduling.volcano.sh"; then echo {}; fi
' 2>/dev/null
# Expect at least one ClusterRole matched (owned by kuberay-operator helm release)

# 5. Helm release revision bumped
helm list -n kube-system | grep kuberay-operator
# Expect REVISION=3 (was 2 before upgrade)
```

**Gate**: all five post-upgrade checks pass.

---

## Rollback plan

If Phase 4 breaks kuberay-operator for any reason:

```bash
# Roll back to prior revision (v2, the state before this plan ran)
helm rollback kuberay-operator 2 -n kube-system
kubectl -n kube-system rollout status deploy/kuberay-operator
```

If Kueue needs to come back (should not be needed, but possible if admin asks):

```bash
# Reinstall upstream Kueue at the same version we had
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.1/manifests.yaml
kubectl apply -f /root/k8s_fix_backup/<date>/kueue_cluster_resources.yaml
kubectl apply -f /root/k8s_fix_backup/<date>/kueue_localqueue.yaml
```

(Workloads are ephemeral; they don't need to be restored.)

---

## What this plan does NOT do

Explicitly out of scope (per user direction, deferred to later phases):

- Training yaml changes (removing `kueue.x-k8s.io/queue-name` label or
  `schedulerName: volcano` from pod templates).
- RayJob smoke tests to verify auto-PodGroup creation.
- Concurrent 2×16-node gang-scheduling verification.
- Node group / queue-binding setup (for me+intern split).

These land in a separate plan after Phase 4 completes successfully.

---

## Execution gates summary

| Gate | Check | Must-pass before proceeding |
|---|---|---|
| 0 | `kubectl get rayjob -A` shows no running jobs | → start Phase 1 |
| 1 | No `clusterqueue/resourceflavor/localqueue` remain | → start Phase 2 |
| 2 | Six sanity checks (ns/crd/webhooks/clusterroles) empty | → start Phase 3 |
| 3 | `helm list -n kube-system` lists kuberay-operator rev=2 | → start Phase 4 |
| 4 | Five post-upgrade checks pass; new arg present | → PLAN COMPLETE |

---

## References

- Kueue uninstall (official): https://kueue.sigs.k8s.io/docs/installation/#uninstall
- Ray Volcano install: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html
- Helm installer: https://helm.sh/docs/intro/install/
- Prior analysis: `k8s_kuberay_kueue_setup/kueue_vs_volcano_deep_analysis.md`
