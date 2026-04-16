# KubeRay + Volcano: Install From Scratch

> End-to-end implementation guide to reproduce the working setup on a new
> Kubernetes cluster. Based on our running config (validated by the voltest
> 10×8-node gang scheduling run, 2026-04-15).
>
> **Stack**: KubeRay `v1.5.1` + Volcano `v1.11.0` with Volcano PodGroup gang
> scheduling enabled at the kuberay-operator level. **No Kueue.** No other
> admission controllers needed for single-tenant GPU training clusters.

---

## 1. Prerequisites

### Cluster
- Kubernetes control plane (any recent version; we run v1.28+)
- kubeconfig on your workstation with write access to `kube-system` namespace and cluster-scoped resources (CRDs, ClusterRoles)
- `kubectl` installed; proxy env configured so kubectl reaches the API (ours: `NO_PROXY=<k8s-api-ip>`)
- `helm` ≥ v3.14 installed locally. If absent:
  ```bash
  curl -fsSL -o /tmp/get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
  chmod +x /tmp/get_helm.sh && bash /tmp/get_helm.sh
  helm version        # expect v3.x
  ```

### GPU nodes
- NVIDIA GPU operator installed → `nvidia.com/gpu` is an allocatable resource
- RDMA NICs exposed (optional, needed for RoCE cross-node NCCL) → `rdma-training/roce` device plugin. Ours: 8 mlx5 NICs per B300 node surfaced as `rdma-training/roce: 1` per whole-node claim.
- Container runtime supports NVIDIA (containerd + nvidia-container-toolkit)

### Registry
- Access to an internal image registry that can pull the Ray image. Ours: `registry.cn-hangzhou.aliyuncs.com/spacegoing/` + `registry.cn-sh-02.sensecore.cn/ccr-kuberay/`.
- `imagePullSecret` configured in the target namespace for private registries (ours: `aliyunsecret` in `default`).

### Storage
- Shared filesystem (NFS / Lustre / PVC) mounted into pods if you need to share code/data across RayJobs. Ours: `pvc-jdwzpnv`. **Note PFS root-squash** — see Appendix A.

---

## 2. Install Volcano

Per Volcano's official Quick Start Guide (https://github.com/volcano-sh/volcano#quick-start-guide):

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano \
    -n volcano-system --create-namespace \
    --version 1.11.0
```

**Our variant**: we use `-n kube-system` instead of `volcano-system` because
that is where the admin placed it. Both work; pick whichever matches your
cluster conventions.

### Verify
```bash
kubectl get pods -n volcano-system           # or kube-system
# Expected (3 deployments + 1 one-shot admission-init Job):
#   volcano-admission-xxxxxx          1/1  Running
#   volcano-controllers-xxxxxx        1/1  Running
#   volcano-scheduler-xxxxxx          1/1  Running
#   volcano-admission-init-xxxxxx     0/1  Completed

kubectl get crd | grep volcano.sh
# Expected 7 CRDs:
#   commands.bus.volcano.sh
#   jobflows.flow.volcano.sh
#   jobs.batch.volcano.sh
#   jobtemplates.flow.volcano.sh
#   numatopologies.nodeinfo.volcano.sh
#   podgroups.scheduling.volcano.sh        ← critical: auto-created gang groups live here
#   queues.scheduling.volcano.sh
```

### Verify the Volcano scheduler has `gang` plugin enabled

The `gang` plugin is what enforces all-or-nothing placement. It is enabled by
default in the Volcano helm chart, but verify the configmap:

```bash
kubectl get cm -n kube-system volcano-scheduler-configmap \
    -o jsonpath='{.data.volcano-scheduler\.conf}'
```

Expected content:
```yaml
actions: "allocate, reclaim"
tiers:
- plugins:
  - name: priority
  - name: gang               # ← this is the gang-scheduling enforcer
    enablePreemptable: false
  - name: conformance
- plugins:
  - name: predicates
  - name: capacity
  - name: nodeorder
  - name: binpack
```

If `gang` is missing, edit the configmap (`kubectl -n kube-system edit cm
volcano-scheduler-configmap`); Volcano scheduler auto-reloads.

**Default `default` Queue** is auto-created by Volcano. Verify:
```bash
kubectl get queues.scheduling.volcano.sh
# Expected: default (56d)  root (56d)
```

---

## 3. Install KubeRay operator WITH Volcano batch scheduling

Per KubeRay's official Volcano integration doc
(https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html):

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# The critical flag is --set batchScheduler.name=volcano
helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.5.1 \
    -n kube-system \
    --set batchScheduler.name=volcano
```

**What `batchScheduler.name=volcano` does**:
1. Adds `--batch-scheduler=volcano` to the operator's container args
2. Grants the operator ClusterRole permissions on `podgroups.scheduling.volcano.sh`
3. Makes the operator automatically create one `PodGroup` per RayCluster/RayJob with `minMember = head + len(workerReplicas)`, and set `schedulerName: volcano` on all pods

Without this flag, KubeRay uses the default kube-scheduler and **no PodGroups are created** — which is the broken config we had to fix (see `fix_kuberay_volcano_config_plan.md`).

### Verify
```bash
kubectl -n kube-system get deploy kuberay-operator
# Expected: 1/1 Ready

kubectl -n kube-system get deploy kuberay-operator \
    -o jsonpath='{.spec.template.spec.containers[0].args}' | python3 -m json.tool
# Must include: "--batch-scheduler=volcano"

kubectl -n kube-system logs -l app.kubernetes.io/name=kuberay-operator --tail=50 \
    | grep -iE "volcano|batch.scheduler|podgroup"
# Expected lines (paraphrased):
#   Feature flag batch-scheduler is enabled    scheduler name: volcano
#   controllers.RayCluster  Starting EventSource   kind source: *v1beta1.PodGroup
```

`kubectl get crd | grep ray.io` should show: `rayclusters`, `rayjobs`, `rayservices`.

### If your cluster uses a private registry (like ours)

The stock helm chart pulls from `quay.io/kuberay/operator:v1.5.1`. If your
cluster can't reach quay.io, override the image at install time:

```bash
helm install kuberay-operator kuberay/kuberay-operator \
    --version 1.5.1 \
    -n kube-system \
    --set batchScheduler.name=volcano \
    --set image.repository=registry.cn-sh-02.sensecore.cn/ccr-kuberay/operator \
    --set image.tag=v1.5.1
```

**Important — upgrading an already-installed operator**:
```bash
helm upgrade kuberay-operator kuberay/kuberay-operator \
    --version 1.5.1 -n kube-system \
    --reuse-values \
    --set batchScheduler.name=volcano
```
`--reuse-values` preserves the admin's image override; without it, the chart
resets image fields to defaults → pods ImagePullBackOff.

---

## 4. End-to-end smoke test

Minimal RayJob that proves the stack works. Save as `smoke-rayjob.yaml`:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: smoke-test
  namespace: default
spec:
  entrypoint: python -c "import ray; ray.init(); print('Ray sees GPUs:', ray.cluster_resources().get('GPU', 0))"
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 30

  rayClusterSpec:
    rayVersion: '2.54.0'
    headGroupSpec:
      rayStartParams:
        num-gpus: '8'
      template:
        spec:
          containers:
          - name: ray-head
            image: <your-ray-image-with-cuda>
            resources:
              requests: { nvidia.com/gpu: '8', cpu: '8', memory: 32Gi }
              limits:   { nvidia.com/gpu: '8', cpu: '8', memory: 32Gi }
    workerGroupSpecs:
    - groupName: workers
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams: { num-gpus: '8' }
      template:
        spec:
          containers:
          - name: ray-worker
            image: <your-ray-image-with-cuda>
            resources:
              requests: { nvidia.com/gpu: '8', cpu: '8', memory: 32Gi }
              limits:   { nvidia.com/gpu: '8', cpu: '8', memory: 32Gi }
```

Apply and verify the PodGroup is auto-created:

```bash
kubectl apply -f smoke-rayjob.yaml

# Within 5-10 seconds:
kubectl get podgroup
# Expected: one PodGroup named "ray-smoke-test-pg" with minMember=2

kubectl describe podgroup ray-smoke-test-pg | grep -E 'MinMember|Phase|Queue'
# MinMember: 2
# Phase:     Running  (or Pending briefly until both nodes free)
# Queue:     default

kubectl get pod -l ray.io/cluster
# Both head + worker Running on 2 distinct nodes

# Wait for completion (30-60s for this trivial job)
kubectl get rayjob smoke-test -o jsonpath='{.status.jobStatus}'     # SUCCEEDED
kubectl get rayjob smoke-test -o jsonpath='{.status.jobDeploymentStatus}'  # Complete
```

If the PodGroup is auto-created and both pods schedule on distinct nodes,
Volcano gang scheduling is working.

---

## 5. Production RayJob template

Based on our working voltest template (see `voltest/yaml/voltest-rayjob-template.yaml`).

### Essential settings

| Field | Value | Why |
|---|---|---|
| `shutdownAfterJobFinishes` | `true` | Tears down the RayCluster when entrypoint exits; otherwise the cluster lingers indefinitely |
| `ttlSecondsAfterFinished` | `30` (short!) | High values (e.g. 300) hold nodes for minutes after completion, blocking gang-promotion of next queued job. See Appendix B. |
| `rayVersion` | match the Ray in your image | KubeRay uses this for compatibility checks; `2.54.0` works for Ray 2.54 |
| `headGroupSpec.rayStartParams.num-gpus` | `'8'` | Tell Ray how many GPUs this pod owns (must match container resources.requests.nvidia.com/gpu) |
| `workerGroupSpecs[].replicas` = `minReplicas` = `maxReplicas` | equal | Fixes the gang size. No autoscaling. Autoscaling with gang scheduling is messy — keep pinned. |
| `imagePullSecrets: [ name: <secret> ]` | your internal registry secret | Present on both head and worker pod specs |
| Resources **requests** = **limits** per pod | whole node (e.g. 176 CPU, 1920Gi RAM, 8 GPU, 1 RDMA) | Forces 1-pod-per-node; avoids co-location |
| `securityContext.capabilities.add: [IPC_LOCK]` | required | For RDMA + shm-based NCCL/NVSHMEM |
| `emptyDir volume for /dev/shm (medium: Memory, sizeLimit: 60Gi)` | required | Ray/PyTorch use /dev/shm heavily; default 64Mi breaks nccl |

### Anti-pattern: do NOT set these

- `ray.io/scheduler-name: volcano` label — **not needed since KubeRay v1.3.0**; the operator's batch-scheduler flag sets `schedulerName: volcano` on pods automatically (https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html).
- `schedulerName: volcano` directly on pod template — same reason; operator-managed.
- `kueue.x-k8s.io/queue-name` label — Kueue not installed in this stack.
- Shell injection into `entrypoint` — keep it as a plain command; use env vars for parameterization.

### Reference full template

See:
- `voltest/yaml/voltest-rayjob-template.yaml` — 8-pod gang scheduling reference
- `iter_kuberay_32nodes_verl_training/yaml/stage10-*.yaml` — 16-node production training jobs

---

## 6. Optional: node-group isolation (multi-user sharing)

If you want to split the cluster (e.g. `me=26 nodes`, `intern=5 nodes`) with
Volcano alone (no Kueue), use the **NodeGroup plugin**
(https://volcano.sh/en/docs/user-guide/how_to_use_nodegroup_plugin/). See
`kueue_vs_volcano_deep_analysis.md` §Q2 for the full procedure:

1. Label nodes with `volcano.sh/nodegroup-name=<group>`.
2. Create Volcano `Queue` objects with `spec.affinity.nodeGroupAffinity.requiredDuringSchedulingIgnoredDuringExecution: [<group>]`.
3. Enable the `nodegroup` plugin in `volcano-scheduler-configmap` with `arguments: { strict: true }`.
4. Users label their RayJob with `volcano.sh/queue-name: <queue>`.

Not needed for single-user clusters.

---

## Appendix A — PFS / shared-storage root-squash

On parallel filesystems exposed through k8s PVCs (NFS v4 with `root_squash`,
some Lustre configurations, etc.), **root on the host ≠ root in the pod**.
Pods run as `anonuid` (often `10000` or `65534`) when opening files.

Symptom: `[Errno 13] Permission denied` when a pod reads a file you
created on the host as root. Example from our voltest run:
```
python3: can't open file '/root/myCodeLab/host/voltest/gpu_burn.py': [Errno 13] Permission denied
```

Fix: grant world-read on any file/dir that pods need to read:
```bash
chmod -R o+rX /path/to/code
# o+rX: world-read on files; world-read-and-execute on dirs (not files)
```

For log output dirs where pods *write*:
```bash
chmod o+w /path/to/logs
# Files the pod creates will be owned by the squashed UID (e.g. 10000)
```

---

## Appendix B — `ttlSecondsAfterFinished` gotcha (Bug we caught)

KubeRay's `spec.ttlSecondsAfterFinished` controls how long the RayCluster
lives after the RayJob entrypoint exits. Setting it too high (e.g. 300 = 5
min) means **the 8 pods keep holding their nodes for 5 min after the job
SUCCEEDED**, blocking Volcano from promoting the next Inqueue PodGroup.

Symptom: RayJob `status.jobStatus=SUCCEEDED` but `kubectl get pod` still
shows 8 Running pods, and the next queued PodGroup stays `Inqueue` for
minutes even though capacity is free.

Fix: set `ttlSecondsAfterFinished: 30` (or lower). For high-throughput
batch workflows, use an external auto-cleanup loop:

```bash
while :; do
  for job in $(kubectl get rayjob --no-headers \
      | awk '$2=="SUCCEEDED" || $2=="FAILED"{print $1}'); do
    kubectl delete rayjob "$job" --wait=false
  done
  sleep 20
done
```

This frees nodes within seconds of job completion.

---

## Appendix C — Why not Kueue?

For single-user GPU clusters, **Kueue adds complexity without solving the
actual failure mode**. See `kueue_vs_volcano_deep_analysis.md` for the full
analysis, but briefly:

- Kueue does **admission control** ("does this workload fit in the tenant's quota?"), not **pod placement**.
- Kueue's gang-scheduling feature (`waitForPodsReady`) is **reactive** — pods partially bind, then get evicted on timeout. Lots of churn under contention.
- Volcano's PodGroup gang is **proactive** — pods never partially bind; they wait until `minMember` nodes are simultaneously free, then bind atomically.
- Our actual failure mode in production (March 2026 Incidents A/B/C) was partial placement deadlock — a physical-placement problem. Volcano fixes it at the right layer; Kueue cannot.

Install Kueue if and only if you need tenant-level priority classes, cohort
borrowing, or multi-team quota fair-sharing. For our single-user B300
cluster, we removed it.

---

## Appendix D — Full dependency versions (our verified stack)

```
Kubernetes         v1.28+ (cluster-dependent)
Helm               v3.20+
Volcano            v1.11.0   (chart and appVersion)
KubeRay operator   v1.5.1    (chart and appVersion)
Ray (in pod image) v2.54.0
NVIDIA GPU driver  matching the container image CUDA (we run CUDA 13.0)
cuDNN              9.11+ (9.18 in our image; required by B300 MLA fused attention)
```

Image sourcing: we use `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` for pods and `registry.cn-sh-02.sensecore.cn/ccr-kuberay/operator:v1.5.1` for the operator (both mirrored from upstream since the cluster cannot reach quay.io/docker.io directly).

---

## References

1. **Volcano quickstart**: https://github.com/volcano-sh/volcano#quick-start-guide
2. **Volcano PodGroup docs**: https://volcano.sh/en/docs/podgroup/
3. **Volcano Queue docs**: https://volcano.sh/en/docs/queue/
4. **Volcano NodeGroup plugin**: https://volcano.sh/en/docs/user-guide/how_to_use_nodegroup_plugin/
5. **KubeRay operator install**: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html
6. **KubeRay RayJob quickstart**: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html
7. **KubeRay Volcano integration**: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html

## Internal references

- `kueue_vs_volcano_deep_analysis.md` — why we chose Volcano alone
- `fix_kuberay_volcano_config_plan.md` — the migration from broken config
- `fix_kuberay_volcano_dev_notes.md` — execution log for the migration
- `voltest/` — 10×8-node gang scheduling verification that validated this stack
