# KubeRay + Kueue + Volcano Setup Tutorial
## For verl Distributed GPU Training — From Vanilla K8s to Running RayJobs

> **Audience**: Complete Kubernetes beginner.
> **Goal**: Replicate the current production setup (32-node B300 cluster) on a new k8s cluster.
> **What you get at the end**: A cluster that can run multi-node verl RayJobs with GPU scheduling, job queuing, and RDMA networking.

---

## Part 0 — K8s Concepts You Must Know First

Before touching any commands, read this section. K8s has a lot of jargon that makes everything confusing until you have a mental model.

### The core idea

Kubernetes (k8s) is a system that manages **containers** across **many machines** as if they were one machine. You describe what you want to run ("I need 16 pods, each with 8 GPUs") and k8s figures out which physical nodes to put them on, restarts them if they crash, etc.

### Key objects (in order of importance to us)

| Object | What it is | Analogy |
|---|---|---|
| **Node** | A physical or virtual machine in the cluster | A server in a rack |
| **Pod** | The smallest runnable unit — one or more containers sharing network/storage | A running Docker container group |
| **Namespace** | A logical partition of the cluster — like folders. Most things live in `default` | A project folder |
| **PersistentVolumeClaim (PVC)** | A request for shared storage — like a network drive mounted into pods | NFS mount |
| **Secret** | Encrypted key-value store — used for image pull passwords, API keys | Passwords file |
| **CRD (Custom Resource Definition)** | Extends k8s with new object types. KubeRay adds `RayJob`, `RayCluster`. Kueue adds `ClusterQueue` etc. | Plugin that adds new YAML types |
| **Operator** | A controller that watches CRDs and acts on them. KubeRay operator watches `RayJob` objects and spawns pods | Background daemon that makes CRDs work |

### The four things we install

| Component | What it does | Why we need it |
|---|---|---|
| **KubeRay operator** | Watches `RayJob` YAML → spins up a Ray head pod + N worker pods → runs your entrypoint script | To run multi-node Ray jobs |
| **Volcano scheduler** | Gang scheduling: allocates ALL N pods at once (not one by one). Prevents deadlock where 15/16 pods got scheduled but the 16th can't fit | GPU training needs all nodes or none |
| **Kueue** | Job queue with quotas: admits jobs only when enough GPU quota is free; queues the rest | Prevents over-scheduling; enables concurrent 16-node jobs to share a 32-node cluster |
| **RDMA device plugin** | Exposes `rdma-training/roce` resource on each node so pods can request a RoCE NIC | High-speed GPU interconnect (NCCL) |

### How a RayJob flows through the system

```
You: kubectl apply -f stage10a-rayjob.yaml
         │
         ▼
   Kueue: "Do I have 128 GPUs + 16 RoCE available in b300-training-queue?"
   → If yes: admit the workload immediately
   → If no:  set status=SUSPENDED, wait in queue
         │
         ▼ (admitted)
   KubeRay operator: create RayCluster (1 head pod + 15 worker pods)
         │
         ▼
   Volcano scheduler: gang-schedule all 16 pods simultaneously onto 16 nodes
         │
         ▼
   Each pod: pull image, mount PVC, set env vars, start Ray
         │
         ▼
   Head pod: run entrypoint (run_40bra_k8s_16node_single_domain.sh c351)
         → starts Gym servers on all nodes via Ray remote tasks
         → ray job submit → verl training loop begins
         │
         ▼
   Training completes → RayJob status=SUCCEEDED → pods deleted
   Kueue: release 128 GPUs quota → next queued job gets admitted
```

### kubectl — the command-line tool

`kubectl` is the CLI for talking to the k8s API server. Every command follows the pattern:

```bash
kubectl <verb> <resource-type> [<name>] [flags]

# Verbs: get, apply, delete, describe, logs, exec
# Resource types: pod, node, rayjob, raycluster, clusterqueue, pvc, secret, ...
```

---

## Part 1 — What's Running on the Current Cluster

This is what you're replicating. Understanding each piece helps you adapt it.

### Cluster facts

| Item | Value |
|---|---|
| K8s distribution | K3s v1.34.2 |
| API server | `https://10.108.109.145:443` (or `180.184.249.201:8021` via NAT) |
| Nodes | 33 total: 32 GPU nodes (8× B300 each) + 1 virtual control-plane |
| GPU per node | 8× NVIDIA B300 (sm_103) |
| CPU / RAM per node | 176 cores / 1920 Gi |
| Network | 8× Mellanox ConnectX-7 RoCEv2 per node (mlx5_10–mlx5_17) |
| Storage | 1 PVC (`pvc-jdwzpnv`), 1 Pi, ReadWriteMany, QuarkFS CSI driver |
| Scheduler | Volcano (gang scheduling) |
| Job queue | Kueue (BestEffortFIFO, 256 GPU quota) |
| Ray version | 2.54.0 |
| Training image | `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` |

### Installed operators (all in `kube-system` except Kueue)

```
kuberay-operator         — watches RayJob/RayCluster CRDs
volcano-scheduler        — gang-schedules pods
volcano-controllers      — manages Volcano's Queue/PodGroup objects
volcano-admission        — webhook that patches pods with PodGroup annotations
kueue-controller-manager — admission controller in kube-system
```

### Kueue objects

```
ResourceFlavor: b300-nodes          — represents the homogeneous B300 nodes
ClusterQueue:   b300-training-queue — quota: 256 GPU, 32 RoCE, 5640 CPU, 61445Gi RAM
LocalQueue:     training-queue      — in namespace default, references ClusterQueue
```

RayJobs opt into Kueue by adding this label:
```yaml
labels:
  kueue.x-k8s.io/queue-name: training-queue
```

---

## Part 2 — Installing Everything on a New Cluster

### Prerequisites checklist

Before starting, your new cluster needs:
- [ ] Kubernetes cluster running (K3s, RKE2, kubeadm, or managed like EKS/GKE — K3s recommended for on-prem)
- [ ] `kubectl` installed on your workstation and `~/.kube/config` pointing at the new cluster
- [ ] `helm` v3 installed
- [ ] GPU nodes with NVIDIA drivers + `nvidia-container-toolkit` installed on nodes
- [ ] NVIDIA device plugin running in cluster (exposes `nvidia.com/gpu` resource)
- [ ] RDMA device plugin running (exposes `rdma-training/roce` resource) — cluster admin sets this up
- [ ] Shared storage accessible from all nodes (NFS, CephFS, or cloud PVC) with ReadWriteMany
- [ ] Container registry reachable from nodes (Aliyun, DockerHub, or private)

### Step 1 — Install KubeRay operator

KubeRay is the operator that turns `RayJob` YAML into actual pods.

```bash
# Add the KubeRay Helm repo
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install KubeRay operator into kube-system (or ray-system — pick one)
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace kube-system \
  --create-namespace \
  --version 1.3.0   # use latest stable; check: helm search repo kuberay

# Verify it's running
kubectl get pods -n kube-system | grep kuberay
# Expected: kuberay-operator-<hash>   1/1   Running
```

**Why KubeRay?** Without it, `kubectl apply -f rayjob.yaml` would fail — k8s doesn't know what a `RayJob` is until the operator registers the CRD.

### Step 2 — Install Volcano scheduler

Volcano is a batch scheduler optimized for AI workloads. It provides **gang scheduling** — all pods in a group are scheduled simultaneously or not at all.

```bash
# Install Volcano via Helm
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update

helm install volcano volcano-sh/volcano \
  --namespace kube-system \
  --create-namespace \
  --version 1.10.0   # use latest stable

# Verify
kubectl get pods -n kube-system | grep volcano
# Expected three pods:
#   volcano-admission-<hash>     1/1   Running
#   volcano-controllers-<hash>   1/1   Running
#   volcano-scheduler-<hash>     1/1   Running
```

**Why Volcano?** Default k8s scheduler schedules pods one by one. For 16 pods that all need 8 GPUs to train together, if only 15 fit, you'd have 15 pods running (wasting 120 GPUs) and 1 pod waiting forever. Volcano holds all 16 until the entire group can start at once.

To use Volcano in a pod spec, add:
```yaml
spec:
  schedulerName: volcano   # instead of the default "default-scheduler"
```

### Step 3 — Install Kueue

Kueue manages job admission — it controls which jobs run now vs. wait in queue, based on available quota.

```bash
# Install Kueue (uses a single manifest, not Helm)
# Check latest version: https://github.com/kubernetes-sigs/kueue/releases
VERSION=v0.10.0   # use latest stable
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/${VERSION}/manifests.yaml

# Verify
kubectl get pods -n kueue-system
# Expected: kueue-controller-manager-<hash>   2/2   Running
```

**Why Kueue?** Without Kueue, submitting 10 RayJobs simultaneously would try to create 10 × 16 = 160 pods at once, all competing for 256 GPUs and deadlocking. Kueue admits only 2 jobs at a time (each needs 128 GPUs, total = 256 = quota) and queues the rest.

### Step 4 — Enable Kueue integration with KubeRay

Kueue needs to know it should manage RayJob objects. This is done by patching the Kueue config:

```bash
kubectl edit configmap kueue-manager-config -n kueue-system
```

Make sure the `integrations.frameworks` section includes `ray.io/v1/RayJob`:

```yaml
apiVersion: config.kueue.x-k8s.io/v1beta1
kind: Configuration
integrations:
  frameworks:
  - "batch/v1/Job"
  - "ray.io/v1/RayJob"    # ADD THIS LINE if not present
  - "kubeflow.org/v1/MPIJob"
```

After editing, restart Kueue:
```bash
kubectl rollout restart deployment kueue-controller-manager -n kueue-system
```

### Step 5 — Create Kueue quota objects

This is the exact config from the current cluster, saved as `kueue-setup.yaml` in `iter_kuberay_32nodes_verl_training/yaml/kueue-setup.yaml`.

**Adapt these numbers to your cluster** (GPU count, node count, etc.):

```yaml
# kueue-setup.yaml
---
# ResourceFlavor — describes your node type (homogeneous GPU nodes)
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: b300-nodes   # rename to match your GPU type, e.g. "h100-nodes"
spec: {}             # empty = all nodes are equivalent

---
# ClusterQueue — total resource quota for the entire cluster
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: b300-training-queue   # can be any name
spec:
  namespaceSelector: {}       # {} = accessible from all namespaces
  queueingStrategy: BestEffortFIFO   # FIFO with best-effort (admitted in order, but can skip if resources fit)
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu", "rdma-training/roce", "cpu", "memory"]
    flavors:
    - name: b300-nodes   # must match ResourceFlavor name above
      resources:
      - name: nvidia.com/gpu
        nominalQuota: "256"      # CHANGE: total GPUs in your cluster (nodes × GPUs/node)
      - name: rdma-training/roce
        nominalQuota: "32"       # CHANGE: total RDMA adapters (1 per node typically)
      - name: cpu
        nominalQuota: "5640"     # CHANGE: total CPUs with small buffer
      - name: memory
        nominalQuota: "61445Gi"  # CHANGE: total RAM with small buffer

---
# LocalQueue — namespace-scoped queue that RayJobs reference
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: training-queue   # RayJobs use this name in their label
  namespace: default
spec:
  clusterQueue: b300-training-queue   # must match ClusterQueue name above
```

Apply it:
```bash
kubectl apply -f kueue-setup.yaml

# Verify
kubectl get resourceflavor
kubectl get clusterqueue
kubectl get localqueue -n default
```

### Step 6 — Set up storage (PVC)

Every pod needs to access your code and model weights. This is done via a PVC (PersistentVolumeClaim) backed by shared storage.

**On the current cluster**: PVC `pvc-jdwzpnv` is backed by QuarkFS (a parallel filesystem). On your new cluster, use whatever storage you have:
- **On-prem**: NFS server → create NFS StorageClass → create PVC
- **AWS**: EFS (elastic filesystem) with `efs.csi.aws.com` driver → PVC with `ReadWriteMany`
- **Azure**: Azure Files → PVC with `ReadWriteMany`
- **GCP**: Filestore → PVC with `ReadWriteMany`

> **Critical**: You must use `ReadWriteMany` (RWX) mode — all 32 pods need to read the same files simultaneously.

Example PVC for NFS-backed storage:
```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-pvc    # replace pvc-jdwzpnv with this name in your RayJob YAMLs
  namespace: default
spec:
  accessModes:
    - ReadWriteMany     # MUST be RWX — all pods read simultaneously
  resources:
    requests:
      storage: 10Ti     # adjust to your needs
  storageClassName: nfs-sc   # your storage class name
```

**PVC permission gotcha**: If your storage provider squashes root UID (common), pods running as root can't write to the PVC. Check:
```bash
# Create a test pod and try writing:
kubectl run pvc-test --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"v","persistentVolumeClaim":{"claimName":"training-pvc"}}],"containers":[{"name":"c","image":"busybox","command":["sh","-c","echo test > /mnt/test.txt && echo OK"],"volumeMounts":[{"name":"v","mountPath":"/mnt"}]}]}}'
kubectl logs pvc-test
# Should print "OK". If permission denied, ask your storage admin to disable root squash.
```

### Step 7 — Create image pull secret

Pods need credentials to pull images from private registries.

```bash
# For Aliyun (current cluster production registry):
kubectl create secret docker-registry aliyunsecret \
  --docker-server=registry.cn-hangzhou.aliyuncs.com \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PASSWORD \
  --docker-email=YOUR_EMAIL \
  -n default

# For any other registry (same command, change server/credentials):
kubectl create secret docker-registry myregistry-secret \
  --docker-server=YOUR_REGISTRY_URL \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PASSWORD \
  -n default
```

Then reference it in your RayJob YAML:
```yaml
imagePullSecrets:
- name: aliyunsecret   # or myregistry-secret
```

---

## Part 3 — Adapting the RayJob YAML for Your Cluster

The hard part of replication is getting the **network/RDMA environment variables** right. These are cluster-specific and wrong values cause silent training hangs or very slow NCCL communication.

### Things you MUST change

| YAML field | Current cluster value | What to change to |
|---|---|---|
| `image` | `registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev` | Your registry/image |
| `imagePullSecrets[0].name` | `aliyunsecret` | Your secret name |
| `claimName` | `pvc-jdwzpnv` | Your PVC name |
| `subPath` (code) | `lichang93/st_verl_dockerfile` | Path within your PVC |
| `subPath` (downloads) | `lichang93/downloads` | Path within your PVC |
| `NCCL_SOCKET_IFNAME` | `eth0` | Your ethernet interface name (run `ip a` on a node) |
| `NCCL_IB_GID_INDEX` | `5` | Your RoCEv2 GID index (see §3.1 below) |
| `NCCL_IB_TC` | `96` | Your RoCE traffic class (ask your network admin) |
| `NCCL_IB_HCA` | `mlx5_10:1,...,mlx5_17:1` | Your IB/RoCE HCAs (run `ibstat` on a node) |
| `nvidia.com/gpu` request | `'8'` | GPUs per node in your cluster |
| `rdma-training/roce` request | `'1'` | RDMA adapters per node (or remove if no RDMA) |
| `cpu` / `memory` request | `'176'` / `1920Gi` | Your node's CPU/RAM |
| `workerGroupSpecs[0].replicas` | `15` (16-node) or `31` (32-node) | N total nodes - 1 |
| `rayVersion` | `'2.54.0'` | Must match Ray version in your image |

### 3.1 — How to find your NCCL network parameters

Run these commands **on a GPU node** (SSH in, or run a debug pod):

```bash
# Find your ethernet interface name:
ip a | grep -E "^[0-9]+:" | grep -v lo
# Look for the interface with an IP on your cluster's subnet
# Common values: eth0, ens3, bond0, ib0

# Find your InfiniBand/RoCE HCAs:
ibstat 2>/dev/null || ibv_devinfo
# Lists all HCAs. Note the names (mlx5_0, mlx5_1, etc.)

# Find the active GID index for RoCEv2 (look for GID with "RoCE v2" type):
for dev in $(ls /sys/class/infiniband/); do
  for port in $(ls /sys/class/infiniband/$dev/ports/); do
    for gid_idx in $(ls /sys/class/infiniband/$dev/ports/$port/gids/); do
      gid=$(cat /sys/class/infiniband/$dev/ports/$port/gids/$gid_idx)
      gid_type=$(cat /sys/class/infiniband/$dev/ports/$port/gid_attrs/types/$gid_idx 2>/dev/null)
      if [[ "$gid_type" == *"RoCE v2"* ]]; then
        echo "$dev port $port GID[$gid_idx] = $gid (TYPE: $gid_type)"
      fi
    done
  done
done
# Note the GID index that shows "RoCE v2" — use that as NCCL_IB_GID_INDEX

# Find traffic class:
# Ask your network admin what QoS TC is reserved for RDMA training traffic.
# Typical values: 96 (dscp24), 106 (dscp26.5), 128 (dscp32)
# If no special QoS: use 0
```

### 3.2 — Annotated RayJob template (current cluster)

This is the canonical template extracted from `stage10a-40bra-16node-sd-math-rayjob.yaml`:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: my-training-job           # CHANGE: unique name for this job
  namespace: default
  labels:
    # These labels are for cluster admin visibility — adjust as you like
    lepton.sensetime.com/submitter: lichang93
    lepton.sensetime.com/framework-type: Ray
    # This label opts into Kueue admission control — REQUIRED for queuing
    kueue.x-k8s.io/queue-name: training-queue

spec:
  # The command to run in the head pod once Ray cluster is up
  entrypoint: bash /root/myCodeLab/host/verl/my_scripts/k8s/run_40bra_k8s_16node_single_domain.sh c351

  # Delete the RayCluster after job finishes (saves resources)
  shutdownAfterJobFinishes: true

  # Keep pods around for 30 min after finish so you can read logs
  ttlSecondsAfterFinished: 1800

  rayClusterSpec:
    # MUST match the Ray version installed in your container image
    # Check: docker run --rm <image> python3 -c "import ray; print(ray.__version__)"
    rayVersion: '2.54.0'

    # ─── HEAD NODE ──────────────────────────────────────────────────
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'   # expose Ray dashboard
        num-gpus: '8'               # GPUs to expose to Ray on this pod

      template:
        spec:
          # Volcano does gang scheduling (all pods start at once)
          schedulerName: volcano

          imagePullSecrets:
          - name: aliyunsecret       # CHANGE: your image pull secret

          containers:
          - name: ray-head
            image: registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
            imagePullPolicy: IfNotPresent   # use cached image if available

            # ── Environment variables ──────────────────────────────
            # YAML anchor — define once, reuse in worker with: env: *full_env
            env: &full_env

            # ── NCCL Network (CHANGE for your cluster) ─────────────
            # Which ethernet interface NCCL uses for node discovery
            - {name: NCCL_SOCKET_IFNAME,              value: eth0}
            # Which GID index is the RoCEv2 GID (run ibstat to find)
            - {name: NCCL_IB_GID_INDEX,               value: '5'}
            # QoS Traffic Class for RDMA (ask network admin; 0 = no QoS)
            - {name: NCCL_IB_TC,                      value: '96'}
            # Which RDMA HCAs to use (run ibstat to find names)
            - {name: NCCL_IB_HCA,                     value: 'mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_14:1,mlx5_15:1,mlx5_16:1,mlx5_17:1'}
            # IB timeout/retry — increase for flaky networks
            - {name: NCCL_IB_TIMEOUT,                 value: '22'}
            - {name: NCCL_IB_RETRY_CNT,               value: '13'}
            # Adaptive routing — use ECMP load balancing (1=on)
            - {name: NCCL_IB_ADAPTIVE_ROUTING,        value: '1'}
            # NCCL buffer + chunk sizes (tuned for B300/Mellanox)
            - {name: NCCL_BUFFSIZE,                   value: '16777216'}
            - {name: NCCL_P2P_NET_CHUNKSIZE,          value: '524288'}
            - {name: NCCL_CROSS_NIC,                  value: '1'}
            - {name: NCCL_MIN_NCHANNELS,              value: '32'}
            - {name: NCCL_IB_QPS_PER_CONNECTION,      value: '8'}
            # Disable NVLS (NVLink Sharp) — not available cross-node in k8s
            - {name: NCCL_NVLS_ENABLE,                value: '0'}

            # ── NVSHMEM / DeepEP (only needed for flex/DeepEP dispatcher) ──
            # If using alltoall dispatcher, you can remove these
            - {name: NVSHMEM_IB_GID_INDEX,            value: '5'}   # SAME as NCCL
            - {name: NVSHMEM_IB_TRAFFIC_CLASS,        value: '96'}  # SAME as NCCL
            - {name: NVSHMEM_IBGDA_NIC_HANDLER,       value: gpu}
            - {name: NVSHMEM_IB_ENABLE_IBGDA,         value: '1'}
            - {name: NVSHMEM_IBGDA_ENABLE_MULTI_PORT, value: '1'}
            - {name: NVSHMEM_HCA_LIST,                value: 'mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17'}
            - {name: MELLANOX_VISIBLE_DEVICES,        value: '10,11,12,13,14,15,16,17'}
            - {name: UCX_NET_DEVICES,                 value: eth0}

            # ── vLLM / Model training ──────────────────────────────
            # B300 is sm_103; vLLM's sm_100 check fails without this
            - {name: VLLM_ATTENTION_BACKEND,          value: CUTLASS_MLA}
            - {name: VLLM_USE_V1,                     value: '1'}
            # Transformer Engine fused attention (MLA)
            - {name: NVTE_FUSED_ATTN,                 value: '1'}
            # Required for Megatron tensor parallelism
            - {name: CUDA_DEVICE_MAX_CONNECTIONS,     value: '1'}

            # ── Python / Ray / Logging ─────────────────────────────
            # Redirect .pyc files off PFS (PFS __pycache__ causes slowness)
            - {name: PYTHONPYCACHEPREFIX,             value: /tmp/pycache}
            # Disable Ray telemetry
            - {name: RAY_USAGE_STATS_ENABLED,         value: '0'}
            - {name: RAY_ENABLE_OPENTELEMETRY,        value: '0'}
            # Surface NCCL errors immediately instead of hanging
            - {name: TORCH_NCCL_ASYNC_ERROR_HANDLING, value: '1'}

            # ── Resources (CHANGE for your nodes) ─────────────────
            resources:
              requests:
                nvidia.com/gpu: '8'          # GPUs per pod
                rdma-training/roce: '1'      # RDMA adapters per pod (remove if no RDMA plugin)
                cpu: '176'                   # CPUs per pod
                memory: 1920Gi              # RAM per pod
              limits:
                nvidia.com/gpu: '8'
                rdma-training/roce: '1'
                cpu: '176'
                memory: 1920Gi

            # ── Security: allow locked memory (needed for GPU RDMA) ──
            securityContext:
              capabilities:
                add: [IPC_LOCK]

            # ── Storage mounts ────────────────────────────────────
            volumeMounts:
            # Code mount: your repo becomes /root/myCodeLab/host inside pod
            - name: afs-pvc
              mountPath: /root/myCodeLab/host
              subPath: lichang93/st_verl_dockerfile   # CHANGE: path within your PVC
            # Data mount: models + datasets
            - name: afs-pvc
              mountPath: /root/myCodeLab/host/downloads
              subPath: lichang93/downloads             # CHANGE: path within your PVC
            # Shared memory (needed for CUDA graphs, large tensors)
            - name: shm
              mountPath: /dev/shm

          volumes:
          - name: afs-pvc
            persistentVolumeClaim:
              claimName: pvc-jdwzpnv    # CHANGE: your PVC name
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 60Gi           # /dev/shm size — keep at least 32Gi for large models

    # ─── WORKER NODES ────────────────────────────────────────────────
    workerGroupSpecs:
    - groupName: workers
      # CHANGE: (total_nodes - 1) workers. For 16-node: 15. For 32-node: 31.
      replicas: 15
      minReplicas: 15   # same as replicas for strict gang scheduling
      maxReplicas: 15   # same as replicas
      rayStartParams:
        num-gpus: '8'
      template:
        spec:
          schedulerName: volcano
          imagePullSecrets:
          - name: aliyunsecret
          containers:
          - name: ray-worker
            image: registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
            imagePullPolicy: IfNotPresent
            env: *full_env    # reuse head's env vars (YAML anchor)
            resources:
              requests:
                nvidia.com/gpu: '8'
                rdma-training/roce: '1'
                cpu: '176'
                memory: 1920Gi
              limits:
                nvidia.com/gpu: '8'
                rdma-training/roce: '1'
                cpu: '176'
                memory: 1920Gi
            securityContext:
              capabilities:
                add: [IPC_LOCK]
            volumeMounts:
            - name: afs-pvc
              mountPath: /root/myCodeLab/host
              subPath: lichang93/st_verl_dockerfile
            - name: afs-pvc
              mountPath: /root/myCodeLab/host/downloads
              subPath: lichang93/downloads
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: afs-pvc
            persistentVolumeClaim:
              claimName: pvc-jdwzpnv
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 60Gi
```

---

## Part 4 — Validation Stages (Run These In Order)

Don't jump straight to 32-node training. These stages find problems early and cheaply.

### Stage 0 — Image pull + PVC access

**Goal**: Confirm your image can be pulled and your PVC is readable.

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: sanity-test
  namespace: default
spec:
  schedulerName: volcano
  imagePullSecrets:
  - name: aliyunsecret       # your secret
  restartPolicy: Never
  containers:
  - name: test
    image: registry.cn-hangzhou.aliyuncs.com/spacegoing/myverl:ncr2602_vllm012.dev
    imagePullPolicy: IfNotPresent
    command:
    - bash
    - -c
    - |
      echo "=== Python + Ray version ==="
      python3 -c "import ray; print('Ray:', ray.__version__)"
      echo "=== Code mount ==="
      ls /root/myCodeLab/host/verl/my_scripts/
      echo "=== Downloads mount ==="
      ls /root/myCodeLab/host/downloads/models/
      echo "=== DONE ==="
    resources:
      requests:
        cpu: '2'
        memory: 4Gi
    securityContext:
      capabilities:
        add: [IPC_LOCK]
    volumeMounts:
    - name: pvc
      mountPath: /root/myCodeLab/host
      subPath: lichang93/st_verl_dockerfile
    - name: pvc
      mountPath: /root/myCodeLab/host/downloads
      subPath: lichang93/downloads
  volumes:
  - name: pvc
    persistentVolumeClaim:
      claimName: pvc-jdwzpnv    # your PVC
EOF

# Wait for it to complete
kubectl wait pod sanity-test --for=condition=Ready --timeout=120s
kubectl logs sanity-test

# Expected output: Ray version, script directory listing, model directory listing, DONE
kubectl delete pod sanity-test
```

**Common failures at this stage**:
- `ImagePullBackOff` → wrong image name or secret name/credentials
- `PVC not found` → wrong PVC name or PVC not in default namespace
- `Permission denied` on PVC → root squash enabled on storage; ask admin to disable
- `ls: cannot access ...` → wrong subPath

### Stage 1 — NCCL connectivity test

**Goal**: Confirm NCCL can communicate between 2 pods over RDMA.

The existing YAML `iter_kuberay_32nodes_verl_training/yaml/stage1-nccl-rayjob.yaml` is ready to apply. Before applying, update the `image` and `imagePullSecrets` and `claimName` if needed.

```bash
# Apply the NCCL test (2 nodes)
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
kubectl apply -f iter_kuberay_32nodes_verl_training/yaml/stage1-nccl-rayjob.yaml

# Watch pods start (should see 2 pods: 1 head, 1 worker)
kubectl get pods -w | grep nccl

# Get head pod name
HEAD=$(kubectl get pods -l ray.io/cluster=nccl-rayjob-test -l ray.io/node-type=head -o name)
kubectl logs -f $HEAD -c ray-head

# Expected output: allreduce bandwidth (e.g., "busbw 744.00 GB/s")
# If you see bandwidth > 100 GB/s, RDMA is working.

# Cleanup
kubectl delete rayjob nccl-rayjob-test
```

**Common failures at this stage**:
- Pods stay `Pending` → Kueue not admitting (check `kubectl get clusterqueue`), or Volcano not scheduling (check `kubectl describe pod <name>`)
- NCCL hangs → wrong `NCCL_IB_GID_INDEX` or `NCCL_SOCKET_IFNAME`; check with `NCCL_DEBUG=INFO`
- Low bandwidth (< 10 GB/s) → NCCL falling back to TCP instead of RDMA; wrong HCA names

### Stage 2 — Small training run (2 nodes)

Apply `stage4-40bra-rayjob.yaml` (2-node, dapo reward, no Gym needed) as a minimal training test.

```bash
kubectl apply -f iter_kuberay_32nodes_verl_training/yaml/stage4-40bra-rayjob.yaml
```

If step 1 completes and logs show `timing_s/step: ~180-300s`, the training loop is working.

### Stage 3 — Scale up

Once 2-node works, apply the 16-node or 32-node YAML.

---

## Part 5 — kubectl Quick Reference

```bash
# ── Cluster state ──────────────────────────────────────────────────
kubectl get nodes -o wide                    # list all nodes + IPs + roles
kubectl top nodes                            # CPU/RAM usage per node
kubectl get pods -A                          # all pods in all namespaces
kubectl get pods -n default -w              # watch pods in default namespace

# ── RayJob lifecycle ──────────────────────────────────────────────
kubectl apply -f my-rayjob.yaml             # submit a job
kubectl get rayjob                          # list all RayJobs
kubectl describe rayjob <name>              # events + status (use when job fails)
kubectl delete rayjob <name>               # cancel + cleanup

# ── Pod logs ──────────────────────────────────────────────────────
kubectl get pods -l ray.io/cluster=<job-name>   # pods for a specific job
kubectl logs -f <pod-name> -c ray-head          # follow head pod logs
kubectl logs -f <pod-name> -c ray-worker        # follow worker pod logs
kubectl logs --previous <pod-name> -c ray-head  # logs from crashed pod

# ── Debugging ──────────────────────────────────────────────────────
kubectl describe pod <pod-name>            # events, resource usage, mounts
kubectl exec -it <pod-name> -c ray-head -- bash   # shell into running pod
kubectl get events --sort-by='.lastTimestamp'     # recent cluster events

# ── Kueue ──────────────────────────────────────────────────────────
kubectl get clusterqueue                   # quota usage
kubectl describe clusterqueue b300-training-queue   # admitted vs pending
kubectl get workload -A                    # all queued/admitted workloads

# ── Secrets + Storage ──────────────────────────────────────────────
kubectl get secret                        # list secrets
kubectl get pvc                           # list PVCs

# ── Cleanup all RayJobs (CAUTION: deletes running jobs too) ──────
kubectl delete rayjob --all -n default
```

**Proxy bypass** (only needed on b32 where system proxy blocks the k8s API):
```bash
# Add to ~/.bashrc:
alias kubectl='NO_PROXY=10.108.109.145 HTTPS_PROXY= HTTP_PROXY= kubectl'
# OR prefix every command:
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl get nodes
```

---

## Part 6 — Checkpoint Directory Setup (Before Every New Job)

This is a common gotcha. The checkpoint dir must exist AND have `o+rwx` permissions before submitting, because k8s pods run as a different UID than the PFS owner.

```bash
# On b32 (host, before kubectl apply):
mkdir -p /mnt/public/lichang93/st_verl_dockerfile/ckpts/<experiment_name>
chmod o+rwx /mnt/public/lichang93/st_verl_dockerfile/ckpts/
chmod -R o+rwx /mnt/public/lichang93/st_verl_dockerfile/ckpts/<experiment_name>
```

If you forget this, the job runs fine until step 1 checkpoint save → fatal `PermissionError` → entire job crashes.

Also ensure scripts are executable:
```bash
chmod o+rx /mnt/public/lichang93/st_verl_dockerfile/verl/my_scripts/k8s/*.sh
chmod o+rx /mnt/public/lichang93/st_verl_dockerfile/verl/my_scripts/gym/*.sh
```

---

## Part 7 — Troubleshooting Reference

| Symptom | Likely Cause | Fix |
|---|---|---|
| Pods stuck `Pending` forever | Kueue not admitting (quota full) | `kubectl describe clusterqueue b300-training-queue` — check admitted vs pending |
| Pods stuck `Pending` after Kueue admits | Volcano can't gang-schedule (not enough nodes free) | Check `kubectl describe pod <name>` → Events; wait for other jobs to finish |
| `ImagePullBackOff` | Wrong image name or secret | Check secret: `kubectl get secret aliyunsecret -o yaml`; check image tag |
| Pod crashes at start | Script not found or not executable | Check `chmod o+rx` on entrypoint script |
| `PermissionError` at checkpoint | PFS permissions | `chmod -R o+rwx ckpts/<name>` before submitting |
| Training hangs at step 0 (no output) | NCCL can't communicate | Enable `NCCL_DEBUG=INFO`, check logs for "NCCL Init" success |
| NCCL very slow (< 10 GB/s) | Falling back to TCP | Wrong `NCCL_IB_HCA` or `NCCL_IB_GID_INDEX`; run ibstat to verify |
| `Session name mismatch` (Gym Ray) | Stale `/tmp/ray_gym` on host | `start_gym_uv.sh` handles this; or `rm -rf /tmp/ray_gym` before job |
| DeepEP flex fails: `create DCT share err` | IBGDA not supported without device passthrough | Use `alltoall` dispatcher (`moe_token_dispatcher_type=alltoall, moe_enable_deepep=False`) |
| `val_max_samples: null → TypeError` | YAML null → Python None | Use `-1` in YAML (verl convention for "no limit") |
| Tokenizer load race (PFS `sys.modules` error) | Multiple workers importing simultaneously on PFS | Already fixed in `hf_tokenizer.py` with retry loop |
| Kueue not managing RayJobs | Integration not enabled | Check `kueue-manager-config`, add `ray.io/v1/RayJob` to frameworks |

---

## Part 8 — File Map: Current Cluster Configs

All configs from the current cluster live here:

```
st_verl_dockerfile/
│
├── iter_kuberay_32nodes_verl_training/
│   ├── yaml/
│   │   ├── kueue-setup.yaml                          ← Kueue ResourceFlavor/ClusterQueue/LocalQueue
│   │   ├── stage1-nccl-rayjob.yaml                   ← NCCL test (start here)
│   │   ├── stage4-40bra-rayjob.yaml                  ← 2-node minimal training test
│   │   ├── stage9-40bra-32node-fapo-curriculum-rayjob.yaml  ← 32-node reference (Stage 9)
│   │   ├── stage10a-40bra-16node-sd-math-rayjob.yaml ← 16-node current (math, c351)
│   │   └── stage10b-40bra-16node-sd-code-rayjob.yaml ← 16-node current (code, c361)
│   ├── training_plan.md                              ← Stage-by-stage progression + bugs
│   ├── dev_notes.md                                  ← Full bug log with root causes
│   └── quickstart.md                                 ← Existing quick reference
│
├── verl/my_scripts/k8s/
│   ├── run_40bra_k8s_16node_single_domain.sh         ← Current Stage 10 entrypoint
│   ├── run_40bra_k8s_32node_fapo_curriculum.sh       ← Stage 9 reference entrypoint
│   ├── my_deepep_env_k8s.yaml                        ← Ray runtime env vars (k8s)
│   └── config/
│       ├── 40bra_16node_sd.yaml                      ← Hydra config (16-node static)
│       ├── env_k8s_b300_16node.yaml                  ← Cluster paths + sharding
│       └── combo_40Bra.yaml                          ← All combo hyperparams
│
└── k8s_kuberay_kueue_setup/
    └── tutorial.md                                   ← This file
```

---

## Summary: Minimum Steps to Get Running on a New Cluster

1. `helm install kuberay-operator` (Step 1)
2. `helm install volcano` (Step 2)
3. `kubectl apply -f kueue-manifest.yaml` (Step 3)
4. Edit Kueue config to add RayJob integration (Step 4)
5. `kubectl apply -f kueue-setup.yaml` with your GPU counts (Step 5)
6. Create PVC backed by shared storage (Step 6)
7. `kubectl create secret docker-registry` (Step 7)
8. Copy `stage1-nccl-rayjob.yaml`, update image/secret/PVC/network vars, apply it
9. Verify NCCL bandwidth > 100 GB/s
10. `kubectl apply -f stage10a-40bra-16node-sd-math-rayjob.yaml` (update image/secret/PVC/network vars first)
