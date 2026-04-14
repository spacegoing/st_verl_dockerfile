# K8s + GPU Training Stack — Learning Plan

> For someone who knows Docker/Compose well but is new to Kubernetes.
> Covers everything in our stack: k8s core, KubeRay, Kueue, Volcano,
> NCCL/RoCE networking, NVIDIA GPU operator, and verl-specific config.
>
> Estimated total: 3-4 weeks at 2-3 hours/day. Can compress to 1-2 weeks
> if you skip the hands-on exercises and focus on reading.

---

## Phase 0: Mental Model — Docker Compose vs Kubernetes (Day 1)

**Goal**: Understand why k8s exists and how it maps to what you already know.

| Docker Compose | Kubernetes |
|---|---|
| `docker-compose.yml` | Multiple YAML files (Pod, Service, Deployment...) |
| `services:` | Pod (one or more containers) |
| `docker compose up -d` | `kubectl apply -f` |
| `docker compose down` | `kubectl delete -f` |
| `docker compose logs` | `kubectl logs <pod>` |
| `docker exec -it` | `kubectl exec -it` |
| Single host | Multi-node cluster |
| Docker daemon manages containers | kubelet (per-node) + control plane (cluster-wide) |
| `network_mode: host` | Pod networking (CNI), Services, host networking via `hostNetwork: true` |
| `volumes:` | PersistentVolumeClaim (PVC), emptyDir, hostPath |
| No scheduling | Scheduler decides which node runs which pod |
| No auto-restart beyond `restart: always` | Controllers (Deployment, Job) ensure desired state |

**Key insight**: In Compose, YOU decide what runs where. In k8s, you declare
WHAT you want (resources, replicas) and the scheduler decides WHERE.

**Read** (30 min):
- [Kubernetes concepts overview](https://kubernetes.io/docs/concepts/overview/) — just the overview page
- [Kubernetes for Docker users](https://kubernetes.io/docs/reference/kubectl/docker-cli-to-kubectl/) — exact mapping

---

## Phase 1: Kubernetes Core Concepts (Days 2-5)

### 1.1 Architecture (Day 2)

**Goal**: Understand the components that make up a k8s cluster.

```
Control Plane (master):
  ├── kube-apiserver     ← all kubectl commands go here
  ├── etcd               ← cluster state database
  ├── kube-scheduler     ← decides which node runs a pod (we replace this with Volcano)
  └── kube-controller-manager ← runs controllers (Deployment, Job, etc.)

Worker Nodes:
  ├── kubelet            ← agent that runs pods on each node
  ├── container runtime  ← containerd (runs actual containers, like Docker daemon)
  └── kube-proxy         ← networking rules
```

**Read** (1 hour):
- [Cluster Architecture](https://kubernetes.io/docs/concepts/architecture/)
- [Nodes](https://kubernetes.io/docs/concepts/architecture/nodes/)

**Hands-on**:
```bash
# These map to what you already do with our cluster
kubectl get nodes                    # like: docker info (but for the whole cluster)
kubectl describe node <node-name>    # resources, conditions, taints
kubectl get pods --all-namespaces    # like: docker ps (but cluster-wide)
```

### 1.2 Pods, Containers, Init Containers (Day 2-3)

**Goal**: Understand what a Pod is (the thing that replaces `docker run`).

A Pod = one or more containers that share:
- Network namespace (same IP, can talk via localhost)
- Storage volumes
- Lifecycle (all containers start/stop together)

**Init containers** (this is what was failing in our `probe-init` issue):
- Run BEFORE the main container starts
- Must complete successfully (exit 0) before next init container starts
- If an init container fails → pod stays in `Init:CrashLoopBackOff`
- Platform admins often inject init containers via admission webhooks (we don't control these)

**Read** (1.5 hours):
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/)
- [Init Containers](https://kubernetes.io/docs/concepts/workloads/pods/init-containers/)
- [Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
  - Focus on: phase, conditions, container states, restart policy

**Hands-on**:
```bash
kubectl get pod <pod-name> -o yaml   # full pod spec — compare to docker inspect
kubectl describe pod <pod-name>      # events, conditions, container statuses
kubectl logs <pod-name> -c <container-name>  # like: docker logs
kubectl logs <pod-name> -c <init-container-name>  # init container logs
```

### 1.3 Resources, Requests, Limits (Day 3)

**Goal**: Understand how k8s allocates CPU, memory, and GPUs.

This is the foundation of scheduling. In Compose, you might set `deploy.resources.limits`.
In k8s, there are TWO fields:

```yaml
resources:
  requests:              # scheduler uses this to PLACE the pod (guaranteed minimum)
    nvidia.com/gpu: 8
    cpu: 176
    memory: 1920Gi
  limits:                # kubelet enforces this as MAXIMUM (OOM-killed if exceeded)
    nvidia.com/gpu: 8
    cpu: 176
    memory: 1920Gi
```

In our yaml, requests == limits (guaranteed QoS class). This means "give me exactly
this much, no overcommit."

**Read** (1 hour):
- [Resource Management for Pods](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/) — skim

**Key for our stack**: `nvidia.com/gpu: 8` is a device plugin extended resource.
The NVIDIA GPU device plugin registers GPUs as schedulable resources. Each GPU
is exclusive — no sharing (unlike CPU/memory which can be overcommitted).

### 1.4 Scheduling: How Pods Land on Nodes (Day 4)

**Goal**: Understand the default scheduler, then why we use Volcano instead.

Default scheduler flow:
1. **Filtering**: eliminate nodes that don't meet requirements (not enough GPU, tainted, etc.)
2. **Scoring**: rank remaining nodes by preference
3. **Binding**: assign pod to the winning node

**Read** (1 hour):
- [Kubernetes Scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/)
- [Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/)
  - Focus on: nodeSelector, affinity, taints/tolerations
- [Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/)
  - This explains `NotReady,SchedulingDisabled` nodes

**Hands-on**:
```bash
# Why are 2 nodes not schedulable?
kubectl get nodes | grep -v " Ready "
kubectl describe node <not-ready-node> | grep -A5 "Taints\|Conditions"
```

### 1.5 Services, DNS, Networking (Day 4-5)

**Goal**: Understand how pods find each other (this is how Ray head/workers connect).

In Compose: containers on the same network find each other by service name.
In k8s: Pods find each other via **Services** (stable DNS name → pod IPs).

```
# In our RayJob yaml, KubeRay creates a head Service:
#   fiberpo-gw-c196-rcr56-head-svc.default.svc.cluster.local:6379
# Workers use this DNS name to connect to the Ray head.
# That's what wait-gcs-ready init container checks:
#   ray health-check --address fiberpo-gw-c196-rcr56-head-svc.default.svc.cluster.local:6379
```

**Read** (1 hour):
- [Services](https://kubernetes.io/docs/concepts/services-networking/service/)
- [DNS for Services and Pods](https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/)

### 1.6 Storage: PVC, Volumes (Day 5)

**Goal**: Understand how our PFS mount works in k8s.

In Compose: `volumes: - /mnt/public/lichang93:/root/myCodeLab/host`
In k8s:
```yaml
volumes:
- name: afs-pvc
  persistentVolumeClaim:
    claimName: pvc-jdwzpnv          # pre-created PVC pointing to PFS

containers:
- volumeMounts:
  - name: afs-pvc
    mountPath: /root/myCodeLab/host
    subPath: lichang93/st_verl_dockerfile   # subdirectory within PVC
```

**Read** (45 min):
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [PersistentVolumeClaims](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#persistentvolumeclaims)
- Skip dynamic provisioning — our PVC is pre-created by admin

---

## Phase 2: kubectl Mastery (Days 6-7)

**Goal**: Be comfortable debugging any pod issue.

### Essential commands (our cluster)

```bash
# Always prefix with proxy bypass:
#   HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl ...

# ── Cluster overview ──
kubectl get nodes                          # node status
kubectl get nodes -o wide                  # + IP, OS, kernel version
kubectl top nodes                          # CPU/memory usage per node

# ── List resources ──
kubectl get pods                           # default namespace
kubectl get pods -A                        # all namespaces
kubectl get pods -o wide                   # + node assignment, IP
kubectl get rayjob                         # KubeRay jobs
kubectl get raycluster                     # Ray clusters (created by RayJob)
kubectl get workload                       # Kueue workloads (admission queue)

# ── Inspect a pod ──
kubectl describe pod <name>                # THE most useful debug command
                                           # Shows: events, conditions, container
                                           # states, resource requests, volumes
kubectl get pod <name> -o yaml             # full spec + status (machine-readable)
kubectl get pod <name> -o jsonpath='{.status.phase}'  # extract specific field

# ── Logs ──
kubectl logs <pod> -c <container>          # main container logs
kubectl logs <pod> -c <init-container>     # init container logs
kubectl logs <pod> -c <container> --previous  # logs from previous crash
kubectl logs <pod> --tail=50               # last 50 lines
kubectl logs -f <pod>                      # follow (like docker logs -f)

# ── Exec into pod ──
kubectl exec -it <pod> -c <container> -- bash

# ── Events (cluster-wide debug) ──
kubectl get events --sort-by='.lastTimestamp' | tail -30
kubectl get events --field-selector reason=Failed

# ── Delete ──
kubectl delete rayjob <name>               # cascading: removes raycluster + pods
kubectl delete pod <name>                   # single pod (controller may recreate it)
```

**Read** (1 hour):
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [kubectl Reference](https://kubernetes.io/docs/reference/kubectl/) — bookmark

**Exercise**: Practice on our cluster. Run `kubectl describe pod` on a running
pod from a past job. Read every section: Containers, Conditions, Volumes, Events.

---

## Phase 3: Jobs, CRDs, and Operators (Days 8-10)

### 3.1 Jobs and CRDs (Day 8)

**Goal**: Understand how k8s is extended beyond built-in resources.

Built-in resources: Pod, Service, Deployment, Job, ConfigMap, Secret, etc.

**Custom Resource Definitions (CRDs)**: Extensions that add new resource types.
This is how KubeRay adds `RayJob`, `RayCluster`; how Kueue adds `ClusterQueue`,
`LocalQueue`, `Workload`; how Volcano adds `PodGroup`, `Queue`.

```bash
# List all CRDs on our cluster
kubectl get crd | grep -E "ray|kueue|volcano"
```

**Read** (45 min):
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Custom Resources](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
- [Operator Pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) — how operators
  watch CRDs and manage resources

### 3.2 Admission Webhooks (Day 8)

**Goal**: Understand why platform-injected init containers appear in our pods.

Admission webhooks intercept API requests (like pod creation) and can:
- **Mutating**: modify the request (inject init containers, add labels, set defaults)
- **Validating**: reject the request if it doesn't meet policy

The `probe-init` init container that's failing? It was injected by a mutating
webhook — we never defined it in our yaml. The platform admin configured it.

**Read** (30 min):
- [Admission Controllers](https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/)
- [Dynamic Admission Control](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/)
  — just the overview, skip implementation

---

## Phase 4: KubeRay — Ray on Kubernetes (Days 11-13)

**Goal**: Understand how Ray clusters are managed on k8s, and how our RayJob yaml works.

KubeRay is the operator that manages Ray clusters on k8s. It watches for
`RayCluster` and `RayJob` CRDs and creates/manages pods accordingly.

### 4.1 Core Concepts (Day 11)

```
RayJob (what we submit)
  └── creates RayCluster
        ├── headGroupSpec → 1 head Pod (runs ray start --head)
        └── workerGroupSpecs → N worker Pods (runs ray start --address=head:6379)
```

**Read** (2 hours):
- [KubeRay overview](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [RayJob](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html)
  — this is exactly what we use
- [RayCluster](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html)
- [RayJob configuration](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html)

### 4.2 Our RayJob YAML Walkthrough (Day 12)

Map every field in our yaml to what it does:

```yaml
apiVersion: ray.io/v1
kind: RayJob                          # CRD managed by KubeRay operator
metadata:
  name: fiberpo-gw-c196               # job name (kubectl get rayjob)
  labels:
    kueue.x-k8s.io/queue-name: training-queue  # Kueue integration
spec:
  entrypoint: bash /root/.../run.sh   # command to run after cluster is ready
  shutdownAfterJobFinishes: true      # delete RayCluster when job ends
  ttlSecondsAfterFinished: 1800       # delete RayJob object 30 min after completion
  rayClusterSpec:
    headGroupSpec:
      rayStartParams:
        num-gpus: '8'                 # Ray sees 8 GPUs on head
      template:
        spec:
          schedulerName: volcano      # use Volcano instead of default scheduler
          containers:
          - resources:
              requests:
                nvidia.com/gpu: '8'   # k8s scheduler requirement
    workerGroupSpecs:
    - replicas: 15                    # 15 workers + 1 head = 16 nodes
      minReplicas: 15                 # no autoscaling
      maxReplicas: 15
```

**Exercise**: Read `iter_kuberay_32nodes_verl_training/yaml/stage10-40bra-sd-mcqa-ts300-e2-rayjob.yaml`
line by line. For each field, find its documentation in the KubeRay docs.

### 4.3 Lifecycle and Debugging (Day 13)

**Read**:
- [Troubleshooting guide](https://docs.ray.io/en/latest/cluster/kubernetes/troubleshooting.html)

**Key debugging flow for RayJobs**:
```
1. kubectl get rayjob <name>                    → job-level status
2. kubectl get raycluster <cluster-name>        → cluster-level status
3. kubectl get pod -l ray.io/cluster=<cluster>  → per-pod status
4. kubectl describe pod <pod>                   → events, init containers
5. kubectl logs <head-pod> -c ray-head          → Ray head startup
6. kubectl exec <head-pod> -- ray job list      → inner Ray job status
7. kubectl exec <head-pod> -- ray job logs <id> → actual training logs
```

---

## Phase 5: Kueue — Job Queuing and Quota (Days 14-15)

**Goal**: Understand how Kueue controls which jobs are admitted to the cluster.

Kueue sits BETWEEN `kubectl apply` and pod creation:
```
kubectl apply -f rayjob.yaml
  → Kueue intercepts (via label kueue.x-k8s.io/queue-name)
  → Checks quota in ClusterQueue
  → If quota available: admits (lets KubeRay create pods)
  → If not: queues (job stays Suspended until quota frees up)
```

### Our Kueue setup:
```
LocalQueue: training-queue (namespace: default)
  └── ClusterQueue: b300-training-queue
        ├── nominalQuota: 256 GPUs, 32 RDMA, 5640 CPUs, 61TB RAM
        ├── queueingStrategy: BestEffortFIFO
        └── preemption: Never (no job can kick another)
```

**Read** (2 hours):
- [Kueue overview](https://kueue.sigs.k8s.io/docs/overview/)
- [Concepts](https://kueue.sigs.k8s.io/docs/concepts/) — ResourceFlavor, ClusterQueue, LocalQueue, Workload
- [Run a RayJob](https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/) — exact integration we use

**Hands-on**:
```bash
kubectl get clusterqueue -o yaml              # our quota config
kubectl get localqueue                        # namespace queue
kubectl get workload                          # admitted/pending workloads
kubectl get workload <name> -o yaml           # detailed admission status
```

**Key understanding**: Kueue does PAPER accounting against `nominalQuota`.
It does NOT check physical node availability. That's why two 16-node jobs
can both be admitted (128+128=256 ≤ 256 quota) but still deadlock if
physical nodes aren't available.

---

## Phase 6: Volcano — GPU-Aware Scheduling (Days 16-17)

**Goal**: Understand why we use Volcano instead of the default scheduler,
and what gang-scheduling means.

### Why Volcano?
Default kube-scheduler doesn't understand:
- GPU topology (which GPUs are on which NUMA node)
- Gang scheduling (all-or-nothing: either ALL pods in a group get placed, or NONE)
- Priority-based preemption for batch workloads

### Our usage (minimal):
We use Volcano **only as a scheduler** (pod placement), NOT for gang-scheduling.
```yaml
spec:
  schedulerName: volcano    # on every pod template
```
This means Volcano replaces kube-scheduler for pod-to-node binding, but
it schedules pods ONE AT A TIME (not as atomic groups).

### What we're missing:
Gang-scheduling via PodGroup would prevent the deadlock issue:
```yaml
# NOT currently configured, but WOULD fix the simultaneous submission problem:
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: fiberpo-gw-c196
spec:
  minMember: 16    # don't place ANY pod until 16 nodes are free
```

**Read** (1.5 hours):
- [Volcano overview](https://volcano.sh/en/docs/)
- [Scheduling](https://volcano.sh/en/docs/schduling/) — focus on gang scheduling
- [PodGroup](https://volcano.sh/en/docs/podgroup/)
- [Queue](https://volcano.sh/en/docs/queue/)

**Hands-on**:
```bash
kubectl get queue -n volcano-system -o yaml   # Volcano queues
kubectl get podgroup                          # any active PodGroups
```

---

## Phase 7: NVIDIA GPU Stack on Kubernetes (Days 18-19)

**Goal**: Understand how GPUs become schedulable k8s resources.

### Stack layers:
```
Your container (CUDA code)
  ↓
NVIDIA Container Toolkit (nvidia-container-runtime)
  ↓ injects GPU devices into container
containerd (container runtime)
  ↓
NVIDIA Device Plugin (DaemonSet on every GPU node)
  ↓ registers nvidia.com/gpu as k8s extended resource
kubelet
  ↓ advertises nvidia.com/gpu: 8 to scheduler
kube-scheduler / Volcano
```

### Device Plugin:
- Runs as a DaemonSet (one pod per GPU node)
- Reports GPU count to kubelet: `nvidia.com/gpu: 8`
- When a pod requests `nvidia.com/gpu: 8`, the device plugin assigns
  specific GPU devices and sets `NVIDIA_VISIBLE_DEVICES`
- GPUs are **exclusive** — once assigned to a pod, no other pod can use them
- This is why `cudaErrorDevicesUnavailable` means another pod has the GPU

**Read** (1 hour):
- [NVIDIA GPU Operator docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html) — overview only
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin#readme)
- [Schedule GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

### RDMA/RoCE in k8s:
Our yaml requests `rdma-training/roce: 1` — this is a custom extended resource
registered by the RDMA device plugin. It ensures the pod gets access to the
RoCE NICs (mlx5_10-17) for NCCL inter-node communication.

```yaml
resources:
  requests:
    nvidia.com/gpu: '8'           # 8 GPUs (NVIDIA device plugin)
    rdma-training/roce: '1'       # RDMA NIC access (RDMA device plugin)
```

---

## Phase 8: NCCL and Multi-Node GPU Communication (Days 20-21)

**Goal**: Understand the NCCL environment variables in our yaml and why they matter.

### What is NCCL?
NVIDIA Collective Communications Library — handles GPU-to-GPU communication
(allreduce, allgather, etc.) across nodes. It's what makes distributed training work.

### Transport layers:
```
Same node:    NVLink/NVSwitch (fastest, ~900 GB/s per GPU)
Cross-node:   RDMA/RoCE via InfiniBand verbs (fast, ~400 Gbps per NIC)
Fallback:     TCP/IP over ethernet (slow, ~100 Gbps)
```

### Our NCCL config explained:

```yaml
# Network interface selection
NCCL_SOCKET_IFNAME: eth0              # TCP socket interface (for bootstrap/OOB)
NCCL_IB_HCA: mlx5_10:1,...,mlx5_17:1 # Which RDMA NICs to use (8 NICs, one per GPU)
NCCL_IB_GID_INDEX: 5                  # RoCEv2 GID index (routing config for k8s)
NCCL_IB_TC: 96                        # Traffic class for QoS

# Reliability
NCCL_IB_TIMEOUT: 22                   # IB timeout (2^22 * 4.096μs ≈ 17 sec)
NCCL_IB_RETRY_CNT: 13                 # retry count on IB errors
NCCL_IB_ADAPTIVE_ROUTING: 1           # enable adaptive routing

# Performance tuning
NCCL_BUFFSIZE: 16777216               # 16 MB buffer (larger = fewer ops, higher latency)
NCCL_P2P_NET_CHUNKSIZE: 524288        # 512 KB chunk size for P2P
NCCL_CROSS_NIC: 1                     # allow cross-NIC communication
NCCL_MIN_NCHANNELS: 32                # minimum channels (parallelism)
NCCL_IB_QPS_PER_CONNECTION: 8         # QPs per IB connection
NCCL_NVLS_ENABLE: 0                   # disable NVLink SHARP (not supported on B300)
```

**Read** (2 hours):
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) — reference
- [NCCL Topology](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/topology.html) — how NCCL discovers GPUs and NICs
- [RoCE/RDMA with NCCL](https://docs.nvidia.com/networking/display/rdmacore60/introduction) — overview of RDMA concepts

### NVSHMEM/DeepEP (advanced):
```yaml
NVSHMEM_IB_ENABLE_IBGDA: 1            # DeepEP uses NVSHMEM with IBGDA transport
NVSHMEM_IBGDA_NIC_HANDLER: gpu        # GPU-direct RDMA (k8s pods support this)
NVSHMEM_HCA_LIST: mlx5_10,...         # which NICs NVSHMEM uses
```
This is for MoE expert parallelism (DeepEP flex dispatcher). Only matters for
MoE models with cross-node expert communication.

---

## Phase 9: Putting It All Together — Our Full Stack (Days 22-23)

**Goal**: Trace a complete job lifecycle through all layers.

### Exercise: Trace `kubectl apply -f rayjob.yaml` end to end

```
1. You run:  kubectl apply -f fiberpo-gw-c196-32node-rayjob.yaml
                    |
2. API server stores RayJob CR in etcd
                    |
3. Kueue webhook intercepts (sees kueue.x-k8s.io/queue-name label)
   → checks ClusterQueue quota: 128 GPUs requested, 256 available → ADMITTED
   → sets RayJob as unsuspended
                    |
4. KubeRay operator sees new RayJob → creates RayCluster CR
   → creates 1 head Pod + 15 worker Pods
                    |
5. Mutating webhook (platform) injects probe-init init container into each pod
                    |
6. Volcano scheduler picks up each Pod (schedulerName: volcano)
   → for each pod: finds a node with 8 free GPUs + 176 CPUs + 1920Gi RAM + 1 RDMA
   → binds pod to node
                    |
7. kubelet on each node:
   → pulls image (if not cached)
   → runs init containers: probe-init → wait-gcs-ready
   → runs main container: ray start --address=head:6379
                    |
8. NVIDIA device plugin assigns 8 GPUs to each container
   → sets NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
                    |
9. Ray cluster forms: head + 15 workers = 16 nodes, 128 GPUs
                    |
10. KubeRay submits entrypoint command as Ray job
    → bash /root/.../run_fiberpo_combo.sh 196
                    |
11. Script runs: RAY_ADDRESS=auto ray job submit --runtime-env=... -- python3 -m verl.trainer.main_ppo ...
                    |
12. verl creates Ray actors (WorkerDict) across all 16 nodes
    → NCCL initializes: discovers GPUs, connects via RoCE (mlx5_10-17)
    → NVSHMEM initializes: DeepEP IBGDA for MoE expert dispatch
                    |
13. Training loop runs: generate → reward → update → checkpoint
                    |
14. Job finishes → KubeRay deletes RayCluster → all pods terminated
    → Kueue releases quota → next queued job can be admitted
```

### Exercise: Read our reference docs

Now re-read these with full context:
- `iter_kuberay_32nodes_verl_training/k8s_scheduling_explained.md` — our 3-layer scheduling analysis
- `iter_kuberay_32nodes_verl_training/training_plan.md` — full stage progression
- `CLAUDE.md` sections 4-6 — container lifecycle, Gym servers, training runs

---

## Phase 10: Advanced Topics (Days 24+, as needed)

Pick based on what you encounter:

### Debugging skills
- [Troubleshooting Applications](https://kubernetes.io/docs/tasks/debug/debug-application/)
- [Troubleshooting Clusters](https://kubernetes.io/docs/tasks/debug/debug-cluster/)
- [Debug Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/)
- [Debug Services](https://kubernetes.io/docs/tasks/debug/debug-application/debug-service/)

### Security and RBAC
- [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) — why some commands need permissions
- [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)
  — `capabilities: {add: [IPC_LOCK]}` in our yaml

### Networking deep dive
- [CNI](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/network-plugins/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)

### Monitoring
- [Metrics Server](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/)
- Prometheus + Grafana for GPU metrics (DCGM exporter)

---

## Quick Reference: Official Docs Links

| Topic | URL |
|-------|-----|
| **Kubernetes** | https://kubernetes.io/docs/ |
| **kubectl** | https://kubernetes.io/docs/reference/kubectl/ |
| **KubeRay** | https://docs.ray.io/en/latest/cluster/kubernetes/ |
| **Kueue** | https://kueue.sigs.k8s.io/docs/ |
| **Volcano** | https://volcano.sh/en/docs/ |
| **NCCL** | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ |
| **NVIDIA Device Plugin** | https://github.com/NVIDIA/k8s-device-plugin |
| **NVIDIA GPU Operator** | https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/ |
| **NVSHMEM** | https://docs.nvidia.com/nvshmem/ |

---

## Reading Order Summary

| Day | Topic | Time |
|-----|-------|------|
| 1 | Mental model: Compose → k8s | 30 min |
| 2-3 | Pods, init containers, lifecycle, resources | 3.5 hrs |
| 4-5 | Scheduling, services, DNS, storage | 3 hrs |
| 6-7 | kubectl mastery + practice | 2 hrs |
| 8-10 | Jobs, CRDs, operators, webhooks | 2 hrs |
| 11-13 | KubeRay: RayJob, RayCluster, debugging | 4 hrs |
| 14-15 | Kueue: queuing, quota, admission | 2.5 hrs |
| 16-17 | Volcano: scheduling, gang-scheduling | 1.5 hrs |
| 18-19 | NVIDIA GPU stack, device plugin, RDMA | 1.5 hrs |
| 20-21 | NCCL env vars, RoCE, NVSHMEM | 2 hrs |
| 22-23 | Full stack trace, re-read project docs | 2 hrs |
