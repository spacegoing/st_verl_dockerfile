# K8s From a Docker User's Perspective

> You know Docker/Compose. You want to understand k8s — not to administer it,
> but to use it effectively and debug problems independently.
>
> This doc answers four questions:
> 1. Where are the admin configs and how do I inspect them?
> 2. Where is k8s physically? How does kubectl reach it?
> 3. Where is the admin/user boundary?
> 4. What should I focus on to reuse my Docker knowledge?

---

## 1. Where Are the Admin Configs and How Do I Inspect Them?

The admin configured "something somewhere" — here's how to find ALL of it.

### 1.1 Your entry point: kubeconfig

```bash
cat ~/.kube/config
```

This file tells kubectl WHERE the cluster is and WHO you are:
```yaml
clusters:
- cluster:
    server: https://180.184.249.201:8021    # ← API server address
    certificate-authority-data: ...          # ← TLS cert to trust
  name: my-cluster

users:
- user:
    token: ...                               # ← your auth token
  name: lichang93

contexts:
- context:
    cluster: my-cluster
    user: lichang93
    namespace: default                       # ← your default namespace
  name: my-context
```

**Docker analogy**: `DOCKER_HOST=tcp://...` + docker login credentials, but for the whole cluster.

### 1.2 Inspect EVERYTHING the admin configured

The cluster is a database of objects. You can list and read (almost) all of them:

```bash
# ── What resource types exist? ──
kubectl api-resources                    # FULL list of every resource type
kubectl api-resources | grep -i gpu      # find GPU-related resources
kubectl api-resources | grep -i ray      # find Ray CRDs
kubectl api-resources | grep -i kueue    # find Kueue CRDs
kubectl api-resources | grep -i volcano  # find Volcano CRDs

# ── Cluster-level configs (set by admin, visible to you) ──
kubectl get nodes -o wide                           # all machines in the cluster
kubectl describe node <name>                        # per-node: GPUs, taints, conditions, capacity
kubectl get namespaces                              # isolation boundaries (you're in "default")
kubectl get storageclass                            # how PVCs get storage
kubectl get pv                                      # physical volumes (admin-created)
kubectl get pvc                                     # your volume claims

# ── CRDs (extensions installed by admin) ──
kubectl get crd                                     # ALL custom resource types
kubectl get crd | grep ray                          # → rayclusters, rayjobs, rayservices
kubectl get crd | grep kueue                        # → clusterqueues, localqueues, workloads, resourceflavors
kubectl get crd | grep volcano                      # → podgroups, queues, jobs

# ── Kueue config (quota/queuing — admin-created, you use) ──
kubectl get clusterqueue -o yaml                    # quota limits (256 GPUs etc.)
kubectl get localqueue -o yaml                      # namespace → clusterqueue mapping
kubectl get resourceflavor -o yaml                  # node selection rules (if any)

# ── Volcano config ──
kubectl get queue -A -o yaml                        # Volcano scheduling queues
kubectl get podgroup -A                             # active gang-scheduling groups

# ── What's injected into your pods? (admission webhooks) ──
kubectl get mutatingwebhookconfiguration            # things that MODIFY your pods
kubectl get validatingwebhookconfiguration          # things that REJECT your pods
# The probe-init container comes from one of these webhooks

# ── System pods (the control plane + operators) ──
kubectl get pods -n kube-system                     # core k8s components
kubectl get pods -A | grep -i ray                   # KubeRay operator
kubectl get pods -A | grep -i kueue                 # Kueue controller
kubectl get pods -A | grep -i volcano               # Volcano scheduler + controller

# ── RBAC: what CAN you do? ──
kubectl auth can-i --list                           # your permissions
kubectl auth can-i create rayjob                    # can I create rayjobs?
kubectl auth can-i delete node                      # can I delete nodes? (probably no)
kubectl auth can-i get pod --subresource=log        # can I read pod logs?

# ── Secrets and ConfigMaps (app config) ──
kubectl get secret                                  # image pull secrets, TLS certs
kubectl get configmap                               # config files mounted into pods
```

### 1.3 The "exhaustive inspection" one-liner

```bash
# Dump EVERYTHING in your namespace to one file (useful for offline reading):
kubectl api-resources --verbs=list --namespaced -o name | \
  xargs -n 1 kubectl get --show-kind --ignore-not-found 2>/dev/null

# Dump cluster-scoped resources:
kubectl api-resources --verbs=list --namespaced=false -o name | \
  xargs -n 1 kubectl get --show-kind --ignore-not-found 2>/dev/null
```

---

## 2. Where Is K8s? How Does kubectl Reach It?

### 2.1 Physical architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Your machine (b32, 10.12.11.6)                             │
│                                                             │
│  ~/.kube/config says:                                       │
│    server: https://180.184.249.201:8021                     │
│                                                             │
│  kubectl ──HTTPS──→ API Server (180.184.249.201:8021)       │
│                         │                                   │
│  IMPORTANT: system proxy blocks this IP!                    │
│  That's why you need:                                       │
│    HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201       │
│  before every kubectl command.                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  K8s Control Plane (somewhere at 180.184.249.201)           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ API Server   │  │ etcd         │  │ Controller Mgr   │  │
│  │ (port 8021)  │  │ (state DB)   │  │ (runs operators) │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│  ┌──────┴───────┐                                           │
│  │ Schedulers   │  Volcano replaces default scheduler       │
│  │ (Volcano)    │  for pods with schedulerName: volcano     │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
         │
         │  kubelet on each worker node polls API server
         │  for pod assignments
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker Nodes (33 machines, B300 GPUs)                      │
│                                                             │
│  host-10-125-1-22  ┌─────────────────────────────────────┐  │
│  host-10-125-1-26  │ kubelet (agent)                     │  │
│  host-10-125-1-36  │ containerd (container runtime)      │  │
│  ...               │ NVIDIA device plugin (GPU mgmt)     │  │
│  host-10-125-1-163 │ kube-proxy (networking)             │  │
│  (33 total)        └─────────────────────────────────────┘  │
│                                                             │
│  You CANNOT SSH to these machines.                          │
│  kubectl exec is your only way in (into a container).       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Docker analogy

| Docker | K8s |
|--------|-----|
| Docker daemon runs on YOUR machine | API server runs on a REMOTE machine |
| `docker` CLI talks to local daemon via unix socket | `kubectl` talks to remote API server via HTTPS |
| You have root on the machine | You have limited RBAC permissions via token |
| `docker inspect` shows everything | `kubectl describe` shows what your token allows |
| `docker exec` = exec into container on your machine | `kubectl exec` = exec into container on a remote node (proxied via API server → kubelet) |

### 2.3 The communication chain for `kubectl exec`

```
kubectl exec -it <pod> -- bash
    │
    ▼
API Server (180.184.249.201:8021)    ← authenticates you, checks RBAC
    │
    ▼
kubelet on the node running the pod  ← API server proxies to kubelet
    │
    ▼
containerd                           ← kubelet asks containerd to exec
    │
    ▼
Your container                       ← you get a shell
```

This is why `kubectl exec` sometimes feels slow — it's 4 hops, not local.

---

## 3. Where Is the Admin/User Boundary?

### 3.1 The clear split

```
ADMIN DOMAIN (you can read, cannot change)          YOUR DOMAIN (you create and manage)
─────────────────────────────────────────            ──────────────────────────────────
Nodes (physical machines)                            RayJob yamls
  - add/remove/drain/reboot nodes                     - spec, entrypoint, replicas
  - install GPU drivers                                - container image, env vars
  - configure RDMA/RoCE NICs                           - resource requests
                                                       - volumes/mounts
Operators (software on the cluster)
  - KubeRay operator (manages RayCluster)            Training scripts
  - Kueue controller (manages queuing)                 - run_fiberpo_combo.sh
  - Volcano scheduler                                 - start_gym_uv.sh
  - NVIDIA device plugin                              - my_deepep_env_k8s.yaml
  - Platform probes (probe-init webhook)
                                                     Docker images
Cluster config                                         - Dockerfile.base
  - ClusterQueue (quota: 256 GPUs)                     - Dockerfile.ncr.26.02.mydev
  - ResourceFlavor (node labels)                       - push to Aliyun registry
  - Volcano Queue (scheduling policy)
  - Admission webhooks                               Data and checkpoints
  - RBAC (your permissions)                            - PFS: /mnt/public/lichang93/
  - Network policies                                   - mounted via PVC in pod
  - StorageClass, PV, PVC (admin creates PV,
    you use PVC)

Node-level issues                                    Pod-level issues
  - containerd crash → FailedCreatePodSandBox          - OOM in your container
  - GPU hardware error → cudaError                     - Python crash in training
  - NIC failure → NCCL timeout                         - wrong config → Hydra error
  - probe-init timeout → Init stuck                    - missing file → FileNotFoundError
  - image pull failure (registry down)                 - CUDA version mismatch (your image)
```

### 3.2 Gray areas (things you might WANT to do but CAN'T)

| Action | Why you'd want it | Why you can't | Workaround |
|--------|-------------------|---------------|------------|
| Restart a pod on a different node | Current node has containerd issues | Can't evict/reschedule directly | Delete rayjob and resubmit (scheduler picks new nodes) |
| Check node-level logs | Debug why pod sandbox creation failed | No SSH, no node log access | `kubectl describe pod` events are your best view |
| Reset a GPU | `cudaErrorDevicesUnavailable` | Need SSH + `nvidia-smi -r` | Delete pod, hope scheduler picks different node |
| Disable probe-init webhook | It's blocking your pods | Cluster-admin manages webhooks | Ask admin, or wait for it to recover |
| Increase quota | Need more concurrent jobs | ClusterQueue is admin-managed | Request increase from admin |
| Cordon a bad node | Keep pods off a flaky node | Needs node RBAC | Report node to admin with evidence |

### 3.3 Rule of thumb

**If `kubectl describe pod` shows the error comes from:**
- Your container name (`ray-head`, `ray-worker`) → **your problem** (code, config, image)
- An init container you didn't define (`probe-init`) → **admin problem**
- kubelet/containerd messages (`FailedCreatePodSandBox`) → **admin problem**
- Scheduler messages (`Insufficient nvidia.com/gpu`) → **could be either** (quota config vs your request size)

---

## 4. Full Stack Tutorial: Reusing Docker Knowledge

### 4.1 The concept mapping (what you already know)

```
DOCKER CONCEPT              K8S EQUIVALENT                    WHAT'S DIFFERENT
──────────────              ──────────────                    ─────────────────
Dockerfile                  Same Dockerfile                   Nothing — you build images the same way
docker build                Same docker build                 Nothing — build is outside k8s
docker push                 Same docker push                  Pods pull from registry, not local daemon
docker-compose.yml          RayJob YAML                       Different syntax, same idea: declare services
services: head/worker       headGroupSpec/workerGroupSpecs    KubeRay creates pods from these
image:                      containers[].image:               Same — container image reference
environment:                containers[].env:                 Same — env vars in container
volumes:                    volumeMounts + volumes            Split into "what" (volume) and "where" (mount)
ports:                      Not needed (pod networking)       Pods get IPs; Services provide DNS names
network_mode: host          Not available by default          Pods have isolated networking (use Services)
depends_on:                 Init containers                   wait-gcs-ready waits for head before workers start
restart: always             Built into Pod restartPolicy      Pods restart failed containers automatically
docker compose up -d        kubectl apply -f                  Submit to remote cluster instead of local daemon
docker compose down         kubectl delete rayjob             Cascading delete (rayjob → raycluster → pods)
docker compose logs         kubectl logs                      Same concept, add -c for specific container
docker exec                 kubectl exec                      Same concept, proxied via API server
docker ps                   kubectl get pods                  Same concept, cluster-wide view
docker inspect              kubectl describe pod              Similar, but split across describe/get -o yaml
docker stats                kubectl top pod                   If metrics-server is installed
```

### 4.2 What's genuinely NEW (no Docker equivalent)

These concepts don't exist in Docker — focus your learning here:

#### A. Scheduling (Day 1-2 focus)

Docker: you choose which machine runs each container.
K8s: you declare resource needs, scheduler chooses the machine.

```yaml
# You say: "I need 8 GPUs, 176 CPUs, 1920Gi RAM"
resources:
  requests:
    nvidia.com/gpu: '8'
    cpu: '176'
    memory: 1920Gi

# Scheduler says: "node host-10-125-1-22 has that available" → pod goes there
# You never specify the node name. This is the fundamental paradigm shift.
```

**Why it matters**: When a pod is Pending, it's because the scheduler can't find
a node that meets your requests. `kubectl describe pod` shows WHY in the Events.

**Read**: [Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/)

#### B. Controllers and Desired State (Day 2-3 focus)

Docker Compose: you run `up` and `down`. That's it.
K8s: you declare desired state, controllers continuously reconcile.

```
You:        "I want 15 worker pods"  (workerGroupSpecs.replicas: 15)
KubeRay:    Creates 15 pods
Reality:    Node crashes, 1 pod dies
KubeRay:    Detects 14/15 → creates a new pod on a different node
```

This is why you can't just "stop" a pod — the controller will recreate it.
You must delete the controller (RayJob) to truly stop everything.

**Read**: [Controllers](https://kubernetes.io/docs/concepts/architecture/controller/)

#### C. Services and DNS (Day 3-4 focus)

Docker Compose: containers find each other by service name via Docker DNS.
K8s: same concept, but via Service objects + cluster DNS.

```
# KubeRay auto-creates a Service for the head pod:
#   fiberpo-gw-c196-rcr56-head-svc.default.svc.cluster.local
#
# Workers use this DNS name to find the head:
#   ray start --address=fiberpo-gw-c196-rcr56-head-svc.default.svc.cluster.local:6379
#
# The wait-gcs-ready init container checks this:
#   ray health-check --address=<head-svc>:6379
```

**Read**: [Services](https://kubernetes.io/docs/concepts/services-networking/service/)

#### D. RBAC and Multi-Tenancy (Day 4 focus)

Docker: you're root, full access.
K8s: you have a token with specific permissions.

```bash
# "Can I do this?"
kubectl auth can-i delete node            # probably: no
kubectl auth can-i create rayjob          # probably: yes
kubectl auth can-i get pod --subresource=log  # probably: yes
```

**Read**: [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) — skim the concepts

### 4.3 Layer-by-layer: our full stack from user perspective

Read in this order. For each layer, I note what you control vs what admin controls.

```
Layer 1: CONTAINER IMAGE (you own 100%)
├── Dockerfile.base, Dockerfile.ncr.26.02.mydev
├── docker build, docker push to Aliyun registry
├── Everything inside the container: Python, verl, vLLM, CUDA, uv, Gym venvs
└── Read: you already know this. Skip.

Layer 2: RAYJOB YAML (you own 100%)
├── Container spec: image, env vars, resource requests, volume mounts
├── Ray config: headGroupSpec, workerGroupSpecs, replicas
├── Job config: entrypoint command, shutdownAfterJobFinishes, ttl
├── Labels: kueue queue name, submitter
└── Read: https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html
    Focus on: RayJob spec fields, rayClusterSpec, workerGroupSpecs

Layer 3: RAY RUNTIME ENV (you own 100%)
├── my_deepep_env_k8s.yaml: env vars propagated to all Ray workers
├── NCCL vars, NVSHMEM vars, WANDB, OTEL, etc.
├── These override container-level env vars for Ray worker processes
└── Read: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments

Layer 4: TRAINING SCRIPT (you own 100%)
├── run_fiberpo_combo.sh: combo loading, wandb setup, ray job submit
├── Hydra config: ppo_megatron_trainer.yaml + overrides
├── start_gym_uv.sh: Gym server management
├── start_gym_all_nodes.py: distribute Gym startup via Ray tasks
└── Read: you already know this. Skip.

Layer 5: KUEUE (admin owns config, you own labels)
├── Admin created: ClusterQueue (256 GPU quota), LocalQueue, ResourceFlavor
├── You set: kueue.x-k8s.io/queue-name: training-queue (label in your yaml)
├── What you can inspect:
│     kubectl get clusterqueue -o yaml    # see quota and current usage
│     kubectl get workload                # see your jobs in the queue
├── What you can't change: quota limits, queuing strategy, preemption policy
└── Read: https://kueue.sigs.k8s.io/docs/concepts/
    Focus on: ClusterQueue, LocalQueue, Workload lifecycle

Layer 6: VOLCANO (admin installed, you set schedulerName)
├── Admin installed: Volcano scheduler + controller
├── You set: schedulerName: volcano (in pod template)
├── What you can inspect:
│     kubectl get queue -A -o yaml        # scheduling queues
│     kubectl get podgroup                # gang-scheduling groups (if any)
├── What you can't change: scheduler config, plugins, queue weights
└── Read: https://volcano.sh/en/docs/schduling/
    Focus on: what gang-scheduling is and why we don't have it

Layer 7: KUBERAY OPERATOR (admin installed, auto-manages your RayJobs)
├── Admin installed: kuberay-operator pod in some namespace
├── It watches RayJob CRDs → creates RayCluster → creates Pods
├── Auto-injects: wait-gcs-ready init container (checks head is ready)
├── Auto-creates: head Service (DNS name for workers to find head)
├── You can inspect:
│     kubectl get rayjob -o yaml          # your job spec + status
│     kubectl get raycluster              # clusters created by your jobs
│     kubectl logs <kuberay-operator-pod> -n <namespace>  # operator logs (if permitted)
└── Read: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html

Layer 8: NVIDIA GPU STACK (admin owns, transparent to you)
├── Admin installed: GPU device plugin (DaemonSet), GPU drivers, container toolkit
├── You request: nvidia.com/gpu: 8 in resource requests
├── Device plugin assigns GPUs, sets NVIDIA_VISIBLE_DEVICES
├── You can inspect:
│     kubectl describe node <name> | grep -A5 "Allocatable"  # GPU count
│     kubectl describe node <name> | grep -A20 "Allocated"   # GPU usage
├── You can't: reset GPUs, update drivers, configure device plugin
└── Read: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/

Layer 9: RDMA/ROCE NETWORKING (admin owns, you configure NCCL vars)
├── Admin installed: RDMA device plugin, configured RoCE NICs
├── You request: rdma-training/roce: 1 in resource requests
├── You configure: NCCL_IB_HCA, NCCL_IB_GID_INDEX etc. in your env vars
├── The NICs (mlx5_10-17) are made available to your container
├── You can inspect:
│     kubectl exec <pod> -- ibstat              # RDMA device status
│     kubectl exec <pod> -- ibv_devinfo         # IB device capabilities
│     kubectl exec <pod> -- show_gids mlx5_10   # GID table
└── Read: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

Layer 10: PLATFORM INFRA (admin owns, invisible to you)
├── Admission webhooks: inject probe-init, modify pod spec
├── Virtual cluster (vcluster): namespace isolation
├── Pod-syncer: syncs pod state between vcluster and host cluster
├── You can inspect:
│     kubectl get mutatingwebhookconfiguration   # what modifies your pods
│     kubectl describe pod → init containers     # see injected containers
├── You can't: disable webhooks, fix probe failures, restart node services
└── No official docs — this is platform-specific (SenseTime Lepton)
```

### 4.4 Debugging decision tree

```
Pod not starting?
  │
  ├── Status: Pending
  │   └── kubectl describe pod → Events
  │       ├── "Insufficient nvidia.com/gpu" → not enough free GPUs/nodes
  │       ├── "didn't match Pod's node affinity" → node selector/taint issue
  │       └── "0/33 nodes available" → cluster full or nodes tainted
  │       → YOUR ACTION: check if another job is hogging nodes, or reduce replicas
  │       → ADMIN ACTION: if nodes are tainted/cordoned, ask admin to uncordon
  │
  ├── Status: Init:0/1 or Init:CrashLoopBackOff
  │   └── kubectl describe pod → check which init container
  │       ├── "probe-init" (image: registry.sensetime.com/...) → PLATFORM ISSUE
  │       │   → kubectl logs <pod> -c probe-init
  │       │   → ADMIN ACTION: platform monitoring probe can't reach backend
  │       │
  │       └── "wait-gcs-ready" (image: your image) → HEAD POD NOT READY
  │           → check head pod status first (it must be Running)
  │           → if head is also stuck → same analysis on head pod
  │           → YOUR ACTION: delete rayjob and resubmit
  │
  ├── Status: ContainerCreating (stuck)
  │   └── kubectl describe pod → Events
  │       ├── "FailedCreatePodSandBox" → CONTAINERD/NODE ISSUE
  │       │   → ADMIN ACTION: node needs containerd restart
  │       └── "pulling image" (slow) → IMAGE PULL
  │           → first pull of a large image (~30GB) takes time
  │           → YOUR ACTION: wait, or ensure image is pre-cached
  │
  ├── Status: Running but job fails
  │   └── This is YOUR code/config
  │       → kubectl exec <head> -- ray job logs <id>
  │       → look for Python errors, OOM, NCCL errors
  │
  └── Status: Error or CrashLoopBackOff (main container)
      └── kubectl logs <pod> -c ray-head (or ray-worker)
          → container startup crash
          → YOUR ACTION: check entrypoint, ray start command, env vars
```

---

## Official Docs Reading List (Prioritized for Docker Users)

### Must-Read (Week 1)

| Priority | Topic | URL | Time |
|----------|-------|-----|------|
| 1 | Docker → kubectl mapping | https://kubernetes.io/docs/reference/kubectl/docker-cli-to-kubectl/ | 15 min |
| 2 | Pod overview | https://kubernetes.io/docs/concepts/workloads/pods/ | 30 min |
| 3 | Pod lifecycle + init containers | https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/ | 30 min |
| 4 | Resource requests/limits | https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/ | 30 min |
| 5 | kubectl cheat sheet | https://kubernetes.io/docs/reference/kubectl/cheatsheet/ | 20 min |
| 6 | Debug pods | https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/ | 30 min |

### Should-Read (Week 2)

| Priority | Topic | URL | Time |
|----------|-------|-----|------|
| 7 | Services and DNS | https://kubernetes.io/docs/concepts/services-networking/service/ | 30 min |
| 8 | Persistent Volumes | https://kubernetes.io/docs/concepts/storage/persistent-volumes/ | 20 min |
| 9 | RayJob quickstart | https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html | 30 min |
| 10 | RayJob config reference | https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html | 45 min |
| 11 | Kueue concepts | https://kueue.sigs.k8s.io/docs/concepts/ | 30 min |
| 12 | Schedule GPUs | https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/ | 15 min |

### Nice-to-Read (Week 3+)

| Priority | Topic | URL | Time |
|----------|-------|-----|------|
| 13 | Cluster architecture | https://kubernetes.io/docs/concepts/architecture/ | 30 min |
| 14 | Scheduling | https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/ | 20 min |
| 15 | Taints and tolerations | https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/ | 20 min |
| 16 | RBAC | https://kubernetes.io/docs/reference/access-authn-authz/rbac/ | 20 min |
| 17 | Volcano gang scheduling | https://volcano.sh/en/docs/schduling/ | 20 min |
| 18 | NCCL env vars | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html | 30 min |
| 19 | KubeRay troubleshooting | https://docs.ray.io/en/latest/cluster/kubernetes/troubleshooting.html | 20 min |
| 20 | Custom resources / operators | https://kubernetes.io/docs/concepts/extend-kubernetes/operator/ | 15 min |

### Reference (bookmark, look up as needed)

| Topic | URL |
|-------|-----|
| kubectl full reference | https://kubernetes.io/docs/reference/kubectl/ |
| k8s API reference | https://kubernetes.io/docs/reference/kubernetes-api/ |
| NCCL all env vars | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html |
| Ray runtime environments | https://docs.ray.io/en/latest/ray-core/handling-dependencies.html |
| Kueue RayJob integration | https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/ |

---

## Quick Sanity Check Commands

Run these now to build your mental model of the cluster:

```bash
P="HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201"

# 1. What cluster am I connected to?
cat ~/.kube/config | grep server

# 2. What permissions do I have?
eval $P kubectl auth can-i --list 2>/dev/null | head -30

# 3. What nodes exist and what state are they in?
eval $P kubectl get nodes -o wide

# 4. What CRDs (extensions) are installed?
eval $P kubectl get crd | grep -E "ray|kueue|volcano"

# 5. What's the GPU quota?
eval $P kubectl get clusterqueue -o jsonpath='{.items[0].spec.resourceGroups[0].flavors[0].resources}' | python3 -m json.tool

# 6. What webhooks modify my pods?
eval $P kubectl get mutatingwebhookconfiguration -o name

# 7. What's currently running?
eval $P kubectl get rayjob
eval $P kubectl get pods -o wide | head -20

# 8. System pods (operators):
eval $P kubectl get pods -A | grep -iE "ray-operator|kueue|volcano" | head -10
```
