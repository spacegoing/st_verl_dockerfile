# Kueue vs Volcano: Our Setup, Past Failures, and What the Official Docs Say

> Three questions about our current KubeRay + Kueue + Volcano stack:
> 1. What exact conflicts have we hit running our 2×16-node jobs?
> 2. Why do we configure both if either alone can gang-schedule?
> 3. Which is actually better per the official docs?
>
> Based on: Ray docs (kueue.html, volcano.html) + our failure history from
> 2026-03-31, 2026-04-03, 2026-04-13 + current yaml inspection.

---

## Question 1: What exact conflicts have we hit?

Three concrete incidents are documented in the conversation history. They are
**all the same root cause**: partial pod placement because neither layer
actually gang-schedules our RayJob pods.

### Incident A — 2026-03-31: `c364 + c369` deadlock (primary example)

Submitted two 16-node RayJobs simultaneously (`c364`, `c369`, 128 GPUs each,
total 256). Quota was 256. Cluster had 33 schedulable nodes.

**What happened, step by step** (from `k8s_scheduling_explained.md` §"March 31 scenario"):

1. **Kueue** accepted both (128+128 = 256 ≤ 256 nominal quota). ✅
2. **KubeRay** created 32 pods (2 heads + 30 workers).
3. **Volcano** bound pods one at a time, interleaving the two RayJobs:
   ```
   T1..T25: c364-head, c369-head, c364-w1, c369-w1, ..., c364-w12  ✓
   T26:     Only 7 nodes free. c364 still needs 2 more, c369 still needs 8.
            Total need = 10, available = 7 → 3 pods stuck Pending forever.
   ```
4. c369 ended up with only 3 running pods (head + 2 workers), 13 Pending.
   Its RayCluster never became ready.
5. **Neither job released its placed pods** — `minReplicas: 15` was unmet but
   already-bound pods stay Running. **Deadlock.**

### Incident B — 2026-04-01 through 2026-04-02: Cascading `cudaErrorDevicesUnavailable`

Third resubmit attempt (`ts500-e2`) crashed at CUDA init with:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

**Cause**: c369's stuck RayCluster from Incident A held 3 nodes for 18+ hours.
When `ts500-e2` was submitted, Volcano could only find 3 free nodes for 15
required workers → partial placement → CUDA device already claimed by a ghost
pod → crash.

**The fix at the time** was manual cleanup:
```bash
kubectl delete raycluster <stuck-cluster-name>
kubectl delete rayjob --all   # nuclear
```

### Incident C — 2026-04-13: `c196 + c197` partial fit

Submitted two 16-node jobs. The cluster had 33 nodes, but 2 were unschedulable
that day, leaving **31 usable — not enough for 2×16 = 32**.

- c196 got 16 pods Running.
- c197 got 13 of 16 pods Running, 3 stuck Pending.
- c197's RayCluster never became ready.

Again, no gang-scheduling meant Volcano happily placed 13 of c197's 16 pods.
Those 13 held nodes indefinitely. This is when we adopted the rule: **submit
sequentially, wait for c196 pods to all be Running before submitting c197**.
That rule is a workaround, not a fix.

### Common root cause (all three incidents)

| Layer | What it did | Was that wrong? |
|-------|-------------|-----------------|
| Kueue | Admitted both jobs based on paper quota math | ❌ No — quota arithmetic was correct |
| KubeRay | Created all 32 pods for admitted jobs | ❌ No — it creates what was admitted |
| Volcano | Bound pods one-by-one, interleaving | ✅ **Yes — this is the failure point.** No gang-scheduling was configured, so partial placement was allowed. |

**The missing feature**: atomic "all 16 pods of a RayJob get placed, or none
do" guarantee. We have neither Kueue's gang-admission nor Volcano's PodGroup
gang-scheduling actually wired up (details in Q2 below).

---

## Question 2: Why both? Are we even configured correctly?

### What our current yaml actually does

```yaml
# RayJob metadata
labels:
  kueue.x-k8s.io/queue-name: training-queue    # triggers Kueue admission

# RayJob spec
spec:
  shutdownAfterJobFinishes: true               # required by Kueue

  rayClusterSpec:
    headGroupSpec.template.spec:
      schedulerName: volcano                   # Volcano as per-pod scheduler
    workerGroupSpecs[*].template.spec:
      schedulerName: volcano                   # Volcano as per-pod scheduler
```

**Kueue is doing**: quota accounting, FIFO queueing, admission.
**Volcano is doing**: per-pod placement decisions (which node to bind each pod to).
**Neither is doing gang-scheduling in our current config.**

### Why we have both — the real reason

Looking at the operator deployments in our cluster:
- `kuberay-operator` — installed by our admin
- `volcano-scheduler` + `volcano-controllers` + `volcano-admission` — installed
- `kueue-controller-manager` — installed

All three were set up by the cloud admin. From the 2026-04-03 analysis:

> Layer 1 (Kueue): **tenant-level multi-team accounting** — the admin's tool
> for giving each team a slice of the cluster. ClusterQueue `b300-training-queue`
> gives our team 256 GPU of nominal quota.
>
> Layer 3 (Volcano): **GPU-aware pod scheduler** — replaces kube-scheduler for
> ML workloads that need topology-aware, resource-aware placement.

In other words: **Kueue and Volcano answer different questions**.

| Question | Who answers it |
|----------|----------------|
| "Is this tenant allowed to consume 128 GPU right now given quotas/priorities across teams?" | **Kueue** |
| "Given this pod spec, which physical node should it bind to?" | **Volcano** (or default kube-scheduler) |
| "Can all 16 pods of this RayJob be placed atomically, or should we wait?" | **Neither — not configured** |

So: having both is *architecturally correct* for a multi-tenant GPU cluster.
The two serve non-overlapping purposes. The bug isn't "both are configured";
the bug is "**gang scheduling is configured in neither**".

### What the two official docs each claim about gang-scheduling

Both docs technically say their layer can do gang-scheduling — **but they mean
different things by "gang"**:

| Doc | What "gang-schedule" means there | Does our yaml trigger it? |
|-----|----------------------------------|---------------------------|
| Ray/Kueue doc | All-or-nothing **admission**: Kueue admits the whole RayJob or none of it. Paper math: "does the tenant have 128 GPU of quota?" | Partial — we have the queue-name label and `shutdownAfterJobFinishes: true` ✅, but this only guards admission, not placement. |
| Ray/Volcano doc | All-or-nothing **physical placement**: KubeRay creates a `PodGroup` with `minMember: N`; Volcano holds all pods `Pending` until `N` nodes are simultaneously free. | **No.** We use `schedulerName: volcano` on pods, but KubeRay is not configured to create PodGroups. The Volcano doc requires KubeRay operator to be installed with `--batch-scheduler=volcano` (or helm value `batchScheduler.name=volcano`). |

I verified our KubeRay operator does NOT have batch-scheduler enabled:

```bash
$ kubectl -n kube-system get deploy kuberay-operator -o jsonpath='{.spec.template.spec.containers[0].args}'
["--feature-gates=...","--enable-leader-election=true","--enable-metrics=true",
 "--reconcile-concurrency=1","--qps=100","--burst=200"]
# ↑ no --batch-scheduler flag

$ kubectl get podgroup -A
No resources found   # ← KubeRay isn't creating PodGroups for our RayJobs
```

**Conclusion on correctness**: Our yaml invokes Kueue correctly for quota, and
invokes Volcano as a per-pod scheduler, but **neither is doing
gang-scheduling**. This is the gap that caused Incidents A/B/C. We need one of:

**Option 1 — Enable Volcano PodGroup gang-scheduling via KubeRay** (recommended
by the Ray/Volcano doc):
  - Admin action: reinstall/patch kuberay-operator with `--batch-scheduler=volcano`.
  - Result: KubeRay auto-creates a PodGroup per RayJob with `minMember = head + workers`.
  - Volcano then refuses to place any pod until `minMember` nodes are free.

**Option 2 — Kueue-based serialization**: lower `nominalQuota` from 256 → 128.
  - Only 1 × 16-node job admitted at a time; no concurrent jobs → no interleaved
    placement → no deadlock.
  - Trade-off: 50% cluster utilization when two jobs are waiting.
  - Tenant-side fix (no admin action).

**Option 3 — Kueue elastic-jobs + Workload all-or-nothing** (newer Kueue feature):
  - The kueue doc mentions `kueue.x-k8s.io/elastic-job: "true"` and that
    "Kueue admits workloads on an 'all or nothing' basis". But "all or nothing"
    here is still admission-level, not placement-level. It prevents partial
    **provisioning by KubeRay**, not partial placement by Volcano. In practice
    this is weaker than Volcano PodGroups for our failure mode.

---

## Question 3: Which is better per the official docs?

**Short answer**: For our exact problem (partial placement deadlock), **Volcano
with KubeRay's `--batch-scheduler=volcano` is the correct fix.** Kueue cannot
solve the physical-placement problem no matter how it's configured.

### Point-by-point

| Criterion | Kueue | Volcano (with KubeRay batchScheduler) |
|-----------|-------|----------------------------------------|
| Multi-tenant quota, priority, preemption | ✅ Native strength | ⚠️ Has queues, but less flexible |
| FIFO / priority queueing | ✅ Built-in (`queueingStrategy`, `priorityClassName`) | ✅ (Volcano `Queue` with `weight`) |
| "All-or-nothing" **admission** (prevent partial RayJob creation) | ✅ That's its core model | ❌ Not its role |
| "All-or-nothing" **physical placement** (prevent interleaved binding) | ❌ Happens after Kueue; Kueue does not see physical nodes | ✅ PodGroup `minMember` enforces this atomically |
| KubeRay integration | Via label `kueue.x-k8s.io/queue-name`; requires `shutdownAfterJobFinishes: true` | Via KubeRay operator flag `--batch-scheduler=volcano`; auto-creates PodGroup |
| Needed for topology-aware (RoCE/NVLink) placement? | ❌ No | ✅ `volcano.sh/network-topology-mode: hard` supports it |
| Who admin-manages the resources? | Admin defines `ClusterQueue`/`ResourceFlavor`; users only set a label | Admin installs Volcano; users set `schedulerName: volcano` + optionally queue label |
| Docs describe as | "Quota management and scheduling for workloads" | "Batch system for running batch, AI/ML, HPC and data workloads" |

### Why not "just pick one and drop the other"

The Kueue doc and Volcano doc each describe a **complete solution** in
isolation, but they solve different halves of multi-tenant GPU scheduling:

- **Kueue without Volcano**: fine if all your pods are happy with the default
  kube-scheduler (independent, stateless pods). Inadequate for coscheduled
  multi-pod workloads — default kube-scheduler still places pods one-by-one.
- **Volcano without Kueue**: fine if you have one team, one queue. Inadequate
  for multi-tenant quota (Volcano `Queue` supports weight but not the richer
  nominal/borrowing-quota model Kueue has).

Our cluster is multi-tenant (multiple teams share 32 B300 nodes) AND our
workloads are coscheduled (RayJob = gang of 16 pods). We genuinely need both.

### The actual recommended config (per both docs combined)

```yaml
# RayJob (same as today — no changes needed here)
metadata:
  labels:
    kueue.x-k8s.io/queue-name: training-queue   # Kueue admission
    volcano.sh/queue-name: kuberay-test-queue   # Optional Volcano queue
spec:
  shutdownAfterJobFinishes: true                # required by Kueue
  rayClusterSpec:
    # ⚠️ No longer need schedulerName: volcano on pods (per Ray/Volcano doc:
    #    "Starting from KubeRay v1.3.0 you no longer need the
    #     ray.io/scheduler-name: volcano label")
    headGroupSpec:
      rayStartParams: {num-gpus: '8'}
      template:
        spec:
          containers: [...]
    workerGroupSpecs: [...]
```

But **the critical change is admin-side**, not yaml-side:

```bash
# Admin needs to install kuberay-operator with batchScheduler enabled:
helm upgrade kuberay-operator kuberay/kuberay-operator \
  --set batchScheduler.name=volcano
```

Once that's done, KubeRay will auto-create a PodGroup for every RayJob with
`minMember = head + len(workers)`, and Incident A/B/C cannot recur — Volcano
will refuse to place any pod of a RayJob until all 16 target nodes are free
simultaneously.

### Verdict

- **If only one can be kept**: Volcano (with KubeRay batchScheduler), because
  our failure mode is physical placement, not admission.
- **If both are kept (realistic)**: Kueue for tenant admission; Volcano
  PodGroup (via KubeRay batchScheduler flag) for atomic placement. This is
  what both docs implicitly assume when they describe their own piece; they
  don't discuss each other because each doc scopes itself to one layer.
- **What blocks the proper fix**: Admin action to enable KubeRay's
  `batchScheduler.name=volcano`. We've never asked for this explicitly — our
  prior workaround has been "submit jobs sequentially, wait for one to be
  Running before submitting the next". That works but leaves ~2 min idle per
  handoff and needs human discipline.

---

## Operational TL;DR

| Situation | What to do today (no admin action) | What to ask admin for |
|-----------|------------------------------------|-----------------------|
| Single 16-node job | Submit. Works fine. | — |
| 2 × 16-node concurrent | **Serialize**: submit A, wait until all 16 pods Running, then submit B. | Enable `batchScheduler.name=volcano` on kuberay-operator → true gang-scheduling, safe concurrent. |
| Stuck RayCluster after partial placement | `kubectl delete raycluster <stuck>` or `kubectl delete rayjob --all` | Same as above — fix removes the failure mode. |
| Need priority between jobs across teams | Use `kueue.x-k8s.io/priority-class: <name>` on RayJob labels | Admin defines `WorkloadPriorityClass` objects. |
| Need topology-aware (single-rack) placement | None today | Admin enables Volcano network-topology plugin; then we add `volcano.sh/network-topology-mode: hard`. |

---

## References

- Ray docs — Kueue gang-scheduling: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kueue.html#step-4-gang-scheduling-with-kueue
- Ray docs — Volcano + KubeRay: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html
- Our internal analysis: `iter_kuberay_32nodes_verl_training/k8s_scheduling_explained.md` (written 2026-04-03, covers the 3-layer breakdown + Incident A in detail)
- Our yamls: `iter_kuberay_32nodes_verl_training/yaml/stage10-*-rayjob.yaml`
