# Kueue vs Volcano — Deep Analysis (single-user, ML-only cluster)

> Revisited after pushback. Earlier doc (`kueue_vs_volcano_analysis.md`) claimed
> "keep both, they do different things." This doc replaces that conclusion.
>
> **Context correction**: the cluster is 31 nodes, **single-user (you)**, single
> workload type (RayJob for GRPO training). Kueue was installed at your request;
> Volcano was installed by the cloud admin as their default GPU-cluster stack.
> The "multi-tenant quota" justification for keeping both does **not** apply.
>
> This doc answers — with direct quotes from official docs and exact mechanism
> descriptions — whether Kueue or Volcano alone is sufficient, and which is
> better for our specific setup.

---

## TL;DR

Both tools can **standalone** do queueing + gang-scheduling for RayJob. You do
not need both. For our single-user ML-training cluster, **Volcano alone is
strictly better** because:

1. Its gang-scheduling is **proactive** (pods never partially bind) — Kueue's
   is **reactive** (pods partially bind, then get evicted on timeout).
2. It has first-class ML features KubeRay is explicitly designed to use
   (`batchScheduler.name: volcano` → auto-PodGroup, topology-aware placement).
3. Kueue's main unique value is multi-tenant cohort/quota-borrowing — which
   you do not use.

Detailed evidence in §3 below.

---

## 1. What each tool does — with official doc evidence

### Volcano

**Volcano Queue** (https://volcano.sh/en/docs/queue/) — queueing + quota.

> "A Queue is a collection of PodGroups, which adopts FIFO. It is also used as
> the basis for resource division."

Quota fields: `capability` (hard upper limit), `deserved` (fair share),
`guarantee` (reserved floor), `weight` (proportional), `reclaimable`.
Priority field controls preemption across queues.

**Volcano PodGroup** (https://volcano.sh/en/docs/podgroup/) — gang scheduling.

> "PodGroup is a group of pods with strong association, mainly used in batch
> scheduling … `minMember` specifies the minimum pod count required; if this
> threshold cannot be met, the entire group remains unscheduled."
>
> "When resource requirements (minMember + minResources) cannot be satisfied,
> no pod or task in the PodGroup will be scheduled."

This is **proactive gang scheduling**: the Volcano scheduler will not bind
*any* pod of the PodGroup until `minMember` nodes satisfying `minResources`
are **simultaneously** free.

Status transitions: `Pending → Inqueue → Running` (or `Unschedulable`).

**Volcano vcjob** (https://volcano.sh/en/docs/vcjob/) — Volcano's own batch
job kind. Not directly relevant to us since we use RayJob, but note:

> "If no PodGroup is specified when a VolcanoJob is created, Volcano will
> create a PodGroup automatically."

KubeRay does the analogous auto-creation when the operator is started with
`--batch-scheduler=volcano`, per the Ray docs
(https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html):

> "KubeRay automatically generates a corresponding PodGroup with `minMember`
> and `minResources` specifications."

### Kueue

**Kueue Concepts** (https://kueue.sigs.k8s.io/docs/concepts/) — admission
control for "Workloads".

> "A Workload is an application that will run to completion. It is the **unit
> of admission** in Kueue."
>
> "Quota reservation is sometimes referred to as workload scheduling or job
> scheduling, but it should **not** be confused with pod scheduling."

This is the critical architectural statement: **Kueue never binds pods to
nodes.** It decides whether a Workload is admitted (allowed to create pods);
the pod-to-node binding is done by kube-scheduler (or whatever scheduler is
configured for the pod).

**Kueue RayJob integration** (https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/):

> "Kueue controls the `spec.suspend` field of the RayJob. When a RayJob is
> admitted by Kueue, Kueue will unsuspend it by setting `spec.suspend` to `false`."

Mechanism: `spec.suspend=true` initially → admission passes → `false` →
KubeRay proceeds to create pods.

**Kueue waitForPodsReady / all-or-nothing**
(https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/) — this
is Kueue's gang-scheduling feature.

> "The workload is monitored by Kueue until all of its Pods are ready (meaning
> scheduled, running, and passing the optional readiness probe). If pods don't
> become ready within the timeout period, the workload is **evicted and
> requeued** … the admission is cancelled, the corresponding job is suspended
> and the Workload is re-queued."
>
> "When `blockAdmission: true` is enabled, workloads are admitted sequentially
> to prevent deadlock situations."

This is **reactive gang scheduling**: Kueue lets the pods be created and
partially placed. If they do not all reach Ready within `timeout` (default
5 min), Kueue sets `spec.suspend=true` → KubeRay tears down all pods →
Workload is requeued.

Kueue does **not** physically reserve nodes. It cannot tell the kube-scheduler
"don't place these pods until all N nodes are free". It only admits, waits,
and evicts.

---

## 2. Side-by-side: the two gang-scheduling mechanisms

| Property | Volcano PodGroup | Kueue waitForPodsReady |
|---|---|---|
| **Mechanism class** | Proactive (hold) | Reactive (evict) |
| **Where it acts** | Pod scheduling — scheduler refuses to bind | Workload admission + post-admission monitoring |
| **What happens on partial fit** | Pods stay `Pending` indefinitely; no side effects | Pods are created & partially placed; after `timeout` the Workload is evicted and all pods deleted; Workload requeued |
| **Resource churn during retry** | None (pods never bind) | High (all pods created, scheduled, then destroyed on timeout) |
| **Deadlock prevention across two concurrent RayJobs** | Built-in — second RayJob's PodGroup simply sits Pending until first frees nodes | Requires `blockAdmission: true` (serializes admission), else both can be evicted in cycles |
| **Requires operator-level config** | Yes — `--batch-scheduler=volcano` on kuberay-operator | Yes — `waitForPodsReady.enable=true` on kueue-controller-manager |
| **Yaml-level user action** | None (auto-PodGroup per RayJob) | Label `kueue.x-k8s.io/queue-name: <name>` + `shutdownAfterJobFinishes: true` |
| **What binds pods to nodes** | Volcano scheduler | kube-scheduler (or Volcano if `schedulerName: volcano`) |
| **Topology-aware (RoCE/NUMA/rack)** | Yes (`volcano.sh/network-topology-mode: hard`) | No native — you'd need affinity rules |
| **Tunables** | `minMember`, `minResources`, `priorityClassName`, `queue` | `timeout` (default 5m), `recoveryTimeout`, `blockAdmission`, `requeuingStrategy`, `backoffLimitCount`, `backoffBaseSeconds`, `backoffMaxSeconds` |

### Applied to our March 31 failure (c364 + c369, 32 pods on 33 nodes, 2 unavailable)

**Under Volcano PodGroup (minMember=16)**:
- c364 PodGroup admitted; Volcano holds pods `Pending`; binds one by one until
  minMember satisfied. Since 31 nodes free, c364 places all 16. ✅
- c369 PodGroup admitted; Volcano needs 16 free nodes but only 15 remain.
  All 16 c369 pods stay `Pending` (no binding yet). ✅
- When c364 finishes, Volcano binds all 16 c369 pods atomically. ✅
- **No deadlock is possible** — partial placement is structurally prevented.

**Under Kueue waitForPodsReady (timeout=5m, blockAdmission=false)**:
- Both c364 and c369 admitted; both sets of 16 pods created; kube-scheduler
  interleaves placement. After 5 min, c369 has only 13/16 pods Ready.
- Kueue evicts c369 → KubeRay tears down c369's pods.
- c369 requeued; on retry, c364 is fully running, so c369's 16 pods fit.
- **No deadlock, but 5+ minutes wasted and high churn.**

**Under Kueue waitForPodsReady + blockAdmission=true**:
- c364 admitted; c369 held at admission until c364's pods are Ready.
- c369 admitted only after c364 is fully Running.
- **Effectively serializes the two jobs** — trades throughput for safety.

### Applied to our April 13 failure (31 usable nodes for 32 pods)

- **Volcano PodGroup**: c196 places 16 (✅). c197 PodGroup waits forever
  (only 15 free) — `Pending` until c196 finishes. Admin visible via
  `kubectl get podgroup`. Operator can cancel c197 if waiting too long.
- **Kueue waitForPodsReady**: c197's 16 pods partially placed (13/16), 5 min
  timeout → evict → requeue → still only 15 nodes → evict → requeue → infinite
  churn. Eventually `backoffLimitCount` reached → workload deactivated.
- **Kueue + blockAdmission**: c197 never admitted while c196 runs. Clean.

---

## 3. Does Kueue require Volcano, or vice versa? Official answer: No.

### Kueue alone suffices

From the docs cited in §1:
- Kueue admits (suspends/unsuspends RayJob).
- `waitForPodsReady` provides all-or-nothing semantics via eviction.
- Pods are bound by kube-scheduler (default) or any scheduler you set.
- The Kueue RayJob doc contains **zero** references to Volcano.

### Volcano alone suffices

From the docs cited in §1:
- Volcano Queue provides FIFO queueing + quota (`capability`/`deserved`/etc.).
- Volcano PodGroup provides proactive gang scheduling.
- KubeRay (with operator flag `--batch-scheduler=volcano`) auto-generates
  PodGroups per RayJob with correct minMember.
- The Ray Volcano doc contains **zero** references to Kueue.

### Neither doc mentions combining them

Both pages describe a complete solution for their use case. The silence is
deliberate: they are alternatives, not layers.

### What our current config actually uses

```bash
# kuberay-operator args — no --batch-scheduler flag:
$ kubectl -n kube-system get deploy kuberay-operator \
    -o jsonpath='{.spec.template.spec.containers[0].args}'
["--feature-gates=...","--enable-leader-election=true",
 "--enable-metrics=true","--reconcile-concurrency=1",
 "--qps=100","--burst=200"]

# No PodGroups exist:
$ kubectl get podgroup -A
No resources found

# Our RayJob yaml uses `schedulerName: volcano` on pods but NO PodGroup
# and NO kueue.x-k8s.io/priority-class:
$ grep -E 'kueue|volcano|scheduler|suspend|PodGroup' stage10-*.yaml
    kueue.x-k8s.io/queue-name: training-queue
    shutdownAfterJobFinishes: true
    schedulerName: volcano
    schedulerName: volcano
```

**What this means**: today we run **both tools in their degraded modes**.
Kueue admits but doesn't gang (no `waitForPodsReady` on the controller).
Volcano binds pods (because `schedulerName: volcano`) but doesn't gang
(no PodGroup). The two tools are stacked, but neither is doing the
gang-scheduling job. That's why Incidents A/B/C (deadlocks) happened.

---

## 4. Which is better for our setup?

### Evaluation criteria (our actual needs)

| Need | Weight | Notes |
|---|---|---|
| Gang scheduling for 16-pod RayJobs | **critical** | Sole cause of past outages |
| Prevent deadlock when 2 concurrent 16-node jobs submit | **critical** | Happened 3 times |
| Multi-tenant quota / cohort borrowing | **irrelevant** | Single user |
| FIFO queueing of pending jobs | moderate | Nice to have when >2 jobs queued |
| Priority / preemption across jobs | low | All our jobs are equal priority |
| Topology-aware placement (RoCE affinity) | moderate | 8-NIC nodes; same-rack better |
| Low operator churn | moderate | Fewer evict/retry cycles preferred |
| Fits Ray's documented paved path | moderate | Ray docs and examples skew to Volcano for batchScheduler |

### Scorecard

| Criterion | Volcano standalone | Kueue standalone |
|---|---|---|
| Prevents partial pod placement (proactive) | ✅ PodGroup `minMember` | ⚠️ Reactive (evict+requeue) |
| Prevents 2-job deadlock without extra flags | ✅ (PodGroups independent) | ⚠️ Needs `blockAdmission=true` (serialization) |
| Resource churn on tight-fit retry | Low | High |
| KubeRay-native integration | ✅ `--batch-scheduler=volcano` auto-PodGroup | ✅ `spec.suspend` control |
| Topology-aware scheduling | ✅ built-in | ❌ requires manual affinity |
| ML/HPC plugins (PyTorch/MPI env) | ✅ | ❌ |
| Multi-tenant quota sophistication | ✅ queues with hierarchy | ✅✅ cohorts, borrowing, fair-share |
| Useful features we don't need | Fewer wasted | Many wasted |
| Operator-level admin action required | Yes (one helm flag) | Yes (waitForPodsReady config) |

### Verdict: **Volcano alone, with KubeRay operator configured for it.**

**Reasoning**:
1. Our actual failure mode is physical placement, not admission. Volcano fixes
   it at the level where the bug exists.
2. Volcano's gang is free of eviction churn. For 16-node GPU jobs that take
   2+ minutes to initialize CUDA/NCCL, eviction on timeout is very costly.
3. Ray's own docs show Volcano as the canonical batch-scheduler integration
   for KubeRay (see the `batchScheduler.name: volcano` helm pattern).
4. Kueue's strongest feature (cohort-based quota-borrowing across tenants) is
   unused in a single-user cluster; we'd be paying complexity cost for no gain.
5. Volcano's topology plugins give a path to better RoCE locality later if we
   want it.

**When Kueue would be the right answer instead**: a shared cluster with
multiple teams, heterogeneous workloads (Ray + TFJob + PyTorchJob + vanilla
Jobs), strict quota enforcement per team, and admission control with borrowing.
That is not us.

---

## 5. Proposed change to our setup

### Admin action (one-time)

```bash
# Reinstall or patch kuberay-operator with batchScheduler enabled:
helm upgrade kuberay-operator kuberay/kuberay-operator \
  --namespace kube-system \
  --set batchScheduler.name=volcano
```

This causes kuberay-operator to auto-create a `PodGroup` (with `minMember =
1 + len(workerGroups)*replicas` and `minResources` aggregated from pod
requests) every time a RayJob/RayCluster is created.

### Tenant action (our yaml)

We can keep our current yaml essentially as-is, or simplify:

```yaml
# Option A: Minimal — remove Kueue entirely
metadata:
  name: bra40-sd-math-c351
  # no labels needed — Volcano handles queue via spec
spec:
  shutdownAfterJobFinishes: true
  rayClusterSpec:
    headGroupSpec:
      template:
        spec: { containers: [...] }      # no schedulerName needed; batchScheduler flag sets it
    workerGroupSpecs:
    - replicas: 15
      minReplicas: 15
      maxReplicas: 15
      template:
        spec: { containers: [...] }
```

Optionally add a Volcano queue label for priority/separation later:
```yaml
metadata:
  labels:
    volcano.sh/queue-name: b300-training   # if admin creates this queue
```

### Kueue: leave installed but remove from yaml

- Drop `kueue.x-k8s.io/queue-name: training-queue` label from RayJobs — this
  stops Kueue from gating our jobs.
- Optionally ask admin to delete the `b300-training-queue` ClusterQueue since
  it's no longer used.

This is a strictly-reducing change: less complexity, fewer failure modes.

### Migration validation

After the admin flips the batchScheduler flag:

```bash
# Verify PodGroups are now created per RayJob
kubectl apply -f stage10-...-rayjob.yaml
kubectl get podgroup -A        # should show one per RayJob
kubectl describe podgroup <name> | grep -E 'MinMember|MinResources|Phase'
# Expected:
#   MinMember: 16
#   MinResources: cpu=2816, memory=30720Gi, nvidia.com/gpu=128
#   Phase: Running (or Pending if cluster full)
```

Submit two 16-node jobs concurrently to reproduce the old failure:
```bash
kubectl apply -f job-A.yaml
kubectl apply -f job-B.yaml
# Expected: one runs, the other's PodGroup stays Pending — not interleaved.
kubectl get podgroup -A
kubectl get pod -l ray.io/is-ray-node=yes
```

---

## 6. Questions for the admin (if you want to adopt §5)

1. Can you re-deploy `kuberay-operator` with `batchScheduler.name=volcano`
   (helm value) or equivalently add `--batch-scheduler=volcano` to the
   operator's container args?
2. After that is live, is it OK to remove the Kueue `ClusterQueue`
   `b300-training-queue` + `LocalQueue` `training-queue`, or leave them
   unused?
3. Do you have a Volcano queue naming convention you want us to use, or is
   `default` fine?

---

## 7. References (all quotes above are sourced from these)

- Ray Kueue integration: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kueue.html
- Ray Volcano integration: https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/volcano.html
- Volcano Queue: https://volcano.sh/en/docs/queue/
- Volcano PodGroup: https://volcano.sh/en/docs/podgroup/
- Volcano vcjob: https://volcano.sh/en/docs/vcjob/
- Kueue Concepts: https://kueue.sigs.k8s.io/docs/concepts/
- Kueue RayJob integration: https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/
- Kueue waitForPodsReady: https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/
- Internal: `iter_kuberay_32nodes_verl_training/k8s_scheduling_explained.md`
