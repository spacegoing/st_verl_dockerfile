# K8s / KubeRay / Volcano operations cheatsheet

Practical reference for day-to-day LLM training ops on this cluster.
Assumes `KP` is already aliased in `.bashrc`:

```bash
alias KP='HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl'
```

All commands below use `KP` as a drop-in for `kubectl`. Every `KP` can
become `KP -n <ns>` if you work outside `default`.

---

## 1. Mental model — the 5 resource types in play

When you submit one training run, five different controllers each
create their own object. Here's the ownership chain:

```
             YOU RUN: bspo_scripts/submit_bspo_md.sh cbmd-v2
                                |
                                ▼
              RayJob (ray.io/v1)     ← this is the user-facing object
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   RayCluster            batch/v1 Job   ← the "driver" job that runs
   (ray.io/v1)                              `ray job submit …` on the head
       │
       ▼
    Pods (head + N×workers)
       │
       ▼
   PodGroup (scheduling.volcano.sh/v1beta1)
       ← volcano creates this for gang scheduling
       ← it tracks "minMember pods must land atomically"
```

**Key invariant**: kubectl's ownerReferences mean `KP delete rayjob X`
cascades downward to RayCluster, Pods, PodGroup, and the driver Job.
All five resource types should converge on the same lifecycle — *in
principle*. In practice, see §5.

---

## 2. What each resource tells you (and when to look)

| Resource         | CRD / API group                   | Look here when…                                                             | Key status fields                                                    |
|------------------|-----------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| `rayjob`         | `ray.io/v1`                       | you want the job-level view: submitted, running, succeeded, failed          | `status.jobStatus`, `status.jobDeploymentStatus`, `status.startTime`, `status.endTime` |
| `raycluster`     | `ray.io/v1`                       | you need the actual ray cluster state, head IP, worker count                | `status.state` (`ready`, `failed`), `status.desiredWorkerReplicas`, `status.availableWorkerReplicas` |
| `podgroup` / `pg`| `scheduling.volcano.sh/v1beta1`   | you want to know whether gang scheduling is satisfied                       | `status.phase` (`Inqueue`, `Pending`, `Running`, `Completed`), `spec.minMember`, `status.running` |
| `pods`           | core v1                           | a training step hangs / OOMs / you need logs or `exec`                      | `status.phase`, `status.containerStatuses[].state`, `metadata.labels`     |
| `jobs` (batch/v1)| `batch/v1`                        | you want to see the driver's own completion / error (e.g. Hydra parse failed) | `status.active`, `status.succeeded`, `status.failed`                  |

**Rule of thumb**:
- *Which jobs am I running?* → `rayjob`
- *Why is my submitted job still not running?* → `podgroup` (queued on `Inqueue`?) then `pods` (stuck `Pending`?).
- *Why did training fail?* → `rayjob` status → `raycluster` status → `pods` (look at head logs).
- *The entrypoint shell script crashed before training started?* → `jobs` (the driver Job's pod logs).

---

## 3. Inspection — top-of-day commands

### Everything at once (our most useful one-liner)

```bash
KP get rayjob,raycluster,podgroup
```

Three resources side-by-side. Each RayJob has one RayCluster and one
PodGroup. Matching prefixes make it easy to eyeball.

### Just what's alive right now

**Important caveat**: Kubernetes' `--field-selector` only supports a
short list of built-in fields — roughly `metadata.name`,
`metadata.namespace`, and for Pods `status.phase` + `spec.nodeName`.
Custom resources like `RayJob` and `PodGroup` do **not** advertise
their status fields as selectable, so
`--field-selector=status.jobStatus!=…` returns
`field label not supported`. Filter client-side with `awk`, `grep`,
or `jq`:

```bash
# RayJobs not in a terminal state (keeps the default wide columns)
KP get rayjob | awk 'NR==1 || ($2 != "SUCCEEDED" && $2 != "FAILED")'

# Same thing, names only (handy for piping to xargs)
KP get rayjob -o json | jq -r '
  .items[] | select((.status.jobStatus // "") != "SUCCEEDED"
                    and (.status.jobStatus // "") != "FAILED")
           | .metadata.name'

# PodGroups still scheduling or running
KP get podgroup | awk 'NR==1 || $2 != "Completed"'
```

Note the `// ""` in jq — a RayJob that is still Initializing has no
`status.jobStatus` field at all, so a raw `!=` comparison would miss
it. `// ""` normalises the missing case to empty string.

### By our project labels

Our submit scripts stamp labels on every object they create:

- `training-type=40bra-md` — all multi-domain BSPO runs
- `training-type=40bra-sd` — all single-domain BSPO runs
- `combo_id=<id>` — specific combo (e.g. `cbmd-v2`)
- `submitter=lichang93`

```bash
KP get rayjob -l training-type=40bra-md       # all md runs ever
KP get rayjob -l combo_id=cbmd-v2             # specific combo
KP get rayjob -l 'training-type in (40bra-sd,40bra-md)'   # both
```

### Per-job deep dive

```bash
# Everything owned by one RayJob
KP describe rayjob bra40-md-cbmd-v2-bvnzq

# Pods from one RayCluster
KP get pods -l ray.io/cluster=bra40-md-cbmd-v2-bvnzq-dvq9l

# Events (why is this pending / why did that pod restart)
KP get events --sort-by=.lastTimestamp --field-selector=involvedObject.name=bra40-md-cbmd-v2-bvnzq
```

---

## 4. Filtering: active / recent / historical

This is your pain point #2. By default `KP get rayjob` shows everything
— including month-old SUCCEEDED/FAILED rayjobs kept around by their
`ttlSecondsAfterFinished`.

### Three filters, from most to least useful

```bash
# (a) Not yet done. Best for "what's burning GPU right now?"
KP get rayjob | awk 'NR==1 || ($2 != "SUCCEEDED" && $2 != "FAILED")'

# (b) Started in last 24h. Good for post-mortem of today's work.
KP get rayjob -o json | jq -r '
  .items
  | sort_by(.status.startTime // "")
  | .[]
  | select((.status.startTime // "") > (now - 86400 | todateiso8601))
  | [.metadata.name, (.status.jobStatus // "-"), (.status.jobDeploymentStatus // "-"), .status.startTime] | @tsv'

# (c) Only mine (if multiple submitters share the cluster).
# Labels ARE field-selectable, so this one works natively:
KP get rayjob -l submitter=lichang93
```

### Why historical entries linger

Two reasons:
1. `shutdownAfterJobFinishes: true` kills the RayCluster + pods when the
   RayJob ends — but the `RayJob` object itself stays for
   `ttlSecondsAfterFinished: 1800` (30 min) so you can read logs.
2. After TTL, the RayJob is also deleted — but k8s Events, PodGroups,
   and orphan pods sometimes survive their owner's deletion if any
   finalizer or controller bug delays the GC.

### Purge historical rayjobs manually

```bash
# Preview: all SUCCEEDED rayjobs (names only)
KP get rayjob -o json | jq -r '.items[] | select(.status.jobStatus == "SUCCEEDED") | .metadata.name'

# Actually delete all completed rayjobs older than 24h
KP get rayjob -o json | jq -r '
  .items[]
  | select(.status.endTime != null)
  | select((.status.endTime | fromdateiso8601) < (now - 86400))
  | .metadata.name
' | xargs -r -n1 KP delete rayjob

# Aggressive: every terminal rayjob (use with care).
# --field-selector would be cleaner but doesn't work on CRD status fields
# (see §4 caveat); we pipe names through jq instead.
KP get rayjob -o json | jq -r '
  .items[] | select(.status.jobStatus == "SUCCEEDED" or .status.jobStatus == "FAILED") | .metadata.name
' | xargs -r -n1 KP delete rayjob
```

---

## 5. State-sync confusion (why "orphaned" PodGroups happen)

This is your pain point #3. Each of the 5 resource types above is
reconciled by a **different controller**:

| resource    | controller reconciling it                                       |
|-------------|-----------------------------------------------------------------|
| RayJob      | `kuberay-operator`                                              |
| RayCluster  | `kuberay-operator`                                              |
| Pod         | `kubelet` (node-local) + `kube-controller-manager` (cluster)   |
| PodGroup    | `volcano-controller-manager` (volcano's control plane)          |
| Job         | `kube-controller-manager` (stock Job controller)                |

These run as separate deployments. Each watches its own CRD and only
indirectly reacts to others via ownerReferences. Three common desync
failure modes:

### A. RayJob deleted, PodGroup lingers

Cause: volcano-controller hit a transient error while processing the
PodGroup deletion event, or the kuberay-operator's cascade-delete
reached the RayCluster before volcano noticed. Symptom:

```bash
KP get rayjob       # nothing
KP get podgroup     # ray-bra40-md-cbmd-v2-* still there
```

Fix:
```bash
KP delete podgroup ray-bra40-md-cbmd-v2-bvnzq-pg
```

### B. RayJob Running, some Pods stuck Pending

Cause: PodGroup is `Inqueue` because `minMember` can't be met — some
other job is holding the resources. Symptom:

```bash
KP get rayjob           # STATUS: (blank), DEPLOYMENT: Initializing
KP get podgroup         # PHASE: Inqueue, RUNNING: <minMember>
```

This is not a bug — it's gang scheduling waiting for a slot. Check
total cluster GPU usage. When another job releases nodes, volcano will
admit this one.

### C. rayjob SUCCEEDED but Pods still Terminating

Cause: pods have a finalizer (e.g. RDMA cleanup) that hasn't fired yet.
Symptom: `KP get pods` shows `Terminating` for minutes after the RayJob
ended. Fix:

```bash
# Wait. If 5+ minutes pass, force-delete:
KP delete pod <name> --grace-period=0 --force
```

### Is this a misconfiguration on our side?

No — this is the expected failure mode of a controller-per-resource
design. The upstream fixes (explicit finalizers, ownerReferences, TTL
controller) cover 99% of cases. The leftover 1% is why triage commands
(§3 + §6) exist.

### The one config choice we can revisit

In `rayjob_md.yaml` and `rayjob.yaml`:

```yaml
shutdownAfterJobFinishes: true
ttlSecondsAfterFinished: 1800   # 30 minutes
```

If you set `ttlSecondsAfterFinished: 0`, the RayJob is garbage-collected
the moment it finishes — no lingering entries, but also no log access
via `kubectl logs` once gone. The current 30-minute window is a fair
compromise. If you want cleaner lists, lower it to `300` (5 min) or
run the purge command in §4 as a cron.

---

## 6. Cleanup — delete safely

### One RayJob (cascades to cluster + pods + podgroup)

```bash
KP delete rayjob bra40-md-cbmd-v2-bvnzq
```

### All jobs for one combo

```bash
KP delete rayjob -l combo_id=cbmd-v2
```

### All of one training-type

```bash
KP delete rayjob -l training-type=40bra-md
```

### Nuke everything md + sd (use with care)

```bash
KP delete rayjob -l 'training-type in (40bra-sd,40bra-md)'
```

### Clean leftover podgroups + pods (post-mortem after a rough session)

```bash
# Podgroups whose rayjob is gone
KP get podgroup -o json | jq -r '
  .items[] | select(.metadata.ownerReferences == null) | .metadata.name
' | xargs -r -n1 KP delete podgroup

# Pods stuck Terminating > 5m
KP get pods --field-selector=status.phase=Unknown -o name | xargs -r -n1 KP delete --grace-period=0 --force
```

---

## 7. Submission workflows (from this repo)

One md combo:
```bash
STAGE=3 NNODES=8 verl/my_scripts/bspo_scripts/submit_bspo_md.sh cbmd-v2
```

All 5 formal cbmd runs:
```bash
verl/my_scripts/bspo_scripts/submit_formal_md.sh
```

One sd Phase-2 preview:
```bash
NNODES=4 verl/my_scripts/bspo_scripts/submit_bspo.sh cbsp302 \
    'actor_rollout_ref.actor.bspo_delta=3.0e-4 actor_rollout_ref.actor.bspo_lambda_tj=1.0e-2'
```

Dry-run preview (see yaml without submitting):
```bash
COMBO_ID=cbmd-v2 NNODES=8 WORKER_REPLICAS=7 EXTRA_OVERRIDES='env=k8s_b300_8node_debug' \
    envsubst '${COMBO_ID} ${NNODES} ${WORKER_REPLICAS} ${EXTRA_OVERRIDES}' \
    < verl/my_scripts/bspo_scripts/rayjob_md.yaml | less
```

Dry-run + let kube-apiserver validate without persisting:
```bash
# ...same envsubst as above... | KP create --dry-run=server -f -
```

---

## 8. Watch / follow — live monitoring

### RayJob status changes

```bash
KP get rayjob bra40-md-cbmd-v2-bvnzq -w
```

### Head-pod training logs (the actual training stdout)

```bash
CLUSTER=$(KP get rayjob bra40-md-cbmd-v2-bvnzq -o jsonpath='{.status.rayClusterName}')
KP logs -l ray.io/cluster=$CLUSTER -c ray-head --tail=200 -f
```

### Tail a specific metric out of the run log

```bash
HEAD=$(KP get pods -l ray.io/cluster=$CLUSTER,ray.io/node-type=head -o name | head -1 | sed 's|^pod/||')
KP exec $HEAD -- bash -c \
  'tail -f /root/myCodeLab/host/verl/ckpts/40bra_k8s_multi_domain/bra40-md-cbmd-v2-bvnzq/run.log' \
  | grep -E 'step:|bspo_md_|grad_norm|pg_loss'
```

### All ray pods across all jobs, with node placement

```bash
KP get pods -l ray.io/node-type --sort-by=.spec.nodeName \
    -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase,AGE:.metadata.creationTimestamp
```

### Cluster-wide GPU usage right now

```bash
KP describe node | grep -E 'Name:|nvidia.com/gpu' | paste - - -
```

---

## 9. Triage quick path when something's wrong

```bash
# Step 1: what do I see?
KP get rayjob,podgroup,raycluster -l combo_id=cbmd-v2

# Step 2: what's wrong with the RayJob?
KP describe rayjob bra40-md-cbmd-v2-bvnzq | tail -50

# Step 3: any pods not Running?
KP get pods -l ray.io/cluster=bra40-md-cbmd-v2-bvnzq-dvq9l \
    --field-selector=status.phase!=Running

# Step 4: pod-level events
KP describe pod <pending-or-failed-pod> | tail -40

# Step 5: driver Job logs (if entrypoint crashed before training)
KP logs job/bra40-md-cbmd-v2-bvnzq | tail -100
```

---

## 10. Recommended `.bashrc` additions

On top of the existing `alias KP=…`:

```bash
# Compact combined view — the one command to run first in the morning
alias KPall='KP get rayjob,raycluster,podgroup'

# Active jobs only (status.jobStatus is NOT field-selectable on CRDs;
# we filter client-side with awk — see §4 caveat).
alias KPr='KP get rayjob | awk '"'"'NR==1 || ($2 != "SUCCEEDED" && $2 != "FAILED")'"'"''
alias KPpg='KP get podgroup | awk '"'"'NR==1 || $2 != "Completed"'"'"''

# By project — replace with your usual filter
alias KPmd='KP get rayjob -l training-type=40bra-md'
alias KPsd='KP get rayjob -l training-type=40bra-sd'

# Tail the head-pod training log for a given rayjob name
# Usage: KPlog bra40-md-cbmd-v2-bvnzq
KPlog() {
    local rj=$1
    local cluster=$(KP get rayjob "$rj" -o jsonpath='{.status.rayClusterName}')
    [ -z "$cluster" ] && { echo "no cluster for $rj" >&2; return 1; }
    KP logs -l ray.io/cluster=$cluster -c ray-head --tail=200 -f
}

# Tail the run.log inside the head pod (more readable than ray output)
# Usage: KPrunlog bra40-md-cbmd-v2-bvnzq
KPrunlog() {
    local rj=$1
    local cluster=$(KP get rayjob "$rj" -o jsonpath='{.status.rayClusterName}')
    local head=$(KP get pods -l ray.io/cluster=$cluster,ray.io/node-type=head -o name | head -1 | sed 's|^pod/||')
    local project_dir=$(KP get rayjob "$rj" -o jsonpath='{.metadata.labels.training-type}')
    local subdir=$( [ "$project_dir" = "40bra-md" ] && echo 40bra_k8s_multi_domain || echo 40bra_k8s_single_domain )
    KP exec "$head" -- bash -c "tail -f /root/myCodeLab/host/verl/ckpts/$subdir/$rj/run.log"
}

# Describe rayjob in full (status + events + pods)
# Usage: KPdesc bra40-md-cbmd-v2-bvnzq
KPdesc() {
    local rj=$1
    KP describe rayjob "$rj" | tail -80
    echo "═══════════════════ Pods ═══════════════════"
    local cluster=$(KP get rayjob "$rj" -o jsonpath='{.status.rayClusterName}')
    KP get pods -l ray.io/cluster=$cluster
    echo "═══════════════════ Recent events ═══════════════════"
    KP get events --sort-by=.lastTimestamp \
        --field-selector=involvedObject.name=$rj | tail -10
}

# Clean one combo (cascades rayjob → raycluster → pods → podgroup)
# Usage: KPrm cbmd-v2
KPrm() { KP delete rayjob -l combo_id="$1"; }

# Clean all terminal rayjobs (both SUCCEEDED and FAILED).
# Same CRD-field caveat: resolve names with jq, then delete.
KPclean() {
    KP get rayjob -o json \
      | jq -r '.items[]
               | select(.status.jobStatus == "SUCCEEDED" or .status.jobStatus == "FAILED")
               | .metadata.name' \
      | xargs -r -n1 KP delete rayjob
}

# Show all ray pods with node placement
alias KPpods='KP get pods -l ray.io/node-type --sort-by=.spec.nodeName -o wide'
```

---

## 10b. k9s — the interactive alternative

`k9s` is a terminal UI over kubectl. For daily exploration (the three
pain points at the top of this doc) it's usually faster than the
kubectl + awk/jq pipelines above. Keep the CLI for scripting / CI; use
k9s for interactive triage.

### Install

Single binary, no root required:

```bash
# Option A: webinstall (simplest)
von && curl -sS https://webinstall.dev/k9s | bash

# Option B: if you have Go installed
go install github.com/derailed/k9s@latest
```

### Alias with the same proxy hygiene as KP

```bash
alias K9='HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 k9s'
```

### Keys you'll actually use

| key                     | does                                                     |
|-------------------------|----------------------------------------------------------|
| `:rayjob` / `:pod` / `:podgroup` / `:raycluster` / `:events` | jump to that resource type |
| `/<text>`               | live-filter the current view (e.g. `/cbmd` → only cbmd-* rows) |
| `0` / `1` / … / `9`     | switch namespace (`0` = all, `1` = first found, etc.)    |
| `l`                     | logs of selected pod (+ `f` to follow, `w` to wrap)      |
| `d`                     | describe                                                 |
| `y`                     | yaml                                                     |
| `e`                     | edit                                                     |
| `s`                     | shell into selected pod (like `kubectl exec -it`)        |
| `ctrl-d`                | delete (prompts for confirmation)                        |
| `enter`                 | drill into children (rayjob → its pods)                  |
| `esc`                   | back up a level                                          |
| `?` or `h`              | show key help for the current view                       |
| `:q` or `ctrl-c`        | quit                                                     |

### Why k9s beats the CLI for each of your pain points

1. **"which resource for what"** — `:` then type the start of any CRD
   name. Tab-completes. No need to remember exact plural / API group.
2. **"historical clutter"** — `/Running` or `/SUCCEEDED` filters the
   current table instantly. Combine with `ctrl-r` to reset.
3. **"state desync"** — the screen auto-refreshes every ~2s by default.
   When a rayjob hits SUCCEEDED you'll see its pods flip to
   `Terminating` in real time on a neighbouring view (`:pod`), making
   lingering objects obvious instead of hidden.

### Caveats

- Default refresh rate is 2s — on a busy cluster bump it to 5s or 10s
  via `~/.k9s/config.yaml`:
  ```yaml
  k9s:
    refreshRate: 5
  ```
- Column layouts for custom resources are generic by default. Tune
  `~/.k9s/skins/*.yaml` and `~/.k9s/views.yaml` to pin useful columns
  for RayJob (e.g. `status.rayClusterName`, `status.jobDeploymentStatus`).
- No piping / scripting. For automation, stay with kubectl + jq.

### Quick workflow mapping (CLI → k9s)

| what you want           | kubectl path                                          | k9s path                              |
|-------------------------|-------------------------------------------------------|---------------------------------------|
| list all rayjobs        | `KP get rayjob`                                       | `:rayjob`                             |
| active only             | `KP get rayjob \| awk …` (§4)                         | `:rayjob` → `/Running\|Initializing`  |
| describe one            | `KP describe rayjob X`                                | select row → `d`                      |
| head pod logs           | `KPlog X` function                                    | `:rayjob` → enter → select head → `l` → `f` |
| delete one rayjob       | `KP delete rayjob X`                                  | select row → `ctrl-d`                 |
| delete a combo          | `KPrm cbmd-v2`                                        | `:rayjob` → `/cbmd-v2` → `ctrl-a` → `ctrl-d` |
| spot stuck PodGroups    | `KP get podgroup \| awk …`                            | `:podgroup` (sorts by age)            |
| shell into head pod     | `KP exec -it <pod> -- bash`                           | select pod → `s`                      |

---

## 11. Summary: what to inspect for what

| You want to know…                            | First command                                              |
|----------------------------------------------|-------------------------------------------------------------|
| What's running right now                     | `KPall` or `KPr`                                            |
| Why is my submit stuck at Initializing       | `KP get podgroup -l combo_id=<id>` (is it `Inqueue`?)       |
| Why did training fail                        | `KPrunlog <rayjob>` then scroll up to last non-warning line |
| Why did the entrypoint fail before training  | `KP logs job/<rayjob-name>`                                 |
| GPU usage                                    | `KPpods` + `KP describe node`                               |
| Clean up after a messy session               | `KPclean` then `KP get podgroup` to spot orphans            |
| Submit a new run                             | `verl/my_scripts/bspo_scripts/submit_bspo_md.sh <combo>`    |
| Kill a specific run                          | `KP delete rayjob <name>`                                   |
| Kill a combo across all attempts             | `KPrm <combo>`                                              |

Keep this page open in a tab during the first week of any new cluster.
Almost every operational question maps to a line in the tables above.
