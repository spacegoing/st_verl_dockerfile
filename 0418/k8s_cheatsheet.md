# K8s cheatsheet — daily observation and management

Targeted at this cluster's workflow: RayJob training on Volcano gang.
All commands assume the `KP` alias defined below.

---

## 0. Setup

The system proxy blocks the cluster's API endpoint. Every `kubectl` call
must unset `HTTPS_PROXY` and `HTTP_PROXY` and include the API IP in
`NO_PROXY`. Two ways to avoid typing the prefix every time:

```bash
# Add BOTH to ~/.bashrc and ~/.zshrc

# Alias — works in interactive shell only (typing at the prompt)
alias KP='HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl'

# Function — works in interactive shell AND inside scripts / pipelines
KP() { HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl "$@"; }
```

The function form is more robust: alias expansion does not happen inside
command substitution, pipelines inside scripts, or `xargs`. If you copy
any of the cleanup commands below into a `.sh` file, the function is
what makes them work.

Verify both work:
```bash
KP get nodes | head -3
KP get rayjob --no-headers | wc -l     # count all RayJobs (piped to wc)
```

---

## 1. "What is going on right now?" — 4 commands

```bash
# Only non-terminal RayJobs (hides SUCCEEDED/FAILED clutter)
KP get rayjob --no-headers | awk '$2 != "SUCCEEDED" && $2 != "FAILED"'

# PodGroups: Running = actually on nodes; Inqueue = waiting for gang capacity
KP get podgroup

# Ray pods only (ignore non-Ray workloads)
KP get pod -l ray.io/cluster --no-headers | awk '{print $1, $3}'

# Cluster capacity at a glance
KP get nodes --no-headers | wc -l    # total node count
KP describe node | grep -E "^(Name:|  nvidia.com/gpu)"  # per-node GPU
```

## 2. "What is blocked and why?" — 3 commands

```bash
# Any Inqueue PodGroup = waiting. Shows why it cannot run.
KP get podgroup -o json | jq -r '.items[] | select(.status.phase=="Inqueue") | [.metadata.name, .spec.minMember, (.status.conditions[-1].message // "")] | @tsv'

# Which nodes are being held right now (by whom)
KP get pod -A -o wide --field-selector=status.phase=Running | awk '$7!=""' | awk '{print $7}' | sort -u

# Volcano scheduler recent events (why a gang is stuck)
KP describe podgroup <name> | tail -30
```

## 3. "What happened to this job?" — per-job inspection

```bash
JOB=bra40-dbg-cdbg5-8n-vmk62   # replace with actual name

# Status, start/end time, failure reason
KP get rayjob $JOB -o jsonpath='{.status}' | jq

# Pods belonging to this job (head + workers)
CLUSTER=$(KP get rayjob $JOB -o jsonpath='{.status.rayClusterName}')
KP get pod -l ray.io/cluster=$CLUSTER -o wide

# Head pod (where training driver runs)
HEAD=$(KP get pod -l ray.io/cluster=$CLUSTER,ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
echo "head pod: $HEAD"

# PodGroup status (gang scheduling info)
KP describe podgroup ray-${JOB}-pg
```

## 4. Logs — "what did the training driver say?"

```bash
# Tail head pod logs (the actual training driver output)
KP logs $HEAD -c ray-head --tail=200

# Follow live
KP logs $HEAD -c ray-head -f

# The Ray driver output via ray job logs (from inside the head pod)
KP exec $HEAD -c ray-head -- bash -c 'ray job logs $(ray job list 2>/dev/null | python3 -c "import sys,re; print([m.group(1) for m in re.finditer(r\"submission_id=\x27(raysubmit_[^\x27]+)\x27\", sys.stdin.read())][0])")'

# Training driver log on PFS (new layout, per-run dir keyed by RayJob):
ls /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/run.log
tail -n 200 /mnt/public/lichang93/st_verl_dockerfile/verl/ckpts/40bra_k8s_single_domain/$JOB/run.log

# Old layout (runs before 2026-04-18 12:45 UTC) — deprecated, still readable:
ls -t /mnt/public/lichang93/st_verl_dockerfile/verl/logs/40bra_k8s_single_domain/ | head -1
```

## 5. Delete and cleanup

### Delete one job
```bash
KP delete rayjob <name>
```

### Delete by label (e.g., everything belonging to a combo)
```bash
KP delete rayjob -l combo_id=cdbg5
```

### Delete terminal-state jobs older than N days
```bash
# List candidates first (SUCCEEDED/FAILED only, older than 7 days)
KP get rayjob -o json | jq -r --arg now "$(date -u +%s)" '
  .items[] | select(.status.jobStatus=="SUCCEEDED" or .status.jobStatus=="FAILED")
    | select((($now|tonumber) - (.metadata.creationTimestamp|fromdateiso8601)) > 86400*7)
    | .metadata.name'

# Then delete them
KP get rayjob -o json | jq -r --arg now "$(date -u +%s)" '
  .items[] | select(.status.jobStatus=="SUCCEEDED" or .status.jobStatus=="FAILED")
    | select((($now|tonumber) - (.metadata.creationTimestamp|fromdateiso8601)) > 86400*7)
    | .metadata.name' \
  | xargs -r KP delete rayjob
```

### Delete orphan Completed pods (RayJob submitter corpses)
```bash
KP delete pod --field-selector=status.phase=Succeeded
```

### Delete orphan PodGroups (no owning RayJob)
```bash
for pg in $(KP get podgroup -o jsonpath='{.items[*].metadata.name}'); do
    owner=$(KP get podgroup $pg -o jsonpath='{.metadata.ownerReferences[0].name}')
    [ -z "$owner" ] && { echo "orphan: $pg"; continue; }
    KP get rayjob "$owner" >/dev/null 2>&1 || { echo "orphan (no RayJob $owner): $pg"; KP delete podgroup $pg; }
done
```

### The "nuke everything terminal" one-liner (use with care)
```bash
# Dry-run: show what would be deleted
KP get rayjob -o json | jq -r '.items[] | select(.status.jobStatus=="SUCCEEDED" or .status.jobStatus=="FAILED") | .metadata.name'
# Real: delete all SUCCEEDED/FAILED
KP get rayjob -o json | jq -r '.items[] | select(.status.jobStatus=="SUCCEEDED" or .status.jobStatus=="FAILED") | .metadata.name' | xargs -r KP delete rayjob
```

## 6. Cluster capacity

```bash
# How many nodes are fully free? (no RayJob pods on them)
BUSY=$(KP get pod -A -l ray.io/cluster -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | sort -u | wc -l)
TOTAL=$(KP get nodes --no-headers | grep -v 'vnode' | wc -l)
echo "busy=$BUSY  total=$TOTAL  free=$((TOTAL - BUSY))"

# Per-node GPU allocation
KP get pod -A -o json | jq -r '
  .items[] | select(.status.phase=="Running")
    | .spec.nodeName + " " + (.spec.containers[0].resources.requests."nvidia.com/gpu" // "0")
' | awk '{sum[$1]+=$2} END {for (n in sum) print n, sum[n]}' | sort -k2 -n -r
```

## 7. Ownership — "who submitted this?"

Useful before deleting something you did not start.

```bash
# RayJob submitter label (if set)
KP get rayjob <name> -o jsonpath='{.metadata.labels}'

# Pod creator via ownerReferences
KP get pod <pod> -o jsonpath='{.metadata.ownerReferences}'

# vcjob (Volcano Job) submitter
KP get vcjob <name> -o jsonpath='{.metadata.labels.lepton\.sensetime\.com/submitter}'
```

## 8. Volcano specifics (gang scheduling)

```bash
# All gang-scheduled workloads
KP get podgroup

# Volcano scheduler logs (why a gang is stuck)
KP -n kube-system logs deploy/volcano-scheduler --tail=100

# Scheduler configmap (gang plugin, binpack weights)
KP get cm -n kube-system volcano-scheduler-configmap -o jsonpath='{.data.volcano-scheduler\.conf}'

# VCJobs (Volcano's native job type — used by non-Ray workloads like nccl-test)
KP get vcjob
KP describe vcjob <name>
KP delete vcjob <name>   # deletes parent → pods vanish permanently
```

## 9. Watch mode (live updates)

```bash
KP get rayjob -w                                  # watch rayjobs
KP get podgroup -w                                # watch gangs
KP get pod -l ray.io/cluster=<cluster> -w         # watch one job's pods
KP logs <pod> -c ray-head -f                      # follow logs
```

Terminate with `Ctrl-C`.

## 10. Troubleshooting patterns

```bash
# A pod is stuck Pending — why?
KP describe pod <pod> | tail -30

# A gang is stuck Inqueue — why?
KP describe podgroup <pg> | tail -30

# An operator is dying — check events
KP -n kube-system logs deploy/kuberay-operator --tail=100

# I want to shell into the head pod
KP exec -it $HEAD -c ray-head -- bash

# Copy a file out of a pod before it terminates
KP cp $HEAD:/path/in/container /local/path
```

## 11. Quick cleanup for the current mess

Your `KP get rayjob` right now shows 15 entries, only a few matter. To clear
everything SUCCEEDED/FAILED and keep the cluster view clean:

```bash
# See what will be deleted
KP get rayjob --no-headers | awk '$2=="SUCCEEDED" || $2=="FAILED" {print $1}'

# Delete them
KP get rayjob --no-headers | awk '$2=="SUCCEEDED" || $2=="FAILED" {print $1}' | xargs -r KP delete rayjob

# Then clean up any orphan pods the RayJob gc left behind
KP delete pod --field-selector=status.phase=Succeeded
```

This is safe for workloads you own. If there are RayJobs from other users
(check the `lepton.sensetime.com/submitter` label), filter them out first.

## 12. One-shot status: "give me the full picture"

```bash
# Copy-paste this block when you need a quick situation report
echo "=== rayjob ==="
KP get rayjob --no-headers | awk '$2!="SUCCEEDED" && $2!="FAILED" {printf "%-40s %-12s %-15s %s\n", $1,$2,$3,$7}'
echo
echo "=== podgroup ==="
KP get podgroup --no-headers
echo
echo "=== ray pods (running only) ==="
KP get pod -l ray.io/cluster --no-headers --field-selector=status.phase=Running | awk '{printf "%-60s %s\n", $1, $6}'
echo
echo "=== capacity ==="
BUSY=$(KP get pod -l ray.io/cluster --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | sort -u | wc -l)
echo "busy=$BUSY  total=31  free=$((31 - BUSY))"
```
