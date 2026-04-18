# 0418 — Plan A Execution Summary: KubeRay + Volcano correctness

> Exhaustive record of what was investigated, written, and verified for
> Plan A. See also `0418/plan_a_kuberay_volcano.md` (the plan itself)
> and `0418/summary.md` (combined high-level summary).

---

## 1. File map — what changed and why

### New files

| Path | Created | Purpose |
|---|---|---|
| `0418/plan_a_kuberay_volcano.md` | 2026-04-17 17:28 | The plan document. 9 sections: cluster state (verified), voltest reference pattern, current production submit flow compliance, legacy yaml drift, expected Volcano behavior, 6 verification checks, risks + mitigations, deliverables, exit criteria. |
| `0418/verify_gang.sh` | 2026-04-17 17:32 | Executable bash: one-shot validator running the 6 Plan-A checks against a named RayJob. Prints per-check pass/fail and an actionable diagnostic on failure. |
| `0418/rayjob_debug.yaml` | 2026-04-17 17:31 | Parameterized debug RayJob template. Conforms to the voltest pattern: `generateName`, no Kueue label, no manual `schedulerName: volcano`, `replicas == minReplicas == maxReplicas`. Used by both Plan A (to produce real RayJobs for verification) and Plan B (as the debug-run template). |
| `0418/submit_debug.sh` | 2026-04-17 17:32 | Submitter CLI wrapping `envsubst` → `kubectl create`. Unsets `HTTPS_PROXY`/`HTTP_PROXY` (k8s API is outside proxy path) and sets `NO_PROXY=180.184.249.201`. Validates combo id and node-count bounds. |

### Modified files

| Path | Change type | Brief |
|---|---|---|
| `iter_kuberay_32nodes_verl_training/yaml/legacy/README.md` | Edit | Expanded the existing "do not resurrect" warning from 3 lines to an enumerated list of the 3 specific drift items with file:line citations and a concrete pointer to the correct pattern (`submit/rayjob.yaml`). |

### Files inspected but NOT changed (already correct)

| Path | Why inspected | Conclusion |
|---|---|---|
| `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml` | Active production submit template | Already conforms to voltest pattern. No change needed. |
| `iter_kuberay_32nodes_verl_training/submit/submit.sh` | Active submit wrapper | Correctly unsets proxy envs and uses `envsubst` whitelist. No change needed. |
| `voltest/submit/rayjob.yaml` | Known-good reference template | Used as canonical comparison point. |
| `k8s_kuberay_kueue_setup/fix_kuberay_volcano_config_plan.md` | Evidence for Phase 4 fix | Verified the fix has been applied on the live cluster. |

---

## 2. Summary — what was accomplished

**Objective**: Prove that every RayJob we submit for verl training uses the correct KubeRay + Volcano gang-scheduling pattern (the one voltest/ verified at 10×8-node scale), so we never re-enter the March-incident partial-placement deadlock class.

**Outcome**: ✅ **Plan A fully satisfied.**

Concretely:
1. Verified on the live cluster that `kuberay-operator` runs with `--batch-scheduler=volcano`, Volcano's `gang` plugin is enabled, and Kueue is completely uninstalled.
2. Audited the active production submit template (`submit/rayjob.yaml` + `submit/submit.sh`) — already conforms to the voltest pattern; no changes required.
3. Identified exact divergences in the archived `yaml/legacy/stage10*.yaml` files (stale Kueue label, manual `schedulerName`, hard-coded `metadata.name`) and documented them in `yaml/legacy/README.md` so they cannot silently leak into new work.
4. Built `0418/verify_gang.sh` — a script with 6 checks that exercises the full Volcano gang contract on any running RayJob. Ran it 4 times against 4 real RayJobs (1 failed, 3 successful) over the course of the Plan B node-count experiments; all 6 checks passed every time once the verifier itself was debugged.
5. Proved in production traffic that Volcano gang works cleanly: on the second P0 submission (`bra40-dbg-cdbg5-16n-4pllg`), capacity was temporarily blocked by the first P0's lingering pods (ttl=1800). The PodGroup entered `Inqueue` with zero pods bound. When the failed run was explicitly deleted, the 16-pod gang promoted `Inqueue → Scheduled` within ~8 seconds, atomically. No partial placement was ever observed.

**What did NOT need to be done**: modify `submit/rayjob.yaml` or `submit/submit.sh`. Active production is already correct. The only content change was the `yaml/legacy/README.md` warning.

---

## 3. Exhaustive step-by-step log

### Step A-1 — Cluster state probe (2026-04-17 ~17:25 UTC)

Launched an agent to run these exact commands (proxy unset required — system proxy blocks direct k8s API access):

```bash
export KC="HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl"
$KC -n kube-system get deploy kuberay-operator -o jsonpath='{.spec.template.spec.containers[0].args}'
$KC -n kube-system get deploy volcano-scheduler
$KC get cm -n kube-system volcano-scheduler-configmap -o jsonpath='{.data.volcano-scheduler\.conf}'
$KC get clusterqueue,resourceflavor,localqueue -A
$KC -n kueue-system get deploy
$KC get rayjob,podgroup,raycluster -A
$KC get nodes --no-headers | wc -l
```

**Findings** (all verified live):
- `kuberay-operator` args array contains `"--batch-scheduler=volcano"` at index 1.
- Operator pod `kuberay-operator-f58bf4ccc-hkb2x` was 1/1 Running (up 2d2h).
- `volcano-scheduler`, `volcano-controllers`, `volcano-admission` all 1/1 (v1.11.2, deployed 58 days).
- `volcano-scheduler-configmap` has `gang` plugin enabled in tier 1 alongside `priority` and `conformance`. Actions: `allocate, reclaim`.
- **No Kueue CRDs registered** (`server doesn't have a resource type "clusterqueue"`). `kueue-system` namespace has zero deployments.
- 0 active RayJobs, 0 PodGroups. 15 historical RayJobs all in terminal state.
- 32 nodes (31 compute × 8 B300 GPU + 1 vnode) fully idle.

### Step A-2 — Plan A drafted (17:28 UTC)

`0418/plan_a_kuberay_volcano.md` written with 9 sections:
1. Cluster state (verified facts above)
2. voltest reference template (9 properties the correct pattern must have)
3. Current production submit flow assessment (fully conforms)
4. Legacy yaml drift (stale Kueue label line 23, manual `schedulerName` lines 38/124, hard-coded `metadata.name` line 18)
5. Expected Volcano behavior (the observable invariants)
6. 6 verification checks for the debug run
7. Risks + mitigations (operator helm rollback, node drain, ttl side-effects, scheduler crash)
8. Deliverables (README warning, verify_gang.sh, this plan)
9. Exit criteria

### Step A-3 — verify_gang.sh initial draft (17:32 UTC)

Wrote the validator with 6 checks:
1. RayJob exists
2. RayCluster name resolved
3. PodGroup exists (matched by `ownerReferences[0].name`)
4. `minMember == 1 + workerReplicas`
5. Pod state coherence (all pods same `status.phase`)
6. No Unschedulable events in `kubectl describe podgroup`
+ RayJob jobStatus sanity

### Step A-4 — Legacy yaml README warning (17:33 UTC)

Tried `Write` — failed because the file already existed and I hadn't `Read` it first. Read existing content (brief 3-line warning), then used `Edit` to expand it. The updated README now enumerates the 3 drift items with file:line citations:

- `stage10a-*.yaml:23` — `kueue.x-k8s.io/queue-name: training-queue` (no-op post-Phase-4, but confusion risk)
- `stage10a-*.yaml:38` head, `:124` worker — manual `schedulerName: volcano` (operator sets this now)
- `stage10a-*.yaml:18` — hard-coded `metadata.name` (blocks concurrent resubmission; should use `generateName`)

### Step A-5 — First real run for verification (17:36 UTC)

Submitted `bra40-dbg-cdbg5-16n-2mvbt` via `0418/submit_debug.sh cdbg5 16 ...`. This was also the P0 baseline for Plan B, so Plan A's verification ran concurrently with the Plan B ladder.

`verify_gang.sh` returned:
- Check 1: ✅ RayJob exists
- Check 2: ✅ RayCluster name resolved
- Check 3: ❌ **"No PodGroup found for RayCluster"** — false failure.

Manual check showed the PodGroup **was** created (`ray-bra40-dbg-cdbg5-16n-2mvbt-pg`, minMember=16, Running) — but its `ownerReferences[0].name` was the **RayJob** (`bra40-dbg-cdbg5-16n-2mvbt`), not the RayCluster (`bra40-dbg-cdbg5-16n-2mvbt-vwcpv`).

**Bug #1 in verify_gang.sh**: checked owner against `$CLUSTER`. The kuberay-operator creates the PodGroup with the parent RayJob as owner, not the cluster.

**Fix**: changed the jsonpath from
```
'{range .items[?(@.metadata.ownerReferences[0].name==\'${CLUSTER}\')]}'
```
to
```
'{range .items[?(@.metadata.ownerReferences[0].name==\'${JOB}\')]}'
```
…and updated all the diagnostic messages accordingly.

### Step A-6 — Second run of verify_gang.sh (17:37 UTC)

Now:
- Check 3: ✅ PodGroup exists: minMember=16, phase=Running
- Check 4: ✅ minMember matches 1+workerReplicas = 16
- Check 5: ❌ **"Mixed pod states (gang failure!): Pending Running"** — false failure.
- Check 6: ❌ **"PodGroup has 3 Unschedulable event(s)"** — false failure.

Manual check revealed:
- Head pod was `1/1 Running` (status.phase=Running).
- Worker pods were `0/1 Init:0/1` (status.phase=Pending until init container completes). All 16 pods had `spec.nodeName` set, i.e. they were all **bound** to nodes — just not done with container init yet.
- PodGroup had transitioned `Unschedulable → Scheduled` within 8 seconds at creation (normal — Volcano logs the initial "14/14 Pending, minAvailable=16" before the atomic admit).

**Bug #2 in verify_gang.sh**: the gang invariant is atomic **placement** (nodeName binding), not atomic **phase**. Container init naturally takes longer on worker pods than on the head.

**Bug #3 in verify_gang.sh**: `kubectl describe podgroup` contains historical events. Transient Unschedulable events before the first admit are expected. The correct check is on current `.status.phase` and whether the `Scheduled` condition is now True.

**Fix (single edit)**:
```bash
# Check 5 — replaced status.phase check with spec.nodeName binding check
POD_COUNT=$(kubectl get pod -l "ray.io/cluster=$CLUSTER" --no-headers | wc -l)
BOUND=$(kubectl get pod -l "ray.io/cluster=$CLUSTER" \
    -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' | grep -c .)
# Pass if BOUND == POD_COUNT (all bound) OR BOUND == 0 (none bound yet, correct gang).
# Fail only if 0 < BOUND < POD_COUNT (partial placement).

# Check 6 — replaced event history grep with current status check
PG_PHASE_NOW=$(kubectl get podgroup "$PG_NAME" -o jsonpath='{.status.phase}')
SCHEDULED_STATUS=$(kubectl get podgroup "$PG_NAME" \
    -o jsonpath="{.status.conditions[?(@.type=='Scheduled')].status}")
# Pass if Running/Inqueue + Scheduled=True; fail only if currently Unschedulable.
```

### Step A-7 — Third run of verify_gang.sh (17:37 UTC)

All 6 checks pass:

```
✓ RayJob exists
✓ RayCluster name resolved: bra40-dbg-cdbg5-16n-2mvbt-vwcpv
✓ PodGroup exists: name=ray-bra40-dbg-cdbg5-16n-2mvbt-pg  minMember=16  phase=Running
✓ minMember matches 1+workerReplicas = 16
✓ All 16 pods bound to nodes (gang placement atomic)
✓ PodGroup Running + Scheduled=True (gang admitted atomically)
✓ RayJob status: jobStatus=  deployment=Initializing
```

### Step A-8 — Observed gang queueing in the wild (17:47 UTC)

When I deleted the failed P0 and resubmitted (`bra40-dbg-cdbg5-16n-4pllg`), the PodGroup initially went to `Inqueue` because the failed run's pods were still terminating. `verify_gang.sh` correctly reported:

```
✓ PodGroup exists: ...  phase=Inqueue
✓ No pods bound yet (16 created but Pending — PodGroup waiting for capacity)
✓ PodGroup Inqueue (no Scheduled condition yet — transitional)
```

Once the failed run's pods finished terminating (~10 s later), the 16-pod gang promoted atomically. This is the exact queueing behavior voltest documented: Volcano admits the whole gang or none of it.

### Step A-9 — Re-verified across P1 and P2 (18:52 + 20:02 UTC)

Every subsequent run — `bra40-dbg-cdbg5-8n-f5d9j` (minMember=8) and `bra40-dbg-cdbg5-4n-7hzjr` (minMember=4) — passed all 6 checks on first run. PodGroup sized correctly for each. Gang admission was atomic.

### Step A-10 — No changes to production submit flow

Audited `iter_kuberay_32nodes_verl_training/submit/rayjob.yaml` line-by-line against the voltest template. Conformance:

- `generateName: bra40-sd-${COMBO_ID}-` ✅ (line 22)
- No Kueue label ✅ (labels at 24-28 are all non-Kueue)
- No manual `schedulerName: volcano` on pod specs ✅ (comment at 47-49 explains why)
- `replicas == minReplicas == maxReplicas == 15` ✅ (lines 110-112)
- `shutdownAfterJobFinishes: true` ✅ (line 31)
- `ttlSecondsAfterFinished: 1800` (intentionally large for post-run inspection)
- YAML anchors `&full_env`, `&full_mounts`, `&full_volumes` for env/mounts/volumes sync between head and worker

Decision: **do not touch production files**. The submit template is correct. The only deliverable was the README warning in `yaml/legacy/`.

---

## 4. Current stage — status & how it was solved

**Status: ✅ SOLVED.**

All Plan A exit criteria satisfied:

- [x] Every active submit path (`submit/submit.sh <combo>`) confirmed correct against voltest pattern (Step A-10)
- [x] Legacy yamls carry a README warning (Step A-4)
- [x] `0418/verify_gang.sh` exists and passes against real debug runs (Steps A-7, A-8, A-9)
- [x] 5-step debug RayJobs (P0, P1, P2) ran end-to-end with all Section 6 checks passing (Steps A-5..A-9; executed jointly with Plan B)

**How it was verified**:
- 4 live RayJobs exercised over a 4-hour period (one failed-then-terminal, three successful full runs)
- Every check was exercised by `verify_gang.sh` and re-verified by manual `kubectl` inspection at least once
- The "gang under capacity pressure" case was caught naturally when the failed P0's pods held 16 nodes hostage for ~10 seconds — Volcano correctly queued the next gang as `Inqueue` and promoted it atomically

**Bugs found and resolved** (all in Plan A instrumentation, not in cluster config):

| # | File | Bug | Root cause | Fix |
|---|---|---|---|---|
| A-1 | `verify_gang.sh` | Check 3 "No PodGroup found" false failure | Checked `ownerReferences[0].name=='${CLUSTER}'`, but kuberay-operator creates PodGroup owned by RayJob | Changed selector to `${JOB}` |
| A-2 | `verify_gang.sh` | Check 5 "Mixed pod states" false failure | Used `status.phase`, which differs during container init even when gang placement is correct | Switched to `spec.nodeName` (all-bound or none-bound check) |
| A-3 | `verify_gang.sh` | Check 6 "Unschedulable event(s)" false failure | Grepped event history; transient pre-gang Unschedulable is normal | Switched to current `.status.phase` + `conditions[?Scheduled].status` |

**No bugs found in cluster config** — the Phase 4 fix (`--batch-scheduler=volcano` + Kueue uninstall) was already applied and working correctly at the time of investigation.

**Follow-ups intentionally NOT done** (scope-limited):

- Not re-writing production `submit/rayjob.yaml` (it was already correct).
- Not editing the legacy yamls themselves (they are archival; a README warning is sufficient).
- Not adding a CI check that runs `verify_gang.sh` on every submit (could be a future enhancement if we add a centralized submit wrapper).
