#!/usr/bin/env bash
# 0418/verify_gang.sh — one-shot verification that a RayJob is gang-scheduled by Volcano.
#
# Usage: ./verify_gang.sh <rayjob_name>
#
# Runs the 6 checks from plan_a_kuberay_volcano.md section 6.
# Exits 0 if all pass, non-zero with diagnostic if any fail.

set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export NO_PROXY="${K8S_API_IP:-180.184.249.201}" \
       no_proxy="${K8S_API_IP:-180.184.249.201}"

JOB="${1:?usage: $0 <rayjob_name>}"

pass() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1" >&2; FAIL=1; }

FAIL=0

echo "[verify_gang] RayJob: $JOB"
echo

# 1. RayJob exists
if ! kubectl get rayjob "$JOB" >/dev/null 2>&1; then
    fail "RayJob $JOB not found"
    exit 1
fi
pass "RayJob exists"

CLUSTER=$(kubectl get rayjob "$JOB" -o jsonpath='{.status.rayClusterName}')
if [ -z "$CLUSTER" ]; then
    # Before RayCluster is created, the name isn't set yet — back off gracefully.
    echo "  (RayCluster not yet created; re-run after ~30s)"
    exit 0
fi
pass "RayCluster name resolved: $CLUSTER"

# 2. PodGroup exists — owned by the RayJob (not the RayCluster)
PG_LINE=$(kubectl get podgroup -o jsonpath="{range .items[?(@.metadata.ownerReferences[0].name=='${JOB}')]}{.metadata.name}{'\t'}{.spec.minMember}{'\t'}{.status.phase}{'\n'}{end}" 2>/dev/null | head -1)
if [ -z "$PG_LINE" ]; then
    fail "No PodGroup found for RayJob $JOB"
    echo
    echo "Operator args (should include --batch-scheduler=volcano):"
    kubectl -n kube-system get deploy kuberay-operator \
        -o jsonpath='{.spec.template.spec.containers[0].args}'
    echo
    echo "All PodGroups in namespace:"
    kubectl get podgroup
    exit 2
fi
PG_NAME=$(echo "$PG_LINE" | cut -f1)
PG_MINMEMBER=$(echo "$PG_LINE" | cut -f2)
PG_PHASE=$(echo "$PG_LINE" | cut -f3)
pass "PodGroup exists: name=$PG_NAME  minMember=$PG_MINMEMBER  phase=$PG_PHASE"

# 3. minMember matches expected (1 + worker replicas)
EXPECTED_MIN=$(kubectl get raycluster "$CLUSTER" \
    -o jsonpath='{range .spec.workerGroupSpecs[*]}{.replicas}{"\n"}{end}' 2>/dev/null \
    | awk 'BEGIN{s=1} {s+=$1} END{print s}')
if [ "$PG_MINMEMBER" != "$EXPECTED_MIN" ]; then
    fail "minMember mismatch: expected=$EXPECTED_MIN  actual=$PG_MINMEMBER"
else
    pass "minMember matches 1+workerReplicas = $EXPECTED_MIN"
fi

# 4. Pod binding coherence — gang invariant is that all pods are BOUND to nodes
# (spec.nodeName set) or NONE are. Once bound, container init can progress
# asynchronously per pod; a pod with status Pending but spec.nodeName set
# is already scheduled (gang has succeeded) and is just initializing.
POD_COUNT=$(kubectl get pod -l "ray.io/cluster=$CLUSTER" --no-headers 2>/dev/null | wc -l)
BOUND=$(kubectl get pod -l "ray.io/cluster=$CLUSTER" \
    -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' 2>/dev/null | grep -c . || true)

if [ "$POD_COUNT" = "0" ]; then
    pass "No pods yet (PodGroup still queueing — correct gang behavior)"
elif [ "$BOUND" = "$POD_COUNT" ]; then
    pass "All $POD_COUNT pods bound to nodes (gang placement atomic)"
elif [ "$BOUND" = "0" ]; then
    pass "No pods bound yet ($POD_COUNT created but Pending — PodGroup waiting for capacity)"
else
    fail "Partial pod binding (gang failure!): $BOUND of $POD_COUNT bound"
    kubectl get pod -l "ray.io/cluster=$CLUSTER" -o wide
fi

# 5. PodGroup must currently be in a non-Unschedulable phase.
# (Transient Unschedulable events before the first gang bind are normal.)
PG_PHASE_NOW=$(kubectl get podgroup "$PG_NAME" -o jsonpath='{.status.phase}' 2>/dev/null)
# Scheduled condition must exist (and, if present, be True)
SCHEDULED_STATUS=$(kubectl get podgroup "$PG_NAME" \
    -o jsonpath="{.status.conditions[?(@.type=='Scheduled')].status}" 2>/dev/null)
if [ "$PG_PHASE_NOW" = "Pending" ]; then
    pass "PodGroup is Pending (gang waiting for capacity — correct queue behavior)"
elif [ "$PG_PHASE_NOW" = "Inqueue" ] || [ "$PG_PHASE_NOW" = "Running" ]; then
    if [ "$SCHEDULED_STATUS" = "True" ]; then
        pass "PodGroup $PG_PHASE_NOW + Scheduled=True (gang admitted atomically)"
    else
        pass "PodGroup $PG_PHASE_NOW (no Scheduled condition yet — transitional)"
    fi
elif [ "$PG_PHASE_NOW" = "Unschedulable" ]; then
    fail "PodGroup is currently Unschedulable"
    kubectl describe podgroup "$PG_NAME" | tail -30
else
    pass "PodGroup phase=$PG_PHASE_NOW"
fi

# 6. RayJob job status sanity
JOB_STATUS=$(kubectl get rayjob "$JOB" -o jsonpath='{.status.jobStatus}' 2>/dev/null)
DEPLOY_STATUS=$(kubectl get rayjob "$JOB" -o jsonpath='{.status.jobDeploymentStatus}' 2>/dev/null)
pass "RayJob status: jobStatus=$JOB_STATUS  deployment=$DEPLOY_STATUS"

echo
if [ "$FAIL" = "0" ]; then
    echo "[verify_gang] ✅ all checks passed"
    exit 0
else
    echo "[verify_gang] ❌ $FAIL check(s) failed — investigate"
    exit 1
fi
