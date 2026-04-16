#!/usr/bin/env bash
# Run a probe pod against a specified PVC and capture the inside/outside UID view.
#
# Usage: bash probe.sh <pvc-name> <pvc-short-name> [host-mount-path-if-known]
#
#   pvc-name        e.g. "squash-test-false"
#   pvc-short       e.g. "false"  (used in pod name: squash-probe-<short>)
#   host-mount-path Optional; if we can't see the PVC from the host directly
#                   (vcluster Loft fake-pv), we leave this blank.
set -euo pipefail
export HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201

PVC="${1:?Usage: probe.sh <pvc-name> <short>}"
SHORT="${2:?Usage: probe.sh <pvc-name> <short>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL="${HERE}/yaml/pod-probe-template.yaml"
GEN="${HERE}/yaml/generated-probe-${SHORT}.yaml"
LOG="${HERE}/logs/probe-${SHORT}.log"

echo "== Probing PVC: ${PVC} (short=${SHORT}) =="
sed -e "s|__PVC__|${PVC}|g" -e "s|__PVC_SHORT__|${SHORT}|g" "$TPL" > "$GEN"

# 1) Apply PVC (if it doesn't exist yet)
PVC_YAML="${HERE}/yaml/pvc-${SHORT}.yaml"
if [ -f "$PVC_YAML" ]; then
    echo "[$(date -u +%FT%TZ)] Applying PVC: $PVC_YAML"
    kubectl apply -f "$PVC_YAML"
fi

# 2) Wait for PVC Bound
for i in {1..30}; do
    PHASE=$(kubectl get pvc "$PVC" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [ "$PHASE" = "Bound" ]; then
        echo "[$(date -u +%FT%TZ)] PVC $PVC is Bound"
        break
    fi
    echo "[$(date -u +%FT%TZ)] PVC $PVC phase=$PHASE (waiting)"
    sleep 2
done

# 3) Apply probe pod
echo "[$(date -u +%FT%TZ)] Applying probe pod"
kubectl apply -f "$GEN"
POD="squash-probe-${SHORT}"

# 4) Wait for pod Running or Succeeded
for i in {1..60}; do
    PHASE=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [ "$PHASE" = "Running" ] || [ "$PHASE" = "Succeeded" ]; then
        echo "[$(date -u +%FT%TZ)] Pod $POD phase=$PHASE"
        break
    fi
    if [ "$PHASE" = "Failed" ]; then
        echo "[$(date -u +%FT%TZ)] Pod $POD FAILED — describe:"
        kubectl describe pod "$POD" | tail -40
        break
    fi
    echo "[$(date -u +%FT%TZ)] Pod $POD phase=$PHASE (waiting)"
    sleep 5
done

# 5) Capture pod logs
echo "[$(date -u +%FT%TZ)] Collecting pod logs -> $LOG"
kubectl logs "$POD" --all-containers=true --tail=-1 > "$LOG" 2>&1 || true

# 6) Show a summary
echo "===== SUMMARY ====="
echo "pod log: $LOG"
echo ""
tail -40 "$LOG"
echo ""
echo "===== END SUMMARY ====="
