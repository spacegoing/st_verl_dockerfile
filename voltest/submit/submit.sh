#!/usr/bin/env bash
# submit.sh — launch one voltest RayJob (8 nodes × 8 GPU GPU-burn).
#
# Usage:
#   ./submit.sh                  # default DURATION_S=300
#   ./submit.sh <duration_s>     # override (e.g. 60 for smoke)
#
# Inspect:
#   kubectl get rayjob -l voltest=true
#   kubectl logs -l ray.io/cluster=$(kubectl get rayjob <name> -o jsonpath='{.status.rayClusterName}') -c ray-head -f
#
# Cancel all voltest jobs:
#   kubectl delete rayjob -l voltest=true

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL="${HERE}/rayjob.yaml"

# Proxy hygiene — the k8s API is outside the usual proxy path.
# Clear BOTH cases (Go HTTP client checks uppercase AND lowercase independently).
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export NO_PROXY="${K8S_API_IP:-180.184.249.201}" \
       no_proxy="${K8S_API_IP:-180.184.249.201}"

# ── args ──────────────────────────────────────────────────────────────────────
DURATION_S="${1:-300}"

[[ "$DURATION_S" =~ ^[0-9]+$ ]] || {
    echo "error: duration_s must be a positive integer, got '$DURATION_S'" >&2
    exit 2
}

[ -r "$TPL" ] || { echo "error: template missing: $TPL" >&2; exit 3; }

# ── submit ────────────────────────────────────────────────────────────────────
# envsubst with explicit whitelist: only ${DURATION_S} is substituted.
export DURATION_S

RAYJOB_NAME=$(envsubst '${DURATION_S}' < "$TPL" \
              | kubectl create -f - -o name \
              | sed 's|^rayjob.ray.io/||')

echo "[submit] duration=${DURATION_S}s  name=$RAYJOB_NAME"
echo
echo "watch progress:"
echo "  kubectl get rayjob $RAYJOB_NAME -w"
echo "head pod logs (once head is Running):"
echo "  kubectl logs -l ray.io/cluster=\$(kubectl get rayjob $RAYJOB_NAME -o jsonpath='{.status.rayClusterName}') -c ray-head -f"
