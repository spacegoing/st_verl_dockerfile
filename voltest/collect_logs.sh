#!/usr/bin/env bash
# Collect pod stdout for each voltest job into voltest/logs/pod-<job>.log.
# Also dumps rayjob / podgroup / pod state snapshots.
set -euo pipefail
export HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/logs"
mkdir -p "$OUT"

TS=$(date -u +%Y%m%dT%H%M%SZ)
kubectl get rayjob -l voltest=true -o wide   > "$OUT/state-rayjob-${TS}.txt"    2>&1 || true
kubectl get podgroup -A                       > "$OUT/state-podgroup-${TS}.txt"  2>&1 || true
kubectl get pod -l voltest=true -o wide      > "$OUT/state-pod-${TS}.txt"       2>&1 || true

# For each voltest pod, fetch stdout
for pod in $(kubectl get pod -l voltest=true -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null); do
    F="$OUT/pod-${pod}.log"
    echo "collecting $pod -> $F"
    kubectl logs "$pod" --all-containers=true --tail=-1 > "$F" 2>&1 || true
done

echo "Collected to $OUT/"
ls -la "$OUT/" | tail -30
