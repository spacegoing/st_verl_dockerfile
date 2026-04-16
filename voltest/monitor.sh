#!/usr/bin/env bash
# One-shot probe of voltest state. Prints counts + detail tables.
set -euo pipefail
export HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201

echo "=== $(date -u +%FT%TZ) ==="
echo ""
echo "-- RayJobs (voltest=true) --"
kubectl get rayjob -l voltest=true -o wide 2>&1 | head -20
echo ""
echo "-- PodGroups (scheduling.volcano.sh) --"
kubectl get podgroup -A 2>&1 | head -20
echo ""
echo "-- Pods (voltest=true) --"
kubectl get pod -l voltest=true -o wide 2>&1 | head -20
echo ""
echo "-- Summary counts --"
JOBS=$(kubectl get rayjob -l voltest=true --no-headers 2>/dev/null | wc -l)
PGS=$(kubectl get podgroup -A --no-headers 2>/dev/null | wc -l)
PODS_RUN=$(kubectl get pod -l voltest=true --no-headers 2>/dev/null | awk '$3=="Running"{c++} END{print c+0}')
PODS_PEND=$(kubectl get pod -l voltest=true --no-headers 2>/dev/null | awk '$3=="Pending"{c++} END{print c+0}')
PODS_OTHER=$(kubectl get pod -l voltest=true --no-headers 2>/dev/null | awk '$3!="Running" && $3!="Pending"{c++} END{print c+0}')
echo "  RayJobs: $JOBS"
echo "  PodGroups: $PGS"
echo "  Pods Running: $PODS_RUN | Pending: $PODS_PEND | Other: $PODS_OTHER"
