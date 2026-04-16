#!/usr/bin/env bash
# Clean up all squash_test objects from the cluster (PVCs, pods, generated yamls).
set -euo pipefail
export HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Delete probe pods =="
kubectl delete pod -l squashtest=true --ignore-not-found=true 2>&1

echo "== Delete test PVCs =="
for pvc in squash-test-true squash-test-false squash-test-default; do
    kubectl delete pvc "$pvc" --ignore-not-found=true 2>&1
done

echo "== Delete generated pod yamls =="
rm -f "${HERE}/yaml/generated-probe-"*.yaml

echo "== Remaining squash_test resources =="
kubectl get pvc,pod -l squashtest=true 2>&1 || true
kubectl get pvc 2>&1 | grep -i squash || echo "(no squash-test PVCs remain)"

echo "== Done =="
