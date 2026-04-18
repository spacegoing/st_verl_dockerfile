#!/usr/bin/env bash
# bisimpo/launch_phase1.sh — submit the 3 Phase-1 ablation RayJobs.
#
# Runs: cbsp101 (Obj 1 simplest), cbsp103 (Obj 3 w_penalty), cbsp104 (Obj 4 w_penalty_only).
# All at default hparams (delta=3e-4, lambda=1e-3), 120 training steps, 4-node C11.
# Expected wall clock per run: ~18 h. Three run concurrently on 12 of ~28 free nodes.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for combo in cbsp101 cbsp103 cbsp104; do
    echo "=== launching $combo ==="
    NNODES=4 "$HERE/submit_bspo.sh" "$combo" 2>&1 | grep -E 'name=|variant=' | head -2
    sleep 3
done

echo
KC() { HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl "$@"; }
echo "=== active Phase-1 jobs ==="
KC get rayjob -l training-type=40bra-sd --no-headers 2>&1 | awk '$1 ~ /cbsp10/ && $2!="SUCCEEDED" && $2!="FAILED"'
