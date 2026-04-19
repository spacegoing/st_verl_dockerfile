#!/usr/bin/env bash
# bisimpo/submit_formal_md.sh — launch all 5 formal cbmd-v{1..5} runs.
#
# Per plan:
#   3 × 8-node  : cbmd-v1 (simplest), cbmd-v2 (hierarchy), cbmd-v3 (strict_wasserstein)
#   1 × 4-node  : cbmd-v4 (w_penalty)
#   cbmd-v5     : queue via submit; volcano gang scheduling waits for 4-node slot to open
#
# Run ONLY from the production nemo_bspo_md branch (after squash_to_prod.sh).
# Total: 3×8 + 1×4 + 1×4 (queued for later) = 28 + 4 (queued) nodes.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "Submitting formal cbmd-v{1..5} with λ_Tj=1e-2, λ_Grp=1e-2, λ_D=1.0"
echo "════════════════════════════════════════════════════════════════"

# 8-node runs (V1, V2, V3)
for v in v1 v2 v3; do
    STAGE=3 NNODES=8 "$HERE/submit_bspo_md.sh" "cbmd-${v}" | tail -3
    sleep 3
done

# 4-node run (V4)
STAGE=3 NNODES=4 "$HERE/submit_bspo_md.sh" cbmd-v4 | tail -3
sleep 3

# V5 — submit as 4-node; volcano will gang-wait for 4 nodes to free
echo
echo "Submitting cbmd-v5 (w_penalty_only, 4-node — will queue via Volcano)"
STAGE=3 NNODES=4 "$HERE/submit_bspo_md.sh" cbmd-v5 | tail -3

echo
echo "════════════════════════════════════════════════════════════════"
echo "All 5 formal runs submitted. Check status:"
echo "  kubectl get rayjob -l training-type=40bra-md"
echo "════════════════════════════════════════════════════════════════"
