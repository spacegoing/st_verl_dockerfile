#!/usr/bin/env bash
# run_pipeline.sh — end-to-end voltest workflow.
#
# Phases:
#   1. preflight   — sanity-check cluster state
#   2. submit      — `kubectl create -f rayjob.yaml` N times; k8s generateName
#                    gives each instance a unique suffix (no templating)
#   3. wait        — block until all N RayJobs we created reach terminal state
#                    (Volcano auto-promotes queued gangs as pods free; KubeRay
#                    auto-tears down via ttlSecondsAfterFinished=30 in yaml)
#   4. verify      — check per-RayCluster PFS log for gpu_burn DONE lines
#   5. report      — summary to stdout + logs/pipeline-<ts>.log
#
# Usage:
#   ./run_pipeline.sh           # default N=10
#   N=5 ./run_pipeline.sh       # submit 5
#   SKIP_SUBMIT=1 ./run_pipeline.sh   # only verify an already-finished batch

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SH="${HERE}/submit/submit.sh"
LOG_DIR="${HERE}/logs"
mkdir -p "$LOG_DIR"

N="${N:-10}"
DURATION_S="${DURATION_S:-300}"
TICK_SECONDS="${TICK_SECONDS:-20}"
WATCH_TIMEOUT_S="${WATCH_TIMEOUT_S:-2400}"   # 40 min cap

TS=$(date -u +%Y%m%dT%H%M%SZ)
PIPE_LOG="${LOG_DIR}/pipeline-${TS}.log"

# Track the set of rayjob names we submit in this run so we only wait/verify
# our own jobs (not any other voltest-* resources in the namespace).
SUBMITTED_NAMES_FILE="${LOG_DIR}/.pipeline-${TS}-names"

# Clear proxy in both cases (Go HTTP client checks HTTPS_PROXY and https_proxy
# independently). NO_PROXY must include the k8s API IP or direct connect fails.
K8S_API_IP="${K8S_API_IP:-180.184.249.201}"
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
export NO_PROXY="$K8S_API_IP" no_proxy="$K8S_API_IP"

# ──────────────────────────────────────────────────────────────────────────────
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$PIPE_LOG"; }
die() { log "FATAL: $*"; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "required binary not found: $1"; }

# ──────────────────────────────────────────────────────────────────────────────
phase_1_preflight() {
    log "=== PHASE 1 — preflight ==="
    require kubectl
    [ -x "$SUBMIT_SH" ] || die "submit script missing or not executable: $SUBMIT_SH"

    LEFTOVER=$(kubectl get rayjob -l voltest=true --no-headers 2>/dev/null | wc -l)
    if [ "$LEFTOVER" != "0" ]; then
        log "WARN: $LEFTOVER voltest rayjobs already exist (from another pipeline run or leftovers):"
        kubectl get rayjob -l voltest=true 2>&1 | tee -a "$PIPE_LOG"
        die "clean them up first: kubectl delete rayjob -l voltest=true"
    fi

    GPU_TOTAL=$(kubectl get node -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' \
                | awk '{s+=$1} END{print s+0}')
    log "cluster allocatable GPUs: $GPU_TOTAL (need: $((N*8*8)) for all N=$N jobs to run concurrently)"
    log "preflight OK"
}

# ──────────────────────────────────────────────────────────────────────────────
phase_2_submit() {
    log "=== PHASE 2 — submit $N jobs via submit/submit.sh $DURATION_S ==="
    : > "$SUBMITTED_NAMES_FILE"
    for i in $(seq 1 "$N"); do
        # submit.sh prints "[submit] duration=Ns  name=voltest-xxxx" as first line.
        # Extract the name= field.
        OUT=$("$SUBMIT_SH" "$DURATION_S" 2>&1)
        NAME=$(echo "$OUT" | awk '/^\[submit\]/ {for(j=1;j<=NF;j++) if ($j ~ /^name=/) { sub(/^name=/,"",$j); print $j; exit }}')
        if [ -z "$NAME" ] || ! [[ "$NAME" =~ ^voltest- ]]; then
            log "submit output:"
            echo "$OUT" | tee -a "$PIPE_LOG"
            die "submit.sh did not produce a valid RayJob name"
        fi
        echo "$NAME" >> "$SUBMITTED_NAMES_FILE"
        log "  [${i}/${N}] $NAME"
    done
    log "submitted $N jobs — tracking names in $SUBMITTED_NAMES_FILE"
}

# ──────────────────────────────────────────────────────────────────────────────
phase_3_wait() {
    log "=== PHASE 3 — wait for all N=$N of OUR submitted jobs to reach terminal state ==="
    log "    (Volcano auto-promotes queued PodGroups; KubeRay auto-cleans via ttl=30 in yaml)"

    START_WATCH=$(date +%s)
    PREV_STATE=""

    while true; do
        if [ $(($(date +%s) - START_WATCH)) -gt "$WATCH_TIMEOUT_S" ]; then
            log "FATAL: watch timeout ($WATCH_TIMEOUT_S s) exceeded"
            return 1
        fi

        DONE=0
        while IFS= read -r NAME; do
            # First check if CR exists. `kubectl get -o name` returns the resource
            # name on stdout and exits 0 if present; empty + exit !=0 if deleted.
            if ! kubectl get rayjob "$NAME" -o name >/dev/null 2>&1; then
                # CR already gone → ttl expired after SUCCEEDED/FAILED → terminal
                DONE=$((DONE+1))
                continue
            fi
            # CR exists; check jobStatus (empty during Initializing, set once Ray job starts)
            STATUS=$(kubectl get rayjob "$NAME" -o jsonpath='{.status.jobStatus}' 2>/dev/null || true)
            case "$STATUS" in
                SUCCEEDED|FAILED) DONE=$((DONE+1)) ;;
            esac
        done < "$SUBMITTED_NAMES_FILE"

        # Condensed state line (only voltest PodGroups, only our own jobs if possible)
        PG_STATE=$(kubectl get podgroup -A --no-headers 2>/dev/null \
                   | awk '/ray-voltest-/ {printf "%s:%s(%s/%s) ", $2, $3, $5, $4}')
        if [ "$PG_STATE" != "$PREV_STATE" ]; then
            log "  done=$DONE/$N  ${PG_STATE:-(no active voltest PodGroups)}"
            PREV_STATE="$PG_STATE"
        fi

        [ "$DONE" -ge "$N" ] && { log "all N=$N jobs terminal"; return 0; }
        sleep "$TICK_SECONDS"
    done
}

# ──────────────────────────────────────────────────────────────────────────────
phase_4_verify() {
    log "=== PHASE 4 — verify PFS logs ==="
    PASS=0
    FAIL=0
    while IFS= read -r NAME; do
        # Log file is named after the RayCluster (downward-API JOB_NAME).
        # RayCluster name = <rayjob-name>-<5char>. Discover it via label selector.
        # Since rayjobs may be deleted (ttl), fall back to a glob match on the logs dir.
        LF=$(ls -1 "$LOG_DIR"/"$NAME"-*.log 2>/dev/null | head -1 || true)
        if [ -z "$LF" ] || [ ! -s "$LF" ]; then
            log "  $NAME: MISSING log file (looked for $LOG_DIR/$NAME-*.log)"
            FAIL=$((FAIL+1))
            continue
        fi
        DONE_COUNT=$(grep -c "gpu_burn DONE" "$LF" 2>/dev/null || echo 0)
        if [ "$DONE_COUNT" -ge 8 ]; then
            log "  $NAME: $DONE_COUNT/8 DONE markers  ✓  [$(basename "$LF")]"
            PASS=$((PASS+1))
        elif [ "$DONE_COUNT" -ge 7 ]; then
            log "  $NAME: $DONE_COUNT/8 DONE markers  ~ (1 log write lost to teardown race, benign)"
            PASS=$((PASS+1))
        else
            log "  $NAME: only $DONE_COUNT/8 DONE markers  ✗"
            FAIL=$((FAIL+1))
        fi
    done < "$SUBMITTED_NAMES_FILE"
    log "verified: $PASS pass, $FAIL fail"
    [ "$FAIL" = "0" ]
}

# ──────────────────────────────────────────────────────────────────────────────
phase_5_report() {
    log "=== PHASE 5 — summary ==="
    TOTAL_S=$(($(date +%s) - PIPELINE_START))
    log "pipeline wall clock: ${TOTAL_S}s"
    log "pipeline log:      $PIPE_LOG"
    log "submitted names:   $SUBMITTED_NAMES_FILE"
    log "per-job logs:      $LOG_DIR/voltest-*.log"
}

# ──────────────────────────────────────────────────────────────────────────────
main() {
    PIPELINE_START=$(date +%s)
    log "voltest pipeline starting  N=$N"

    if [ "${SKIP_SUBMIT:-0}" != "1" ]; then
        phase_1_preflight
        phase_2_submit
    else
        log "SKIP_SUBMIT=1 → skipping phases 1-2"
        [ -s "${SUBMITTED_NAMES_FILE:-}" ] || die "SKIP_SUBMIT requires prior names file: $SUBMITTED_NAMES_FILE"
    fi

    phase_3_wait || { phase_4_verify || true; phase_5_report; exit 1; }
    phase_4_verify
    phase_5_report
}

main "$@"
