#!/usr/bin/env bash
# Launch N voltest RayJobs (default 10) by substituting __JOBNAME__ in the template.
# Records submission output to voltest/logs/submit.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPL="${HERE}/yaml/voltest-rayjob-template.yaml"
LOG="${HERE}/logs/submit.log"
N="${N:-10}"
START="${START:-0}"

: > "$LOG"
echo "[$(date -u +%FT%TZ)] Launching ${N} RayJobs (voltest-${START}..voltest-$((START+N-1)))" | tee -a "$LOG"

for i in $(seq "$START" $((START + N - 1))); do
    JOB="voltest-${i}"
    YAML_OUT="${HERE}/yaml/generated-${JOB}.yaml"
    sed "s/__JOBNAME__/${JOB}/g" "$TPL" > "$YAML_OUT"
    echo "[$(date -u +%FT%TZ)] Submitting ${JOB} from ${YAML_OUT}" | tee -a "$LOG"
    HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 \
        kubectl apply -f "$YAML_OUT" 2>&1 | tee -a "$LOG"
done

echo "[$(date -u +%FT%TZ)] All ${N} submissions complete." | tee -a "$LOG"
