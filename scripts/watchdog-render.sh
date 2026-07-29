#!/bin/bash
# Resilient render: run mvgen render + assemble, restarting ComfyUI and
# resuming whenever progress stalls (no state.json update for STALL_SECS).
# Usage: watchdog-render.sh <manifest> <workdir> <track> <out.mp4>
set -u
MANIFEST=$1; WORK=$2; TRACK=$3; OUT=$4
STALL_SECS=900
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
LOG_DIR="$WORK"; mkdir -p "$LOG_DIR"

comfy_up() {
  curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8188/system_stats 2>/dev/null | grep -q 200
}

restart_comfy() {
  echo "[watchdog] restarting comfyui"
  pkill -9 -f 'main.py --listen 127.0.0.1 --port 8188' 2>/dev/null
  sleep 3
  nohup ~/programs/comfyui/start.sh > "$LOG_DIR/comfyui-watchdog.log" 2>&1 &
  for i in $(seq 1 45); do comfy_up && return 0; sleep 2; done
  echo "[watchdog] comfyui failed to come up"; return 1
}

attempt=0
while [ $attempt -lt 8 ]; do
  attempt=$((attempt+1))
  comfy_up || restart_comfy || exit 1
  echo "[watchdog] render attempt $attempt"
  "$PY" -m mvgen.render "$MANIFEST" "$WORK" > "$LOG_DIR/render-$attempt.log" 2>&1 &
  RPID=$!
  while kill -0 $RPID 2>/dev/null; do
    sleep 60
    if [ -f "$WORK/state.json" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$WORK/state.json") ))
      if [ $age -gt $STALL_SECS ]; then
        echo "[watchdog] stalled (${age}s since progress) — killing and restarting"
        kill -9 $RPID 2>/dev/null
        restart_comfy || exit 1
        break
      fi
    fi
  done
  if wait $RPID 2>/dev/null; then
    if "$PY" -m mvgen.assemble "$MANIFEST" "$WORK" "$OUT"; then
      echo "WATCHDOG_COMPLETE"; exit 0
    fi
  fi
done
echo "[watchdog] giving up after $attempt attempts"; exit 1
