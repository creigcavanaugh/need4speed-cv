#!/usr/bin/env bash
# Wrapper that restarts car_speed_tracker_polygon.py if it exits or is killed.
# Usage: ./run_tracker.sh
# Stop: Ctrl+C (sends SIGINT to this script, which propagates to the tracker)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER="$SCRIPT_DIR/car_speed_tracker_polygon.py"
PYTHON="${SCRIPT_DIR}/venv/bin/python"
RETRY_DELAY=5

trap 'echo "[wrapper] Caught interrupt, stopping."; kill "$CHILD_PID" 2>/dev/null; exit 0' INT TERM

echo "[wrapper] Starting tracker. Ctrl+C to stop."

while true; do
    "$PYTHON" "$TRACKER" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 || $EXIT_CODE -eq 130 ]]; then
        # Clean exit (0) or Ctrl+C in child (130) — don't restart
        echo "[wrapper] Tracker exited cleanly."
        break
    fi

    echo "[wrapper] Tracker exited with code $EXIT_CODE at $(date '+%Y-%m-%d %H:%M:%S') — restarting in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done
