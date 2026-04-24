#!/usr/bin/env bash
# Start the queue manager in tmux if GPU0 utilization is below threshold.
# Usage: run from cron. Outputs nothing on success.
set -euo pipefail

THRESH=20 # percent GPU utilization threshold to consider "idle"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# echo "Script directory: $SCRIPT_DIR" 
CHECK="$SCRIPT_DIR/print_vram_usage.sh"

if [ ! -r "$CHECK" ]; then
  echo "missing $CHECK" >&2
  exit 1
fi

# call the checker script (it should emit a single integer)
# capture only stdout and normalize whitespace/carriage returns
percent=$(bash "$CHECK" 2>/dev/null || true)
percent=$(printf '%s' "$percent" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

if [ -z "$percent" ]; then
  echo "error: $CHECK returned no output" >&2
  exit 1
fi

# ensure it's an integer
if ! printf '%s' "$percent" | grep -Eq '^[0-9]+$'; then
  echo "error: $CHECK returned non-integer output" >&2
  exit 1
fi
echo "GPU0 utilization: $percent%"

if [ "$percent" -lt "$THRESH" ]; then
  # start/send keys (exact command provided)
  tmux new-session -d -s queue 2>/dev/null || true
  tmux send-keys -t queue "bash" C-m
  tmux send-keys -t queue "$SCRIPT_DIR/../.venv/bin/python3 $SCRIPT_DIR/queue_manager.py" C-m
fi

exit 0
