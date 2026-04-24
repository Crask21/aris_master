#!/usr/bin/env bash
# Print only the integer GPU0 VRAM usage percentage (stdout contains only the number).
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found." >&2
  exit 1
fi

# Query GPU0 utilization in percent (no header, no units)
out=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null || true)
if [ -z "${out}" ]; then
  exit 1
fi

# Trim non-numeric characters (spaces, percent signs) and get first token
val=$(echo "$out" | tr -d ' %' | awk '{gsub(/^ +| +$/,"",$1); print $1}')
if [ -z "${val}" ]; then
  exit 1
fi

# Round to nearest integer and print
perc=$(awk -v v="$val" 'BEGIN{printf "%.0f", v+0}')
printf "%s\n" "$perc"
