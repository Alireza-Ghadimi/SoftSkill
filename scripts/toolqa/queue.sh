#!/usr/bin/env bash
# Run several run_toolqa.sh jobs back to back on one GPU.
# Usage: queue.sh <gpu> "<split> <mode> <tag> [args...]" ["..."]
set -uo pipefail
GPU="$1"; shift
for job in "$@"; do
  # shellcheck disable=SC2086
  "$(dirname "$0")/run_toolqa.sh" "${GPU}" ${job} || echo "job failed: ${job}" >&2
done
