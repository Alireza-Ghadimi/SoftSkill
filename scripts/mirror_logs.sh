#!/usr/bin/env bash
# Mirror compact run logs (tqdm progress stripped) + the aggregate summary into the repo's logs/.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${SOFTSKILL_ROOT:?set SOFTSKILL_ROOT to your storage directory (datasets, checkpoints, results, venvs)}"
mkdir -p "${REPO}/logs/runs"
for f in "${ROOT}"/results/logs/*.log; do
  tr '\r' '\n' < "$f" | grep -a -v -E '^\s*$|it/s\]|ex/s\]|batch/s\]|s/batch\]|s/ex\]|s/it\]|epoch/s\]|Loading weights|\|[ #█▏▎▍▌▋▊▉]*\| *[0-9]+/[0-9]+' > "${REPO}/logs/runs/$(basename "$f")"
done
cp -f "${ROOT}"/results/logs/launch_*.out "${REPO}/logs/" 2>/dev/null || true
"${ROOT}/venv/bin/python" "${REPO}/scripts/summarize.py" --root "${ROOT}" --out_dir "${REPO}/logs" > /dev/null
