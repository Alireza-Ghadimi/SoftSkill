#!/usr/bin/env bash
# Recreate the environment used for this reproduction (as run on 2026-10-05).
# usage: SOFTSKILL_ROOT=/path/to/storage scripts/setup_env.sh [--patches-only]
#   --patches-only  init submodules and apply patches, then stop (no venvs, no Docker image)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${SOFTSKILL_ROOT:?set SOFTSKILL_ROOT to your storage directory (datasets, checkpoints, results, venvs)}"
mkdir -p "${ROOT}"/{data,checkpoints,results/logs}
git -C "${REPO}" submodule update --init
# Apply our patches to the pinned upstream checkout (idempotent), in this (glob) order:
#   pooled_init.patch                 init_strategy text_max_pool / text_mean_pool
#   qwen3.patch                       opt-in gradient checkpointing (single-round + agentic)
#   spreadsheet_docker_sandbox.patch  opt-in Docker sandbox for SpreadsheetBench generated code
#   window_pooled_init.patch          init_strategy text_mean_pool_window (applied after pooled_init)
# Later patches edit lines touched by earlier ones, so a re-run is detected with a stamp (checksum of
# the patch set) kept in the submodule's git dir rather than by reverse-checking each patch.
UP="${REPO}/third_party/SoftSkill"
STAMP="$(git -C "${UP}" rev-parse --absolute-git-dir)/softskill_patches.sha256"
SUM="$(cat "${REPO}"/patches/*.patch | sha256sum | cut -d' ' -f1)"
if [[ -f "${STAMP}" && "$(cat "${STAMP}")" == "${SUM}" ]] && ! git -C "${UP}" diff --quiet; then
  echo "patches already applied to third_party/SoftSkill"
else
  for patch in "${REPO}"/patches/*.patch; do
    git -C "${UP}" apply --check "${patch}" 2>/dev/null || {
      echo "cannot apply ${patch}; reset the submodule with: git -C third_party/SoftSkill checkout -- ." >&2; exit 1; }
    git -C "${UP}" apply "${patch}" && echo "applied $(basename "${patch}")"
  done
  echo "${SUM}" > "${STAMP}"
fi
"${REPO}/scripts/toolqa/setup_skim.sh"   # patches/skim/toolqa.patch -> third_party/SKIM
[[ "${1:-}" == "--patches-only" ]] && { echo "patches applied; stopping (--patches-only)"; exit 0; }
[[ -x "${ROOT}/venv/bin/python" ]] || virtualenv "${ROOT}/venv"   # python3-venv/ensurepip is not installed on this host
P="${ROOT}/venv/bin/pip"
"$P" install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
"$P" install transformers==5.17.0 accelerate peft datasets huggingface_hub pillow qwen-vl-utils pytest tqdm flash-linear-attention ninja packaging
# Fast Qwen3.5 linear-attention conv kernel (only matters for the Qwen3.5-4B sanity check).
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST=8.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE "$P" install --no-build-isolation causal-conv1d
"$P" install -e "${REPO}/third_party/SoftSkill"
# Agentic tasks: ALFWorld (fast-downward's build calls a bare `python`, so the venv must be on PATH).
PATH="${ROOT}/venv/bin:${PATH}" "$P" install "alfworld>=0.4.0" "gymnasium>=0.29.0"
# Separate vLLM venv for rollouts / prompt-embeds serving (vLLM pins its own torch).
[[ -x "${ROOT}/venv_vllm/bin/python" ]] || virtualenv "${ROOT}/venv_vllm"
PATH="${ROOT}/venv_vllm/bin:${PATH}" "${ROOT}/venv_vllm/bin/pip" install vllm==0.31.0
"${ROOT}/venv_vllm/bin/pip" install --no-deps -e "${REPO}/third_party/SoftSkill"
# SpreadsheetBench code sandbox image (enable with SOFTSKILL_SHEET_SANDBOX_IMAGE=softskill-sheet-sandbox:1).
docker build -t softskill-sheet-sandbox:1 "${REPO}/docker/spreadsheet_sandbox"
