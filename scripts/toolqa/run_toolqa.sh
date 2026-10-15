#!/usr/bin/env bash
# Run SKIM's ToolQA runner on the base Qwen3-8B (no SKIM checkpoint, no LoRA).
# Usage: run_toolqa.sh <gpu> <split: pilot|full|smoke> <skill_mode> <tag> [extra runner args...]
#   skill_mode=naive runs method=naive; anything else runs method=golden_skill.
set -euo pipefail
GPU="$1"; SPLIT="$2"; MODE="$3"; TAG="$4"; shift 4
[[ "${GPU}" == "0" ]] && { echo "refusing GPU 0" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT=/data/a84460786/softskill_qwen3/toolqa
export TMPDIR="${ROOT}/tmp" HF_HOME="${ROOT}/data/hf_home" TOOLQA_EMBED_CACHE_DIR="${ROOT}/data/embed_cache"
export CUDA_VISIBLE_DEVICES="${GPU}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/scripts/toolqa${PYTHONPATH:+:${PYTHONPATH}}"  # softskill_vectors.py
PY=/data/a84460786/softskill_qwen3/venv/bin/python
SRA="${ROOT}/data/SR-Agents"
mkdir -p "${ROOT}/data/no_checkpoint" "${ROOT}/results/${SPLIT}" "${ROOT}/logs"

case "${SPLIT}" in
  full)  INST="${SRA}/data/bench/instances" ;;
  pilot) INST="${ROOT}/data/instances_pilot" ;;
  smoke) INST="${ROOT}/data/instances_smoke" ;;
  *) echo "bad split ${SPLIT}" >&2; exit 1 ;;
esac
METHOD=golden_skill; [[ "${MODE}" == "naive" ]] && METHOD=naive

OUT="${ROOT}/results/${SPLIT}/${TAG}.jsonl"
LOG="${ROOT}/logs/${SPLIT}_${TAG}.log"
echo "GPU=${GPU} split=${SPLIT} mode=${MODE} method=${METHOD} -> ${OUT}" | tee -a "${LOG}"
cd "${REPO}/third_party/SKIM"
"${PY}" skill/run_skill_compiler_inference.py \
  --checkpoint "${ROOT}/data/no_checkpoint" \
  --llm_model /data/models/huggingface/qwen3-8b --compressor_model none \
  --dataset toolqa --method "${METHOD}" --skill_mode "${MODE}" \
  --skillrag_root "${SRA}" --instances_dir "${INST}" \
  --corpus_path "${SRA}/data/bench/corpus/corpus.json" \
  --toolqa_data_dir "${ROOT}/data/ext_unzip/external_corpus" \
  --toolqa_stop_at_observation --disable_parallel --resume \
  --result_path "${OUT}" "$@" >> "${LOG}" 2>&1
echo "done $(date -u)" >> "${LOG}"
