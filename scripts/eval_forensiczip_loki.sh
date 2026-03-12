#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH_7B="${MODEL_PATH_7B:-}"
LOKI_JSON_DIR="${LOKI_JSON_DIR:-}"
LOKI_MEDIA_ROOT="${LOKI_MEDIA_ROOT:-}"
LOKI_VIDEO_FRAMES="${LOKI_VIDEO_FRAMES:-4}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
WORKERS="${WORKERS:-16}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
RETENTION_RATIOS_STR="${RETENTION_RATIOS_STR:-0.50 0.35 0.30 0.25 0.15 0.125 0.10}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EFFICIENCY_PROFILE_BATCHES="${EFFICIENCY_PROFILE_BATCHES:-1}"
PROFILE_BATCHES="${PROFILE_BATCHES:-0}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
PROCESSOR_REVISION="${PROCESSOR_REVISION:-a272c74}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/forensiczip/loki}"

FORENSICZIP_SELECT_LAYER="${FORENSICZIP_SELECT_LAYER:--2}"
FORENSICZIP_BIRTH_COST="${FORENSICZIP_BIRTH_COST:-0.35}"
FORENSICZIP_DEATH_COST="${FORENSICZIP_DEATH_COST:-0.35}"
FORENSICZIP_SINKHORN_EPS="${FORENSICZIP_SINKHORN_EPS:-0.1}"
FORENSICZIP_SINKHORN_ITERS="${FORENSICZIP_SINKHORN_ITERS:-20}"
FORENSICZIP_EMA_BETA="${FORENSICZIP_EMA_BETA:-0.6}"
FORENSICZIP_BIRTH_WEIGHT="${FORENSICZIP_BIRTH_WEIGHT:-0.75}"
FORENSICZIP_POS_LAMBDA="${FORENSICZIP_POS_LAMBDA:-0.0}"
FORENSICZIP_FORENSIC_ETA="${FORENSICZIP_FORENSIC_ETA:-0.0}"

require_path() {
  local value="$1"
  local name="$2"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] Missing required variable: ${name}" >&2
    exit 1
  fi
}

require_path "${MODEL_PATH_7B}" MODEL_PATH_7B
require_path "${LOKI_JSON_DIR}" LOKI_JSON_DIR
require_path "${LOKI_MEDIA_ROOT}" LOKI_MEDIA_ROOT

mkdir -p "${OUTPUT_ROOT}"
read -r -a RETENTION_RATIOS <<< "${RETENTION_RATIOS_STR}"

echo "[ForensicZip/LOKI] model=${MODEL_PATH_7B}"
echo "[ForensicZip/LOKI] output=${OUTPUT_ROOT}"
echo "[ForensicZip/LOKI] ratios=${RETENTION_RATIOS[*]}"

for ratio in "${RETENTION_RATIOS[@]}"; do
  out_dir="${OUTPUT_ROOT}/r${ratio}"
  mkdir -p "${out_dir}"
  processor_args=()
  if [[ -n "${PROCESSOR_PATH}" ]]; then
    processor_args=(--processor_path "${PROCESSOR_PATH}" --processor_revision "${PROCESSOR_REVISION}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" scripts/eval_forensiczip.py \
    --dataset_type loki \
    --model_path "${MODEL_PATH_7B}" \
    --val_batch_size "${VAL_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --output_path "${out_dir}" \
    --test_json_file "${LOKI_JSON_DIR}" \
    --data_base_test "${LOKI_MEDIA_ROOT}" \
    --loki_media_root "${LOKI_MEDIA_ROOT}" \
    --video_num_frames "${LOKI_VIDEO_FRAMES}" \
    --exp_name "forensiczip_loki_r${ratio}" \
    "${processor_args[@]}" \
    --forensiczip_retention "${ratio}" \
    --forensiczip_select_layer "${FORENSICZIP_SELECT_LAYER}" \
    --forensiczip_birth_cost "${FORENSICZIP_BIRTH_COST}" \
    --forensiczip_death_cost "${FORENSICZIP_DEATH_COST}" \
    --forensiczip_sinkhorn_eps "${FORENSICZIP_SINKHORN_EPS}" \
    --forensiczip_sinkhorn_iters "${FORENSICZIP_SINKHORN_ITERS}" \
    --forensiczip_ema_beta "${FORENSICZIP_EMA_BETA}" \
    --forensiczip_birth_weight "${FORENSICZIP_BIRTH_WEIGHT}" \
    --forensiczip_pos_lambda "${FORENSICZIP_POS_LAMBDA}" \
    --forensiczip_forensic_eta "${FORENSICZIP_FORENSIC_ETA}" \
    --max_length "${MAX_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --efficiency_profile_batches "${EFFICIENCY_PROFILE_BATCHES}" \
    --profile_batches "${PROFILE_BATCHES}"
done
