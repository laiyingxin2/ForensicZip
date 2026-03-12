#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH_7B="${MODEL_PATH_7B:-}"
FAKECLUE_TEST_JSON="${FAKECLUE_TEST_JSON:-}"
FAKECLUE_DATA_BASE="${FAKECLUE_DATA_BASE:-}"
LOKI_JSON_DIR="${LOKI_JSON_DIR:-}"
LOKI_MEDIA_ROOT="${LOKI_MEDIA_ROOT:-}"
LOKI_VIDEO_FRAMES="${LOKI_VIDEO_FRAMES:-4}"

BATCH_SIZE="${BATCH_SIZE:-}"
WORKERS="${WORKERS:-16}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
DATASETS="${DATASETS:-fakeclue loki}"
RETENTION_RATIOS_STR="${RETENTION_RATIOS_STR:-0.50 0.35 0.30 0.25 0.15 0.125 0.10}"
EFFICIENCY_PROFILE_BATCHES="${EFFICIENCY_PROFILE_BATCHES:-1}"
PROFILE_BATCHES="${PROFILE_BATCHES:-0}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
PROCESSOR_REVISION="${PROCESSOR_REVISION:-main}"

FORENSICZIP_SELECT_LAYER="${FORENSICZIP_SELECT_LAYER:--2}"
FORENSICZIP_BIRTH_COST="${FORENSICZIP_BIRTH_COST:-0.35}"
FORENSICZIP_DEATH_COST="${FORENSICZIP_DEATH_COST:-0.35}"
FORENSICZIP_SINKHORN_EPS="${FORENSICZIP_SINKHORN_EPS:-0.1}"
FORENSICZIP_SINKHORN_ITERS="${FORENSICZIP_SINKHORN_ITERS:-20}"
FORENSICZIP_EMA_BETA="${FORENSICZIP_EMA_BETA:-0.6}"
FORENSICZIP_BIRTH_WEIGHT="${FORENSICZIP_BIRTH_WEIGHT:-0.75}"
FORENSICZIP_POS_LAMBDA="${FORENSICZIP_POS_LAMBDA:-0.0}"
FORENSICZIP_FORENSIC_ETA="${FORENSICZIP_FORENSIC_ETA:-0.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/forensiczip}"
mkdir -p "${OUTPUT_ROOT}"

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a RETENTION_RATIOS <<< "${RETENTION_RATIOS_STR}"

require_path() {
  local value="$1"
  local name="$2"
  if [[ -z "$value" ]]; then
    echo "[ERROR] Missing required variable: ${name}" >&2
    exit 1
  fi
}

require_path "${MODEL_PATH_7B}" MODEL_PATH_7B

echo "[ForensicZip] CUDA_DEVICES=${CUDA_DEVICES}"
echo "[ForensicZip] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[ForensicZip] datasets=${DATASET_LIST[*]}"
echo "[ForensicZip] retention_ratios=${RETENTION_RATIOS[*]}"
echo "[ForensicZip] batch_size=${BATCH_SIZE} workers=${WORKERS}"
echo "[ForensicZip] max_length=${MAX_LENGTH} max_new_tokens=${MAX_NEW_TOKENS}"
echo "[ForensicZip] select_layer=${FORENSICZIP_SELECT_LAYER} eps=${FORENSICZIP_SINKHORN_EPS} iters=${FORENSICZIP_SINKHORN_ITERS} ema=${FORENSICZIP_EMA_BETA} eta=${FORENSICZIP_FORENSIC_ETA}"

run_block() {
  local model_name="$1"
  local model_path="$2"
  local batch_size="$3"

  if [[ -z "$model_path" || ! -e "$model_path" ]]; then
    echo "[SKIP] ${model_name} model path not found: ${model_path}"
    return
  fi

  for dataset in "${DATASET_LIST[@]}"; do
    local test_json="${FAKECLUE_TEST_JSON}"
    local data_base="${FAKECLUE_DATA_BASE}"
    local extra_args=()
    if [[ "${dataset}" == "loki" ]]; then
      require_path "${LOKI_JSON_DIR}" LOKI_JSON_DIR
      require_path "${LOKI_MEDIA_ROOT}" LOKI_MEDIA_ROOT
      test_json="${LOKI_JSON_DIR}"
      extra_args=(--dataset_type loki --loki_media_root "${LOKI_MEDIA_ROOT}" --video_num_frames "${LOKI_VIDEO_FRAMES}")
    else
      require_path "${FAKECLUE_TEST_JSON}" FAKECLUE_TEST_JSON
      require_path "${FAKECLUE_DATA_BASE}" FAKECLUE_DATA_BASE
      extra_args=(--dataset_type fakeclue)
    fi

    declare -A seen=()
    for ratio in "${RETENTION_RATIOS[@]}"; do
      seen["$ratio"]=$(( ${seen["$ratio"]:-0} + 1 ))
      suffix=""
      if (( ${seen["$ratio"]} > 1 )); then
        suffix="_rep${seen["$ratio"]}"
      fi
      ratio_tag="${ratio}"
      out_dir="${OUTPUT_ROOT}/${dataset}/${model_name}/r${ratio_tag}${suffix}"
      mkdir -p "${out_dir}"

      processor_args=()
      if [[ -n "${PROCESSOR_PATH}" ]]; then
        processor_args+=(--processor_path "${PROCESSOR_PATH}" --processor_revision "${PROCESSOR_REVISION}")
      fi

      CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" scripts/eval_forensiczip.py             --model_path "${model_path}"             --val_batch_size "${batch_size}"             --workers "${WORKERS}"             --output_path "${out_dir}"             --test_json_file "${test_json}"             --data_base_test "${data_base}"             --exp_name "fakevlm_forensiczip_${dataset}_${model_name}_r${ratio_tag}${suffix}"             "${processor_args[@]}"             --forensiczip_retention "${ratio}"             --forensiczip_select_layer "${FORENSICZIP_SELECT_LAYER}"             --forensiczip_birth_cost "${FORENSICZIP_BIRTH_COST}"             --forensiczip_death_cost "${FORENSICZIP_DEATH_COST}"             --forensiczip_sinkhorn_eps "${FORENSICZIP_SINKHORN_EPS}"             --forensiczip_sinkhorn_iters "${FORENSICZIP_SINKHORN_ITERS}"             --forensiczip_ema_beta "${FORENSICZIP_EMA_BETA}"             --forensiczip_birth_weight "${FORENSICZIP_BIRTH_WEIGHT}"             --forensiczip_pos_lambda "${FORENSICZIP_POS_LAMBDA}"             --forensiczip_forensic_eta "${FORENSICZIP_FORENSIC_ETA}"             --max_length "${MAX_LENGTH}"             --max_new_tokens "${MAX_NEW_TOKENS}"             "${extra_args[@]}"             --efficiency_profile_batches "${EFFICIENCY_PROFILE_BATCHES}"             --profile_batches "${PROFILE_BATCHES}"
    done
  done
}

run_block "7b" "${MODEL_PATH_7B}" "${BATCH_SIZE}"
