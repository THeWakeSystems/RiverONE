#!/usr/bin/env bash
set -euo pipefail

PV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${PV_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/weights/miniViT_v2_distilled}"
DATA_DIR="${DATA_DIR:-${PV_DIR}/QcalEval}"
OUTPUT_DIR="${OUTPUT_DIR:-${PV_DIR}/outputs_v2/pv_tuned_qcaleval}"
DEVICE="${DEVICE:-cuda:0}"

ARGS=()
if [[ "${ALLOW_MISSING_IMAGES:-0}" == "1" ]]; then
  ARGS+=(--allow_missing_images)
fi
if [[ "${UPDATE_NON_QUANTIZED:-0}" == "1" ]]; then
  ARGS+=(--update_non_quantized_parameters)
fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  ARGS+=(--gradient_checkpointing)
fi

python3 "${PV_DIR}/validate_qcaleval.py" \
  --data_dir "${DATA_DIR}" \
  "${ARGS[@]/--update_non_quantized_parameters/}" \
  "${ARGS[@]/--gradient_checkpointing/}"

python3 "${PV_DIR}/train_pv_tuning.py" \
  --model_dir "${MODEL_DIR}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --max_length "${MAX_LENGTH:-4096}" \
  --lr "${LR:-3e-4}" \
  --code_lr "${CODE_LR:-1e-2}" \
  --max_code_change_per_step "${MAX_CODE_CHANGE_PER_STEP:-1e-3}" \
  --beam_size "${BEAM_SIZE:-1}" \
  --log_every_steps "${LOG_EVERY_STEPS:-1}" \
  --save_every_steps "${SAVE_EVERY_STEPS:-0}" \
  "${ARGS[@]}"
