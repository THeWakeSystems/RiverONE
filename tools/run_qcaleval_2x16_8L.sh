#!/bin/bash
# QCalEval inference — AQLM 2x16 8L MLP-only on 4 GPUs
# Q1-Q4 parallel, Q5-Q6 after (sequential on same GPUs)

set -euo pipefail

REPORTS_TAG="qcaleval_aqlm_2x16_8L_mlp_only"
MODEL_PATH="/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly"

SCRIPTS_DIR="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/scripts"
DATA_ROOT="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/Data"
REPORTS_ROOT="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/reports/${REPORTS_TAG}"

export LD_PRELOAD="$HOME/.local/cuda-compat/libnvrtc.so.13.0:$HOME/.local/cuda-compat/libnvrtc-builtins.so.13.0"

RUNNER="${SCRIPTS_DIR}/run_qcaleval_aqlm.py"

echo "=== Phase 1: Q1-Q4 on 4 GPUs ==="

# Q1 on GPU 0
python3 "$RUNNER" --question q1 --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --reports-root "$REPORTS_ROOT" \
  --device cuda:0 > "${REPORTS_ROOT}/Q1/run.log" 2>&1 &

# Q2 on GPU 1
python3 "$RUNNER" --question q2 --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --reports-root "$REPORTS_ROOT" \
  --device cuda:1 > "${REPORTS_ROOT}/Q2/run.log" 2>&1 &

# Q3 on GPU 2
python3 "$RUNNER" --question q3 --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --reports-root "$REPORTS_ROOT" \
  --device cuda:2 > "${REPORTS_ROOT}/Q3/run.log" 2>&1 &

# Q4 on GPU 3
python3 "$RUNNER" --question q4 --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --reports-root "$REPORTS_ROOT" \
  --device cuda:3 > "${REPORTS_ROOT}/Q4/run.log" 2>&1 &

echo "Phase 1 launched. Waiting for Q1-Q4 to finish..."
wait
echo "Phase 1 done."

echo ""
echo "=== Phase 2: Q5-Q6 on 2 GPUs ==="

# Q5 on GPU 0
python3 "$RUNNER" --question q5 --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --reports-root "$REPORTS_ROOT" \
  --device cuda:0 > "${REPORTS_ROOT}/Q5/run.log" 2>&1 &

# Q6 on GPU 1
python3 "$RUNNER" --question q6 --model-path "$MODEL_PATH" \
  --data-root "$DATA_ROOT" --reports-root "$REPORTS_ROOT" \
  --device cuda:1 > "${REPORTS_ROOT}/Q6/run.log" 2>&1 &

wait
echo "Phase 2 done."
echo "All QCalEval inference complete!"
