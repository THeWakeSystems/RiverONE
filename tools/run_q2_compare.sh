#!/bin/bash
# QCalEval Q2 comparison across 4 models
set -euo pipefail

SCRIPTS_DIR="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/scripts"
DATA_ROOT="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/Data"
AQLM_ROOT="/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM"

declare -A MODELS
MODELS[baseline]="$AQLM_ROOT/RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly"
MODELS[shared]="$AQLM_ROOT/RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly-shared"
MODELS[nredo5]="$AQLM_ROOT/RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly-nredo5"
MODELS[mixed12L]="$AQLM_ROOT/RiverOne-QC-4B-v2-AQLM-2x16-12L-MLPonly-mixed"

export LD_PRELOAD="$HOME/.local/cuda-compat/libnvrtc.so.13.0:$HOME/.local/cuda-compat/libnvrtc-builtins.so.13.0"

for tag in baseline shared nredo5 mixed12L; do
    MODEL_PATH="${MODELS[$tag]}"
    REPORTS="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/reports/qcaleval_${tag}/Q2"
    mkdir -p "$REPORTS"
    echo "=== Q2: $tag ==="
    python3 "$SCRIPTS_DIR/run_qcaleval_aqlm.py" \
        --question q2 \
        --model-path "$MODEL_PATH" \
        --data-root "$DATA_ROOT" \
        --reports-root "/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/reports/qcaleval_${tag}" \
        --device cuda 2>&1 | tail -5
    echo ""
done
echo "All Q2 done!"
