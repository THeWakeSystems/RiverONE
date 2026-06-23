#!/usr/bin/env bash
# ============================================================================
# PV-Tuning 完整训练流程 —— 后台运行 & 实时监控命令
# GPU 分配: 单卡 → GPU7, 多卡 → GPU7,6,5,4
# ============================================================================
#
#  使用方法:
#    bash pv_tuning_workflow.sh <STEP>
#
#  STEP:
#    check      环境检查
#    data       数据准备
#    dryrun     冒烟测试 (GPU7, 10步)
#    train      正式训练 (GPU7, 后台)
#    eval       训练后评估
#    all        一键执行 check → data → dryrun → train → eval
# ============================================================================
set -euo pipefail

PV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${PV_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/weights/miniViT_distilled"
DATA_DIR="${PV_DIR}/QcalEval"
OUTPUT_DIR="${PV_DIR}/outputs/pv_tuned_qcaleval"
LOG_DIR="${PV_DIR}/logs"
RUN_NAME="pv_tune_$(date +%Y%m%d_%H%M%S)"

# ── GPU 配置 ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=7          # 单卡用 GPU7
MULTI_GPU="7,6,5,4"                    # 多卡顺序

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: 环境检查
# ══════════════════════════════════════════════════════════════════════════════
step_check() {
    echo "============================================"
    echo " Step 1/5: 环境检查"
    echo "============================================"

    echo ""
    echo "→ Python: $(python3 --version)"
    echo "→ PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "→ CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
    echo "→ GPU 7 名称: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
    echo "→ GPU 7 显存: $(python3 -c 'import torch; p=torch.cuda.get_device_properties(0); print(f\"{p.total_memory/1e9:.1f} GB\")' 2>/dev/null || echo 'N/A')"
    echo ""

    # 检查关键包
    for pkg in torch transformers safetensors tqdm pillow; do
        if python3 -c "import ${pkg}" 2>/dev/null; then
            echo "  ✓ ${pkg}"
        else
            echo "  ✗ ${pkg} — 请运行: pip install -r requirements.txt"
        fi
    done

    # 检查 AQLM 库
    if python3 -c "import sys; sys.path.insert(0,'${PROJECT_DIR}/engine'); from src.utils import _dequantize_weight" 2>/dev/null; then
        echo "  ✓ AQLM lib (dequantize)"
    else
        echo "  ✗ AQLM lib 不可用"
    fi

    # 检查模型目录
    if [[ -f "${MODEL_DIR}/model.safetensors" ]]; then
        MODEL_MB=$(du -m "${MODEL_DIR}/model.safetensors" | cut -f1)
        echo "  ✓ 模型文件: ${MODEL_MB} MB"
    else
        echo "  ✗ 模型文件缺失: ${MODEL_DIR}"
    fi

    echo ""
    echo "→ 当前可用 GPU:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null | head -8
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: 数据准备
# ══════════════════════════════════════════════════════════════════════════════
step_data() {
    echo "============================================"
    echo " Step 2/5: 数据准备与校验"
    echo "============================================"
    cd "${PV_DIR}"
    python3 validate_qcaleval.py --allow_missing_images

    echo ""
    echo "→ 如需正式训练，请先将图片放入:"
    echo "    ${DATA_DIR}/images/"
    echo "→ 当前冒烟测试使用灰色占位图 (--allow_missing_images)"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: 冒烟测试 (GPU7, 10步)
# ══════════════════════════════════════════════════════════════════════════════
step_dryrun() {
    local LOG_FILE="${LOG_DIR}/${RUN_NAME}_dryrun.log"

    echo "============================================"
    echo " Step 3/5: 冒烟测试 (GPU7, 10步)"
    echo "============================================"
    echo "→ 日志: ${LOG_FILE}"
    echo "→ 预计耗时: 2-5 分钟"
    echo ""

    cd "${PV_DIR}"

    nohup python3 -u train_pv_tuning.py \
        --model_dir "${MODEL_DIR}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda:0 \
        --epochs 1 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_length 4096 \
        --max_quantized_layers 4 \
        --dry_run_steps 10 \
        --allow_missing_images \
        --log_every_steps 1 \
        > "${LOG_FILE}" 2>&1 &

    local PID=$!
    echo "→ 后台 PID: ${PID}"
    echo ""
    echo "── 实时查看 ────────────────────────────────"
    echo "  tail -f ${LOG_FILE}"
    echo "  watch -n 2 nvidia-smi"
    echo "──────────────────────────────────────────────"
    echo ""
    echo "→ 等待完成 (约 2-5 分钟)..."
    wait ${PID} 2>/dev/null || true
    echo "→ 冒烟测试完成"
    tail -20 "${LOG_FILE}"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: 正式训练 (GPU7, 后台)
# ══════════════════════════════════════════════════════════════════════════════
step_train() {
    local LOG_FILE="${LOG_DIR}/${RUN_NAME}_train.log"

    echo "============================================"
    echo " Step 4/5: 正式 PV-Tuning 训练 (GPU7)"
    echo "============================================"
    echo "→ 日志: ${LOG_FILE}"
    echo "→ 模型输出: ${OUTPUT_DIR}"
    echo "→ 预计耗时: 数小时 (取决于数据量与GPU)"
    echo ""

    cd "${PV_DIR}"

    # 后台启动训练
    nohup python3 -u train_pv_tuning.py \
        --model_dir "${MODEL_DIR}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --device cuda:0 \
        --epochs 1 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_length 4096 \
        --lr 3e-4 \
        --code_lr 1e-2 \
        --max_code_change_per_step 1e-3 \
        --beam_size 1 \
        --gradient_checkpointing \
        --log_every_steps 1 \
        --save_every_steps 200 \
        > "${LOG_FILE}" 2>&1 &

    local PID=$!
    echo "${PID}" > "${LOG_DIR}/${RUN_NAME}.pid"
    echo "→ 后台 PID: ${PID}"
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  实时监控命令"
    echo "══════════════════════════════════════════════"
    echo ""
    echo "  [训练日志]"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "  [GPU 状态]"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "  [只监控 GPU7]"
    echo "  watch -n 1 'nvidia-smi -i 7'"
    echo ""
    echo "  [Loss 提取]"
    echo "  grep '\[PV\]' ${LOG_FILE} | tail -20"
    echo ""
    echo "  [Code 更新率]"
    echo "  grep 'mean_code_change' ${LOG_FILE} | tail -20"
    echo ""
    echo "  [检查进程]"
    echo "  ps aux | grep train_pv_tuning"
    echo ""
    echo "  [终止训练]"
    echo "  kill \$(cat ${LOG_DIR}/${RUN_NAME}.pid)"
    echo ""
    echo "══════════════════════════════════════════════"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: 评估
# ══════════════════════════════════════════════════════════════════════════════
step_eval() {
    local LOG_FILE="${LOG_DIR}/${RUN_NAME}_eval.log"

    echo "============================================"
    echo " Step 5/5: 评估 PV-Tuned 模型"
    echo "============================================"
    echo "→ 日志: ${LOG_FILE}"
    echo ""

    cd "${PV_DIR}"

    if [[ ! -f "${OUTPUT_DIR}/model.safetensors" ]]; then
        echo "  ✗ 未找到训练输出: ${OUTPUT_DIR}"
        echo "  请先完成训练 (step_train)"
        return 1
    fi

    echo "→ 评估基线模型 (PV-Tuning 前)..."
    python3 -u evaluate_perplexity.py \
        --model_dir "${MODEL_DIR}" \
        --device cuda:0 \
        --allow_missing_images \
        --output_json "${LOG_DIR}/${RUN_NAME}_before.json" \
        > "${LOG_DIR}/${RUN_NAME}_eval_before.log" 2>&1

    echo "→ 评估 PV-Tuned 模型 (训练后)..."
    python3 -u evaluate_perplexity.py \
        --model_dir "${OUTPUT_DIR}" \
        --device cuda:0 \
        --allow_missing_images \
        --output_json "${LOG_DIR}/${RUN_NAME}_after.json" \
        > "${LOG_DIR}/${RUN_NAME}_eval_after.log" 2>&1

    echo ""
    echo "══════════════════════════════════════════════"
    echo "  评估结果"
    echo "══════════════════════════════════════════════"
    python3 -c "
import json
try:
    before = json.load(open('${LOG_DIR}/${RUN_NAME}_before.json'))
    after  = json.load(open('${LOG_DIR}/${RUN_NAME}_after.json'))
    delta  = before['perplexity'] - after['perplexity']
    print(f'  PV-Tuning 前  PPL: {before[\"perplexity\"]:.2f}')
    print(f'  PV-Tuning 后  PPL: {after[\"perplexity\"]:.2f}')
    print(f'  PPL 改善:        {delta:.2f}  ({delta/before[\"perplexity\"]*100:.1f}%)')
except Exception as e:
    print(f'  无法计算改善: {e}')
"
    echo "══════════════════════════════════════════════"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
case "${1:-}" in
    check)   step_check ;;
    data)    step_data ;;
    dryrun)  step_dryrun ;;
    train)   step_train ;;
    eval)    step_eval ;;
    all)
        step_check
        step_data
        step_dryrun
        step_train
        echo ""
        echo "训练在后台运行。完成后执行:"
        echo "  bash pv_tuning_workflow.sh eval"
        ;;
    *)
        echo "用法: bash pv_tuning_workflow.sh <STEP>"
        echo ""
        echo "  check   环境检查"
        echo "  data    数据准备"
        echo "  dryrun  冒烟测试 (GPU7, 10步)"
        echo "  train   正式训练 (GPU7, 后台运行)"
        echo "  eval    训练后评估"
        echo "  all     一键: check → data → dryrun → train → (手动) eval"
        ;;
esac
