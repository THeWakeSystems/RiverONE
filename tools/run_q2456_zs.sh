#!/bin/bash
# RiverONE-2B-ZS QCalEval Q2/Q4/Q5/Q6 推理+评分 (4-GPU并行, score-as-you-go)
set -euo pipefail

MODEL="/home/lxy/workspace/RiverONE-2B-ZS"
DATA_ROOT="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/Data"
REPORTS="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test/reports/RiverONE-2B-ZS"
EVAL_DIR="/home/lxy/workspace/RiverOne_ZeroShot_v1.2_Test"
CONDA_ENV="riverone"

mkdir -p "$REPORTS/Q2" "$REPORTS/Q4" "$REPORTS/Q5" "$REPORTS/Q6"
mkdir -p "$REPORTS/logs"

run_and_score() {
    local q="$1" gpu="$2" max_tokens="$3"
    local log="$REPORTS/logs/${q}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date)] START $q on GPU $gpu" | tee -a "$log"

    # === 推理 ===
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$CONDA_ENV" --no-capture-output \
        python3 -u "$EVAL_DIR/scripts/${q^^}/run_${q}_inference.py" \
        --model-path "$MODEL" \
        --device cuda \
        --data-root "$DATA_ROOT" \
        --data-file "$DATA_ROOT/${q}.jsonl" \
        --reports-root "$REPORTS" \
        --max-tokens "$max_tokens" \
        >> "$log" 2>&1
    local rc=$?
    echo "[$(date)] INFERENCE $q DONE (exit=$rc)" | tee -a "$log"
    if [ $rc -ne 0 ]; then
        echo "[$(date)] SKIP SCORING $q (inference failed)" | tee -a "$log"
        return $rc
    fi

    # === 评分 ===
    case "$q" in
        q2)
            conda run -n "$CONDA_ENV" python3 "$EVAL_DIR/scripts/Q2/eval_q2_acc_with_judge.py" \
                --pred-file "$REPORTS/Q2/predictions_q2.jsonl" \
                --reports-root "$REPORTS" \
                >> "$log" 2>&1
            ;;
        q4)
            conda run -n "$CONDA_ENV" python3 "$EVAL_DIR/scripts/Q4/eval_q4_acc_with_judge.py" \
                --pred-file "$REPORTS/Q4/predictions_q4.jsonl" \
                --reports-root "$REPORTS" \
                >> "$log" 2>&1
            ;;
        q5)
            conda run -n "$CONDA_ENV" python3 "$EVAL_DIR/scripts/Q5/eval_q5_acc_with_judge.py" \
                --pred-file "$REPORTS/Q5/predictions_q5.jsonl" \
                --data-file "$DATA_ROOT/q5.jsonl" \
                --reports-root "$REPORTS" \
                >> "$log" 2>&1
            ;;
        q6)
            conda run -n "$CONDA_ENV" python3 "$EVAL_DIR/scripts/Q6/eval_q6_acc_with_judge.py" \
                --pred-file "$REPORTS/Q6/predictions_q6.jsonl" \
                --reports-root "$REPORTS" \
                >> "$log" 2>&1
            ;;
    esac
    echo "[$(date)] SCORING $q DONE (exit=$?)" | tee -a "$log"
}

# 并行启动 (Q5最慢→GPU0, Q6→GPU1, Q2→GPU2, Q4最快→GPU3)
echo "=== RiverONE-2B-ZS Q2/Q4/Q5/Q6 推理评分开始 ==="
echo "Model: $MODEL"
echo "Reports: $REPORTS"

run_and_score q5 0 1024 &
PID_Q5=$!
run_and_score q6 1 512 &
PID_Q6=$!
run_and_score q2 2 512 &
PID_Q2=$!
run_and_score q4 3 256 &
PID_Q4=$!

# 等待全部完成
wait $PID_Q5; echo "Q5 DONE"
wait $PID_Q6; echo "Q6 DONE"
wait $PID_Q2; echo "Q2 DONE"
wait $PID_Q4; echo "Q4 DONE"

echo ""
echo "=== 全部完成 ==="
echo "结果目录: $REPORTS"
echo ""
echo "=== 评分摘要 ==="
for q in q2 q4 q5 q6; do
    report_md="$REPORTS/${q^^}/judged_${q}_results_report.md"
    if [ -f "$report_md" ]; then
        grep -E "MeanScore|Total" "$report_md" | head -5
    else
        echo "${q^^}: 报告未生成"
    fi
    echo "---"
done
