#!/usr/bin/env bash
# ============================================================================
# QcalEval 数据准备脚本
# ============================================================================
# QcalEval JSONL 引用的图片应放在 QcalEval/images/ 目录下。
# 如果图片尚未就绪，本脚本会列出缺失文件并提供指导。
# ============================================================================
set -euo pipefail

PV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PV_DIR}/QcalEval"
IMAGES_DIR="${DATA_DIR}/images"

echo "============================================"
echo " QcalEval Data Preparation"
echo "============================================"
echo ""
echo "Data directory: ${DATA_DIR}"
echo "Images directory: ${IMAGES_DIR}"
echo ""

# 1. 检查 JSONL 文件
echo "[1/4] Checking JSONL files..."
ZS_FILE="${DATA_DIR}/qcaleval_zs_sft.jsonl"
ICL_FILE="${DATA_DIR}/qcaleval_icl_sft.jsonl"

for f in "${ZS_FILE}" "${ICL_FILE}"; do
    if [[ -f "${f}" ]]; then
        lines=$(wc -l < "${f}")
        size=$(du -h "${f}" | cut -f1)
        echo "  ✓ $(basename "${f}"): ${lines} lines, ${size}"
    else
        echo "  ✗ MISSING: ${f}"
        exit 1
    fi
done

# 2. 提取所有被引用的图片路径
echo ""
echo "[2/4] Extracting image references..."

python3 -c "
import json, sys
from pathlib import Path

data_dir = Path('${DATA_DIR}')
all_images = set()
for fname in ['qcaleval_zs_sft.jsonl', 'qcaleval_icl_sft.jsonl']:
    with (data_dir / fname).open('r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            img = record.get('image', [])
            if isinstance(img, str):
                all_images.add(img)
            elif isinstance(img, list):
                all_images.update(img)

print(f'Total unique image references: {len(all_images)}')

# Categorize
missing = []
existing = []
for rel_path in sorted(all_images):
    full_path = Path(rel_path)
    if not full_path.is_absolute():
        full_path = data_dir / full_path
    if full_path.exists():
        existing.append(str(rel_path))
    else:
        missing.append(str(rel_path))

print(f'Existing images: {len(existing)}')
print(f'Missing images:  {len(missing)}')

if missing:
    print()
    print('Missing image files (first 20):')
    for p in missing[:20]:
        print(f'  - {p}')
    if len(missing) > 20:
        print(f'  ... and {len(missing) - 20} more')

# Write missing list to file
missing_file = data_dir / 'missing_images.txt'
with missing_file.open('w') as f:
    for p in missing:
        f.write(p + '\n')
if missing:
    print(f'Full list written to: {missing_file}')
"

# 3. 检查图片目录
echo ""
echo "[3/4] Images directory status..."

if [[ -d "${IMAGES_DIR}" ]]; then
    img_count=$(ls -1 "${IMAGES_DIR}" 2>/dev/null | wc -l)
    echo "  Images directory exists with ${img_count} files"
else
    echo "  Images directory DOES NOT exist"
    mkdir -p "${IMAGES_DIR}"
    echo "  → Created empty directory: ${IMAGES_DIR}"
fi

# 4. 总结
echo ""
echo "[4/4] Summary"
echo "============================================"
echo ""
echo "If images are missing, you have several options:"
echo ""
echo "A) Copy the actual QcalEval images into:"
echo "     ${IMAGES_DIR}/"
echo ""
echo "B) Run PV-tuning with placeholder gray images (for plumbing test only):"
echo "     ALLOW_MISSING_IMAGES=1 bash run_pv_tuning.sh"
echo ""
echo "C) Validate current state:"
echo "     python3 validate_qcaleval.py"
echo ""
echo "============================================"
