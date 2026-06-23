#!/bin/bash
# ============================================================================
# Launch all 3 AQLM quantization configs
#
# Config 1 (shared codebooks):  CPU post-processing (~5 min)
# Config 2 (nredo=5):           GPU 0 (~4h, quantize_2x16_8L_nredo5.py)
# Config 3 (12L mixed nbits):   GPU 1 (~70min, quantize_2x16_12L_mixed_nbits.py)
#
# All run under conda env "riverone", with NVRTC LD_PRELOAD fix.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="riverone"
export LD_PRELOAD="$HOME/.local/cuda-compat/libnvrtc.so.13.0:$HOME/.local/cuda-compat/libnvrtc-builtins.so.13.0"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Launching 3 AQLM Configs ==="
log "Config 1: Shared Codebooks (post-processing, ~5 min)"
log "Config 2: FAISS nredo=5 (GPU 0, ~4h)"
log "Config 3: 12L Mixed nbits (GPU 1, ~70 min)"

cd "$SCRIPT_DIR"

# Config 1: Shared Codebooks (post-processing — uses torch, needs conda)
log "Launching Config 1..."
CUDA_VISIBLE_DEVICES="" conda run -n "$CONDA_ENV" --no-capture-output \
  python3 post_share_codebooks.py > post_share_codebooks_run.log 2>&1 &
PID1=$!
log "  PID=$PID1"

# Config 2: FAISS nredo=5 (GPU 0, ~4h)
log "Launching Config 2..."
CUDA_VISIBLE_DEVICES=0 conda run -n "$CONDA_ENV" --no-capture-output \
  python3 quantize_2x16_8L_nredo5.py > quantize_2x16_8L_nredo5_run.log 2>&1 &
PID2=$!
log "  PID=$PID2"

# Config 3: 12L Mixed nbits (GPU 1, ~70min)
log "Launching Config 3..."
CUDA_VISIBLE_DEVICES=1 conda run -n "$CONDA_ENV" --no-capture-output \
  python3 quantize_2x16_12L_mixed_nbits.py > quantize_2x16_12L_mixed_nbits_run.log 2>&1 &
PID3=$!
log "  PID=$PID3"

log "All launched. Waiting for completion..."
log "  PIDs: Config1=$PID1  Config2=$PID2  Config3=$PID3"

wait $PID1 $PID2 $PID3

log ""
log "=== ALL DONE ==="
log "Outputs:"
log "  Config 1: $(dirname "$SCRIPT_DIR")/RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly-shared"
log "  Config 2: $(dirname "$SCRIPT_DIR")/RiverOne-QC-4B-v2-AQLM-2x16-8L-MLPonly-nredo5"
log "  Config 3: $(dirname "$SCRIPT_DIR")/RiverOne-QC-4B-v2-AQLM-2x16-12L-MLPonly-mixed"
