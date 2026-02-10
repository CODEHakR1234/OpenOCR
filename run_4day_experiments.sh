#!/bin/bash
# =============================================================================
# 4-Day Experiment Package: Deep SVTRv2Mamba Ablation Study
# =============================================================================
# Experiment 1: [6,6,8] + CTC only (No FRM, No SGM) - ~1.5 days
# Experiment 2: [6,8,8] + GTC (FRM + SGM)           - ~2.5 days
# =============================================================================
# Total estimated time: ~4 days
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_DIR="./output/rec/u14m_filter/ablation"
mkdir -p $LOG_DIR
MAIN_LOG="$LOG_DIR/4day_experiment.log"

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $MAIN_LOG
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a $MAIN_LOG
}

log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a $MAIN_LOG
}

# =============================================================================
echo ""
echo "============================================================================="
echo "  4-Day Experiment Package: Deep SVTRv2Mamba Ablation Study"
echo "============================================================================="
echo ""
log "Starting 4-day experiment package..."
log "Experiment 1: svtrv2_mamba_deep_668_ctc (CTC only, batch=256)"
log "Experiment 2: svtrv2_mamba_deep_688_gtc (GTC, batch=192)"
echo ""

# Check if we're in the right directory
if [ ! -f "tools/train_rec.py" ]; then
    log_error "Please run this script from the OpenOCR root directory!"
    exit 1
fi

# =============================================================================
# Experiment 1: [6,6,8] + CTC only
# =============================================================================
echo ""
echo "============================================================================="
log "${YELLOW}EXPERIMENT 1/2: svtrv2_mamba_deep_668_ctc${NC}"
echo "  - Depths: [6, 6, 8]"
echo "  - d_state: 16"
echo "  - Decoder: CTC only (No FRM, No SGM)"
echo "  - Batch size: 256"
echo "  - Estimated time: ~1.5 days"
echo "============================================================================="
echo ""

START_TIME_1=$(date +%s)
log "Starting Experiment 1..."

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    tools/train_rec.py \
    -c configs/rec/svtrv2/ablation/svtrv2_mamba_deep_668_ctc.yml \
    2>&1 | tee -a "$LOG_DIR/svtrv2_mamba_deep_668_ctc_run.log"

END_TIME_1=$(date +%s)
DURATION_1=$((END_TIME_1 - START_TIME_1))
HOURS_1=$((DURATION_1 / 3600))
MINUTES_1=$(((DURATION_1 % 3600) / 60))

log "Experiment 1 completed in ${HOURS_1}h ${MINUTES_1}m"
echo ""

# =============================================================================
# Experiment 2: [6,8,8] + GTC
# =============================================================================
echo ""
echo "============================================================================="
log "${YELLOW}EXPERIMENT 2/2: svtrv2_mamba_deep_688_gtc${NC}"
echo "  - Depths: [6, 8, 8]"
echo "  - d_state: 16"  
echo "  - Decoder: GTC (FRM + SGM)"
echo "  - Batch size: 192"
echo "  - Estimated time: ~2.5 days"
echo "============================================================================="
echo ""

START_TIME_2=$(date +%s)
log "Starting Experiment 2..."

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
    tools/train_rec.py \
    -c configs/rec/svtrv2/ablation/svtrv2_mamba_deep_688_gtc.yml \
    2>&1 | tee -a "$LOG_DIR/svtrv2_mamba_deep_688_gtc_run.log"

END_TIME_2=$(date +%s)
DURATION_2=$((END_TIME_2 - START_TIME_2))
HOURS_2=$((DURATION_2 / 3600))
MINUTES_2=$(((DURATION_2 % 3600) / 60))

log "Experiment 2 completed in ${HOURS_2}h ${MINUTES_2}m"

# =============================================================================
# Summary
# =============================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - START_TIME_1))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "============================================================================="
echo "  EXPERIMENT PACKAGE COMPLETED!"
echo "============================================================================="
log "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "Results:"
echo "  1. svtrv2_mamba_deep_668_ctc: $LOG_DIR/svtrv2_mamba_deep_668_ctc/"
echo "  2. svtrv2_mamba_deep_688_gtc: $LOG_DIR/svtrv2_mamba_deep_688_gtc/"
echo ""
echo "Logs:"
echo "  - Main log: $MAIN_LOG"
echo "  - Exp 1 log: $LOG_DIR/svtrv2_mamba_deep_668_ctc_run.log"
echo "  - Exp 2 log: $LOG_DIR/svtrv2_mamba_deep_688_gtc_run.log"
echo "============================================================================="
