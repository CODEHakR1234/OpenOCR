#!/bin/bash
# =============================================================================
# 688 GTC (Mamba) + h64 - Batch Size 96
# =============================================================================

cd /mnt/ssd0/hmlee/OpenOCR

LOG_DIR="./output/rec/u14m_filter/ablation"
LOG_FILE="$LOG_DIR/svtrv2_mamba_deep_688_gtc_h64_run.log"
mkdir -p $LOG_DIR

echo "============================================================================="
echo "  688 GTC (Mamba) + h64 - Batch Size 96"
echo "============================================================================="
echo "  - Encoder: SVTRv2Mamba [6, 8, 8]"
echo "  - Resolution: 64 x (64~384)"
echo "  - Batch size: 96 per GPU (192 total)"
echo "  - GPUs: 2"
echo "============================================================================="
echo ""
echo "Starting training with nohup..."
echo "Log file: $LOG_FILE"
echo ""

nohup bash -c "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    tools/train_rec.py \
    -c configs/rec/svtrv2/ablation/svtrv2_mamba_deep_688_gtc_h64.yml" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "$PID" > "$LOG_DIR/688_gtc_h64_pid.txt"
echo ""
echo "============================================================================="
echo "  로그 보는 방법:"
echo "    tail -f $LOG_FILE"
echo ""
echo "  프로세스 확인:"
echo "    ps aux | grep train_rec"
echo ""
echo "  프로세스 종료:"
echo "    kill $PID"
echo "============================================================================="
