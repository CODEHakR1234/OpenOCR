#!/bin/bash
set -e

BUILD_MARKER="/workspace/OpenOCR/.cuda_extensions_built"

# CUDA extension 빌드 (최초 1회만)
if [ ! -f "$BUILD_MARKER" ]; then
    echo "========================================"
    echo "Building CUDA extensions (first run)..."
    echo "========================================"
    
    # DCNv3 빌드 및 설치
    echo "[1/2] Building and installing DCNv3..."
    cd /workspace/OpenOCR/reference/DAMamba/classification/models/ops_dcnv3
    pip install --no-build-isolation -e .
    echo "DCNv3 installed!"
    
    # Selective Scan 빌드 및 설치
    echo "[2/2] Building and installing Selective Scan..."
    cd /workspace/OpenOCR/reference/DAMamba/classification/models/selective_scan
    pip install --no-build-isolation -e .
    echo "Selective Scan installed!"
    
    # 빌드 완료 마커 생성
    touch "$BUILD_MARKER"
    
    echo "========================================"
    echo "CUDA extensions build completed!"
    echo "========================================"
fi

cd /workspace/OpenOCR

# 전달된 명령 실행
exec "$@"
