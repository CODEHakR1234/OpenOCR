# =============================================================================
# SVTRv2-Mamba Docker Environment
# =============================================================================
# DA-Mamba (DCNv3 + Selective Scan) CUDA Extension 빌드를 포함한 환경
#
# 빌드:
#   docker build -t openocr-mamba .
#
# 실행:
#   docker run --gpus all -it --rm \
#       -v $(pwd):/workspace \
#       openocr-mamba bash
# =============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0"
ENV FORCE_CUDA="1"

# 시스템 패키지 + Python 3.10 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    wget \
    curl \
    vim \
    ca-certificates \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10을 기본으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# PyTorch 설치 (CUDA 11.8)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Python 패키지 업그레이드
RUN pip install --upgrade pip setuptools wheel

# OpenOCR 기본 의존성
RUN pip install --no-cache-dir \
    imgaug \
    lmdb \
    "numpy<2" \
    opencv-python \
    pyclipper \
    pyyaml \
    rapidfuzz \
    tqdm

# DA-Mamba 의존성
RUN pip install --no-cache-dir \
    packaging \
    triton \
    timm==0.6.11 \
    pytest \
    chardet \
    yacs \
    termcolor \
    submitit \
    tensorboardX \
    fvcore \
    seaborn \
    tensorboard \
    Cython \
    ninja \
    einops

# 프로젝트 복사
COPY . /workspace/OpenOCR

WORKDIR /workspace/OpenOCR

# Entrypoint 스크립트 설정
RUN chmod +x /workspace/OpenOCR/docker-entrypoint.sh

# PYTHONPATH 설정
ENV PYTHONPATH="/workspace/OpenOCR"

# Entrypoint (첫 실행 시 CUDA extension 빌드)
ENTRYPOINT ["/workspace/OpenOCR/docker-entrypoint.sh"]

# 기본 명령어
CMD ["bash"]
