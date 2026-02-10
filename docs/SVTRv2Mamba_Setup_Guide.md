# SVTRv2Mamba 환경 설정 가이드

이 가이드는 Docker 환경에서 SVTRv2Mamba 모델을 학습하기 위한 전체 환경 설정 과정을 설명합니다.

## 목차
1. [사전 요구사항](#1-사전-요구사항)
2. [Docker 이미지 빌드](#2-docker-이미지-빌드)
3. [데이터셋 준비](#3-데이터셋-준비)
4. [Docker 컨테이너 실행](#4-docker-컨테이너-실행)
5. [학습 실행](#5-학습-실행)
6. [문제 해결](#6-문제-해결)

---

## 1. 사전 요구사항

### 하드웨어
- NVIDIA GPU (CUDA Compute Capability 7.0+)
- 최소 16GB VRAM 권장
- 충분한 디스크 공간 (데이터셋 ~100GB)

### 소프트웨어
- Docker (19.03+)
- NVIDIA Container Toolkit (`nvidia-docker2`)
- Git

### NVIDIA Docker 설치 확인
```bash
# Docker에서 GPU 사용 가능 여부 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. Docker 이미지 빌드

### 2.1 프로젝트 클론
```bash
git clone <repository_url> OpenOCR
cd OpenOCR
```

### 2.2 Dockerfile 확인
프로젝트 루트에 `Dockerfile`이 있어야 합니다:

```dockerfile
# 주요 구성:
# - Base: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# - Python 3.10
# - PyTorch 2.0.1 + CUDA 11.8
# - 필요 패키지들 (timm, einops, fvcore 등)
```

### 2.3 이미지 빌드
```bash
docker build -t openocr-mamba .
```

> ⏱️ 빌드 시간: 약 10-20분 (네트워크 속도에 따라 다름)

---

## 3. 데이터셋 준비

### 3.1 데이터셋 구조
```
OpenOCR/
├── Union14M-L-LMDB-Filtered/    # 학습 데이터 (상위 디렉토리에 위치)
│   ├── filter_train_easy/
│   ├── filter_train_normal/
│   ├── filter_train_medium/
│   ├── filter_train_hard/
│   └── filter_train_challenging/
└── evaluation/                   # 평가 데이터 (상위 디렉토리에 위치)
    ├── CUTE80/
    ├── IC13_857/
    ├── IC15_1811/
    ├── IIIT5k/
    ├── SVT/
    └── SVTP/
```

### 3.2 HuggingFace에서 다운로드

데이터셋은 HuggingFace Hub에서 다운로드할 수 있습니다:

```bash
# huggingface-cli 설치 (호스트에서)
pip install huggingface_hub

# 데이터셋 다운로드
huggingface-cli download topdu/OpenOCR-Data --repo-type dataset --local-dir ./OpenOCR-Data

# 또는 Python으로:
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='topdu/OpenOCR-Data',
    repo_type='dataset',
    local_dir='./OpenOCR-Data'
)
"
```

### 3.3 데이터셋 배치
다운로드 후 압축 해제하여 OpenOCR 상위 디렉토리에 배치:
```bash
# OpenOCR과 같은 레벨에 위치해야 함
/path/to/
├── OpenOCR/
├── Union14M-L-LMDB-Filtered/
└── evaluation/
```

---

## 4. Docker 컨테이너 실행

### 4.1 기본 실행 (영구 컨테이너)
```bash
docker run -it \
    --gpus all \
    --shm-size=16g \
    --name openocr-dev \
    -v /path/to/OpenOCR:/workspace/OpenOCR \
    -v /path/to/Union14M-L-LMDB-Filtered:/workspace/Union14M-L-LMDB-Filtered \
    -v /path/to/evaluation:/workspace/evaluation \
    openocr-mamba
```

### 4.2 옵션 설명
| 옵션 | 설명 |
|------|------|
| `--gpus all` | 모든 GPU 사용 |
| `--shm-size=16g` | 공유 메모리 크기 (DataLoader worker용) |
| `--name openocr-dev` | 컨테이너 이름 지정 |
| `-v` | 볼륨 마운트 (호스트:컨테이너) |

### 4.3 컨테이너 재접속
```bash
# 컨테이너 시작
docker start openocr-dev

# 컨테이너 접속
docker exec -it openocr-dev bash
```

### 4.4 CUDA Extension 자동 빌드
컨테이너 최초 실행 시 `docker-entrypoint.sh`가 자동으로 CUDA extension을 빌드합니다:
- **DCNv3**: Deformable Convolution v3
- **Selective Scan**: Mamba SSM용 CUDA kernel

빌드 완료 후 `.cuda_extensions_built` 마커 파일이 생성됩니다.

---

## 5. 학습 실행

### 5.1 단일 GPU 학습
```bash
cd /workspace/OpenOCR

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/train_rec.py \
    --c configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml
```

### 5.2 다중 GPU 학습
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train_rec.py \
    --c configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml
```

### 5.3 학습 재개
```bash
# config 파일에서 checkpoints 경로 설정 또는:
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/train_rec.py \
    --c configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml \
    --o Global.checkpoints=./output/rec/u14m_filter/svtrv2_mamba_gtc_rctc/latest.pth
```

### 5.4 평가 실행
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_rec.py \
    --c configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml \
    --o Global.pretrained_model=./output/rec/u14m_filter/svtrv2_mamba_gtc_rctc/best.pth
```

---

## 6. 문제 해결

### 6.1 NumPy 버전 오류
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
**해결:**
```bash
pip install "numpy<2"
```

### 6.2 공유 메모리 오류
```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```
**해결:** 컨테이너 실행 시 `--shm-size=16g` 또는 `--ipc=host` 추가

### 6.3 CUDA Extension 빌드 실패
```
ModuleNotFoundError: No module named 'DCNv3'
```
**해결:**
```bash
# 마커 파일 삭제 후 컨테이너 재시작
rm /workspace/OpenOCR/.cuda_extensions_built
# 컨테이너 재시작하면 자동 빌드
```

### 6.4 torch.distributed 오류
```
ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected
```
**해결:** `python -m torch.distributed.launch`로 실행

### 6.5 bfloat16 미지원 GPU
```
RuntimeError: Current CUDA Device does not support bfloat16
```
**해결:** `tools/engine/trainer.py`에서 `torch.bfloat16` → `torch.float16` 변경

### 6.6 GradScaler 오류
```
AttributeError: module 'torch.amp' has no attribute 'GradScaler'
```
**해결:** `torch.amp.GradScaler()` → `torch.cuda.amp.GradScaler()` 변경

---

## 7. Config 파일 설명

### 7.1 주요 설정
```yaml
Global:
  epoch_num: 20           # 학습 에폭 수
  eval_batch_step: [0, 500]  # 평가 주기

Architecture:
  Encoder:
    name: SVTRv2Mamba
    dims: [128, 256, 384]   # Stage별 채널 수
    depths: [6, 6, 6]       # Stage별 블록 수
    d_state: 16             # Mamba SSM state dimension

Train:
  loader:
    batch_size_per_card: 128  # GPU당 배치 크기
    num_workers: 8            # DataLoader worker 수
    pin_memory: True          # 빠른 데이터 전송
```

### 7.2 Ablation Study Configs
```
configs/rec/svtrv2/ablation/
├── svtrv2_mamba_ctc.yml          # CTC only (baseline)
├── svtrv2_mamba_rctc.yml         # FRM only
├── svtrv2_mamba_smtr.yml         # SGM only
├── svtrv2_mamba_ctc_smtr.yml     # CTC + SGM (no FRM)
└── svtrv2_mamba_gtc_rctc_d16.yml # FRM + SGM (full)
```

---

## 8. 성능 최적화 팁

### 8.1 학습 속도 향상
- `num_workers` 증가 (CPU 코어 수에 맞게)
- `pin_memory: True` 설정
- `batch_size` 조정 (GPU 메모리에 맞게)

### 8.2 메모리 최적화
- `use_amp: True`로 Mixed Precision 사용
- `batch_size` 감소
- `max_ratio` 감소 (긴 이미지 제외)

---

## 9. 참고 자료

- [OpenOCR GitHub](https://github.com/Topdu/OpenOCR)
- [DAMamba Paper](https://arxiv.org/abs/2502.12627)
- [SVTRv2 Paper](https://arxiv.org/abs/2411.15858)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)

---

## 10. 연락처 및 이슈

문제가 발생하면 GitHub Issues에 보고해주세요.
