# SVTRv2 + DA-Mamba Integration

## 개요

SVTRv2의 Global Attention을 **DA-Mamba (Dynamic Adaptive Mamba)**로 교체한 실험적 인코더입니다.

- **파일 위치**: `openrec/modeling/encoders/svtrv2_mamba.py`
- **Config 예시**: `configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml`
- **SVTRv2 기반 코드**: `openrec/modeling/encoders/svtrv2_lnconv_two33.py`
- **DA-Mamba 기반 코드**: `reference/DAMamba/classification/models/DAMamba.py`

### 코드 출처

| 컴포넌트 | 출처 파일 | 설명 |
|----------|----------|------|
| `SVTRv2Mamba` | `svtrv2_lnconv_two33.py` | 전체 인코더 구조 |
| `ConvBlock` | `svtrv2_lnconv_two33.py` | Local feature 블록 |
| `POPatchEmbed` | `svtrv2_lnconv_two33.py` | Patch Embedding |
| `SubSample2DTo2D` | `svtrv2_lnconv_two33.py` | 다운샘플링 |
| `Dynamic_Adaptive_Scan` | `reference/DAMamba/.../DAMamba.py` | DCNv3 기반 적응적 스캔 |
| `DASSM` | `reference/DAMamba/.../DAMamba.py` | DAS + SSM 결합 |
| `LayerNorm2d` | `reference/DAMamba/.../DAMamba.py` | 2D LayerNorm |
| `ResDWC` | `reference/DAMamba/.../DAMamba.py` | Residual DW Conv |
| `ConvFFN` | `reference/DAMamba/.../DAMamba.py` | Conv 기반 FFN |
| `build_norm_layer` | `reference/DAMamba/.../DAMamba.py` | Norm layer 빌더 |
| `build_act_layer` | `reference/DAMamba/.../DAMamba.py` | Activation 빌더 |
| `DASSMBlock` | **신규 작성** | DASSM을 Block 형태로 래핑 |

### 의존성 모듈

DA-Mamba CUDA 커널 (빌드 필요):
- `reference/DAMamba/classification/models/ops_dcnv3/` → `DCNv3Function`
- `reference/DAMamba/classification/models/utils.py` → `selective_scan_fn`

---

## 빌드 요구사항

DA-Mamba는 CUDA 커널이 필요합니다. **Linux 환경에서 빌드**해야 합니다.

```bash
# 1. DCNv3 빌드
cd reference/DAMamba/classification/models/ops_dcnv3
python setup.py build_ext --inplace

# 2. Selective Scan 빌드
cd reference/DAMamba/classification/models/selective_scan
python setup.py build_ext --inplace
```

---

## 아키텍처

### 전체 구조

```
Input (B, 3, 32, 128)
       ↓
POPatchEmbed (4× downsample)
       ↓ (B, 64, 8, 32)
Stage 1: ConvBlock × 6 (Local)
       ↓ (B, 128, 8, 32)
Stage 2: ConvBlock × 2 + DASSMBlock × 4 (Local → Global)
       ↓ (B, 256, 4, 32)
Stage 3: DASSMBlock × 6 (Global)
       ↓ (B, 256, 4, 32)
CTC Decoder
```

### Mixer 타입

| Mixer | 블록 | 입출력 | 용도 |
|-------|------|--------|------|
| `'Conv'` | ConvBlock | 4D → 4D | Local feature |
| `'Mamba'` | DASSMBlock | 4D → 4D | Global (DA-Mamba) |
| `'Global'` | Block (Attention) | 3D → 3D | Global (기존) |
| `'FGlobal'` | Flatten + Block | 4D → 3D | Conv→Attn 전환 |

---

## 핵심 컴포넌트

### 1. DASSM (Dynamic Adaptive State Space Model)

DA-Mamba의 핵심 모듈입니다.

```python
class DASSM(nn.Module):
    def __init__(self, d_model, head_dim=16, d_state=1, expand=1, ...):
        self.in_proj = nn.Conv2d(d_model, d_inner, 1)
        self.conv2d = nn.Conv2d(d_inner, d_inner, d_conv, groups=d_inner)
        self.da_scan = Dynamic_Adaptive_Scan(channels=d_inner, group=num_group)
        # SSM 파라미터들...
```

**Forward 흐름:**
1. Input Projection (1×1 Conv)
2. Local Conv (3×3 DW Conv) + SiLU
3. Dynamic Adaptive Scan (DCNv3 기반)
4. SSM (Selective Scan)
5. Output Projection + LayerNorm

### 2. Dynamic_Adaptive_Scan (DAS)

DCNv3 기반의 적응적 샘플링 모듈입니다.

```python
class Dynamic_Adaptive_Scan(nn.Module):
    def __init__(self, channels, group, ...):
        self.dw_conv = nn.Sequential(DWConv, Norm, Act)
        self.offset = nn.Linear(channels, group * k * k * 2)
    
    def forward(self, input, x):
        offset = self.offset(self.dw_conv(input))
        x = DCNv3Function.apply(x, offset, ...)
        return x
```

**핵심 아이디어:** 고정된 스캔 순서 대신, 입력에 따라 "어디를 볼지" 학습

### 3. DASSMBlock

DASSM을 Transformer Block 형태로 감싼 모듈입니다.

```python
class DASSMBlock(nn.Module):
    def __init__(self, dim, head_dim=16, mlp_ratio=4.0, drop=0.0, ...):
        self.pos_embed = ResDWC(dim, 3)      # Position Embedding
        self.norm1 = LayerNorm2d(dim)
        self.token_mixer = DASSM(dim, head_dim=head_dim, dropout=drop)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = ConvFFN(dim, int(mlp_ratio * dim))
    
    def forward(self, x):
        x = self.pos_embed(x)
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

---

## 기존 SVTRv2 대비 변경점

### 1. 새로 추가된 모듈

| 모듈 | 역할 | 출처 |
|------|------|------|
| `Dynamic_Adaptive_Scan` | DCNv3 기반 적응적 샘플링 | DA-Mamba |
| `DASSM` | DAS + SSM 결합 | DA-Mamba |
| `DASSMBlock` | DASSM을 Block으로 감싸기 | 신규 |
| `LayerNorm2d` | 2D 입력용 LayerNorm | DA-Mamba |
| `ResDWC` | Residual Depthwise Conv | DA-Mamba |
| `ConvFFN` | Conv 기반 FFN | DA-Mamba |

### 2. SVTRStage 수정

```python
# 기존
if mixer[i] == 'Global':
    self.blocks.append(Block(...))  # Attention

# 추가
elif mixer[i] in ['Mamba', 'FMamba']:
    self.blocks.append(DASSMBlock(...))  # DA-Mamba
```

### 3. SVTRv2Mamba 클래스

```python
class SVTRv2Mamba(nn.Module):
    def __init__(self, ..., head_dim=16, ...):  # head_dim 파라미터 추가
        # POPatchEmbed flatten 조건 수정
        flatten=(first_mixer not in ['Conv', 'Mamba', 'FMamba'])
```

### 4. 파라미터 이름 수정 (버그 픽스)

```python
# 기존 (svtrv2_lnconv_two33.py) - 버그
stage = SVTRStage(drop=drop_rate, attn_drop=attn_drop_rate, ...)

# 수정 (svtrv2_mamba.py) - 올바름
stage = SVTRStage(drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, ...)
```

---

## Config 설정

### 기본 설정

```yaml
Encoder:
  name: SVTRv2Mamba
  max_sz: [32, 128]
  dims: [64, 128, 256]
  depths: [6, 6, 6]
  num_heads: [2, 4, 8]
  head_dim: 16
  mixer:
    - ['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv']
    - ['Conv', 'Conv', 'Mamba', 'Mamba', 'Mamba', 'Mamba']
    - ['Mamba', 'Mamba', 'Mamba', 'Mamba', 'Mamba', 'Mamba']
```

### Mixer 조합 예시

```yaml
# 보수적 (Conv 많이)
mixer:
  - ['Conv'] * 6
  - ['Conv'] * 4 + ['Mamba'] * 2
  - ['Mamba'] * 6

# 공격적 (Mamba 많이)
mixer:
  - ['Conv'] * 6
  - ['Mamba'] * 6
  - ['Mamba'] * 6
```

---

## 기술적 분석

### DA-Mamba가 OCR에 적합한 이유

1. **효율성**: O(N²) → O(N) 복잡도
2. **DAS**: 불규칙한 텍스트 배열 적응
3. **SSM**: 순차적 맥락 축적 (언어 모델 효과)
4. **4D 유지**: 형태 변환 오버헤드 없음

### 잠재적 한계

1. **정보 압축**: SSM의 hidden state가 개별 feature를 희석시킬 수 있음
2. **DAS 학습**: 적응적 샘플링 패턴 학습에 시간 필요
3. **역방향 텍스트**: 완전히 뒤집힌 텍스트 처리 어려움

### 완화 메커니즘

- **Skip Connection (D 파라미터)**: 원본 feature 직접 전달
- **Residual Connection**: Block 수준 원본 보존
- **Selective Mechanism**: 입력에 따른 적응적 혼합

---

## 학습 가이드

### From Scratch 학습

```bash
python tools/train_rec.py -c configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml
```

### 권장 설정

```yaml
Global:
  epochs: 20

Optimizer:
  name: AdamW
  lr: 0.00065
  weight_decay: 0.05

LRScheduler:
  name: OneCycleLR
  warmup_epoch: 1.5  # DAS 안정화를 위해 약간 길게
```

---

## 파일 구조

```
openrec/modeling/encoders/
├── __init__.py              # SVTRv2Mamba 등록
├── svtrv2_mamba.py          # DA-Mamba 통합 인코더
├── svtrv2_lnconv_two33.py   # 기존 SVTRv2 (참조용)
└── ...

configs/rec/svtrv2/
├── svtrv2_mamba_gtc_rctc.yml  # Mamba 설정
├── svtrv2_smtr_gtc_rctc.yml   # 기존 설정 (참조용)
└── ...

reference/DAMamba/             # DA-Mamba 원본 코드
├── classification/models/
│   ├── DAMamba.py
│   ├── ops_dcnv3/            # DCNv3 CUDA 커널
│   └── selective_scan/       # Selective Scan CUDA 커널
└── ...
```

---

## 참고 자료

- [DA-Mamba Paper (NeurIPS 2025)](https://arxiv.org/abs/...)
- [SVTRv2 Paper](https://arxiv.org/abs/...)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-12 | 초기 버전 생성 |
| 2026-01-12 | DASSMBlock에 dropout 전달 수정 |
| 2026-01-12 | FMamba → Mamba로 통일 권장 |
