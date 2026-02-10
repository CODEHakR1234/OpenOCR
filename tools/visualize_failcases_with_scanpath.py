#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fail Case Visualization with Scan Path Analysis

Fail case의 이미지 + GT + Prediction + Scan Path를 함께 시각화

Usage:
    python tools/visualize_failcases_with_scanpath.py \
        --config configs/rec/svtrv2/ablation/svtrv2_mamba_deep_668_ctc_h64.yml \
        --checkpoint output/rec/u14m_filter/ablation/svtrv2_mamba_deep_668_ctc_h64/best.pth \
        --json ./failcase_analysis_multioriented/analysis_results.json \
        --lmdb ../u14m/multi_oriented \
        --output_dir ./failcase_scanpath \
        --num_samples 20
"""

import os
import sys
import argparse
import io
import json
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from PIL import Image
import lmdb

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine import Config
from openrec.modeling import build_model
from openrec.postprocess import build_post_process


class ScanPathExtractor:
    """DASSM의 offset을 추출하는 hook 기반 extractor"""
    def __init__(self, model):
        self.model = model
        self.offsets = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Dynamic_Adaptive_Scan의 offset을 캡처하는 hook 등록"""
        for name, module in self.model.named_modules():
            if 'da_scan' in name and hasattr(module, 'offset'):
                hook = module.offset.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.offsets[name] = output.detach().cpu()
        return hook
    
    def extract(self, image_tensor):
        """이미지에서 offset 추출"""
        self.offsets = {}
        with torch.no_grad():
            _ = self.model(image_tensor)
        return self.offsets
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def load_analysis_results(json_path):
    """분석 결과 JSON 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_label_to_index_map(lmdb_path):
    """LMDB에서 label -> index 매핑 생성"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    label_to_indices = {}
    
    with env.begin(write=False) as txn:
        num_samples = int(txn.get('num-samples'.encode()).decode())
        
        for idx in range(1, num_samples + 1):
            label_key = f'label-{idx:09d}'.encode()
            label_data = txn.get(label_key)
            
            if label_data:
                label = label_data.decode()
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
    
    env.close()
    return label_to_indices


def get_image_from_lmdb(env, index):
    """LMDB에서 특정 인덱스의 이미지 가져오기"""
    with env.begin(write=False) as txn:
        img_key = f'image-{index:09d}'.encode()
        img_data = txn.get(img_key)
        
        if img_data is None:
            return None
        
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        return img


def preprocess_image(img_pil, base_height=32, max_ratio=12):
    """이미지 전처리"""
    from torchvision import transforms as T
    
    w, h = img_pil.size
    ratio = w / float(h)
    
    target_h = base_height
    if ratio > max_ratio:
        target_w = base_height * max_ratio
    else:
        target_w = int(math.ceil(base_height * ratio))
    
    target_w = max(32, (target_w // 4) * 4)
    
    img_resized = img_pil.resize((target_w, target_h), Image.BILINEAR)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    return img_tensor, img_resized


def visualize_scanpath_on_image(img, offsets, layer_name, ax, title=""):
    """이미지 위에 scan path 시각화"""
    ax.imshow(img)
    
    if layer_name not in offsets:
        ax.set_title(f"{title}\n(No offset data)", fontsize=8)
        ax.axis('off')
        return
    
    offset = offsets[layer_name]  # [B, H, W, groups*k*k*2]
    
    if offset.dim() == 4:
        offset = offset[0]  # [H, W, groups*k*k*2]
    
    H, W, total_offset = offset.shape
    img_h, img_w = img.size[1], img.size[0]
    
    # Scale factors
    scale_y = img_h / H
    scale_x = img_w / W
    
    # Offset을 dx, dy로 분리 (첫 번째 그룹만 사용)
    num_groups = total_offset // 2
    k_sq = 1  # 단순화
    
    # 첫 번째 그룹의 dx, dy
    dx = offset[:, :, 0].numpy()
    dy = offset[:, :, 1].numpy()
    
    # 화살표 그리기
    for i in range(0, H, max(1, H // 4)):
        for j in range(0, W, max(1, W // 4)):
            x = (j + 0.5) * scale_x
            y = (i + 0.5) * scale_y
            
            dx_val = dx[i, j] * scale_x * 0.5
            dy_val = dy[i, j] * scale_y * 0.5
            
            ax.arrow(x, y, dx_val, dy_val, 
                    head_width=2, head_length=1,
                    fc='yellow', ec='red', alpha=0.8, linewidth=0.5)
    
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def visualize_single_failcase(case, img_pil, model, extractor, post_process, 
                              output_path, base_height, max_ratio, device):
    """단일 fail case에 대한 종합 시각화"""
    
    # 전처리
    img_tensor, img_resized = preprocess_image(img_pil, base_height, max_ratio)
    img_tensor = img_tensor.to(device)
    
    # 예측 및 offset 추출
    with torch.no_grad():
        preds = model(img_tensor)
    offsets = extractor.offsets
    
    # Post process
    post_result = post_process(preds)
    if isinstance(post_result, list) and len(post_result) == 2:
        if isinstance(post_result[1], list):
            pred_text = post_result[1][0][0]
        else:
            pred_text = post_result[0][0]
    else:
        pred_text = post_result[0][0] if post_result else ""
    
    gt = case.get('gt', '')
    ed = case.get('edit_distance', '?')
    
    # Stage별 3번째, 6번째 block offset 찾기
    # offset 이름 형식: encoder.stages.2.blocks.5.da_scan.offset 등
    target_blocks = [2, 5]  # 3번째(idx 2), 6번째(idx 5) block
    
    # Stage 1과 Stage 2에서 찾기
    stage_block_layers = {}
    for stage_idx in [1, 2]:
        for block_idx in target_blocks:
            key = f"S{stage_idx+1}_B{block_idx+1}"  # S2_B3, S2_B6, S3_B3, S3_B6
            for name in offsets.keys():
                if f'stages.{stage_idx}' in name and f'blocks.{block_idx}' in name:
                    stage_block_layers[key] = name
                    break
    
    # 시각화 생성: Original + 4개 scan path (Stage1 B3, B6 / Stage2 B3, B6)
    num_plots = 1 + len(stage_block_layers)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]
    
    # 원본 이미지 + 결과
    axes[0].imshow(img_pil)
    axes[0].set_title(
        f"✗ ED:{ed}\n"
        f"GT: {gt[:25]}\n"
        f"Pred: {pred_text[:25]}",
        fontsize=9,
        color='red'
    )
    axes[0].axis('off')
    
    # Stage별 scan path 시각화
    plot_order = ['S2_B3', 'S2_B6', 'S3_B3', 'S3_B6']  # Stage 1(idx 1) = S2, Stage 2(idx 2) = S3
    plot_idx = 1
    for key in plot_order:
        if key in stage_block_layers and plot_idx < len(axes):
            layer_name = stage_block_layers[key]
            # S2_B3 -> "Stage 1 / Block 3" (S2 means stages.1, B3 means block 3)
            parts = key.split('_')  # ['S2', 'B3']
            stage_num = int(parts[0][1])  # S2 -> 2
            block_num = int(parts[1][1])  # B3 -> 3
            visualize_scanpath_on_image(
                img_resized, offsets, layer_name, axes[plot_idx],
                title=f"Stage {stage_num} / Block {block_num}"
            )
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fail Case with Scan Path Visualization')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--json', '-j', type=str, required=True,
                       help='Path to analysis_results.json')
    parser.add_argument('--lmdb', type=str, required=True,
                       help='Path to LMDB dataset')
    parser.add_argument('--output_dir', '-o', type=str, default='./failcase_scanpath',
                       help='Output directory')
    parser.add_argument('--num_samples', '-n', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Specific dataset name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    print(f"🔍 Fail Case + Scan Path Visualization")
    print(f"   Config: {args.config}")
    print(f"   JSON: {args.json}")
    print(f"   LMDB: {args.lmdb}")
    print(f"   Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and model
    print("\n🔧 Loading model...")
    cfg = Config(args.config)
    
    post_process = build_post_process(cfg.cfg['PostProcess'], cfg.cfg['Global'])
    char_num = post_process.get_character_num()
    cfg.cfg['Architecture']['Decoder']['out_channels'] = char_num
    
    model = build_model(cfg.cfg['Architecture'])
    
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(args.device)
    model.eval()
    
    # Get resolution from config
    base_height = 32
    max_ratio = 12
    try:
        sampler_cfg = cfg.cfg.get('Eval', {}).get('sampler', {})
        scales = sampler_cfg.get('scales', [[128, 32]])
        base_height = scales[0][1] if scales else 32
        max_ratio = cfg.cfg.get('Eval', {}).get('loader', {}).get('max_ratio', 12)
    except:
        pass
    print(f"   Resolution: height={base_height}, max_ratio={max_ratio}")
    
    # Create extractor
    extractor = ScanPathExtractor(model)
    print(f"   Registered {len(extractor.hooks)} hooks for scan path extraction")
    
    # Load analysis results
    print("\n📊 Loading analysis results...")
    results = load_analysis_results(args.json)
    
    datasets = results.get('datasets', {})
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = list(datasets.keys())[0]
    
    dataset_data = datasets[dataset_name]
    failcases = dataset_data.get('failcases', [])
    
    print(f"   Dataset: {dataset_name}")
    print(f"   Fail cases: {len(failcases)}")
    
    # Build label index map
    print("\n🗂️ Building label-to-index map...")
    label_to_indices = build_label_to_index_map(args.lmdb)
    
    # Open LMDB
    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    
    # Visualize fail cases
    print(f"\n🎨 Visualizing {min(args.num_samples, len(failcases))} fail cases...")
    
    for i, case in enumerate(failcases[:args.num_samples]):
        gt = case.get('gt', '')
        
        # Find image from LMDB
        if gt not in label_to_indices:
            print(f"  [{i+1}] Not found in LMDB: {gt[:20]}...")
            continue
        
        lmdb_idx = label_to_indices[gt][0]
        img_pil = get_image_from_lmdb(env, lmdb_idx)
        
        if img_pil is None:
            print(f"  [{i+1}] Image load failed: {gt[:20]}...")
            continue
        
        # Visualize
        output_path = os.path.join(args.output_dir, f'failcase_{i+1:03d}_{gt[:15].replace("/", "_")}.png')
        visualize_single_failcase(
            case, img_pil, model, extractor, post_process,
            output_path, base_height, max_ratio, args.device
        )
        print(f"  [{i+1}/{args.num_samples}] Saved: {os.path.basename(output_path)}")
    
    extractor.remove_hooks()
    env.close()
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
