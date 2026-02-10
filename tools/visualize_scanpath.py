#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVTRv2Mamba Scan Path Visualization

Dynamic Adaptive Scan (DASSM)의 offset을 시각화합니다.
DCNv3 기반 적응적 스캔 패턴을 arrow로 표현합니다.

Usage:
    # 이미지 파일 사용
    python tools/visualize_scanpath.py \
        --config configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml \
        --checkpoint output/rec/u14m_filter/svtrv2_mamba_gtc_rctc/best.pth \
        --image path/to/image.jpg \
        --output_dir ./visualization_output
    
    # LMDB 데이터셋에서 이미지 추출 (권장)
    python tools/visualize_scanpath.py \
        --config configs/rec/svtrv2/svtrv2_mamba_gtc_rctc.yml \
        --checkpoint output/rec/u14m_filter/svtrv2_mamba_gtc_rctc/best.pth \
        --lmdb /workspace/evaluation/CUTE80 \
        --lmdb_index 1 \
        --num_images 3 \
        --output_dir ./visualization_output
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from PIL import Image
import io
import lmdb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openrec.modeling import build_model
from tools.engine import Config


def extract_images_from_lmdb(lmdb_path, start_index=1, num_images=1):
    """
    LMDB 데이터셋에서 이미지와 라벨 추출
    
    Args:
        lmdb_path: LMDB 데이터셋 경로
        start_index: 시작 인덱스 (1부터 시작)
        num_images: 추출할 이미지 개수
    
    Returns:
        list of (image_pil, label, index) tuples
    """
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    images = []
    with env.begin(write=False) as txn:
        # 샘플 개수 확인
        num_samples = int(txn.get('num-samples'.encode()).decode())
        print(f"LMDB contains {num_samples} samples")
        
        end_index = min(start_index + num_images, num_samples + 1)
        
        for idx in range(start_index, end_index):
            img_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            
            img_data = txn.get(img_key)
            label_data = txn.get(label_key)
            
            if img_data is None:
                print(f"Warning: Image {idx} not found, skipping...")
                continue
            
            # 이미지 디코딩
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            label = label_data.decode() if label_data else f"unknown_{idx}"
            
            images.append((img, label, idx))
            print(f"  Extracted [{idx}]: '{label}' ({img.size[0]}x{img.size[1]})")
    
    env.close()
    return images


def list_available_lmdb_datasets(base_path='/workspace/evaluation'):
    """사용 가능한 LMDB 데이터셋 목록 출력"""
    if not os.path.exists(base_path):
        print(f"Base path not found: {base_path}")
        return []
    
    datasets = []
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if os.path.isdir(full_path):
            # LMDB인지 확인
            if os.path.exists(os.path.join(full_path, 'data.mdb')):
                datasets.append((name, full_path))
    
    if datasets:
        print("\n📁 Available LMDB datasets:")
        for name, path in datasets:
            print(f"  - {name}: {path}")
    
    return datasets


class ScanPathExtractor:
    """DASSM의 offset을 추출하는 hook 기반 extractor"""
    
    def __init__(self, model):
        self.model = model
        self.offsets = {}
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Dynamic_Adaptive_Scan의 offset을 캡처하는 hook 등록"""
        for name, module in self.model.named_modules():
            if 'da_scan' in name and hasattr(module, 'offset'):
                # offset layer에 hook 등록
                hook = module.offset.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
                print(f"Registered hook for: {name}")
    
    def _make_hook(self, name):
        def hook(module, input, output):
            # output: (B, H, W, group*k*k*2)
            self.offsets[name] = output.detach().cpu()
            if len(input) > 0:
                self.features[name] = input[0].detach().cpu()
        return hook
    
    def extract(self, image_tensor):
        """이미지에서 offset 추출"""
        self.offsets = {}
        self.features = {}
        
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        return self.offsets, self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def preprocess_image(image_path, target_size=(128, 32)):
    """이미지 전처리"""
    img = Image.open(image_path).convert('RGB')
    
    # Resize maintaining aspect ratio
    w, h = img.size
    ratio = w / h
    target_w, target_h = target_size
    
    if ratio > target_w / target_h:
        new_w = target_w
        new_h = int(target_w / ratio)
    else:
        new_h = target_h
        new_w = int(target_h * ratio)
    
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Pad to target size
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    new_img.paste(img, (0, 0))
    
    # To tensor
    img_array = np.array(new_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, new_img


def visualize_offset_arrows(image, offset, save_path, stage_name="", 
                           downsample_factor=4, arrow_scale=2.0, 
                           max_arrows=500, colormap='viridis'):
    """
    Offset을 화살표로 시각화
    
    Args:
        image: PIL Image 또는 numpy array
        offset: (H, W, num_offsets*2) 형태의 offset tensor
        save_path: 저장 경로
        stage_name: 시각화 제목에 표시할 stage 이름
        downsample_factor: 화살표 밀도 조절 (클수록 적은 화살표)
        arrow_scale: 화살표 크기 스케일
        max_arrows: 최대 화살표 개수
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    H, W = offset.shape[:2]
    num_offsets = offset.shape[2] // 2
    
    # 이미지 크기에 맞게 offset 좌표 스케일링
    img_h, img_w = image.shape[:2]
    scale_h = img_h / H
    scale_w = img_w / W
    
    fig, axes = plt.subplots(1, min(num_offsets, 4), figsize=(16, 4))
    if num_offsets == 1:
        axes = [axes]
    
    for idx in range(min(num_offsets, 4)):
        ax = axes[idx]
        ax.imshow(image)
        
        # Offset 추출 (dx, dy)
        dx = offset[:, :, idx * 2].numpy()
        dy = offset[:, :, idx * 2 + 1].numpy()
        
        # Grid points
        y_coords, x_coords = np.meshgrid(
            np.arange(0, H, downsample_factor),
            np.arange(0, W, downsample_factor),
            indexing='ij'
        )
        
        # Flatten
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        
        # Subsample offsets
        dx_sub = dx[::downsample_factor, ::downsample_factor].flatten()
        dy_sub = dy[::downsample_factor, ::downsample_factor].flatten()
        
        # Limit number of arrows
        if len(x_flat) > max_arrows:
            indices = np.random.choice(len(x_flat), max_arrows, replace=False)
            x_flat = x_flat[indices]
            y_flat = y_flat[indices]
            dx_sub = dx_sub[indices]
            dy_sub = dy_sub[indices]
        
        # Scale to image coordinates
        x_img = x_flat * scale_w + scale_w / 2
        y_img = y_flat * scale_h + scale_h / 2
        dx_img = dx_sub * scale_w * arrow_scale
        dy_img = dy_sub * scale_h * arrow_scale
        
        # Color by magnitude
        magnitudes = np.sqrt(dx_sub**2 + dy_sub**2)
        
        # Draw arrows
        ax.quiver(x_img, y_img, dx_img, dy_img, magnitudes,
                 cmap=colormap, angles='xy', scale_units='xy', scale=1,
                 width=0.003, headwidth=3, headlength=4, alpha=0.8)
        
        ax.set_title(f'Offset {idx+1}\nMean magnitude: {magnitudes.mean():.3f}')
        ax.axis('off')
    
    plt.suptitle(f'Scan Path Visualization - {stage_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_sampling_points(image, offset, save_path, stage_name="",
                              downsample_factor=2, point_size=10):
    """
    DCNv3의 실제 샘플링 위치를 시각화
    
    각 그리드 위치 (i, j)에서 실제로 샘플링하는 위치 (i + dx, j + dy)를 표시
    - 빨간 점: 원래 그리드 위치
    - 파란 점: 실제 샘플링 위치 (그리드 + offset)
    - 선: 원래 위치 → 샘플링 위치
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    H, W = offset.shape[:2]
    img_h, img_w = image.shape[:2]
    scale_h = img_h / H
    scale_w = img_w / W
    
    # 첫 번째 offset 사용
    dx = offset[:, :, 0].numpy()
    dy = offset[:, :, 1].numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.imshow(image)
    
    # Grid points (downsampled for visibility)
    for j in range(0, H, downsample_factor):
        for i in range(0, W, downsample_factor):
            # 원래 그리드 위치 (이미지 좌표)
            orig_x = i * scale_w + scale_w / 2
            orig_y = j * scale_h + scale_h / 2
            
            # 샘플링 위치 = 그리드 + offset (이미지 좌표)
            sample_x = (i + dx[j, i]) * scale_w + scale_w / 2
            sample_y = (j + dy[j, i]) * scale_h + scale_h / 2
            
            # 선: 원래 위치 → 샘플링 위치
            ax.plot([orig_x, sample_x], [orig_y, sample_y], 
                   'g-', linewidth=0.8, alpha=0.6)
            
            # 원래 위치 (빨간 점)
            ax.plot(orig_x, orig_y, 'ro', markersize=point_size * 0.4, alpha=0.7)
            
            # 샘플링 위치 (파란 점)
            ax.plot(sample_x, sample_y, 'b^', markersize=point_size * 0.5, alpha=0.8)
    
    ax.set_title(f'Sampling Points - {stage_name}\n'
                f'Red ●: Grid position, Blue ▲: Actual sampling position (grid + offset)')
    ax.axis('off')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Grid Position'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='b', markersize=8, label='Sampling Position'),
        Line2D([0], [0], color='g', linewidth=2, label='Offset Direction')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_actual_scanpath(image, offset, save_path, stage_name="",
                              max_points=200, line_width=1.5):
    """
    실제 Mamba Scan Path 시각화
    
    파란 점(샘플링 위치)들을 Mamba 처리 순서대로 연결
    이것이 실제 SSM이 보는 "sequence"
    
    Raster scan order: 왼쪽→오른쪽, 위→아래로 그리드를 순회하면서
    각 위치의 샘플링 포인트를 순서대로 연결
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    H, W = offset.shape[:2]
    img_h, img_w = image.shape[:2]
    scale_h = img_h / H
    scale_w = img_w / W
    
    # 첫 번째 offset 사용
    dx = offset[:, :, 0].numpy()
    dy = offset[:, :, 1].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # === Left: Full scan path ===
    ax = axes[0]
    ax.imshow(image)
    
    # 샘플링 포인트들을 raster scan 순서로 수집
    sampling_points = []
    for j in range(H):
        for i in range(W):
            # 샘플링 위치 (이미지 좌표)
            sample_x = (i + dx[j, i]) * scale_w + scale_w / 2
            sample_y = (j + dy[j, i]) * scale_h + scale_h / 2
            sampling_points.append((sample_x, sample_y))
    
    sampling_points = np.array(sampling_points)
    
    # 너무 많으면 서브샘플링
    if len(sampling_points) > max_points:
        indices = np.linspace(0, len(sampling_points)-1, max_points, dtype=int)
        sampling_points_sub = sampling_points[indices]
    else:
        sampling_points_sub = sampling_points
    
    # 순서대로 연결 (색상 gradient로 순서 표현)
    n_points = len(sampling_points_sub)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    for i in range(n_points - 1):
        ax.plot([sampling_points_sub[i, 0], sampling_points_sub[i+1, 0]],
               [sampling_points_sub[i, 1], sampling_points_sub[i+1, 1]],
               '-', color=colors[i], linewidth=line_width, alpha=0.7)
    
    # 시작점과 끝점 표시
    ax.plot(sampling_points_sub[0, 0], sampling_points_sub[0, 1], 
           'go', markersize=12, label='Start', zorder=5)
    ax.plot(sampling_points_sub[-1, 0], sampling_points_sub[-1, 1], 
           'rs', markersize=12, label='End', zorder=5)
    
    ax.set_title(f'Actual Mamba Scan Path - {stage_name}\n'
                f'(Color: dark→light = scan order, {n_points} points shown)')
    ax.legend(loc='upper right')
    ax.axis('off')
    
    # === Right: Zoomed region ===
    ax = axes[1]
    ax.imshow(image)
    
    # 중앙 영역만 확대해서 보기
    center_h, center_w = H // 2, W // 2
    region_size = min(H, W) // 3
    
    start_j = max(0, center_h - region_size)
    end_j = min(H, center_h + region_size)
    start_i = max(0, center_w - region_size)
    end_i = min(W, center_w + region_size)
    
    # 해당 영역의 샘플링 포인트들
    zoom_points = []
    zoom_indices = []
    idx = 0
    for j in range(H):
        for i in range(W):
            if start_j <= j < end_j and start_i <= i < end_i:
                sample_x = (i + dx[j, i]) * scale_w + scale_w / 2
                sample_y = (j + dy[j, i]) * scale_h + scale_h / 2
                zoom_points.append((sample_x, sample_y))
                zoom_indices.append(idx)
            idx += 1
    
    zoom_points = np.array(zoom_points)
    
    # 순서대로 연결
    n_zoom = len(zoom_points)
    colors_zoom = plt.cm.plasma(np.linspace(0, 1, n_zoom))
    
    for i in range(n_zoom - 1):
        ax.plot([zoom_points[i, 0], zoom_points[i+1, 0]],
               [zoom_points[i, 1], zoom_points[i+1, 1]],
               '-', color=colors_zoom[i], linewidth=2, alpha=0.8)
    
    # 포인트들 표시
    ax.scatter(zoom_points[:, 0], zoom_points[:, 1], 
              c=np.arange(n_zoom), cmap='plasma', s=30, zorder=5)
    
    # 숫자로 순서 표시 (일부만)
    for i in range(0, n_zoom, max(1, n_zoom // 10)):
        ax.annotate(str(i), (zoom_points[i, 0], zoom_points[i, 1]),
                   fontsize=8, color='white', ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
    
    # Zoom 영역 표시
    rect_x = start_i * scale_w
    rect_y = start_j * scale_h
    rect_w = (end_i - start_i) * scale_w
    rect_h = (end_j - start_j) * scale_h
    ax.add_patch(patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                   linewidth=2, edgecolor='yellow', 
                                   facecolor='none', linestyle='--'))
    
    ax.set_xlim(rect_x - 10, rect_x + rect_w + 10)
    ax.set_ylim(rect_y + rect_h + 10, rect_y - 10)
    ax.set_title(f'Zoomed Scan Path (Center Region)\n'
                f'Numbers show processing order')
    ax.axis('off')
    
    plt.suptitle(f'Mamba Sequential Processing Order - {stage_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_sampling_density(image, offset, save_path, stage_name=""):
    """
    샘플링 밀도를 heatmap으로 시각화
    어느 영역이 많이 샘플링되는지 (attention과 유사한 개념)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    H, W = offset.shape[:2]
    img_h, img_w = image.shape[:2]
    
    # 첫 번째 offset 사용
    dx = offset[:, :, 0].numpy()
    dy = offset[:, :, 1].numpy()
    
    # 샘플링 밀도 계산 (각 위치가 몇 번 샘플링되는지)
    density = np.zeros((H, W), dtype=np.float32)
    
    for j in range(H):
        for i in range(W):
            # 샘플링 위치
            sample_x = int(np.clip(np.round(i + dx[j, i]), 0, W-1))
            sample_y = int(np.clip(np.round(j + dy[j, i]), 0, H-1))
            density[sample_y, sample_x] += 1
    
    # Normalize
    density = density / density.max() if density.max() > 0 else density
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Sampling density heatmap
    density_resized = cv2.resize(density, (img_w, img_h))
    im = axes[1].imshow(density_resized, cmap='hot')
    axes[1].set_title('Sampling Density\n(Where features are sampled FROM)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(density_resized, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'Sampling Density Analysis - {stage_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_offset_heatmap(image, offset, save_path, stage_name=""):
    """
    Offset magnitude를 heatmap으로 시각화
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    H, W = offset.shape[:2]
    num_offsets = offset.shape[2] // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Magnitude heatmap (average over all offsets)
    magnitudes = []
    for idx in range(num_offsets):
        dx = offset[:, :, idx * 2].numpy()
        dy = offset[:, :, idx * 2 + 1].numpy()
        mag = np.sqrt(dx**2 + dy**2)
        magnitudes.append(mag)
    
    avg_magnitude = np.mean(magnitudes, axis=0)
    
    # Resize magnitude to image size
    avg_magnitude_resized = cv2.resize(avg_magnitude, (image.shape[1], image.shape[0]))
    
    im = axes[0, 1].imshow(avg_magnitude_resized, cmap='hot')
    axes[0, 1].set_title('Offset Magnitude (Average)')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
    
    # X direction
    dx_avg = np.mean([offset[:, :, idx * 2].numpy() for idx in range(num_offsets)], axis=0)
    dx_resized = cv2.resize(dx_avg, (image.shape[1], image.shape[0]))
    im = axes[1, 0].imshow(dx_resized, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 0].set_title('X Offset (horizontal)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Y direction
    dy_avg = np.mean([offset[:, :, idx * 2 + 1].numpy() for idx in range(num_offsets)], axis=0)
    dy_resized = cv2.resize(dy_avg, (image.shape[1], image.shape[0]))
    im = axes[1, 1].imshow(dy_resized, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 1].set_title('Y Offset (vertical)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    plt.suptitle(f'Offset Analysis - {stage_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def process_single_image(model, extractor, img_pil, image_name, output_dir, device, 
                         base_height=32, max_ratio=12):
    """단일 이미지에 대해 시각화 수행 (Dynamic Width 지원)"""
    
    # Preprocess image with Dynamic Width (학습과 동일하게)
    w, h = img_pil.size
    ratio = w / float(h)
    
    # Dynamic width 계산
    target_h = base_height
    if ratio > max_ratio:
        target_w = base_height * max_ratio  # max_ratio 제한
    else:
        target_w = int(math.ceil(base_height * ratio))
    
    # 4의 배수로 맞춤 (divided_factor)
    target_w = max(32, (target_w // 4) * 4)
    
    print(f"    Dynamic resize: {w}x{h} → {target_w}x{target_h} (ratio: {ratio:.2f})")
    
    img_resized = img_pil.resize((target_w, target_h), Image.BILINEAR)
    new_img = img_resized  # padding 없이 동적 크기 사용
    
    # To tensor
    img_array = np.array(new_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Extract offsets
    print(f"Extracting offsets for: {image_name}")
    offsets, features = extractor.extract(img_tensor)
    
    if not offsets:
        print("Warning: No offsets extracted. Make sure the model has DASSM layers.")
        return False
    
    # Visualize each stage
    for stage_name, offset in offsets.items():
        print(f"  Visualizing: {stage_name}")
        print(f"    Offset shape: {offset.shape}")
        
        # Remove batch dimension
        offset = offset[0]  # (H, W, num_offsets*2)
        
        # Offset 통계 출력
        num_offsets = offset.shape[-1] // 2
        print(f"    Offset statistics ({num_offsets} groups):")
        for i in range(min(4, num_offsets)):  # 최대 4개 group만 출력
            dx = offset[:, :, i*2].numpy()
            dy = offset[:, :, i*2+1].numpy()
            magnitude = np.sqrt(dx**2 + dy**2)
            print(f"      Group {i+1}:")
            print(f"        dx: mean={dx.mean():+.3f}, std={dx.std():.3f}, range=[{dx.min():.2f}, {dx.max():.2f}]")
            print(f"        dy: mean={dy.mean():+.3f}, std={dy.std():.3f}, range=[{dy.min():.2f}, {dy.max():.2f}]")
            print(f"        magnitude: mean={magnitude.mean():.3f}, max={magnitude.max():.3f}")
        
        # Clean stage name for filename
        clean_name = stage_name.replace('.', '_').replace('/', '_')
        
        # 1. Actual Scan Path (파란 점들의 처리 순서) - 핵심!
        save_path = os.path.join(output_dir, f'{image_name}_{clean_name}_scanpath.png')
        visualize_actual_scanpath(new_img, offset, save_path, stage_name)
        
        # 2. Sampling points (그리드 → 샘플링 위치)
        save_path = os.path.join(output_dir, f'{image_name}_{clean_name}_sampling.png')
        visualize_sampling_points(new_img, offset, save_path, stage_name)
        
        # 3. Arrow visualization (offset 방향)
        save_path = os.path.join(output_dir, f'{image_name}_{clean_name}_arrows.png')
        visualize_offset_arrows(new_img, offset, save_path, stage_name)
        
        # 4. Sampling density (샘플링 밀도)
        save_path = os.path.join(output_dir, f'{image_name}_{clean_name}_density.png')
        visualize_sampling_density(new_img, offset, save_path, stage_name)
        
        # 5. Heatmap (offset 분석)
        save_path = os.path.join(output_dir, f'{image_name}_{clean_name}_heatmap.png')
        visualize_offset_heatmap(new_img, offset, save_path, stage_name)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='SVTRv2Mamba Scan Path Visualization')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (optional)')
    
    # 이미지 소스 옵션 (둘 중 하나 선택)
    parser.add_argument('--image', '-i', type=str, default=None,
                       help='Path to input image file')
    parser.add_argument('--lmdb', type=str, default=None,
                       help='Path to LMDB dataset (e.g., /workspace/evaluation/CUTE80)')
    parser.add_argument('--lmdb_index', type=int, default=1,
                       help='Start index for LMDB extraction (default: 1)')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Number of images to extract from LMDB (default: 1)')
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available LMDB datasets and exit')
    
    parser.add_argument('--output_dir', '-o', type=str, default='./visualization_output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    args = parser.parse_args()
    
    # 데이터셋 목록만 출력
    if args.list_datasets:
        list_available_lmdb_datasets()
        return
    
    # 이미지 소스 확인
    if args.image is None and args.lmdb is None:
        print("Error: Either --image or --lmdb must be specified")
        print("\nExamples:")
        print("  # Using image file:")
        print("  python tools/visualize_scanpath.py -c config.yml --image test.png")
        print("\n  # Using LMDB dataset:")
        print("  python tools/visualize_scanpath.py -c config.yml --lmdb /workspace/evaluation/CUTE80")
        print("\n  # List available datasets:")
        print("  python tools/visualize_scanpath.py -c config.yml --list_datasets")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    cfg = Config(args.config)
    
    # Visualization은 inference 모드가 아닌 training 구조로 빌드
    # infer_gtc=True면 out_channels가 리스트여야 하는데, config에는 int로 되어 있음
    if 'Decoder' in cfg.cfg['Architecture']:
        decoder_cfg = cfg.cfg['Architecture']['Decoder']
        if decoder_cfg.get('name') == 'GTCDecoder' and decoder_cfg.get('infer_gtc', False):
            print("Note: Setting infer_gtc=False for visualization (encoder-only)")
            decoder_cfg['infer_gtc'] = False
    
    # Checkpoint에서 d_state 추론 (config와 checkpoint 불일치 방지)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        ckpt_state = ckpt.get('state_dict', ckpt)
        
        # A_logs shape에서 d_state 추론
        for key in ckpt_state:
            if 'A_logs' in key:
                d_state_from_ckpt = ckpt_state[key].shape[-1]
                encoder_cfg = cfg.cfg['Architecture'].get('Encoder', {})
                config_d_state = encoder_cfg.get('d_state', 1)
                
                if config_d_state != d_state_from_ckpt:
                    print(f"Note: Adjusting d_state from {config_d_state} to {d_state_from_ckpt} (from checkpoint)")
                    encoder_cfg['d_state'] = d_state_from_ckpt
                break
    
    # Build model
    print("Building model...")
    model = build_model(cfg.cfg['Architecture'])
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Visualization은 encoder만 필요하므로 decoder 가중치와 shape mismatch 필터링
        model_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    skipped_keys.append(f"{key}: ckpt {value.shape} vs model {model_state[key].shape}")
            else:
                skipped_keys.append(f"{key}: not in model")
        
        if skipped_keys:
            print(f"Note: Skipped {len(skipped_keys)} keys with shape mismatch (decoder weights)")
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)} / {len(state_dict)} weights (encoder weights)")
    else:
        print("Warning: No checkpoint loaded, using random weights")
    
    model = model.to(args.device)
    model.eval()
    
    # Config에서 해상도 설정 추출
    try:
        sampler_cfg = cfg.cfg.get('Train', {}).get('sampler', {})
        scales = sampler_cfg.get('scales', [[128, 32]])  # [width, height]
        base_height = scales[0][1] if scales else 32
        max_ratio = cfg.cfg.get('Train', {}).get('loader', {}).get('max_ratio', 12)
        print(f"📐 Resolution from config: height={base_height}, max_ratio={max_ratio}")
    except Exception as e:
        print(f"Warning: Could not extract resolution from config: {e}")
        base_height = 32
        max_ratio = 12
    
    # Create extractor
    extractor = ScanPathExtractor(model)
    
    # 이미지 수집
    images_to_process = []
    
    if args.lmdb:
        # LMDB에서 이미지 추출
        print(f"\n📦 Extracting images from LMDB: {args.lmdb}")
        extracted = extract_images_from_lmdb(args.lmdb, args.lmdb_index, args.num_images)
        for img_pil, label, idx in extracted:
            # 파일명에 사용할 수 없는 문자 제거
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            image_name = f"lmdb_{idx:04d}_{safe_label}"
            images_to_process.append((img_pil, image_name, label))
    else:
        # 이미지 파일 로드
        print(f"\n📷 Loading image: {args.image}")
        img_pil = Image.open(args.image).convert('RGB')
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        images_to_process.append((img_pil, image_name, None))
    
    # 각 이미지 처리
    print(f"\n🎨 Processing {len(images_to_process)} image(s)...")
    
    for i, (img_pil, image_name, label) in enumerate(images_to_process):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(images_to_process)}] {image_name}")
        if label:
            print(f"    Label: '{label}'")
        print(f"    Size: {img_pil.size[0]}x{img_pil.size[1]}")
        print(f"{'='*60}")
        
        success = process_single_image(
            model, extractor, img_pil, image_name, 
            args.output_dir, args.device,
            base_height=base_height, max_ratio=max_ratio
        )
        
        if not success:
            print(f"    ⚠️ Failed to process {image_name}")
    
    # Cleanup
    extractor.remove_hooks()
    
    print(f"\n{'='*60}")
    print(f"✅ Visualization complete!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
