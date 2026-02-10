#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qualitative Fail Case Visualization

기존 분석 결과(JSON)를 바탕으로 이미지 + GT + Prediction 정성 분석
GT 텍스트로 LMDB 샘플을 매칭

Usage:
    python tools/visualize_failcases_qualitative.py \
        --json ./failcase_analysis_multioriented/analysis_results.json \
        --lmdb ../u14m/multi_oriented \
        --output_dir ./failcase_qualitative \
        --num_samples 50 \
        --cols 5
"""

import os
import sys
import argparse
import io
import json
import math

import matplotlib.pyplot as plt
from PIL import Image
import lmdb


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
        print(f"   Building label index map for {num_samples} samples...")
        
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
    """LMDB에서 특정 인덱스의 이미지와 라벨 가져오기"""
    with env.begin(write=False) as txn:
        img_key = f'image-{index:09d}'.encode()
        label_key = f'label-{index:09d}'.encode()
        
        img_data = txn.get(img_key)
        label_data = txn.get(label_key)
        
        if img_data is None:
            return None, None
        
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        label = label_data.decode() if label_data else ""
        return img, label


def create_grid_visualization(cases, lmdb_env, label_to_indices, output_dir, prefix, cols, title=""):
    """그리드 시각화 생성"""
    if not cases:
        print(f"No cases to visualize for {prefix}")
        return
    
    num_cases = len(cases)
    
    # 여러 페이지로 나누기 (페이지당 최대 25개)
    cases_per_page = cols * 5  # 5 rows per page
    num_pages = math.ceil(num_cases / cases_per_page)
    
    matched = 0
    not_matched = 0
    
    for page in range(num_pages):
        start_idx = page * cases_per_page
        end_idx = min(start_idx + cases_per_page, num_cases)
        page_cases = cases[start_idx:end_idx]
        
        page_rows = math.ceil(len(page_cases) / cols)
        
        fig = plt.figure(figsize=(cols * 4, page_rows * 2.8))
        
        for i, case in enumerate(page_cases):
            ax = fig.add_subplot(page_rows, cols, i + 1)
            
            gt = case.get('gt', '')
            pred = case.get('pred', '')
            ed = case.get('edit_distance', '?')
            
            # GT로 LMDB 인덱스 찾기
            img = None
            lmdb_idx = None
            
            if gt in label_to_indices:
                indices = label_to_indices[gt]
                lmdb_idx = indices[0]  # 첫 번째 매칭 사용
                img, _ = get_image_from_lmdb(lmdb_env, lmdb_idx)
                matched += 1
            else:
                not_matched += 1
            
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f'Image Not Found\nGT: {gt[:15]}...', 
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow'))
            
            ax.axis('off')
            
            # 제목: GT / Pred
            gt_display = gt[:20] + '...' if len(gt) > 20 else gt
            pred_display = pred[:20] + '...' if len(pred) > 20 else pred
            
            idx_str = f"#{lmdb_idx}" if lmdb_idx else f"[{case['index']}]"
            
            ax.set_title(
                f"✗ {idx_str} ED:{ed}\n"
                f"GT: {gt_display}\n"
                f"Pred: {pred_display}",
                fontsize=8,
                color='red',
                loc='left',
                fontfamily='monospace'
            )
        
        plt.suptitle(f"{title} (Page {page + 1}/{num_pages})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{prefix}_page{page + 1:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    print(f"   Matched: {matched}, Not matched: {not_matched}")


def main():
    parser = argparse.ArgumentParser(description='Qualitative Fail Case Visualization from JSON')
    parser.add_argument('--json', '-j', type=str, required=True,
                       help='Path to analysis_results.json')
    parser.add_argument('--lmdb', type=str, required=True,
                       help='Path to LMDB dataset')
    parser.add_argument('--output_dir', '-o', type=str, default='./failcase_qualitative',
                       help='Output directory')
    parser.add_argument('--num_samples', '-n', type=int, default=50,
                       help='Number of samples to visualize')
    parser.add_argument('--cols', type=int, default=5,
                       help='Number of columns in grid')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Specific dataset name (default: first in JSON)')
    args = parser.parse_args()
    
    print(f"🔍 Qualitative Fail Case Visualization")
    print(f"   JSON: {args.json}")
    print(f"   LMDB: {args.lmdb}")
    print(f"   Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load analysis results
    print("\n📊 Loading analysis results...")
    results = load_analysis_results(args.json)
    
    # Get dataset to visualize
    datasets = results.get('datasets', {})
    if not datasets:
        print("Error: No datasets found in JSON")
        return
    
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
            return
        dataset_name = args.dataset
    else:
        dataset_name = list(datasets.keys())[0]
    
    dataset_data = datasets[dataset_name]
    failcases = dataset_data.get('failcases', [])
    
    print(f"\n📈 Dataset: {dataset_name}")
    print(f"   Total: {dataset_data['total']}")
    print(f"   Correct: {dataset_data['correct']}")
    print(f"   Wrong: {dataset_data['wrong']}")
    print(f"   Accuracy: {dataset_data['accuracy']:.2f}%")
    print(f"   Fail cases to visualize: {min(args.num_samples, len(failcases))}")
    
    # Check LMDB
    if not os.path.exists(args.lmdb):
        print(f"Error: LMDB path not found: {args.lmdb}")
        return
    
    # Build label -> index map
    print(f"\n🗂️ Building label-to-index map from LMDB...")
    label_to_indices = build_label_to_index_map(args.lmdb)
    print(f"   Unique labels: {len(label_to_indices)}")
    
    # Open LMDB for image retrieval
    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    
    # Visualize fail cases
    print(f"\n🎨 Creating visualizations...")
    cases_to_show = failcases[:args.num_samples]
    
    create_grid_visualization(
        cases_to_show,
        env,
        label_to_indices,
        args.output_dir,
        f"{dataset_name}_failcases",
        args.cols,
        title=f"{dataset_name} Fail Cases ({len(failcases)} total, showing {len(cases_to_show)})"
    )
    
    env.close()
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
