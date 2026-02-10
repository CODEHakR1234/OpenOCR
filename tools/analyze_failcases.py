#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fail Case Analysis Script for OCR Models

데이터셋별로 틀린 예측을 분석하고 저장합니다.

Usage:
    # Common benchmarks (CUTE80, IC13, IC15, IIIT5k, SVT, SVTP)
    python tools/analyze_failcases.py \
        --config configs/rec/svtrv2/ablation/svtrv2_mamba_deep_668_ctc_h64.yml \
        --checkpoint output/rec/u14m_filter/ablation/svtrv2_mamba_deep_668_ctc_h64/best.pth \
        --output_dir ./failcase_analysis \
        --benchmark_type common
    
    # Union14M-Benchmark (7 categories: Artistic, Contextless, Curve, General, MultiOriented, MultiWords, Salient)
    python tools/analyze_failcases.py \
        --config configs/rec/svtrv2/ablation/svtrv2_mamba_deep_668_ctc_h64.yml \
        --checkpoint output/rec/u14m_filter/ablation/svtrv2_mamba_deep_668_ctc_h64/best.pth \
        --output_dir ./failcase_analysis_u14m \
        --benchmark_type u14m
    
    # Specific U14M categories only
    python tools/analyze_failcases.py \
        --config ... \
        --benchmark_type u14m \
        --datasets U14M-Curve U14M-MultiOriented
    
    # All benchmarks (common + u14m + other)
    python tools/analyze_failcases.py \
        --config ... \
        --benchmark_type all
"""

import os
import sys
import argparse
import json
import copy
import string
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import Levenshtein

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine import Config
from tools.data import build_dataloader
from tools.utils.logging import get_logger
from openrec.modeling import build_model
from openrec.postprocess import build_post_process

# Initialize logger
logger = get_logger()


# Common English Benchmarks (6 standard)
COMMON_BENCHMARKS = {
    'CUTE80': '../evaluation/CUTE80',
    'IC13_857': '../evaluation/IC13_857',
    'IC15_1811': '../evaluation/IC15_1811',
    'IIIT5k': '../evaluation/IIIT5k',
    'SVT': '../evaluation/SVT',
    'SVTP': '../evaluation/SVTP',
}

# Union14M-Benchmark (7 categories)
U14M_BENCHMARKS = {
    'U14M-Artistic': '../u14m/artistic',
    'U14M-Contextless': '../u14m/contextless',
    'U14M-Curve': '../u14m/curve',
    'U14M-General': '../u14m/general',
    'U14M-MultiOriented': '../u14m/multi_oriented',
    'U14M-MultiWords': '../u14m/multi_words',
    'U14M-Salient': '../u14m/salient',
}

# Additional test sets
OTHER_BENCHMARKS = {
    'ArT': '../OpenOCR-Data/test/ArT',
    'WordArt': '../OpenOCR-Data/wordart_test',
}

# All benchmarks combined
ALL_BENCHMARKS = {**COMMON_BENCHMARKS, **U14M_BENCHMARKS, **OTHER_BENCHMARKS}

# Backward compatibility
BENCHMARKS = COMMON_BENCHMARKS


def normalize_text(text, ignore_case=True, ignore_space=True, filter_symbol=True):
    """텍스트 정규화 (공식 eval과 동일)
    
    Args:
        text: 입력 텍스트
        ignore_case: 대소문자 무시 (default: True)
        ignore_space: 공백 무시 (default: True)
        filter_symbol: 특수문자 필터링 - 숫자/알파벳만 유지 (default: True)
    """
    if ignore_space:
        text = text.replace(' ', '')
    if filter_symbol:
        # 공식 metric의 _normalize_text와 동일: 숫자와 알파벳만 유지
        text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
    if ignore_case:
        text = text.lower()
    return text


def normalized_edit_distance(pred, gt):
    """정규화된 편집 거리 계산"""
    dist = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    return dist / max_len


def analyze_error_type(pred, gt):
    """에러 유형 분석: 삽입, 삭제, 치환"""
    try:
        ops = Levenshtein.editops(pred, gt)
        error_types = {'insert': 0, 'delete': 0, 'replace': 0}
        for op, _, _ in ops:
            error_types[op] += 1
        return error_types
    except:
        return {'insert': 0, 'delete': 0, 'replace': 0}


def get_confusion_pairs(pred, gt):
    """혼동 문자쌍 추출"""
    pairs = []
    try:
        ops = Levenshtein.editops(pred, gt)
        for op, src_pos, dest_pos in ops:
            if op == 'replace':
                if src_pos < len(pred) and dest_pos < len(gt):
                    pairs.append((pred[src_pos], gt[dest_pos]))
    except:
        pass
    return pairs


class FailCaseAnalyzer:
    def __init__(self, config_path, checkpoint_path, output_dir, device='cuda'):
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)
        
        # Load config
        self.cfg = Config(config_path)
        
        # Build model
        print("Building model...")
        self.post_process = build_post_process(self.cfg.cfg['PostProcess'], 
                                                self.cfg.cfg['Global'])
        char_num = self.post_process.get_character_num()
        self.cfg.cfg['Architecture']['Decoder']['out_channels'] = char_num
        
        self.model = build_model(self.cfg.cfg['Architecture'])
        
        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(device)
        self.model.eval()
        
    def evaluate_dataset(self, dataset_name, dataset_path, save_images=False):
        """단일 데이터셋 평가 및 fail case 수집"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"Path: {dataset_path}")
        print(f"{'='*60}")
        
        if not os.path.exists(dataset_path):
            print(f"  ⚠️ Dataset not found, skipping...")
            return None
        
        # Update config for this dataset (deep copy to avoid modifying original)
        config_copy = copy.deepcopy(self.cfg.cfg)
        
        # Check if using RatioDataSet (MSR)
        if 'RatioDataSet' in config_copy['Eval']['dataset']['name']:
            config_copy['Eval']['dataset']['data_dir_list'] = [dataset_path]
        else:
            config_copy['Eval']['dataset']['data_dir'] = dataset_path
        
        # Build dataloader
        dataloader = build_dataloader(config_copy, 'Eval', logger)
        
        results = {
            'dataset': dataset_name,
            'total': 0,
            'correct': 0,
            'wrong': 0,
            'accuracy': 0.0,
            'failcases': [],
            'error_types': {'insert': 0, 'delete': 0, 'replace': 0},
            'confusion_pairs': defaultdict(int),
            'length_stats': defaultdict(lambda: {'total': 0, 'correct': 0}),
        }
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(self.output_dir, dataset_name)
        if save_images:
            os.makedirs(dataset_output_dir, exist_ok=True)
        
        sample_idx = 0
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                    if len(batch) > 1:
                        labels = batch[1] if isinstance(batch[1], list) else None
                else:
                    images = batch
                    labels = None
                
                if isinstance(images, torch.Tensor):
                    images = images.to(self.device)
                
                # Forward pass
                preds = self.model(images)
                
                # Post process - don't pass batch to avoid type issues
                post_result = self.post_process(preds)
                
                # Handle GTC format [gtc_results, ctc_results]
                if isinstance(post_result, list) and len(post_result) == 2:
                    if isinstance(post_result[1], list):
                        pred_results = post_result[1]  # Use CTC results
                    else:
                        pred_results = post_result
                else:
                    pred_results = post_result
                
                # Get ground truth labels from batch using postprocess decode
                gt_texts = []
                if len(batch) >= 2:
                    label_indices = batch[1]
                    
                    # Use postprocess's decode method for consistency
                    if hasattr(self.post_process, 'decode'):
                        try:
                            # Convert to numpy if tensor
                            if isinstance(label_indices, torch.Tensor):
                                label_indices_np = label_indices.cpu().numpy()
                            else:
                                label_indices_np = label_indices
                            
                            decoded = self.post_process.decode(label_indices_np)
                            gt_texts = decoded
                        except Exception as e:
                            # Fallback: manual decoding
                            if hasattr(self.post_process, 'character') and len(batch) >= 3:
                                char_list = self.post_process.character
                                label_lengths = batch[2]
                                for b_idx in range(len(label_lengths)):
                                    length = int(label_lengths[b_idx])
                                    if isinstance(label_indices, torch.Tensor):
                                        indices = label_indices[b_idx][:length].cpu().numpy()
                                    else:
                                        indices = label_indices[b_idx][:length]
                                    text = ''.join([char_list[idx] for idx in indices if 0 < idx < len(char_list)])
                                    gt_texts.append((text, 1.0))
                            else:
                                gt_texts = [('', 1.0)] * len(pred_results)
                    else:
                        gt_texts = [('', 1.0)] * len(pred_results)
                else:
                    gt_texts = [('', 1.0)] * len(pred_results)
                
                # Compare predictions with ground truth
                for i, (pred_item, gt_item) in enumerate(zip(pred_results, gt_texts)):
                    # Extract text from results
                    if isinstance(pred_item, (list, tuple)):
                        pred_text = pred_item[0]
                        pred_conf = pred_item[1] if len(pred_item) > 1 else 0.0
                    else:
                        pred_text = str(pred_item)
                        pred_conf = 0.0
                    
                    if isinstance(gt_item, (list, tuple)):
                        gt_text = gt_item[0]
                    else:
                        gt_text = str(gt_item)
                    
                    # Normalize for comparison
                    pred_norm = normalize_text(pred_text)
                    gt_norm = normalize_text(gt_text)
                    
                    results['total'] += 1
                    text_len = len(gt_text)
                    results['length_stats'][text_len]['total'] += 1
                    
                    if pred_norm == gt_norm:
                        results['correct'] += 1
                        results['length_stats'][text_len]['correct'] += 1
                    else:
                        results['wrong'] += 1
                        
                        # Analyze error
                        edit_dist = Levenshtein.distance(pred_norm, gt_norm)
                        norm_edit_dist = normalized_edit_distance(pred_norm, gt_norm)
                        error_types = analyze_error_type(pred_norm, gt_norm)
                        confusion_pairs = get_confusion_pairs(pred_norm, gt_norm)
                        
                        # Update stats
                        for err_type, count in error_types.items():
                            results['error_types'][err_type] += count
                        
                        for pair in confusion_pairs:
                            results['confusion_pairs'][pair] += 1
                        
                        # Save fail case info
                        failcase = {
                            'index': sample_idx,
                            'gt': gt_text,
                            'pred': pred_text,
                            'confidence': float(pred_conf),
                            'edit_distance': edit_dist,
                            'normalized_edit_distance': float(norm_edit_dist),
                            'error_types': error_types,
                            'gt_length': len(gt_text),
                        }
                        results['failcases'].append(failcase)
                        
                        # Print fail case
                        print(f"  [{sample_idx}] GT: '{gt_text}' | Pred: '{pred_text}' | ED: {edit_dist}")
                    
                    sample_idx += 1
        
        # Calculate accuracy
        if results['total'] > 0:
            results['accuracy'] = results['correct'] / results['total'] * 100
        
        # Convert defaultdict to regular dict for JSON serialization
        # confusion_pairs has tuple keys like ('a', 'b'), convert to string "a->b"
        results['confusion_pairs'] = {f"{k[0]}->{k[1]}": v for k, v in results['confusion_pairs'].items()}
        results['length_stats'] = {str(k): dict(v) for k, v in results['length_stats'].items()}
        
        print(f"\n📊 {dataset_name} Results:")
        print(f"   Total: {results['total']}")
        print(f"   Correct: {results['correct']}")
        print(f"   Wrong: {results['wrong']}")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   Error Types: Insert={results['error_types']['insert']}, "
              f"Delete={results['error_types']['delete']}, "
              f"Replace={results['error_types']['replace']}")
        
        return results
    
    def run_analysis(self, benchmarks=None, save_images=False):
        """모든 벤치마크에 대해 분석 실행"""
        if benchmarks is None:
            benchmarks = BENCHMARKS
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': str(self.cfg.cfg.get('Global', {}).get('config', '')),
            'datasets': {},
            'summary': {
                'total': 0,
                'correct': 0,
                'wrong': 0,
                'accuracy': 0.0,
            }
        }
        
        for dataset_name, dataset_path in benchmarks.items():
            result = self.evaluate_dataset(dataset_name, dataset_path, save_images)
            if result:
                all_results['datasets'][dataset_name] = result
                all_results['summary']['total'] += result['total']
                all_results['summary']['correct'] += result['correct']
                all_results['summary']['wrong'] += result['wrong']
        
        # Calculate overall accuracy
        if all_results['summary']['total'] > 0:
            all_results['summary']['accuracy'] = (
                all_results['summary']['correct'] / all_results['summary']['total'] * 100
            )
        
        # Print summary
        print(f"\n{'='*60}")
        print("📈 OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Samples: {all_results['summary']['total']}")
        print(f"Correct: {all_results['summary']['correct']}")
        print(f"Wrong: {all_results['summary']['wrong']}")
        print(f"Overall Accuracy: {all_results['summary']['accuracy']:.2f}%")
        
        print(f"\n📊 Per-Dataset Accuracy:")
        for name, result in all_results['datasets'].items():
            print(f"   {name}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results):
        """결과 저장"""
        # Save full JSON results
        json_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full results saved to: {json_path}")
        
        # Save human-readable summary
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Fail Case Analysis Report\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"OVERALL SUMMARY\n")
            f.write(f"-" * 40 + "\n")
            f.write(f"Total: {results['summary']['total']}\n")
            f.write(f"Correct: {results['summary']['correct']}\n")
            f.write(f"Wrong: {results['summary']['wrong']}\n")
            f.write(f"Accuracy: {results['summary']['accuracy']:.2f}%\n\n")
            
            for dataset_name, data in results['datasets'].items():
                f.write(f"\n{'='*60}\n")
                f.write(f"{dataset_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Accuracy: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})\n")
                f.write(f"Error Types: I={data['error_types']['insert']}, "
                       f"D={data['error_types']['delete']}, "
                       f"R={data['error_types']['replace']}\n\n")
                
                f.write(f"Fail Cases ({len(data['failcases'])}):\n")
                f.write(f"-" * 40 + "\n")
                for fc in data['failcases']:
                    f.write(f"[{fc['index']}] GT: '{fc['gt']}' | Pred: '{fc['pred']}' | ED: {fc['edit_distance']}\n")
                
                # Top confusion pairs
                if data['confusion_pairs']:
                    f.write(f"\nTop Confusion Pairs:\n")
                    sorted_pairs = sorted(data['confusion_pairs'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]
                    for pair_str, count in sorted_pairs:
                        f.write(f"  {pair_str}: {count}\n")
        
        print(f"📄 Summary saved to: {summary_path}")
        
        # Save per-dataset fail case files
        for dataset_name, data in results['datasets'].items():
            failcase_path = os.path.join(self.output_dir, f'{dataset_name}_failcases.txt')
            with open(failcase_path, 'w', encoding='utf-8') as f:
                f.write(f"# {dataset_name} Fail Cases\n")
                f.write(f"# Accuracy: {data['accuracy']:.2f}%\n")
                f.write(f"# Format: index | GT | Pred | EditDistance | Confidence\n\n")
                for fc in data['failcases']:
                    f.write(f"{fc['index']}\t{fc['gt']}\t{fc['pred']}\t{fc['edit_distance']}\t{fc['confidence']:.4f}\n")
            print(f"📝 {dataset_name} fail cases: {failcase_path}")


def main():
    parser = argparse.ArgumentParser(description='Fail Case Analysis for OCR Models')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--output_dir', '-o', type=str, default='./failcase_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save_images', action='store_true',
                       help='Save images of fail cases')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to analyze (default: all)')
    parser.add_argument('--benchmark_type', type=str, default='common',
                       choices=['common', 'u14m', 'other', 'all'],
                       help='Benchmark type: common (6 standard), u14m (Union14M-Benchmark 7 categories), other (ArT/WordArt), all')
    args = parser.parse_args()
    
    # Filter specific datasets if provided (search across ALL benchmarks)
    if args.datasets:
        benchmarks = {k: v for k, v in ALL_BENCHMARKS.items() if k in args.datasets}
    else:
        # Select benchmarks based on type
        if args.benchmark_type == 'common':
            benchmarks = COMMON_BENCHMARKS
        elif args.benchmark_type == 'u14m':
            benchmarks = U14M_BENCHMARKS
        elif args.benchmark_type == 'other':
            benchmarks = OTHER_BENCHMARKS
        else:  # all
            benchmarks = ALL_BENCHMARKS
    
    print(f"🔍 Fail Case Analysis")
    print(f"   Config: {args.config}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Output: {args.output_dir}")
    print(f"   Datasets: {list(benchmarks.keys())}")
    
    analyzer = FailCaseAnalyzer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )
    
    analyzer.run_analysis(benchmarks=benchmarks, save_images=args.save_images)


if __name__ == '__main__':
    main()
