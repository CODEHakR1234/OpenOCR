#!/usr/bin/env python
"""
Simple FPS measurement script for OCR models.
Usage:
    python tools/measure_fps.py -c CONFIG_FILE -o Global.pretrained_model=PATH
"""

import os
import sys
import time
import torch
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.utility import ArgsParser
from tools.engine import Config
from openrec.modeling import build_model
from openrec.preprocess import create_operators, transform


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Fix GTCDecoder issue for inference
    if cfg['Architecture'].get('Decoder', {}).get('name') == 'GTCDecoder':
        cfg['Architecture']['Decoder']['infer_gtc'] = False
    
    # Build model
    model = build_model(cfg['Architecture'])
    model.to(device)
    model.eval()
    
    # Load pretrained weights
    pretrained_model = cfg['Global'].get('pretrained_model', None)
    if pretrained_model and os.path.exists(pretrained_model):
        state_dict = torch.load(pretrained_model, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter out mismatched decoder weights
        model_state = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
            else:
                print(f"Skipping {k}: not in model")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained model from {pretrained_model}")
    else:
        print("WARNING: No pretrained model loaded!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Prepare test data - batch 64, various widths
    test_sizes = [
        (64, 3, 32, 128),   # 짧은 텍스트
        (64, 3, 32, 256),   # 중간 텍스트
        (64, 3, 32, 384),   # 긴 텍스트
        (64, 3, 32, 512),   # 매우 긴 텍스트
    ]
    
    print("\n" + "="*60)
    print("FPS Measurement (warmup: 50, measure: 200 iterations)")
    print("="*60)
    
    all_fps = []
    
    for size in test_sizes:
        B, C, H, W = size
        dummy_input = torch.randn(size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_input)
        
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        num_iters = 200
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iters):
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        fps = num_iters / elapsed
        avg_time_ms = (elapsed / num_iters) * 1000
        
        all_fps.append(fps)
        throughput = fps * B  # images per second
        print(f"Batch={B:3d}, {H}x{W}: {fps:.1f} batches/s, {throughput:.1f} images/s ({avg_time_ms:.2f} ms/batch)")
    
    print("="*60)
    print(f"Average batches/s: {np.mean(all_fps):.1f}")
    print("="*60)


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
