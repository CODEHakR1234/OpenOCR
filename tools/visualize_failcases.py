#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fail Case Analysis Visualization

분석 결과 JSON을 시각화합니다.

Usage:
    python tools/visualize_failcases.py \
        --json ./failcase_analysis_multioriented/analysis_results.json \
        --output_dir ./failcase_analysis_multioriented/plots
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import Counter

# 한글 폰트 설정 (없으면 기본 폰트 사용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_results(json_path):
    """JSON 결과 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_dataset_accuracy(results, output_dir):
    """데이터셋별 정확도 바 차트"""
    datasets = []
    accuracies = []
    totals = []
    
    for name, data in results['datasets'].items():
        datasets.append(name)
        accuracies.append(data['accuracy'])
        totals.append(data['total'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(datasets, accuracies, color='steelblue', edgecolor='navy')
    
    # 각 바 위에 정확도 표시
    for bar, acc, total in zip(bars, accuracies, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%\n(n={total})', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_title('Recognition Accuracy by Dataset', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=results['summary']['accuracy'], color='red', linestyle='--', 
               label=f"Overall: {results['summary']['accuracy']:.1f}%")
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_accuracy.png'), dpi=150)
    plt.close()
    print(f"  Saved: dataset_accuracy.png")


def plot_error_types(results, output_dir):
    """에러 유형 분포 파이 차트"""
    for dataset_name, data in results['datasets'].items():
        error_types = data['error_types']
        
        if sum(error_types.values()) == 0:
            continue
        
        labels = ['Insert', 'Delete', 'Replace']
        sizes = [error_types['insert'], error_types['delete'], error_types['replace']]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.05, 0.05, 0.05)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        
        ax.set_title(f'Error Types Distribution - {dataset_name}', fontsize=14, fontweight='bold')
        
        # 범례에 실제 개수 표시
        legend_labels = [f'{l}: {s}' for l, s in zip(labels, sizes)]
        ax.legend(wedges, legend_labels, title="Error Types", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_error_types.png'), dpi=150)
        plt.close()
        print(f"  Saved: {dataset_name}_error_types.png")


def plot_confusion_matrix(results, output_dir, top_n=20):
    """혼동 문자쌍 히트맵"""
    for dataset_name, data in results['datasets'].items():
        confusion = data.get('confusion_pairs', {})
        
        if not confusion:
            continue
        
        # Top N 혼동 쌍
        sorted_pairs = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_pairs:
            continue
        
        pairs, counts = zip(*sorted_pairs)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        y_pos = np.arange(len(pairs))
        bars = ax.barh(y_pos, counts, color='coral', edgecolor='darkred')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title(f'Top {top_n} Confusion Pairs - {dataset_name}\n(Pred → GT)', 
                    fontsize=14, fontweight='bold')
        
        # 각 바 옆에 개수 표시
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{count}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_confusion_pairs.png'), dpi=150)
        plt.close()
        print(f"  Saved: {dataset_name}_confusion_pairs.png")


def plot_length_accuracy(results, output_dir):
    """텍스트 길이별 정확도"""
    for dataset_name, data in results['datasets'].items():
        length_stats = data.get('length_stats', {})
        
        if not length_stats:
            continue
        
        lengths = []
        accuracies = []
        totals = []
        
        for length, stats in sorted(length_stats.items(), key=lambda x: int(x[0])):
            if stats['total'] > 0:
                lengths.append(int(length))
                acc = stats['correct'] / stats['total'] * 100
                accuracies.append(acc)
                totals.append(stats['total'])
        
        if not lengths:
            continue
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 정확도 라인
        color = 'steelblue'
        ax1.set_xlabel('Text Length', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12)
        line1 = ax1.plot(lengths, accuracies, 'o-', color=color, linewidth=2, 
                        markersize=8, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 105)
        
        # 샘플 수 바
        ax2 = ax1.twinx()
        color = 'lightcoral'
        ax2.set_ylabel('Sample Count', color='darkred', fontsize=12)
        bar = ax2.bar(lengths, totals, alpha=0.3, color=color, label='Sample Count')
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        ax1.set_title(f'Accuracy by Text Length - {dataset_name}', fontsize=14, fontweight='bold')
        
        # 범례
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1 + [bar], labels1 + ['Sample Count'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_length_accuracy.png'), dpi=150)
        plt.close()
        print(f"  Saved: {dataset_name}_length_accuracy.png")


def plot_edit_distance_distribution(results, output_dir):
    """편집 거리 분포 히스토그램"""
    for dataset_name, data in results['datasets'].items():
        failcases = data.get('failcases', [])
        
        if not failcases:
            continue
        
        edit_distances = [fc['edit_distance'] for fc in failcases]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        max_ed = max(edit_distances) if edit_distances else 10
        bins = range(0, min(max_ed + 2, 20))
        
        ax.hist(edit_distances, bins=bins, color='mediumpurple', edgecolor='purple', alpha=0.7)
        ax.set_xlabel('Edit Distance', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Edit Distance Distribution - {dataset_name}\n(Wrong Predictions Only)', 
                    fontsize=14, fontweight='bold')
        
        # 평균 표시
        mean_ed = np.mean(edit_distances)
        ax.axvline(x=mean_ed, color='red', linestyle='--', 
                  label=f'Mean: {mean_ed:.2f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_edit_distance.png'), dpi=150)
        plt.close()
        print(f"  Saved: {dataset_name}_edit_distance.png")


def create_summary_dashboard(results, output_dir):
    """전체 요약 대시보드"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 전체 정확도 (왼쪽 상단)
    ax1 = fig.add_subplot(2, 2, 1)
    datasets = list(results['datasets'].keys())
    accuracies = [results['datasets'][d]['accuracy'] for d in datasets]
    
    bars = ax1.bar(range(len(datasets)), accuracies, color='steelblue')
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Dataset Accuracy', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=results['summary']['accuracy'], color='red', linestyle='--')
    
    # 2. 전체 통계 (오른쪽 상단)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis('off')
    
    summary_text = f"""
    OVERALL SUMMARY
    ================
    
    Total Samples: {results['summary']['total']:,}
    Correct: {results['summary']['correct']:,}
    Wrong: {results['summary']['wrong']:,}
    
    Overall Accuracy: {results['summary']['accuracy']:.2f}%
    
    Datasets Evaluated: {len(results['datasets'])}
    """
    
    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=14,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 에러 유형 합계 (왼쪽 하단)
    ax3 = fig.add_subplot(2, 2, 3)
    
    total_errors = {'insert': 0, 'delete': 0, 'replace': 0}
    for data in results['datasets'].values():
        for k, v in data['error_types'].items():
            total_errors[k] += v
    
    if sum(total_errors.values()) > 0:
        labels = ['Insert', 'Delete', 'Replace']
        sizes = [total_errors['insert'], total_errors['delete'], total_errors['replace']]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Total Error Types', fontweight='bold')
    
    # 4. 정확도 분포 (오른쪽 하단)
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax4.hist(accuracies, bins=10, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax4.set_xlabel('Accuracy (%)')
    ax4.set_ylabel('Count')
    ax4.set_title('Accuracy Distribution', fontweight='bold')
    ax4.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
               label=f'Mean: {np.mean(accuracies):.1f}%')
    ax4.legend()
    
    plt.suptitle('Fail Case Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize Fail Case Analysis Results')
    parser.add_argument('--json', '-j', type=str, required=True,
                       help='Path to analysis_results.json')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for plots (default: same as json)')
    args = parser.parse_args()
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.json), 'plots')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📊 Visualizing Fail Case Analysis")
    print(f"   Input: {args.json}")
    print(f"   Output: {args.output_dir}")
    
    # Load results
    results = load_results(args.json)
    
    print(f"\n🎨 Generating plots...")
    
    # Generate all plots
    plot_dataset_accuracy(results, args.output_dir)
    plot_error_types(results, args.output_dir)
    plot_confusion_matrix(results, args.output_dir)
    plot_length_accuracy(results, args.output_dir)
    plot_edit_distance_distribution(results, args.output_dir)
    create_summary_dashboard(results, args.output_dir)
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
