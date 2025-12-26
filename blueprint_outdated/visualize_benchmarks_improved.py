#!/usr/bin/env python3
"""
Enhanced Benchmark Visualization Script for Julia PolyBench Results

Features:
- CSV benchmark visualization
- Scaling study log parsing (ASCII table format)
- Flame graph integration preparation
- Multi-benchmark comparison
- Thread scaling analysis with Amdahl's law overlay

Usage:
    python3 visualize_benchmarks.py results.csv
    python3 visualize_benchmarks.py results.csv --output-dir ./plots
    python3 visualize_benchmarks.py --parse-log scaling_study.log
    python3 visualize_benchmarks.py --compare results1.csv results2.csv
"""

import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def parse_scaling_log(filepath):
    """
    Parse ASCII-table format scaling study logs into DataFrame.
    
    Expected format:
    Strategy         |    Min(ms) | Median(ms) |   Mean(ms) |  GFLOP/s |  Speedup | Eff(%)
    ------------------------------------------------------------------------------------------
    sequential       |      7.403 |      7.761 |      8.095 |     2.23 |     1.00x |  100.0
    """
    data = []
    current_config = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Parse configuration headers
        if line.startswith('Threads:'):
            current_config['threads'] = int(re.search(r'Threads:\s*(\d+)', line).group(1))
        elif line.startswith('Dataset:'):
            match = re.search(r'Dataset:\s*(\w+)', line)
            if match:
                current_config['dataset'] = match.group(1)
        elif line.startswith('======') and 'BENCHMARK' in lines[lines.index(line + '\n') + 1] if line + '\n' in lines else False:
            pass
        
        # Parse data rows
        if '|' in line and not line.startswith('-') and not line.startswith('Strategy'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    strategy = parts[0].lower().replace(' ', '_').replace('(', '').replace(')', '')
                    min_ms = float(parts[1])
                    median_ms = float(parts[2])
                    mean_ms = float(parts[3])
                    gflops = float(parts[4])
                    speedup_str = parts[5].replace('x', '')
                    speedup = float(speedup_str)
                    efficiency = float(parts[6])
                    
                    row = {
                        'strategy': strategy,
                        'min_ms': min_ms,
                        'median_ms': median_ms,
                        'mean_ms': mean_ms,
                        'gflops': gflops,
                        'speedup': speedup,
                        'efficiency': efficiency,
                        **current_config
                    }
                    data.append(row)
                except (ValueError, IndexError):
                    continue
    
    if not data:
        print(f"Warning: No data parsed from {filepath}")
        return None
    
    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} rows from {filepath}")
    return df


def load_and_validate_csv(filepath):
    """Load CSV and validate expected columns."""
    df = pd.read_csv(filepath)
    
    # Normalize column names
    column_map = {
        'min(ms)': 'min_ms',
        'median(ms)': 'median_ms',
        'mean(ms)': 'mean_ms',
        'std(ms)': 'std_ms',
        'gflop/s': 'gflops',
        'eff(%)': 'efficiency'
    }
    df.columns = [column_map.get(c.lower(), c.lower()) for c in df.columns]
    
    required = ['strategy', 'min_ms', 'median_ms']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
    
    return df


def create_performance_heatmap(df, output_path, title_prefix=""):
    """Create heatmap of execution times across strategies and datasets."""
    if 'dataset' not in df.columns or df['dataset'].nunique() < 2:
        print("Skipping heatmap: insufficient dataset variation")
        return
    
    pivot = df.pivot_table(values='median_ms', index='strategy', columns='dataset')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > pivot.values.max() / 2 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)
    
    ax.set_title(f'{title_prefix}Execution Time (ms) by Strategy and Dataset')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Strategy')
    
    plt.colorbar(im, ax=ax, label='Time (ms)')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_speedup_chart(df, output_path, dataset=None, title_prefix=""):
    """Create bar chart of speedup values with baseline reference."""
    if dataset and 'dataset' in df.columns:
        df_plot = df[df['dataset'] == dataset].copy()
        title_suffix = f" ({dataset})"
    else:
        df_plot = df.copy()
        title_suffix = ""
    
    if 'speedup' not in df_plot.columns:
        print("Skipping speedup chart: no 'speedup' column")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = df_plot['strategy'].values
    speedups = df_plot['speedup'].values
    
    # Color by performance tier
    colors = []
    for s in speedups:
        if s >= 8:
            colors.append('#27ae60')  # Excellent (green)
        elif s >= 4:
            colors.append('#f39c12')  # Good (orange)
        elif s >= 1:
            colors.append('#3498db')  # Baseline+ (blue)
        else:
            colors.append('#e74c3c')  # Below baseline (red)
    
    bars = ax.bar(range(len(strategies)), speedups, color=colors, edgecolor='black', linewidth=0.5)
    
    # Value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (1x)')
    
    # Add thread count reference if available
    if 'threads' in df_plot.columns:
        max_threads = df_plot['threads'].max()
        ax.axhline(y=max_threads, color='gray', linestyle=':', alpha=0.5, 
                   label=f'Ideal ({max_threads}x)')
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'{title_prefix}Speedup vs Sequential{title_suffix}')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_thread_scaling_chart(df, output_path, title_prefix=""):
    """Create thread scaling analysis with Amdahl's law overlay."""
    if 'threads' not in df.columns:
        print("Skipping scaling chart: no 'threads' column")
        return
    
    # Get unique thread counts and strategies
    thread_counts = sorted(df['threads'].unique())
    if len(thread_counts) < 2:
        print("Skipping scaling chart: need multiple thread counts")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Speedup vs Threads
    parallel_strategies = [s for s in df['strategy'].unique() 
                          if any(x in s.lower() for x in ['thread', 'tiled', 'task'])]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(parallel_strategies)))
    
    for idx, strategy in enumerate(parallel_strategies):
        df_strat = df[df['strategy'] == strategy].sort_values('threads')
        if len(df_strat) > 1:
            ax1.plot(df_strat['threads'], df_strat['speedup'], 
                    marker='o', color=colors[idx], label=strategy, linewidth=2, markersize=8)
    
    # Ideal scaling line
    max_t = max(thread_counts)
    ax1.plot([1, max_t], [1, max_t], 'k--', alpha=0.5, label='Ideal')
    
    # Amdahl's law curves for different parallel fractions
    threads_fine = np.linspace(1, max_t, 100)
    for p in [0.9, 0.95, 0.99]:
        amdahl = 1 / ((1 - p) + p / threads_fine)
        ax1.plot(threads_fine, amdahl, ':', alpha=0.3, label=f"Amdahl p={p}")
    
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title(f'{title_prefix}Thread Scaling (Speedup)')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.set_xlim(0, max_t + 1)
    ax1.set_ylim(0, max_t + 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency vs Threads
    for idx, strategy in enumerate(parallel_strategies):
        df_strat = df[df['strategy'] == strategy].sort_values('threads')
        if len(df_strat) > 1:
            ax2.plot(df_strat['threads'], df_strat['efficiency'], 
                    marker='s', color=colors[idx], label=strategy, linewidth=2, markersize=8)
    
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='100% Efficiency')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title(f'{title_prefix}Thread Scaling (Efficiency)')
    ax2.set_xlim(0, max_t + 1)
    ax2.set_ylim(0, max(110, df['efficiency'].max() * 1.1))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_gflops_chart(df, output_path, dataset=None, title_prefix=""):
    """Create GFLOP/s throughput chart with memory bandwidth reference."""
    if 'gflops' not in df.columns:
        print("Skipping GFLOP/s chart: no 'gflops' column")
        return
    
    if dataset and 'dataset' in df.columns:
        df_plot = df[df['dataset'] == dataset].copy()
        title_suffix = f" ({dataset})"
    else:
        df_plot = df.copy()
        title_suffix = ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_plot['strategy'].values
    gflops = df_plot['gflops'].values
    
    # Gradient colors
    norm = plt.Normalize(vmin=min(gflops), vmax=max(gflops))
    colors = plt.cm.viridis(norm(gflops))
    
    bars = ax.bar(range(len(strategies)), gflops, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, g in zip(bars, gflops):
        height = bar.get_height()
        ax.annotate(f'{g:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('GFLOP/s')
    ax.set_title(f'{title_prefix}Computational Throughput{title_suffix}')
    
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm), ax=ax, label='GFLOP/s')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_allocation_chart(df, output_path, title_prefix=""):
    """Create memory allocation chart (zero = green, non-zero = red)."""
    if 'allocations' not in df.columns:
        print("Skipping allocation chart: no 'allocations' column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df['strategy'].values
    allocations = df['allocations'].values
    
    colors = ['#27ae60' if a == 0 else '#e74c3c' for a in allocations]
    
    bars = ax.bar(range(len(strategies)), allocations, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Memory Allocations')
    ax.set_title(f'{title_prefix}Memory Allocations (green=0, red>0)')
    
    # Legend
    green_patch = mpatches.Patch(color='#27ae60', label='Zero allocations (optimal)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Non-zero allocations')
    ax.legend(handles=[green_patch, red_patch])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_dashboard(df, output_path, title_prefix=""):
    """Create multi-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 12))
    
    strategies = df['strategy'].values
    
    # Panel 1: Execution Time
    ax1 = fig.add_subplot(2, 2, 1)
    times = df['median_ms'].values
    bars1 = ax1.barh(range(len(strategies)), times, color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(strategies)))
    ax1.set_yticklabels(strategies)
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Execution Time (median)')
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, t in zip(bars1, times):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{t:.1f}', va='center', fontsize=8)
    
    # Panel 2: Speedup
    if 'speedup' in df.columns:
        ax2 = fig.add_subplot(2, 2, 2)
        speedups = df['speedup'].values
        colors = ['#27ae60' if s > 1 else '#e74c3c' for s in speedups]
        bars2 = ax2.barh(range(len(strategies)), speedups, color=colors, edgecolor='black')
        ax2.axvline(x=1.0, color='black', linestyle='--')
        ax2.set_yticks(range(len(strategies)))
        ax2.set_yticklabels(strategies)
        ax2.set_xlabel('Speedup (x)')
        ax2.set_title('Speedup vs Sequential')
        ax2.invert_yaxis()
    
    # Panel 3: Efficiency
    if 'efficiency' in df.columns:
        ax3 = fig.add_subplot(2, 2, 3)
        efficiencies = df['efficiency'].values
        # Clip for color mapping (some BLAS values > 100%)
        eff_clipped = np.clip(efficiencies, 0, 200)
        colors = plt.cm.RdYlGn(eff_clipped / 200)
        bars3 = ax3.barh(range(len(strategies)), efficiencies, color=colors, edgecolor='black')
        ax3.axvline(x=100, color='black', linestyle='--', alpha=0.5)
        ax3.set_yticks(range(len(strategies)))
        ax3.set_yticklabels(strategies)
        ax3.set_xlabel('Efficiency (%)')
        ax3.set_title('Parallel Efficiency')
        ax3.invert_yaxis()
    
    # Panel 4: GFLOP/s
    if 'gflops' in df.columns:
        ax4 = fig.add_subplot(2, 2, 4)
        gflops = df['gflops'].values
        bars4 = ax4.barh(range(len(strategies)), gflops, color='darkorange', edgecolor='black')
        ax4.set_yticks(range(len(strategies)))
        ax4.set_yticklabels(strategies)
        ax4.set_xlabel('GFLOP/s')
        ax4.set_title('Computational Throughput')
        ax4.invert_yaxis()
    
    plt.suptitle(f'{title_prefix}Benchmark Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def convert_log_to_csv(log_path, output_path):
    """Convert scaling study log to CSV format."""
    df = parse_scaling_log(log_path)
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Converted {log_path} to {output_path}")
        return output_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Julia PolyBench Benchmark Visualization')
    parser.add_argument('files', nargs='*', help='CSV or log files to process')
    parser.add_argument('--output-dir', '-o', default='./benchmark_plots', help='Output directory')
    parser.add_argument('--parse-log', action='store_true', help='Parse scaling study log format')
    parser.add_argument('--compare', action='store_true', help='Compare multiple result files')
    parser.add_argument('--title', default='', help='Title prefix for charts')
    
    args = parser.parse_args()
    
    if not args.files:
        parser.print_help()
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    
    for filepath in args.files:
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue
        
        # Load data
        if args.parse_log or filepath.suffix == '.log':
            df = parse_scaling_log(filepath)
            if df is not None:
                # Also save as CSV
                csv_path = output_dir / f"{filepath.stem}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved CSV: {csv_path}")
        else:
            df = load_and_validate_csv(filepath)
        
        if df is None or df.empty:
            continue
        
        all_dfs.append((filepath.stem, df))
        
        # Generate visualizations for this file
        base_name = filepath.stem
        prefix = f"{args.title} " if args.title else ""
        
        create_summary_dashboard(df, output_dir / f"{base_name}_summary.png", prefix)
        create_performance_heatmap(df, output_dir / f"{base_name}_heatmap.png", prefix)
        
        # Determine best dataset for single-dataset charts
        if 'dataset' in df.columns:
            datasets = df['dataset'].unique()
            largest = datasets[-1] if len(datasets) > 0 else None
        else:
            largest = None
        
        create_speedup_chart(df, output_dir / f"{base_name}_speedup.png", largest, prefix)
        create_gflops_chart(df, output_dir / f"{base_name}_gflops.png", largest, prefix)
        create_thread_scaling_chart(df, output_dir / f"{base_name}_scaling.png", prefix)
        create_allocation_chart(df, output_dir / f"{base_name}_allocations.png", prefix)
    
    # Compare mode: overlay multiple files
    if args.compare and len(all_dfs) > 1:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_offset = 0
        bar_width = 0.8 / len(all_dfs)
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_dfs)))
        
        all_strategies = set()
        for _, df in all_dfs:
            all_strategies.update(df['strategy'].unique())
        all_strategies = sorted(all_strategies)
        
        for idx, (name, df) in enumerate(all_dfs):
            positions = [all_strategies.index(s) + idx * bar_width for s in df['strategy']]
            ax.bar(positions, df['median_ms'], bar_width, label=name, color=colors[idx], edgecolor='black')
        
        ax.set_xticks([i + bar_width * (len(all_dfs) - 1) / 2 for i in range(len(all_strategies))])
        ax.set_xticklabels(all_strategies, rotation=45, ha='right')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Median Time (ms)')
        ax.set_title('Benchmark Comparison')
        ax.legend()
        
        plt.tight_layout()
        comparison_path = output_dir / "comparison.png"
        plt.savefig(comparison_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {comparison_path}")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
