#!/usr/bin/env python3
"""
Benchmark Visualization Script for Julia PolyBench Results
Generates publication-quality performance charts from CSV benchmark data.
Usage:
    python3 visualize_benchmarks.py results.csv
    python3 visualize_benchmarks.py results.csv --output-dir ./plots
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Use a clean style suitable for publications
plt.style.use('seaborn-v0_8-whitegrid')

def load_and_validate_csv(filepath):
    """Load CSV and validate expected columns."""
    df = pd.read_csv(filepath)
    
    required_columns = ['strategy', 'min_ms', 'median_ms']
    optional_columns = ['dataset', 'speedup', 'efficiency', 'gflops', 'threads', 'allocations']
    
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        # Try alternate column names
        column_map = {
            'min(ms)': 'min_ms',
            'median(ms)': 'median_ms',
            'mean(ms)': 'mean_ms',
            'gflop/s': 'gflops',
            'eff(%)': 'efficiency'
        }
        df.columns = [column_map.get(c.lower(), c.lower()) for c in df.columns]
    
    return df

def create_performance_heatmap(df, output_path):
    """Create heatmap of execution times across strategies and datasets."""
    if 'dataset' not in df.columns:
        print("Skipping heatmap: no 'dataset' column")
        return
    
    pivot = df.pivot_table(values='median_ms', index='strategy', columns='dataset')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    # Add labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                              color='white' if val > pivot.values.max()/2 else 'black')
    
    ax.set_title('Execution Time (ms) by Strategy and Dataset')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Strategy')
    
    plt.colorbar(im, ax=ax, label='Time (ms)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_speedup_chart(df, output_path, dataset=None):
    """Create bar chart of speedup values."""
    if dataset and 'dataset' in df.columns:
        df_plot = df[df['dataset'] == dataset]
        title_suffix = f" ({dataset})"
    else:
        df_plot = df
        title_suffix = ""
    
    if 'speedup' not in df_plot.columns:
        print("Skipping speedup chart: no 'speedup' column")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = df_plot['strategy'].values
    speedups = df_plot['speedup'].values
    
    # Color based on speedup (green > 1, red < 1)
    colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in speedups]
    
    bars = ax.bar(range(len(strategies)), speedups, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.annotate(f'{speedup:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'Speedup vs Sequential{title_suffix}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_efficiency_chart(df, output_path):
    """Create line chart of parallel efficiency across datasets."""
    if 'dataset' not in df.columns or 'efficiency' not in df.columns:
        print("Skipping efficiency chart: missing columns")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = df['dataset'].unique()
    strategies = df['strategy'].unique()
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for idx, strategy in enumerate(strategies):
        df_strat = df[df['strategy'] == strategy]
        if len(df_strat) > 0:
            ax.plot(df_strat['dataset'], df_strat['efficiency'], 
                   marker=markers[idx % len(markers)], 
                   color=colors[idx],
                   label=strategy, linewidth=2, markersize=8)
    
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Efficiency by Strategy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, max(110, df['efficiency'].max() * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_gflops_chart(df, output_path, dataset=None):
    """Create bar chart of GFLOP/s performance."""
    if 'gflops' not in df.columns:
        print("Skipping GFLOP/s chart: no 'gflops' column")
        return
    
    if dataset and 'dataset' in df.columns:
        df_plot = df[df['dataset'] == dataset]
        title_suffix = f" ({dataset})"
    else:
        df_plot = df
        title_suffix = ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_plot['strategy'].values
    gflops = df_plot['gflops'].values
    
    # Gradient colors based on performance
    norm = plt.Normalize(vmin=min(gflops), vmax=max(gflops))
    colors = plt.cm.viridis(norm(gflops))
    
    bars = ax.bar(range(len(strategies)), gflops, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('GFLOP/s')
    ax.set_title(f'Computational Throughput{title_suffix}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='GFLOP/s')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_scaling_chart(df, output_path):
    """Create thread scaling chart if thread data available."""
    if 'threads' not in df.columns:
        print("Skipping scaling chart: no 'threads' column")
        return
    
    # Filter to parallel strategies
    parallel_strategies = [s for s in df['strategy'].unique() 
                          if any(x in s.lower() for x in ['thread', 'parallel', 'task'])]
    
    if not parallel_strategies:
        print("Skipping scaling chart: no parallel strategies found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(parallel_strategies)))
    
    for idx, strategy in enumerate(parallel_strategies):
        df_strat = df[df['strategy'] == strategy].sort_values('threads')
        if len(df_strat) > 1:
            ax.plot(df_strat['threads'], df_strat['speedup'], 
                   marker='o', color=colors[idx], label=strategy, linewidth=2)
    
    # Add ideal scaling line
    max_threads = df['threads'].max()
    ax.plot([1, max_threads], [1, max_threads], 
           linestyle='--', color='gray', label='Ideal Scaling', alpha=0.7)
    
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Thread Scaling Analysis')
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_allocation_chart(df, output_path):
    """Create bar chart showing memory allocations."""
    if 'allocations' not in df.columns:
        print("Skipping allocation chart: no 'allocations' column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df['strategy'].values
    allocations = df['allocations'].values
    
    # Color zero allocations green, non-zero red
    colors = ['#2ecc71' if a == 0 else '#e74c3c' for a in allocations]
    
    bars = ax.bar(range(len(strategies)), allocations, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Memory Allocations')
    ax.set_title('Memory Allocations by Strategy (green=0, red>0)')
    
    # Add legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Zero allocations')
    red_patch = mpatches.Patch(color='#e74c3c', label='Non-zero allocations')
    ax.legend(handles=[green_patch, red_patch])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_comparison(df, output_path):
    """Create multi-panel summary figure."""
    fig = plt.figure(figsize=(16, 10))
    
    # Panel 1: Execution Time
    ax1 = fig.add_subplot(2, 2, 1)
    strategies = df['strategy'].values
    times = df['median_ms'].values
    ax1.barh(range(len(strategies)), times, color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(strategies)))
    ax1.set_yticklabels(strategies)
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Execution Time (median)')
    ax1.invert_yaxis()
    
    # Panel 2: Speedup
    if 'speedup' in df.columns:
        ax2 = fig.add_subplot(2, 2, 2)
        speedups = df['speedup'].values
        colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in speedups]
        ax2.barh(range(len(strategies)), speedups, color=colors, edgecolor='black')
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
        colors = plt.cm.RdYlGn(efficiencies / 100)
        ax3.barh(range(len(strategies)), efficiencies, color=colors, edgecolor='black')
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
        ax4.barh(range(len(strategies)), gflops, color='darkorange', edgecolor='black')
        ax4.set_yticks(range(len(strategies)))
        ax4.set_yticklabels(strategies)
        ax4.set_xlabel('GFLOP/s')
        ax4.set_title('Computational Throughput')
        ax4.invert_yaxis()
    
    plt.suptitle('Benchmark Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_benchmarks.py <csv_file> [--output-dir <dir>]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = Path("./benchmark_plots")
    
    # Parse arguments
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_validate_csv(csv_path)
    print(f"Loaded {len(df)} benchmark results from {csv_path}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Determine dataset for single-dataset charts
    if 'dataset' in df.columns:
        datasets = df['dataset'].unique()
        largest_dataset = datasets[-1] if len(datasets) > 0 else None
    else:
        largest_dataset = None
    
    # Generate all visualizations
    base_name = Path(csv_path).stem
    
    create_summary_comparison(df, output_dir / f"{base_name}_summary.png")
    create_performance_heatmap(df, output_dir / f"{base_name}_heatmap.png")
    create_speedup_chart(df, output_dir / f"{base_name}_speedup.png", largest_dataset)
    create_efficiency_chart(df, output_dir / f"{base_name}_efficiency.png")
    create_gflops_chart(df, output_dir / f"{base_name}_gflops.png", largest_dataset)
    create_scaling_chart(df, output_dir / f"{base_name}_scaling.png")
    create_allocation_chart(df, output_dir / f"{base_name}_allocations.png")
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
