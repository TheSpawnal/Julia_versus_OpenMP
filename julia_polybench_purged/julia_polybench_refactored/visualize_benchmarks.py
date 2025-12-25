#!/usr/bin/env python3
"""
Julia PolyBench Visualization Suite
REFACTORED: Patagonia 80s Retro Pastel Color Palette

Works EXCLUSIVELY with CSV data. No log parsing nonsense.

Features:
- Thread scaling analysis with Amdahl's law overlays
- Multi-benchmark comparison
- Efficiency vs Speedup separation
- Strong/Weak scaling plots
- Heatmaps for strategy comparison

Usage:
    python3 visualize_benchmarks.py results/*.csv
    python3 visualize_benchmarks.py results/*.csv --output-dir ./plots
    python3 visualize_benchmarks.py --compare bench1.csv bench2.csv
    python3 visualize_benchmarks.py --scaling results/scaling_*.csv
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATAGONIA 80s RETRO PASTEL COLOR PALETTE
# =============================================================================
COLORS = {
    # Primary palette - soft retro pastels
    'coral': '#E8967A',        # Warm coral/salmon
    'teal': '#7EBDC2',         # Soft teal
    'gold': '#E5C07B',         # Muted gold
    'sage': '#98C1A6',         # Sage green
    'lavender': '#B8A9C9',     # Soft lavender
    'peach': '#F4B183',        # Peach
    'sky': '#87CEEB',          # Sky blue
    'rose': '#DDA0A0',         # Dusty rose
    
    # Extended palette
    'slate': '#708090',        # Slate gray
    'sand': '#D4B896',         # Sand/tan
    'rust': '#C17F59',         # Rust orange
    'seafoam': '#9FD5D1',      # Seafoam
    'mauve': '#C9A9A6',        # Mauve
    'olive': '#9CAF88',        # Olive green
    
    # Neutrals
    'dark': '#2D3436',         # Near black
    'medium': '#636E72',       # Medium gray
    'light': '#DFE6E9',        # Light gray
    'cream': '#FAF3E3',        # Cream background
}

# Strategy color mapping
STRATEGY_COLORS = {
    'sequential': COLORS['slate'],
    'seq': COLORS['slate'],
    'threads_static': COLORS['teal'],
    'threads': COLORS['teal'],
    'threads_dynamic': COLORS['coral'],
    'dynamic': COLORS['coral'],
    'tiled': COLORS['sage'],
    'blocked': COLORS['sage'],
    'blas': COLORS['gold'],
    'tasks': COLORS['lavender'],
    'wavefront': COLORS['peach'],
    'simd': COLORS['sky'],
    'redblack': COLORS['rose'],
    'colmajor': COLORS['sand'],
}

# Benchmark colors for comparison charts
BENCHMARK_COLORS = {
    '2mm': COLORS['teal'],
    '3mm': COLORS['coral'],
    'cholesky': COLORS['sage'],
    'correlation': COLORS['gold'],
    'jacobi2d': COLORS['lavender'],
    'nussinov': COLORS['peach'],
}

def get_strategy_color(strategy):
    """Get color for strategy, with fallback"""
    return STRATEGY_COLORS.get(strategy.lower(), COLORS['medium'])

def get_benchmark_color(benchmark):
    """Get color for benchmark, with fallback"""
    return BENCHMARK_COLORS.get(benchmark.lower(), COLORS['medium'])

# =============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# =============================================================================
def setup_style():
    """Configure matplotlib for retro Patagonia aesthetic"""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': COLORS['cream'],
        'figure.edgecolor': COLORS['dark'],
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.facecolor': COLORS['cream'],
        
        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['medium'],
        'axes.labelcolor': COLORS['dark'],
        'axes.titlecolor': COLORS['dark'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'grid.color': COLORS['light'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # Text
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        
        # Legend
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': COLORS['light'],
        'legend.fontsize': 9,
        
        # Ticks
        'xtick.color': COLORS['dark'],
        'ytick.color': COLORS['dark'],
    })

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================
def load_csv(filepath):
    """Load and validate CSV benchmark data"""
    df = pd.read_csv(filepath)
    
    # Normalize column names (lowercase, underscore)
    column_map = {
        'min(ms)': 'min_ms',
        'median(ms)': 'median_ms',
        'mean(ms)': 'mean_ms',
        'std(ms)': 'std_ms',
        'gflop/s': 'gflops',
        'eff(%)': 'efficiency_pct',
        'efficiency': 'efficiency_pct',
    }
    df.columns = [column_map.get(c.lower(), c.lower()) for c in df.columns]
    
    # Validate required columns
    required = ['strategy', 'min_ms']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing} in {filepath}")
        return None
    
    return df

def load_multiple_csvs(filepaths):
    """Load and combine multiple CSV files"""
    dfs = []
    for fp in filepaths:
        df = load_csv(fp)
        if df is not None:
            df['source_file'] = Path(fp).stem
            dfs.append(df)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

# =============================================================================
# AMDAHL'S LAW
# =============================================================================
def amdahl_speedup(threads, parallel_fraction):
    """Compute theoretical speedup using Amdahl's law"""
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / threads)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_speedup_chart(df, output_path, title_prefix=""):
    """Bar chart of speedup by strategy"""
    if 'speedup' not in df.columns:
        print("Skipping speedup chart: no 'speedup' column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df['strategy'].unique()
    speedups = [df[df['strategy'] == s]['speedup'].iloc[0] for s in strategies]
    colors = [get_strategy_color(s) for s in strategies]
    
    y_pos = np.arange(len(strategies))
    bars = ax.barh(y_pos, speedups, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    
    # Reference line at speedup = 1
    ax.axvline(x=1.0, color=COLORS['dark'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Value labels
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{speedup:.2f}x', va='center', fontsize=9, color=COLORS['dark'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies)
    ax.set_xlabel('Speedup (relative to sequential)')
    ax.set_title(f'{title_prefix}Speedup by Strategy')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_efficiency_chart(df, output_path, title_prefix=""):
    """Bar chart of parallel efficiency (only for parallel strategies)"""
    if 'efficiency_pct' not in df.columns and 'efficiency' not in df.columns:
        print("Skipping efficiency chart: no efficiency column")
        return
    
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    
    # Filter to parallel strategies only (non-NaN efficiency)
    df_parallel = df[df[eff_col].notna() & (df[eff_col] != '')]
    if df_parallel.empty:
        print("Skipping efficiency chart: no parallel strategies")
        return
    
    # Handle string conversion
    if df_parallel[eff_col].dtype == object:
        df_parallel = df_parallel[df_parallel[eff_col] != '']
        df_parallel = df_parallel.copy()
        df_parallel[eff_col] = pd.to_numeric(df_parallel[eff_col], errors='coerce')
        df_parallel = df_parallel.dropna(subset=[eff_col])
    
    if df_parallel.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df_parallel['strategy'].values
    efficiencies = df_parallel[eff_col].values
    colors = [get_strategy_color(s) for s in strategies]
    
    y_pos = np.arange(len(strategies))
    bars = ax.barh(y_pos, efficiencies, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    
    # Reference line at 100%
    ax.axvline(x=100, color=COLORS['dark'], linestyle='--', alpha=0.5, linewidth=1)
    
    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{eff:.1f}%', va='center', fontsize=9, color=COLORS['dark'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies)
    ax.set_xlabel('Parallel Efficiency (%)')
    ax.set_title(f'{title_prefix}Parallel Efficiency (threaded strategies only)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_gflops_chart(df, output_path, title_prefix=""):
    """Bar chart of computational throughput"""
    if 'gflops' not in df.columns:
        print("Skipping GFLOP/s chart: no 'gflops' column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = df['strategy'].unique()
    gflops = [df[df['strategy'] == s]['gflops'].iloc[0] for s in strategies]
    colors = [get_strategy_color(s) for s in strategies]
    
    y_pos = np.arange(len(strategies))
    bars = ax.barh(y_pos, gflops, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    
    for bar, gf in zip(bars, gflops):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{gf:.1f}', va='center', fontsize=9, color=COLORS['dark'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies)
    ax.set_xlabel('GFLOP/s')
    ax.set_title(f'{title_prefix}Computational Throughput')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_thread_scaling_chart(df, output_path, title_prefix=""):
    """Thread scaling chart with Amdahl's law overlays"""
    if 'threads' not in df.columns:
        print("Skipping scaling chart: no 'threads' column")
        return
    
    thread_counts = sorted(df['threads'].unique())
    if len(thread_counts) < 2:
        print("Skipping scaling chart: need multiple thread counts")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter to parallel strategies
    parallel_strategies = [s for s in df['strategy'].unique() 
                          if any(x in s.lower() for x in ['thread', 'tiled', 'task', 'wavefront'])]
    
    max_threads = max(thread_counts)
    threads_fine = np.linspace(1, max_threads, 100)
    
    # Plot 1: Speedup vs Threads
    for strategy in parallel_strategies:
        df_s = df[df['strategy'] == strategy].sort_values('threads')
        if len(df_s) > 1:
            ax1.plot(df_s['threads'], df_s['speedup'],
                    marker='o', color=get_strategy_color(strategy),
                    label=strategy, linewidth=2, markersize=8)
    
    # Ideal scaling
    ax1.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal', linewidth=1.5)
    
    # Amdahl's law curves
    for p, alpha in [(0.90, 0.3), (0.95, 0.4), (0.99, 0.5)]:
        amdahl = amdahl_speedup(threads_fine, p)
        ax1.plot(threads_fine, amdahl, ':', color=COLORS['slate'],
                alpha=alpha, label=f'Amdahl p={p}')
    
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title(f'{title_prefix}Strong Scaling (Speedup)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_xlim(0, max_threads + 1)
    ax1.set_ylim(0, max_threads + 1)
    
    # Plot 2: Efficiency vs Threads
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    if eff_col in df.columns:
        for strategy in parallel_strategies:
            df_s = df[df['strategy'] == strategy].sort_values('threads')
            df_s = df_s[df_s[eff_col].notna()]
            if df_s[eff_col].dtype == object:
                df_s = df_s[df_s[eff_col] != '']
                df_s = df_s.copy()
                df_s[eff_col] = pd.to_numeric(df_s[eff_col], errors='coerce')
            df_s = df_s.dropna(subset=[eff_col])
            
            if len(df_s) > 1:
                ax2.plot(df_s['threads'], df_s[eff_col],
                        marker='s', color=get_strategy_color(strategy),
                        label=strategy, linewidth=2, markersize=8)
        
        ax2.axhline(y=100, color=COLORS['dark'], linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title(f'{title_prefix}Parallel Efficiency')
        ax2.set_xlim(0, max_threads + 1)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_timing_heatmap(df, output_path, title_prefix=""):
    """Heatmap of execution times across strategies and datasets"""
    if 'dataset' not in df.columns or df['dataset'].nunique() < 2:
        print("Skipping heatmap: need multiple datasets")
        return
    
    pivot = df.pivot_table(values='median_ms', index='strategy', columns='dataset')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create heatmap with custom colormap (warm tones)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('patagonia',
        [COLORS['cream'], COLORS['peach'], COLORS['coral'], COLORS['rust']])
    
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val > pivot.values.max() * 0.6 else COLORS['dark']
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                       color=color, fontsize=9)
    
    ax.set_title(f'{title_prefix}Execution Time (ms)')
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Strategy')
    
    plt.colorbar(im, ax=ax, label='Time (ms)')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_dashboard(df, output_path, title_prefix=""):
    """4-panel summary dashboard"""
    fig = plt.figure(figsize=(14, 10))
    
    strategies = df['strategy'].unique()
    colors = [get_strategy_color(s) for s in strategies]
    y_pos = np.arange(len(strategies))
    
    # Panel 1: Execution Time
    ax1 = fig.add_subplot(2, 2, 1)
    times = df.groupby('strategy')['median_ms'].mean().reindex(strategies)
    bars1 = ax1.barh(y_pos, times, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(strategies)
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Execution Time')
    ax1.invert_yaxis()
    
    # Panel 2: Speedup
    if 'speedup' in df.columns:
        ax2 = fig.add_subplot(2, 2, 2)
        speedups = df.groupby('strategy')['speedup'].mean().reindex(strategies)
        bar_colors = [COLORS['sage'] if s > 1 else COLORS['rose'] for s in speedups]
        ax2.barh(y_pos, speedups, color=bar_colors, edgecolor=COLORS['dark'], linewidth=0.5)
        ax2.axvline(x=1.0, color=COLORS['dark'], linestyle='--', alpha=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(strategies)
        ax2.set_xlabel('Speedup')
        ax2.set_title('Speedup vs Sequential')
        ax2.invert_yaxis()
    
    # Panel 3: GFLOP/s
    if 'gflops' in df.columns:
        ax3 = fig.add_subplot(2, 2, 3)
        gflops = df.groupby('strategy')['gflops'].mean().reindex(strategies)
        ax3.barh(y_pos, gflops, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(strategies)
        ax3.set_xlabel('GFLOP/s')
        ax3.set_title('Computational Throughput')
        ax3.invert_yaxis()
    
    # Panel 4: Efficiency (parallel only)
    eff_col = 'efficiency_pct' if 'efficiency_pct' in df.columns else 'efficiency'
    if eff_col in df.columns:
        ax4 = fig.add_subplot(2, 2, 4)
        df_eff = df.copy()
        if df_eff[eff_col].dtype == object:
            df_eff = df_eff[df_eff[eff_col] != '']
            df_eff[eff_col] = pd.to_numeric(df_eff[eff_col], errors='coerce')
        
        df_eff = df_eff.dropna(subset=[eff_col])
        eff_by_strat = df_eff.groupby('strategy')[eff_col].mean()
        
        parallel_strats = [s for s in strategies if s in eff_by_strat.index]
        if parallel_strats:
            eff_vals = [eff_by_strat.get(s, 0) for s in parallel_strats]
            eff_colors = [get_strategy_color(s) for s in parallel_strats]
            y_pos_eff = np.arange(len(parallel_strats))
            ax4.barh(y_pos_eff, eff_vals, color=eff_colors, edgecolor=COLORS['dark'], linewidth=0.5)
            ax4.axvline(x=100, color=COLORS['dark'], linestyle='--', alpha=0.5)
            ax4.set_yticks(y_pos_eff)
            ax4.set_yticklabels(parallel_strats)
            ax4.set_xlabel('Efficiency (%)')
            ax4.set_title('Parallel Efficiency')
            ax4.invert_yaxis()
    
    plt.suptitle(f'{title_prefix}Benchmark Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_benchmark_comparison(df, output_path, title_prefix=""):
    """Compare multiple benchmarks side by side"""
    if 'benchmark' not in df.columns:
        print("Skipping comparison: no 'benchmark' column")
        return
    
    benchmarks = df['benchmark'].unique()
    if len(benchmarks) < 2:
        print("Skipping comparison: need multiple benchmarks")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Best speedup per benchmark
    best_speedup = df.groupby('benchmark')['speedup'].max()
    colors = [get_benchmark_color(b) for b in best_speedup.index]
    ax1.bar(best_speedup.index, best_speedup.values, color=colors,
            edgecolor=COLORS['dark'], linewidth=0.5)
    ax1.set_ylabel('Best Speedup')
    ax1.set_title('Best Speedup by Benchmark')
    ax1.tick_params(axis='x', rotation=45)
    
    # Best GFLOP/s per benchmark
    if 'gflops' in df.columns:
        best_gflops = df.groupby('benchmark')['gflops'].max()
        colors = [get_benchmark_color(b) for b in best_gflops.index]
        ax2.bar(best_gflops.index, best_gflops.values, color=colors,
                edgecolor=COLORS['dark'], linewidth=0.5)
        ax2.set_ylabel('Best GFLOP/s')
        ax2.set_title('Best Throughput by Benchmark')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{title_prefix}Benchmark Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Julia PolyBench Visualization (Patagonia 80s Style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 visualize_benchmarks.py results/*.csv
    python3 visualize_benchmarks.py results/*.csv --output-dir ./plots
    python3 visualize_benchmarks.py --scaling results/scaling_*.csv
        """
    )
    
    parser.add_argument('files', nargs='*', help='CSV files to visualize')
    parser.add_argument('--output-dir', '-o', default='./benchmark_plots', help='Output directory')
    parser.add_argument('--title', '-t', default='', help='Title prefix')
    parser.add_argument('--scaling', action='store_true', help='Generate scaling study plots')
    parser.add_argument('--compare', action='store_true', help='Generate comparison plots')
    
    args = parser.parse_args()
    
    if not args.files:
        parser.print_help()
        sys.exit(1)
    
    # Setup style
    setup_style()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_multiple_csvs(args.files)
    if df is None or df.empty:
        print("No valid data loaded")
        sys.exit(1)
    
    print(f"Loaded {len(df)} records from {len(args.files)} file(s)")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Generate prefix
    prefix = f"{args.title} " if args.title else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate visualizations
    create_summary_dashboard(df, output_dir / f"summary_{timestamp}.png", prefix)
    create_speedup_chart(df, output_dir / f"speedup_{timestamp}.png", prefix)
    create_efficiency_chart(df, output_dir / f"efficiency_{timestamp}.png", prefix)
    create_gflops_chart(df, output_dir / f"gflops_{timestamp}.png", prefix)
    create_timing_heatmap(df, output_dir / f"heatmap_{timestamp}.png", prefix)
    
    if args.scaling or 'threads' in df.columns:
        create_thread_scaling_chart(df, output_dir / f"scaling_{timestamp}.png", prefix)
    
    if args.compare or 'benchmark' in df.columns:
        create_benchmark_comparison(df, output_dir / f"comparison_{timestamp}.png", prefix)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
