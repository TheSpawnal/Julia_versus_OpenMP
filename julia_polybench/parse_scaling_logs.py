

#!/usr/bin/env python3
"""
Scaling Study Log Parser

Converts ASCII-table format scaling study logs to CSV for visualization.
Handles the specific format output by Julia PolyBench benchmarks.

Usage:
    python3 parse_scaling_logs.py scaling_study.log
    python3 parse_scaling_logs.py scaling_study*.log --output combined.csv
    python3 parse_scaling_logs.py results/ --recursive
"""

import re
import sys
import argparse
from pathlib import Path
import pandas as pd


def parse_scaling_log(filepath):
    """
    Parse a scaling study log file into structured data.
    
    Expected format sections:
    ======================================================================
    CORRELATION MATRIX BENCHMARK
    ======================================================================
    Julia version: 1.11.5
    Threads: 4
    BLAS threads: 1
    CPU threads: 8
    Dataset: MEDIUM (m=240, n=260)
    Memory: 0.99 MB

    ------------------------------------------------------------------------------------------
    Strategy         |    Min(ms) | Median(ms) |   Mean(ms) |  GFLOP/s |  Speedup | Eff(%)
    ------------------------------------------------------------------------------------------
    sequential       |      7.403 |      7.761 |      8.095 |     2.23 |     1.00x |  100.0
    threads          |      0.859 |      0.891 |      0.896 |    19.25 |     8.62x |  215.5
    """
    records = []
    current_context = {
        'benchmark': None,
        'threads': None,
        'dataset': None,
        'blas_threads': None,
        'cpu_threads': None
    }
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect benchmark section header
        if 'BENCHMARK' in line and '=====' in line:
            # Next non-empty line might have benchmark name
            for j in range(max(0, i-2), min(len(lines), i+3)):
                check_line = lines[j].strip()
                if 'CORRELATION' in check_line.upper():
                    current_context['benchmark'] = 'correlation'
                elif '2MM' in check_line.upper():
                    current_context['benchmark'] = '2mm'
                elif '3MM' in check_line.upper():
                    current_context['benchmark'] = '3mm'
                elif 'CHOLESKY' in check_line.upper():
                    current_context['benchmark'] = 'cholesky'
                elif 'JACOBI' in check_line.upper():
                    current_context['benchmark'] = 'jacobi2d'
                elif 'NUSSINOV' in check_line.upper():
                    current_context['benchmark'] = 'nussinov'
        
        # Parse configuration lines
        if line.startswith('Threads:') and 'BLAS' not in line:
            match = re.search(r'Threads:\s*(\d+)', line)
            if match:
                current_context['threads'] = int(match.group(1))
        
        elif line.startswith('BLAS threads:'):
            match = re.search(r'BLAS threads:\s*(\d+)', line)
            if match:
                current_context['blas_threads'] = int(match.group(1))
        
        elif line.startswith('CPU threads:'):
            match = re.search(r'CPU threads:\s*(\d+)', line)
            if match:
                current_context['cpu_threads'] = int(match.group(1))
        
        elif line.startswith('Dataset:'):
            match = re.search(r'Dataset:\s*(\w+)', line)
            if match:
                current_context['dataset'] = match.group(1)
        
        # Parse data rows
        if '|' in line and not line.startswith('-') and not line.startswith('Strategy'):
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) >= 7:
                try:
                    # Clean strategy name
                    strategy = parts[0].lower()
                    strategy = re.sub(r'\s+', '_', strategy)
                    strategy = re.sub(r'[()]', '', strategy)
                    
                    # Parse numeric values
                    min_ms = float(parts[1])
                    median_ms = float(parts[2])
                    mean_ms = float(parts[3])
                    gflops = float(parts[4])
                    speedup_str = parts[5].replace('x', '').strip()
                    speedup = float(speedup_str)
                    efficiency = float(parts[6])
                    
                    record = {
                        'benchmark': current_context.get('benchmark', 'unknown'),
                        'dataset': current_context.get('dataset', 'unknown'),
                        'strategy': strategy,
                        'threads': current_context.get('threads', 1),
                        'min_ms': min_ms,
                        'median_ms': median_ms,
                        'mean_ms': mean_ms,
                        'std_ms': 0.0,  # Not available in log format
                        'gflops': gflops,
                        'speedup': speedup,
                        'efficiency': efficiency,
                        'verified': 'PASS'  # Assume verified if in results
                    }
                    
                    records.append(record)
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    pass
        
        i += 1
    
    if not records:
        print(f"Warning: No data parsed from {filepath}")
        return None
    
    df = pd.DataFrame(records)
    return df


def find_log_files(path, recursive=False):
    """Find all .log files in a directory."""
    path = Path(path)
    
    if path.is_file():
        return [path]
    
    if recursive:
        return list(path.rglob('*.log'))
    else:
        return list(path.glob('*.log'))


def main():
    parser = argparse.ArgumentParser(
        description='Parse Julia PolyBench scaling study logs to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 parse_scaling_logs.py scaling_study.log
    python3 parse_scaling_logs.py *.log --output combined.csv
    python3 parse_scaling_logs.py results/ --recursive --output all_results.csv
        """
    )
    
    parser.add_argument('inputs', nargs='+', help='Log files or directories to parse')
    parser.add_argument('--output', '-o', help='Output CSV file (default: <input>_parsed.csv)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search directories recursively')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    all_files = []
    for input_path in args.inputs:
        all_files.extend(find_log_files(input_path, args.recursive))
    
    if not all_files:
        print("No log files found")
        sys.exit(1)
    
    print(f"Found {len(all_files)} log file(s)")
    
    all_dfs = []
    for filepath in all_files:
        if args.verbose:
            print(f"Parsing: {filepath}")
        
        df = parse_scaling_log(filepath)
        if df is not None:
            df['source_file'] = str(filepath.name)
            all_dfs.append(df)
            print(f"  -> Parsed {len(df)} records")
    
    if not all_dfs:
        print("No data parsed from any files")
        sys.exit(1)
    
    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif len(all_files) == 1:
        output_path = all_files[0].with_suffix('.csv')
    else:
        output_path = Path('combined_scaling_results.csv')
    
    # Save to CSV
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} total records to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Benchmarks: {', '.join(combined['benchmark'].unique())}")
    print(f"Datasets: {', '.join(combined['dataset'].unique())}")
    print(f"Thread counts: {sorted(combined['threads'].unique())}")
    print(f"Strategies: {', '.join(combined['strategy'].unique())}")
    
    # Quick stats
    print("\n" + "-"*60)
    print("Best performance per benchmark:")
    print("-"*60)
    for bench in combined['benchmark'].unique():
        bench_data = combined[combined['benchmark'] == bench]
        best = bench_data.loc[bench_data['gflops'].idxmax()]
        print(f"  {bench}: {best['gflops']:.2f} GFLOP/s ({best['strategy']}, {best['threads']}t)")


if __name__ == '__main__':
    main()
