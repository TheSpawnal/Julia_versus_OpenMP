
# Flame Graphs vs CSV Benchmark Data

## Short Answer

**No, you cannot create flame graphs from CSV benchmark results.**

CSV files contain aggregate metrics (time, speedup, GFLOP/s), not call stack profiles.

---

## What Each Data Type Provides

### CSV Benchmark Results (What you have)
```csv
strategy,time_s,speedup,gflops
sequential,0.0234,1.00,2.45
threads_static,0.0058,4.03,9.88
blas,0.0021,11.14,27.31
```

**Good for:**
- Bar charts (time comparison)
- Line plots (scaling analysis)
- Speedup/Efficiency charts
- GFLOP/s comparison
- Publication-quality figures (Python matplotlib/seaborn)

**NOT suitable for:**
- Flame graphs (no call stack information)
- Identifying hotspots within code
- Understanding where time is spent

---

### Flame Graph Data (What you need for flame graphs)

Flame graphs require **stack sampling data** - records of which functions were executing at sample intervals.

**Format example (perf folded stacks):**
```
main;kernel_2mm;mul!;dgemm_ 1234
main;kernel_2mm;mul!;dgemm_;axpy 567
main;kernel_2mm;inner_loop 890
```

This shows: function call path → sample count

---

## Correct Approach for Your Project

### 1. Publication-Quality Comparison Charts (Python)

Use your CSV data for Julia vs OpenMP comparison figures:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
julia_df = pd.read_csv('results_2mm_LARGE_8t.csv')
omp_df = pd.read_csv('results_2mm_omp_LARGE_8t.csv')

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(julia_df))
width = 0.35
ax.bar([i - width/2 for i in x], julia_df['gflops'], width, label='Julia')
ax.bar([i + width/2 for i in x], omp_df['gflops'], width, label='OpenMP')
ax.set_xlabel('Strategy')
ax.set_ylabel('GFLOP/s')
ax.legend()
plt.savefig('julia_vs_omp_gflops.pdf')
```

### 2. True Flame Graphs (Separate Profiling Runs)

#### For Julia (ProfileView.jl):
```julia
using Profile
using ProfileView

# Profile a single strategy
include("src/kernels/TwoMM.jl")
using .TwoMM

# Setup
params = TwoMM.DATASETS_2MM["LARGE"]
ni, nj, nk, nl = params.ni, params.nj, params.nk, params.nl
alpha, beta = Ref(1.5), Ref(1.2)
A = rand(ni, nk); B = rand(nk, nj)
tmp = zeros(ni, nj); C = rand(nj, nl); D = rand(ni, nl)

# Warmup
TwoMM.kernel_2mm_threads_static!(alpha[], beta[], A, B, tmp, C, D)

# Profile
@profile for _ in 1:100
    fill!(tmp, 0.0)
    TwoMM.kernel_2mm_threads_static!(alpha[], beta[], A, B, tmp, C, D)
end

# View flame graph
ProfileView.view()

# Or save to file
using ProfileSVG
ProfileSVG.save("flamegraph_2mm_threads.svg")
```

#### For OpenMP/C (Linux perf):
```bash
# Compile with debug symbols
gcc -O3 -g -fopenmp -o benchmark_2mm benchmark_2mm.c -lm

# Record profile
perf record -g ./benchmark_2mm

# Convert to flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph_2mm_omp.svg
```

---

## Recommended Visualization Strategy

| Data Source | Visualization | Tool | Purpose |
|-------------|---------------|------|---------|
| CSV results | Bar charts | Python matplotlib | Compare Julia vs OpenMP metrics |
| CSV results | Line plots | Python matplotlib | Scaling analysis (threads vs speedup) |
| CSV results | Tables | LaTeX | Publication metrics |
| Julia Profile | Flame graph | ProfileView/ProfileSVG | Identify Julia hotspots |
| perf record | Flame graph | FlameGraph (Brendan Gregg) | Identify OpenMP hotspots |

---

## DAS-5 Profiling Workflow

```bash
# 1. Run benchmarks, collect CSV
srun -N 1 -c 16 julia -t 16 scripts/run_2mm.jl --dataset LARGE --output csv

# 2. Profile Julia (separate run)
srun -N 1 -c 16 julia -t 16 scripts/profile_2mm.jl --strategy threads_static

# 3. Profile OpenMP (separate run)  
srun -N 1 -c 16 perf record -g ./benchmark_2mm_omp
```

---

## Summary

| Question | Answer |
|----------|--------|
| Can I make flame graphs from CSV? | No |
| What are CSV files good for? | Comparison charts, tables, metrics |
| How do I get flame graphs? | Separate profiling runs with Profile.jl or perf |
| Can I compare Julia vs OpenMP flame graphs? | Yes, but they're separate visualizations |

