# DAS-5 Julia PolyBench Deployment Guide
# ========================================
# Speed, Violence, Momentum 

## PART 0: JULIA ON DAS-5 - NO MULTI-INSTALL NEEDED

DAS-5 uses a **module system**. Julia is installed ONCE on a shared filesystem
and made available to ALL nodes via `module load`. When you submit a SLURM job,
the compute node loads Julia from the same shared location. No per-node installation.

```bash
# Check available Julia versions on DAS-5 headnode
module avail julia

# Load Julia (once per session or in SLURM script)
module load julia/1.10

# Verify
julia --version

# Julia packages are stored in your HOME (shared across nodes)
# Default: ~/.julia/
# Force specific depot:
export JULIA_DEPOT_PATH="$HOME/.julia"
```

For multi-node: Julia's `--machine-file` flag connects workers via SSH.
Each worker uses the same Julia from the module system.

---

## PART 1: TRANSFER FROM WSL2 TO DAS-5

### Option A: Direct SCP (Recommended)
```bash
# From WSL2 terminal:
cd /mnt/c/Users/aldej/Desktop/Julia_versus_OpenMP

# Transfer entire project
scp -r julia_polybench YOUR_USERNAME@fs0.das5.cs.vu.nl:~/Julia_versus_OpenMP/

# Or use rsync for incremental updates (faster for subsequent syncs)
rsync -avz --progress julia_polybench/ \
    YOUR_USERNAME@fs0.das5.cs.vu.nl:~/Julia_versus_OpenMP/julia_polybench/
```

### Option B: Using SSH Config (Convenience)
```bash
# Add to ~/.ssh/config in WSL2:
Host das5
    HostName fs0.das5.cs.vu.nl
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_rsa

# Then simply:
scp -r julia_polybench das5:~/Julia_versus_OpenMP/
rsync -avz julia_polybench/ das5:~/Julia_versus_OpenMP/julia_polybench/
```

### Option C: Git Clone on DAS-5 (If repo is on GitHub)
```bash
# SSH into DAS-5
ssh YOUR_USERNAME@fs0.das5.cs.vu.nl

# Clone directly on the cluster
cd ~
git clone https://github.com/TheSpawnal/Julia_versus_OpenMP.git
```

---

## PART 2: DAS-5 SETUP (First Time)

```bash
# SSH into DAS-5 headnode
ssh YOUR_USERNAME@fs0.das5.cs.vu.nl

# Create project structure
mkdir -p ~/Julia_versus_OpenMP/julia_polybench/results

# Check available modules
module avail

# Load Julia and verify
module load prun
module load julia/1.10
julia --version

# Optional: Add to ~/.bashrc for auto-load
echo 'module load prun' >> ~/.bashrc
echo 'module load julia/1.10' >> ~/.bashrc

# Precompile packages (do this ONCE on headnode)
cd ~/Julia_versus_OpenMP/julia_polybench
julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "Statistics", "Dates", "Printf"])'

# Test locally (short test, headnode is for development only)
julia -t 4 scripts/run_2mm.jl --dataset MINI --iterations 2
```

---

## PART 3: SLURM JOB SUBMISSION

### 3.1 Check Cluster Status
```bash
# Available partitions and nodes
sinfo

# Your current jobs
squeue -u $USER

# Detailed node info
sinfo -o "%40N  %40f"
```

### 3.2 Submit All Benchmarks (MEDIUM Dataset)
```bash
cd ~/Julia_versus_OpenMP/julia_polybench

# Create the SLURM script (copy from artifacts or use the ones below)
# Then submit:
sbatch das5_all_benchmarks.slurm

# Monitor
watch -n 5 squeue -u $USER

# View output in real-time
tail -f julia_bench_*.out
```

### 3.3 Submit Individual Benchmarks
```bash
# Quick single benchmark (15 min limit)
sbatch --job-name=julia_2mm \
       --output=2mm_%j.out \
       --error=2mm_%j.err \
       --time=00:15:00 \
       -N 1 \
       --cpus-per-task=16 \
       --partition=defq \
       --wrap='. /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun julia/1.10; export JULIA_NUM_THREADS=16; export OPENBLAS_NUM_THREADS=1; cd ~/Julia_versus_OpenMP/julia_polybench; julia -t 16 scripts/run_2mm.jl --dataset MEDIUM --output csv'

# Same for 3mm
sbatch --job-name=julia_3mm \
       --output=3mm_%j.out \
       --time=00:15:00 \
       -N 1 --cpus-per-task=16 --partition=defq \
       --wrap='. /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun julia/1.10; export JULIA_NUM_THREADS=16; export OPENBLAS_NUM_THREADS=1; cd ~/Julia_versus_OpenMP/julia_polybench; julia -t 16 scripts/run_3mm.jl --dataset MEDIUM --output csv'
```

### 3.4 Interactive Node Session (For Debugging)
```bash
# Request interactive node (15 min max during daytime)
srun -N 1 --cpus-per-task=16 --time=00:15:00 --partition=defq --pty bash

# Once on compute node:
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load prun julia/1.10
export JULIA_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1

cd ~/Julia_versus_OpenMP/julia_polybench
julia -t 16 scripts/run_2mm.jl --dataset MEDIUM --output csv
```

---

## PART 4: CLEVER ONE-LINERS

### 4.1 Submit All 6 Benchmarks in Parallel
```bash
cd ~/Julia_versus_OpenMP/julia_polybench

for bench in 2mm 3mm cholesky correlation jacobi2d nussinov; do
    sbatch --job-name="jl_${bench}" \
           --output="${bench}_%j.out" \
           --time=00:15:00 \
           -N 1 --cpus-per-task=16 --partition=defq \
           --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun julia/1.10; export JULIA_NUM_THREADS=16; export OPENBLAS_NUM_THREADS=1; cd ~/Julia_versus_OpenMP/julia_polybench; julia -t 16 scripts/run_${bench}.jl --dataset MEDIUM --output csv"
done
```

### 4.2 Thread Scaling Study (1,2,4,8,16 threads)
```bash
cd ~/Julia_versus_OpenMP/julia_polybench

for threads in 1 2 4 8 16; do
    for bench in 2mm 3mm cholesky; do
        sbatch --job-name="jl_${bench}_t${threads}" \
               --output="${bench}_t${threads}_%j.out" \
               --time=00:15:00 \
               -N 1 --cpus-per-task=${threads} --partition=defq \
               --wrap=". /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun julia/1.10; export JULIA_NUM_THREADS=${threads}; export OPENBLAS_NUM_THREADS=1; cd ~/Julia_versus_OpenMP/julia_polybench; julia -t ${threads} scripts/run_${bench}.jl --dataset MEDIUM --output csv"
    done
done
```

### 4.3 Weekend/Night Long Runs (LARGE dataset)
```bash
# Submit for off-hours execution
sbatch --job-name=jl_all_LARGE \
       --output=all_LARGE_%j.out \
       --time=02:00:00 \
       --begin=22:00 \
       -N 1 --cpus-per-task=16 --partition=defq \
       --wrap='. /etc/bashrc; . /etc/profile.d/lmod.sh; module load prun julia/1.10; export JULIA_NUM_THREADS=16; export OPENBLAS_NUM_THREADS=1; cd ~/Julia_versus_OpenMP/julia_polybench; for b in 2mm 3mm cholesky correlation jacobi2d nussinov; do julia -t 16 scripts/run_${b}.jl --dataset LARGE --output csv; done'
```

### 4.4 Cancel All Your Jobs
```bash
scancel -u $USER
```

### 4.5 Aggregate All CSV Results
```bash
cd ~/Julia_versus_OpenMP/julia_polybench/results

# Combine all CSVs into one (keep first header only)
head -1 $(ls *.csv | head -1) > all_results.csv
for f in *.csv; do
    tail -n +2 "$f" >> all_results.csv
done
```

---

## PART 5: RETRIEVE RESULTS

```bash
# From WSL2:
scp 'YOUR_USERNAME@fs0.das5.cs.vu.nl:~/Julia_versus_OpenMP/julia_polybench/results/*.csv' \
    /mnt/c/Users/aldej/Desktop/Julia_versus_OpenMP/julia_polybench/results/

# Or rsync for efficiency
rsync -avz das5:~/Julia_versus_OpenMP/julia_polybench/results/ \
    /mnt/c/Users/aldej/Desktop/Julia_versus_OpenMP/julia_polybench/results/
```

---

## PART 6: VISUALIZE RESULTS

```bash
# On local machine (WSL2 or Windows)
cd /mnt/c/Users/aldej/Desktop/Julia_versus_OpenMP/julia_polybench

# Install deps if needed
pip install pandas matplotlib seaborn numpy

# Generate plots
python3 visualize_benchmarks.py results/*.csv
```

---

## PART 7: DAS-5 RULES REMINDER

| Rule | Value |
|------|-------|
| Max cores per node | 16 |
| Default partition | defq |
| Max daytime job | 15 minutes |
| Night/weekend | Up to 4+ hours |
| NEVER run on headnode | Use SLURM/srun |

---

## PART 8: MULTI-NODE (FUTURE)

For later multi-node expansion using Julia Distributed:

```bash
# Request 4 nodes
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Create machine file
scontrol show hostnames $SLURM_NODELIST > machines.txt

# Run with distributed Julia
julia --machine-file=machines.txt your_distributed_script.jl
```

This uses SSH internally - Julia workers spawn on each node using the
shared module system. No separate installation needed.

---

# Quick Reference Card

```
# Transfer project
rsync -avz julia_polybench/ das5:~/Julia_versus_OpenMP/julia_polybench/

# Submit all benchmarks
sbatch das5_all_benchmarks.slurm

# Monitor
squeue -u $USER

# View output
tail -f julia_bench_*.out

# Cancel
scancel -u $USER

# Retrieve results
scp 'das5:~/Julia_versus_OpenMP/julia_polybench/results/*.csv' ./results/
```
