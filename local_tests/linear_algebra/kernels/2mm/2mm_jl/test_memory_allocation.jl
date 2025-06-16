# Test script to verify memory allocation improvements
# This demonstrates that the improved implementation has near-zero allocation per iteration

include("PolyBench2MM_Improved.jl")
using .PolyBench2MM_Improved
using Printf

# Test with SMALL dataset
ni, nj, nk, nl = 40, 50, 70, 80

# Pre-allocate arrays
A = Matrix{Float32}(undef, ni, nk)
B = Matrix{Float32}(undef, nk, nj)
C = Matrix{Float32}(undef, nj, nl)
D = Matrix{Float32}(undef, ni, nl)
D_original = Matrix{Float32}(undef, ni, nl)
tmp = Matrix{Float32}(undef, ni, nj)

alpha = Ref(1.5f0)
beta = Ref(1.2f0)

# Initialize arrays
PolyBench2MM_Improved.init_arrays!(alpha, beta, tmp, A, B, C, D_original)

println("Memory Allocation Test")
println("="^60)
println("Dataset: SMALL (40×50×70×80)")
println()

# Test each implementation
implementations = [
    ("Sequential", PolyBench2MM_Improved.kernel_2mm_seq!),
    ("Threads (static)", PolyBench2MM_Improved.kernel_2mm_threads_static!),
    ("BLAS", PolyBench2MM_Improved.kernel_2mm_blas!),
    ("Tiled", PolyBench2MM_Improved.kernel_2mm_tiled!),
]

println("Testing allocations per iteration:")
println("-"^60)
println("Implementation       | Allocations | Bytes Allocated")
println("-"^60)

for (name, func) in implementations
    # Warm up
    copyto!(D, D_original)
    fill!(tmp, 0.0f0)
    func(alpha[], beta[], tmp, A, B, C, D)
    
    # Measure allocations
    copyto!(D, D_original)
    fill!(tmp, 0.0f0)
    
    allocs = @allocated func(alpha[], beta[], tmp, A, B, C, D)
    
    # Count number of allocations
    # Create a simple function to count allocations
    function count_allocs()
        copyto!(D, D_original)
        fill!(tmp, 0.0f0)
        func(alpha[], beta[], tmp, A, B, C, D)
    end
    
    # Use @timed to get allocation count
    stats = @timed count_allocs()
    
    @printf("%-20s | %11d | %15d\n", name, stats.alloc, allocs)
end

println("-"^60)
println()

# Now test the full benchmark function to show total overhead
println("Testing full benchmark iteration (including setup):")
println("-"^60)

for (name, func) in implementations
    # Create the full benchmark iteration
    function bench_iter()
        copyto!(D, D_original)
        fill!(tmp, 0.0f0)
        func(alpha[], beta[], tmp, A, B, C, D)
    end
    
    # Warm up
    bench_iter()
    
    # Measure
    stats = @timed bench_iter()
    bytes = @allocated bench_iter()
    
    @printf("%-20s | %11d | %15d | %.3f ms\n", 
            name, stats.alloc, bytes, stats.time * 1000)
end

println("-"^60)
println()
println("Note: Near-zero allocations indicate proper in-place operations")
println("      Any allocations are likely from internal BLAS or thread overhead")

# Demonstrate the problem with the old approach
println()
println("Comparison with allocation-heavy approach:")
println("-"^60)

function bad_kernel_2mm(alpha, beta, A, B, C, D)
    # This demonstrates what NOT to do
    tmp = alpha * A * B  # Allocates new matrix
    return tmp * C + beta * D  # Allocates another new matrix
end

# Measure bad implementation
D_bad = copy(D_original)
bad_stats = @timed bad_result = bad_kernel_2mm(alpha[], beta[], A, B, C, D_bad)
bad_bytes = @allocated bad_kernel_2mm(alpha[], beta[], A, B, C, D_bad)

@printf("%-20s | %11d | %15d | %.3f ms\n", 
        "BAD (allocating)", bad_stats.alloc, bad_bytes, bad_stats.time * 1000)
println("-"^60)
println()
println("The 'BAD' implementation allocates ~$(round(bad_bytes/1024, digits=1)) KB per iteration!")
println("This would cause significant GC pressure in a real benchmark.")