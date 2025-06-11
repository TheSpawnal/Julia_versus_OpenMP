# test_simple.jl
# Simple test to verify the module loads and works

println("Testing PolyBench2MM module...")

# Test 1: Load the module
println("\n1. Loading module...")
try
    include("PolyBench2MM.jl")
    using .PolyBench2MM
    println("✓ Module loaded successfully")
catch e
    println("✗ Failed to load module: $e")
    exit(1)
end

# Test 2: Run a simple benchmark
println("\n2. Running simple benchmark...")
try
    metrics = PolyBench2MM.main(implementation="blas", datasets=["MINI"])
    println("✓ Benchmark completed successfully")
catch e
    println("✗ Benchmark failed: $e")
    exit(1)
end

# Test 3: Verify implementations
println("\n3. Verifying implementations...")
try
    results = PolyBench2MM.verify_implementations("MINI")
    println("✓ Verification completed")
catch e
    println("✗ Verification failed: $e")
    exit(1)
end

println("\n✓ All tests passed!")
println("\nFor distributed testing, run:")
println("  1. julia> include(\"setup_distributed.jl\")")
println("  2. julia> include(\"test_simple.jl\")")