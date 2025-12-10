#!/usr/bin/env julia
#=
Test Suite for Julia PolyBench
Verifies correctness of all implementations
=#

using Test
using Printf
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using PolyBenchJulia
using PolyBenchJulia.Config
using PolyBenchJulia.TwoMM
using PolyBenchJulia.ThreeMM
using PolyBenchJulia.Cholesky
using PolyBenchJulia.Correlation
using PolyBenchJulia.Nussinov

const TEST_DATASET = "SMALL"
const TOLERANCE = 1e-10

@testset "Julia PolyBench Suite" begin
    
    @testset "2MM Kernel" begin
        params = DATASETS_2MM[TEST_DATASET]
        ni, nj, nk, nl = params.ni, params.nj, params.nk, params.nl
        
        alpha = Ref(0.0)
        beta = Ref(0.0)
        A = Matrix{Float64}(undef, ni, nk)
        B = Matrix{Float64}(undef, nk, nj)
        tmp = Matrix{Float64}(undef, ni, nj)
        C = Matrix{Float64}(undef, nj, nl)
        D = Matrix{Float64}(undef, ni, nl)
        D_orig = Matrix{Float64}(undef, ni, nl)
        
        init_2mm!(alpha, beta, A, B, tmp, C, D)
        copyto!(D_orig, D)
        
        # Reference
        D_ref = copy(D_orig)
        tmp_ref = zeros(ni, nj)
        kernel_2mm_seq!(alpha[], beta[], A, B, tmp_ref, C, D_ref)
        
        for strategy in STRATEGIES_2MM
            @testset "$strategy" begin
                kernel_fn = TwoMM.get_kernel(strategy)
                
                fill!(tmp, 0.0)
                copyto!(D, D_orig)
                kernel_fn(alpha[], beta[], A, B, tmp, C, D)
                
                @test maximum(abs.(D - D_ref)) < TOLERANCE
            end
        end
    end
    
    @testset "3MM Kernel" begin
        params = DATASETS_3MM[TEST_DATASET]
        ni, nj, nk, nl, nm = params.ni, params.nj, params.nk, params.nl, params.nm
        
        A = Matrix{Float64}(undef, ni, nk)
        B = Matrix{Float64}(undef, nk, nj)
        C = Matrix{Float64}(undef, nj, nm)
        D = Matrix{Float64}(undef, nm, nl)
        E = Matrix{Float64}(undef, ni, nj)
        F = Matrix{Float64}(undef, nj, nl)
        G = Matrix{Float64}(undef, ni, nl)
        
        init_3mm!(A, B, C, D, E, F, G)
        
        # Reference
        E_ref = zeros(ni, nj)
        F_ref = zeros(nj, nl)
        G_ref = zeros(ni, nl)
        kernel_3mm_seq!(A, B, C, D, E_ref, F_ref, G_ref)
        
        for strategy in STRATEGIES_3MM
            @testset "$strategy" begin
                kernel_fn = ThreeMM.get_kernel(strategy)
                
                fill!(E, 0.0)
                fill!(F, 0.0)
                fill!(G, 0.0)
                kernel_fn(A, B, C, D, E, F, G)
                
                @test maximum(abs.(G - G_ref)) < TOLERANCE
            end
        end
    end
    
    @testset "Cholesky Kernel" begin
        n = DATASETS_CHOLESKY[TEST_DATASET].n
        
        A = Matrix{Float64}(undef, n, n)
        A_orig = Matrix{Float64}(undef, n, n)
        
        init_cholesky!(A)
        copyto!(A_orig, A)
        
        # Reference (using LAPACK)
        A_ref = copy(A_orig)
        LAPACK.potrf!('L', A_ref)
        L_ref = LowerTriangular(A_ref)
        
        for strategy in STRATEGIES_CHOLESKY
            @testset "$strategy" begin
                kernel_fn = Cholesky.get_kernel(strategy)
                
                copyto!(A, A_orig)
                kernel_fn(A)
                L = LowerTriangular(A)
                
                @test maximum(abs.(L - L_ref)) < 1e-8
            end
        end
    end
    
    @testset "Correlation Kernel" begin
        params = DATASETS_CORRELATION[TEST_DATASET]
        m, n = params.m, params.n
        
        data = Matrix{Float64}(undef, m, n)
        data_orig = Matrix{Float64}(undef, m, n)
        mean = Vector{Float64}(undef, n)
        stddev = Vector{Float64}(undef, n)
        corr = Matrix{Float64}(undef, n, n)
        
        init_correlation!(data, mean, stddev, corr)
        copyto!(data_orig, data)
        
        # Reference
        data_ref = copy(data_orig)
        mean_ref = zeros(n)
        stddev_ref = zeros(n)
        corr_ref = zeros(n, n)
        for i in 1:n
            corr_ref[i, i] = 1.0
        end
        kernel_correlation_seq!(data_ref, mean_ref, stddev_ref, corr_ref)
        
        for strategy in STRATEGIES_CORRELATION
            @testset "$strategy" begin
                kernel_fn = Correlation.get_kernel(strategy)
                
                copyto!(data, data_orig)
                fill!(mean, 0.0)
                fill!(stddev, 0.0)
                fill!(corr, 0.0)
                for i in 1:n
                    corr[i, i] = 1.0
                end
                
                kernel_fn(data, mean, stddev, corr)
                
                @test maximum(abs.(corr - corr_ref)) < 1e-8
                
                # Check diagonal is 1
                for i in 1:n
                    @test abs(corr[i, i] - 1.0) < 1e-10
                end
            end
        end
    end
    
    @testset "Nussinov Kernel" begin
        n = DATASETS_NUSSINOV[TEST_DATASET].n
        
        seq = Vector{Int8}(undef, n)
        table = Matrix{Int}(undef, n, n)
        
        init_nussinov!(seq, table)
        
        # Reference
        table_ref = zeros(Int, n, n)
        kernel_nussinov_seq!(seq, table_ref)
        
        for strategy in STRATEGIES_NUSSINOV
            @testset "$strategy" begin
                kernel_fn = Nussinov.get_kernel(strategy)
                
                fill!(table, 0)
                kernel_fn(seq, table)
                
                # Check final score
                @test table[1, n] == table_ref[1, n]
                
                # Check entire upper triangle
                all_match = true
                for i in 1:n, j in (i+1):n
                    if table[i, j] != table_ref[i, j]
                        all_match = false
                        break
                    end
                end
                @test all_match
            end
        end
    end
    
end

println("\nAll tests completed!")
