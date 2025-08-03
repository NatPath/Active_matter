#!/usr/bin/env julia

# Simple script to test the built-in benchmarking in update!
include("potentials.jl")
include("modules_run_and_tumble.jl")
using .FP
using Random

function test_internal_benchmarking()
    println("Testing internal benchmarking of update! function")
    
    # Setup a small simulation for quick testing
    L = 16
    ρ₀ = 50.0
    α = 0.1
    γ = 0.25
    ϵ = 0.0
    T = 1.0
    D = 1.0
    potential_type = "xy_slides"
    fluctuation_type = "profile_switch" 
    potential_magnitude = 4.0
    ffr = 0.0
    
    bond_indices = ([L÷2, L÷2], [L÷2+1, L÷2])
    forcing = Potentials.setBondForce(bond_indices, true, 0.0)
    dims = (L, L)
    
    param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffr)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    
    rng = MersenneTwister(42)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=false)
    state = FP.setState(0, rng, param, T, potential, forcing)
    
    println("Setup: $(param.N) particles in 2D system")
    
    # Test with internal benchmarking enabled
    println("\n" * "="*50)
    println("Running update! with benchmark=true")
    println("="*50)
    
    # Run one update with benchmarking to see the breakdown
    result = FP.update!(param, state, rng; benchmark=true)
    
    if result !== nothing
        println("\nInternal benchmark completed successfully!")
    else
        println("\nNo benchmark results returned - benchmarking may not be fully implemented.")
    end
end

test_internal_benchmarking()
