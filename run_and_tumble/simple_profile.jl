#!/usr/bin/env julia

############### simple_profile.jl ###############

# Include necessary modules
include("potentials.jl")
include("modules_run_and_tumble.jl")
using .FP
using Random

function simple_timing_test()
    println("Simple timing test for update! function components")
    
    # Setup a small simulation
    L = 16
    ρ₀ = 50.0
    D = 1.0
    α = 0.1
    γ = 0.25
    ϵ = 0.0
    T = 1.0
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
    
    println("Setup complete: $(param.N) particles")
    
    # Test the benchmark feature that's built into your update! function
    println("\nRunning update with internal benchmarking...")
    
    # Run a few updates with benchmarking enabled
    for i in 1:5
        println("\n--- Update $i with benchmarking ---")
        FP.update!(param, state, rng; benchmark=true)
    end
    
    println("\nDone!")
end

simple_timing_test()
