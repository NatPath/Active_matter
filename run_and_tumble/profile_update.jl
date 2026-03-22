#!/usr/bin/env julia

############### profile_update.jl ###############

using BenchmarkTools
using Random
using Printf
using Statistics
using Profile

# Include necessary modules
include("potentials.jl")
include("modules_run_and_tumble.jl")
using .FP

function setup_simple_simulation(system_size=16, dim=2)
    """Set up a simple simulation for profiling."""
    println("Setting up simulation for profiling...")
    
    # Simple parameters for quick setup
    L = system_size
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
    
    # Setup forcing
    if dim == 1
        bond_indices = ([L÷2], [L÷2+1])
    elseif dim == 2
        bond_indices = ([L÷2, L÷2], [L÷2+1, L÷2])
    end
    
    forcing = Potentials.setBondForce(bond_indices, true, 0.0)
    dims = ntuple(i -> L, dim)
    
    # Create simulation objects
    param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffr)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    
    rng = MersenneTwister(42)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=false)
    state = FP.setState(0, rng, param, T, potential, forcing)
    
    println("Setup complete: $(param.N) particles in $(dim)D system")
    return param, state, rng
end

function benchmark_update_with_internal_timing(param, state, rng; n_samples=100)
    """Benchmark update! function with internal timing using the built-in benchmark flag."""
    println("\n=== Internal Timing Benchmark ===")
    
    # Clone state to avoid side effects
    state_copy = deepcopy(state)
    
    # Warm up
    for _ in 1:10
        FP.update!(param, state_copy, rng)
    end
    
    println("Running $n_samples updates with internal timing...")
    
    # Run updates with benchmarking enabled
    for i in 1:n_samples
        if i % 10 == 0
            println("  Sample $i/$n_samples")
        end
        FP.update!(param, state_copy, rng; benchmark=true)
    end
    
    println("Internal timing benchmark completed.")
end

function profile_update_function(param, state, rng; n_updates=1000)
    """Profile the update! function to see where time is spent."""
    println("\n=== Profiling update! Function ===")
    
    # Clone state to avoid side effects
    state_copy = deepcopy(state)
    
    # Warm up
    for _ in 1:10
        FP.update!(param, state_copy, rng)
    end
    
    println("Profiling $n_updates updates...")
    
    # Clear previous profile data
    Profile.clear()
    
    # Profile the updates
    @profile for _ in 1:n_updates
        FP.update!(param, state_copy, rng)
    end
    
    println("\nProfile results:")
    Profile.print(mincount=10)
    
    return Profile.fetch()
end

function time_update_components_manually(param, state, rng; n_samples=100)
    """Manually time different components of the update process."""
    println("\n=== Manual Component Timing ===")
    
    # Clone state
    state_copy = deepcopy(state)
    
    # Storage for timing results
    times = Dict(
        "total" => Float64[],
        "action_selection" => Float64[],
        "probability_calc" => Float64[],
        "tower_sampling" => Float64[],
        "state_update" => Float64[]
    )
    
    println("Manually timing $n_samples updates...")
    
    for i in 1:n_samples
        if i % 20 == 0
            println("  Sample $i/$n_samples")
        end
        
        total_start = time_ns()
        
        # This is a simplified version - we'd need to modify the actual update! function
        # to get detailed internal timings. For now, let's just time the whole update
        FP.update!(param, state_copy, rng)
        
        total_time = (time_ns() - total_start) / 1e6  # Convert to ms
        push!(times["total"], total_time)
    end
    
    # Print results
    println("\nManual timing results (averaged over $n_samples samples):")
    println("  Total update time: $(round(mean(times["total"]), digits=3)) ± $(round(std(times["total"]), digits=3)) ms")
    
    return times
end

function benchmark_individual_operations(param, state, rng)
    """Benchmark individual operations that happen during update."""
    println("\n=== Individual Operation Benchmarks ===")
    
    # Get some typical values from the state
    particle = state.particles[1]
    V = state.potential.V
    T = state.T
    
    if length(param.dims) == 2
        i, j = particle.position
        
        # Benchmark random action selection
        println("1. Random action selection:")
        b1 = @benchmark rand($rng, 1:5*($(param.N)+2))
        display(b1)
        
        # Benchmark position calculations
        println("\n2. Position calculations:")
        Lx, Ly = param.dims
        b2 = @benchmark begin
            left = (mod1($i-1, $Lx), $j)
            right = (mod1($i+1, $Lx), $j)
            down = ($i, mod1($j-1, $Ly))
            up = ($i, mod1($j+1, $Ly))
        end
        display(b2)
        
        # Benchmark jump probability calculation
        println("\n3. Jump probability calculation:")
        dirvec = [1.0, 0.0]
        cand = (mod1(i+1, Lx), j)
        b3 = @benchmark FP.calculate_jump_probability(
            $(particle.direction), $dirvec, $(param.D),
            $(V[cand...] - V[i,j]), $T, $(state.exp_table)
        )
        display(b3)
        
        # Benchmark tower sampling
        println("\n4. Tower sampling:")
        p_arr = [0.3, 0.7]
        b4 = @benchmark FP.tower_sampling($p_arr, $(sum(p_arr)), $rng)
        display(b4)
        
        # Benchmark density updates
        println("\n5. Density updates:")
        b5 = @benchmark begin
            $(state.ρ)[$i,$j] -= 1
            $(state.ρ)[$cand...] += 1
            # Restore
            $(state.ρ)[$i,$j] += 1
            $(state.ρ)[$cand...] -= 1
        end
        display(b5)
        
    elseif length(param.dims) == 1
        spot_idx = particle.position[1]
        
        # Similar benchmarks for 1D case
        println("1. Random action selection (1D):")
        b1 = @benchmark rand($rng, 1:3*($(param.N)+2))
        display(b1)
        
        println("\n2. Jump probability calculation (1D):")
        cand_idx = mod1(spot_idx + 1, param.dims[1])
        b2 = @benchmark FP.calculate_jump_probability(
            $(particle.direction[1]), 1, $(param.D), 
            $(V[cand_idx] - V[spot_idx]), $T, $(state.exp_table); 
            ϵ=$(param.ϵ), bond_forcing=0.0
        )
        display(b2)
    end
end

function run_complete_profile(; system_size=16, dim=2)
    """Run a complete profiling suite."""
    println("="^60)
    println("         PROFILING update! FUNCTION")
    println("="^60)
    
    # Setup
    param, state, rng = setup_simple_simulation(system_size, dim)
    
    try
        # 1. Profile with Julia's built-in profiler
        profile_data = profile_update_function(param, state, rng; n_updates=1000)
        
        # 2. Benchmark with internal timing (if available)
        benchmark_update_with_internal_timing(param, state, rng; n_samples=50)
        
        # 3. Manual component timing
        manual_times = time_update_components_manually(param, state, rng; n_samples=100)
        
        # 4. Individual operation benchmarks
        benchmark_individual_operations(param, state, rng)
        
        println("\n" * "="^60)
        println("         PROFILING COMPLETED")
        println("="^60)
        
        return profile_data, manual_times
        
    catch e
        println("\nError during profiling: $e")
        rethrow(e)
    end
end

# Main execution
function main()
    println("Julia update! Function Profiler")
    println("Julia version: $(VERSION)")
    
    # Check for command line arguments
    system_size = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 16
    dim = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 2
    
    println("System size: $system_size, Dimensions: $dim")
    
    # Run the complete profiling suite
    profile_data, manual_times = run_complete_profile(; system_size=system_size, dim=dim)
    
    return profile_data, manual_times
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
