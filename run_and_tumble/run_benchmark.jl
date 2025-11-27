#!/usr/bin/env julia

############### run_benchmark.jl ###############

using BenchmarkTools
using Random
using Printf
using Statistics
using JLD2
using YAML
using Dates

# Include necessary modules
include("potentials.jl")
include("modules_run_and_tumble.jl")
include("benchmark_utils.jl")
using .FP

function setup_benchmark_simulation(system_size=16, dim=2; config_file=nothing)
    """Set up a simulation for benchmarking with specified parameters."""
    println("Setting up benchmark simulation...")
    
    if config_file !== nothing && isfile(config_file)
        println("Loading configuration from: $config_file")
        params = YAML.load_file(config_file)
    else
        println("Using default benchmark parameters")
        params = Dict(
            "dim_num" => dim,
            "L" => system_size,
            "ρ₀" => 50.0,  # Lower density for faster benchmarking
            "D" => 1.0,
            "α" => 0.1,
            "γ" => 0.25,
            "ϵ" => 0.0,
            "T" => 1.0,
            "potential_type" => "xy_slides",
            "fluctuation_type" => "profile_switch",
            "potential_magnitude" => 4.0,
            "forcing_type" => "center_bond_x",
            "ffr" => 0.0,
            "forcing_magnitude" => 0.0
        )
    end
    
    # Setup simulation parameters
    dim_num = params["dim_num"]
    L = params["L"]
    ρ₀ = params["ρ₀"]
    D = params["D"]
    α = params["α"]
    γ = params["γ"]
    ϵ = params["ϵ"]
    T = params["T"]
    potential_type = params["potential_type"]
    fluctuation_type = params["fluctuation_type"]
    potential_magnitude = params["potential_magnitude"]
    forcing_type = params["forcing_type"]
    ffr = params["ffr"]
    forcing_magnitude = params["forcing_magnitude"]
    
    # Setup forcing
    if forcing_type == "center_bond_x"
        if dim_num == 1
            bond_indices = ([L÷2], [L÷2+1])
        elseif dim_num == 2
            bond_indices = ([L÷2, L÷2], [L÷2+1, L÷2])
        end
    elseif forcing_type == "center_bond_y"
        if dim_num == 1
            bond_indices = ([L÷2], [L÷2+1])
        elseif dim_num == 2
            bond_indices = ([L÷2, L÷2], [L÷2, L÷2+1])
        end
    else
        bond_indices = ([1], [2])
    end
    
    forcing = Potentials.setBondForce(bond_indices, true, forcing_magnitude)
    dims = ntuple(i -> L, dim_num)
    
    # Create simulation objects
    param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffr)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    
    rng = MersenneTwister(42)  # Fixed seed for reproducible benchmarks
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=false)
    state = FP.setState(0, rng, param, T, potential, forcing)
    
    println("Benchmark setup complete:")
    println("  Dimensions: $dim_num D")
    println("  System size: $L")
    println("  Particles: $(param.N)")
    println("  Density: $ρ₀")
    
    return param, state, rng
end

function benchmark_single_update(param, state, rng; samples=1000)
    """Benchmark a single update step."""
    println("\n=== Benchmarking Single Update Step ===")
    
    # Warm up
    for _ in 1:10
        FP.update!(param, state, rng)
    end
    
    # Benchmark
    result = @benchmark FP.update!($param, $state, $rng) samples=samples
    
    println("Single update statistics:")
    println("  Median time: $(round(median(result).time / 1e6, digits=3)) ms")
    println("  Mean time: $(round(mean(result).time / 1e6, digits=3)) ms")
    println("  Min time: $(round(minimum(result).time / 1e6, digits=3)) ms")
    println("  Max time: $(round(maximum(result).time / 1e6, digits=3)) ms")
    println("  Samples: $(length(result.times))")
    
    return result
end

function benchmark_component_details(param, state, rng; samples=1000)
    """Run detailed component benchmarking."""
    println("\n=== Detailed Component Benchmarking ===")
    benchmark_update_components(param, state, rng; samples=samples)
end

function benchmark_sweep_performance(param, state, rng; n_sweeps=100)
    """Benchmark multiple sweeps."""
    println("\n=== Benchmarking Multiple Sweeps ===")
    
    # Clone state to avoid affecting the original
    state_copy = deepcopy(state)
    
    # Warm up
    for _ in 1:5
        FP.update!(param, state_copy, rng)
    end
    
    # Benchmark multiple sweeps
    println("Running $n_sweeps sweeps...")
    result = @benchmark begin
        for _ in 1:$n_sweeps
            FP.update!($param, $state_copy, $rng)
        end
    end samples=10
    
    avg_time_per_sweep = median(result).time / (n_sweeps * 1e6)  # Convert to ms
    
    println("Sweep performance:")
    println("  Total time for $n_sweeps sweeps: $(round(median(result).time / 1e9, digits=3)) s")
    println("  Average time per sweep: $(round(avg_time_per_sweep, digits=3)) ms")
    println("  Estimated sweeps per second: $(round(1000/avg_time_per_sweep, digits=1))")
    
    return result, avg_time_per_sweep
end

function benchmark_memory_usage(param, state, rng; n_sweeps=100)
    """Benchmark memory allocation during simulation."""
    println("\n=== Memory Usage Benchmarking ===")
    
    # Single update memory usage
    state_copy = deepcopy(state)
    single_update_result = @benchmark FP.update!($param, $state_copy, $rng) samples=100
    
    println("Memory usage per update:")
    println("  Allocations: $(single_update_result.allocs)")
    println("  Memory: $(round(single_update_result.memory / 1024, digits=2)) KB")
    
    # Multiple sweeps memory usage
    state_copy = deepcopy(state)
    multi_sweep_result = @benchmark begin
        for _ in 1:$n_sweeps
            FP.update!($param, $state_copy, $rng)
        end
    end samples=5
    
    avg_allocs_per_sweep = multi_sweep_result.allocs / n_sweeps
    avg_memory_per_sweep = (multi_sweep_result.memory / n_sweeps) / 1024  # KB
    
    println("Memory usage for $n_sweeps sweeps:")
    println("  Total allocations: $(multi_sweep_result.allocs)")
    println("  Total memory: $(round(multi_sweep_result.memory / 1024, digits=2)) KB")
    println("  Average allocations per sweep: $(round(avg_allocs_per_sweep, digits=1))")
    println("  Average memory per sweep: $(round(avg_memory_per_sweep, digits=2)) KB")
    
    return single_update_result, multi_sweep_result
end

function benchmark_scaling(; system_sizes=[8, 16, 32], config_file=nothing)
    """Benchmark performance scaling with system size."""
    println("\n=== System Size Scaling Benchmark ===")
    
    results = Dict()
    
    for L in system_sizes
        println("\nBenchmarking system size L=$L...")
        param, state, rng = setup_benchmark_simulation(L, 2; config_file=config_file)
        
        # Benchmark single update
        single_result = @benchmark FP.update!($param, $state, $rng) samples=100
        median_time_ms = median(single_result).time / 1e6
        
        # Store results
        results[L] = Dict(
            "particles" => param.N,
            "median_time_ms" => median_time_ms,
            "time_per_particle_μs" => (median_time_ms * 1000) / param.N,
            "memory_kb" => single_result.memory / 1024,
            "allocations" => single_result.allocs
        )
        
        println("  L=$L: $(param.N) particles, $(round(median_time_ms, digits=3)) ms/update")
    end
    
    # Print scaling summary
    println("\nScaling Summary:")
    println("System Size | Particles | Time/Update (ms) | Time/Particle (μs) | Memory (KB)")
    println("-"^80)
    for L in sort(collect(keys(results)))
        r = results[L]
        @printf("     %2d     |   %4d    |      %.3f       |       %.2f        |   %.2f\n", 
                L, r["particles"], r["median_time_ms"], r["time_per_particle_μs"], r["memory_kb"])
    end
    
    return results
end

function benchmark_exp_lookup_performance(param, state, rng)
    """Benchmark exponential lookup table performance."""
    println("\n=== Exponential Lookup Table Benchmark ===")
    benchmark_exp_lookup(param, state, rng; samples=10000)
end

function save_benchmark_results(results, filename="benchmark_results.jld2")
    """Save benchmark results to file."""
    println("\nSaving benchmark results to: $filename")
    timestamp = now()
    
    benchmark_data = Dict(
        "timestamp" => timestamp,
        "results" => results,
        "julia_version" => VERSION,
        "hostname" => gethostname()
    )
    
    @save filename benchmark_data
    println("Benchmark results saved.")
end

function run_full_benchmark(; system_size=16, config_file=nothing, save_results=true)
    """Run a comprehensive benchmark suite."""
    println("="^60)
    println("         RUNNING COMPREHENSIVE BENCHMARK SUITE")
    println("="^60)
    
    # Setup
    param, state, rng = setup_benchmark_simulation(system_size, 2; config_file=config_file)
    
    # Store all results
    all_results = Dict()
    
    # Run benchmarks
    try
        # 1. Single update benchmark
        all_results["single_update"] = benchmark_single_update(param, state, rng; samples=1000)
        
        # 2. Component details
        benchmark_component_details(param, state, rng; samples=1000)
        
        # 3. Sweep performance
        sweep_result, avg_time = benchmark_sweep_performance(param, state, rng; n_sweeps=100)
        all_results["sweep_performance"] = Dict("result" => sweep_result, "avg_time_ms" => avg_time)
        
        # 4. Memory usage
        single_mem, multi_mem = benchmark_memory_usage(param, state, rng; n_sweeps=100)
        all_results["memory_usage"] = Dict("single" => single_mem, "multiple" => multi_mem)
        
        # 5. Exponential lookup performance
        benchmark_exp_lookup_performance(param, state, rng)
        
        # 6. Scaling benchmark
        scaling_results = benchmark_scaling(; system_sizes=[8, 16, 24], config_file=config_file)
        all_results["scaling"] = scaling_results
        
        # Save results if requested
        if save_results
            save_benchmark_results(all_results)
        end
        
        println("\n" * "="^60)
        println("         BENCHMARK SUITE COMPLETED SUCCESSFULLY")
        println("="^60)
        
        return all_results
        
    catch e
        println("\nError during benchmarking: $e")
        rethrow(e)
    end
end

# Main execution
function main()
    println("Julia Run and Tumble Simulation Benchmark")
    println("Julia version: $(VERSION)")
    println("Number of threads: $(Threads.nthreads())")
    
    # Check if config file is provided as argument
    config_file = length(ARGS) > 0 ? ARGS[1] : nothing
    
    if config_file !== nothing
        if isfile(config_file)
            println("Using configuration file: $config_file")
        else
            println("Warning: Configuration file not found: $config_file")
            config_file = nothing
        end
    end
    
    # Run the full benchmark suite
    results = run_full_benchmark(; system_size=16, config_file=config_file, save_results=true)
    
    return results
end

main()
