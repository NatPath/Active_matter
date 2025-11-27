using BenchmarkTools
using Profile
include("modules_run_and_tumble.jl")
using .FP

"""
Benchmark specific components of the update! function
"""
function benchmark_update_components(param, state, rng; samples=1000)
    println("=== Detailed Component Benchmarking ===")
    
    # Benchmark random number generation
    println("Random number generation:")
    if length(param.dims) == 1
        b1 = @benchmark rand($rng, 1:$(3*(param.N+2))) samples=samples
    else
        b1 = @benchmark rand($rng, 1:$(5*(param.N+2))) samples=samples
    end
    display(b1)
    
    # Benchmark jump probability calculation
    println("\nJump probability calculation:")
    particle = state.particles[1]
    V = state.potential.V
    T = state.T
    if length(param.dims) == 1
        spot_idx = particle.position[1]
        cand_idx = mod1(spot_idx + 1, param.dims[1])
        b2 = @benchmark FP.calculate_jump_probability(
            $(particle.direction[1]), 1, $(param.D), 
            $(V[cand_idx] - V[spot_idx]), $T, $(state.exp_table); 
            ϵ=$(param.ϵ), bond_forcing=0.0
        ) samples=samples
    else
        i, j = particle.position
        dirvec = [1.0, 0.0]
        cand = (mod1(i+1, param.dims[1]), j)
        b2 = @benchmark FP.calculate_jump_probability(
            $(particle.direction), $dirvec, $(param.D),
            $(V[cand...] - V[i,j]), $T, $(state.exp_table);
            bond_forcing=0.0
        ) samples=samples
    end
    display(b2)
    
    # Benchmark tower sampling
    println("\nTower sampling:")
    p_arr = [0.3, 0.7]
    b3 = @benchmark FP.tower_sampling($p_arr, $(sum(p_arr)), $rng) samples=samples
    display(b3)
    
    # Benchmark density updates
    println("\nDensity updates:")
    if length(param.dims) == 1
        old_pos = 1
        new_pos = 2
        b4 = @benchmark begin
            $(state.ρ)[$old_pos] -= 1
            $(state.ρ)[$new_pos] += 1
            $(state.ρ)[$old_pos] += 1  # restore
            $(state.ρ)[$new_pos] -= 1  # restore
        end samples=samples
    else
        old_pos = (1, 1)
        new_pos = (1, 2)
        b4 = @benchmark begin
            $(state.ρ)[$old_pos...] -= 1
            $(state.ρ)[$new_pos...] += 1
            $(state.ρ)[$old_pos...] += 1  # restore
            $(state.ρ)[$new_pos...] -= 1  # restore
        end samples=samples
    end
    display(b4)
    
    println("=====================================")
end

"""
Profile the update! function for a specified number of iterations
"""
function profile_update!(param, state, rng; n_updates=1000)
    println("Profiling update! function for $n_updates iterations...")
    
    # Warm up
    for _ in 1:10
        FP.update!(param, state, rng)
    end
    
    # Profile
    Profile.clear()
    @profile for _ in 1:n_updates
        FP.update!(param, state, rng)
    end
    
    Profile.print()
    return Profile.fetch()
end

"""
Compare performance with and without exponential lookup table
"""
function benchmark_exp_lookup(param, state, rng; samples=10000)
    println("=== Exponential Lookup Table Benchmark ===")
    
    # Test values
    test_values = randn(samples) * 5  # Random values around 0
    
    # Direct exp calculation
    println("Direct exp() calculation:")
    b1 = @benchmark [exp(x) for x in $test_values]
    display(b1)
    
    # Lookup table
    println("Lookup table:")
    exp_table = state.exp_table
    b2 = @benchmark [FP.lookup_exp($exp_table, x) for x in $test_values]
    display(b2)
    
    println("Speed improvement: $(round(median(b1).time / median(b2).time, digits=2))x")
    println("=====================================")
end
