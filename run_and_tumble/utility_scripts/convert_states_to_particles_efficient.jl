#!/usr/bin/env julia
using JLD2
include("../modules_run_and_tumble.jl")   # provides FP and the new Particle/State

# Minimal compatibility types matching the old saved layout
module CompatFP
    mutable struct Particle
        position::Vector{Int64}
        direction::Vector{Float64}
    end
    mutable struct State{N,C}
        t::Int64
        particles::Vector{Particle}
        ρ::Array{Int64,N}
        ρ₊::Array{Int64,N}
        ρ₋::Array{Int64,N}
        ρ_avg::Array{Float64,N}
        ρ_matrix_avg_cuts::C
        T::Float64
        potential::Any
        forcing::Any
        exp_table::Any
    end
end

function convert_file(path::String)
    # Map old on-disk type names to the compatibility types
    typemap = Dict(
        "FP.Particle"     => CompatFP.Particle,
        "Main.FP.Particle"=> CompatFP.Particle,
        "FP.State"        => CompatFP.State,
        "Main.FP.State"   => CompatFP.State,
    )

    data = load(path; typemap=typemap)
    old_state = data["state"]
    param     = data["param"]
    potential = get(data, "potential", nothing)

    D = length(param.dims)
    particles = Vector{FP.Particle{D}}(undef, length(old_state.particles))
    for (i, p) in enumerate(old_state.particles)
        particles[i] = FP.Particle{D}(Tuple(p.position), Tuple(p.direction))
    end

    new_state = FP.State(
        old_state.t,
        particles,
        old_state.ρ,
        old_state.ρ₊,
        old_state.ρ₋,
        old_state.ρ_avg,
        old_state.ρ_matrix_avg_cuts,
        old_state.T,
        potential === nothing ? old_state.potential : potential,
        old_state.forcing,
        old_state.exp_table,
    )

    out = joinpath(dirname(path), splitext(basename(path))[1] * "_converted.jld2")
    @save out state=new_state param potential
    println("Converted -> $out")
end

function main()
    isempty(ARGS) && error("Usage: julia convert_states.jl file1.jld2 [file2.jld2 ...]")
    for f in ARGS
        try
            convert_file(f)
        catch e
            @warn "Failed to convert $f" exception=e
        end
    end
end

main()
