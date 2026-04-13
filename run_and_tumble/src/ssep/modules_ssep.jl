using LinearAlgebra
using Statistics
using ProgressMeter
isdefined(@__MODULE__, :Potentials) || include(joinpath(@__DIR__, "..", "common", "potentials.jl"))

module FPSSEP
    using ..Potentials: AbstractPotential, BondForce, bondforce_update!, potential_update!, setPotential, setBondForce
    using LinearAlgebra
    using Random: rand, randperm

    export Param, Particle, State, setParam, setState, setDummyState, reset_statistics!
    export get_state_forcings!, averaged_sample_count, validate_exclusion_state, sync_redundant_ssep_views!

    struct ExpLookupTable
        values::Vector{Float64}
        min_val::Float64
        max_val::Float64
        step::Float64
        inv_step::Float64
    end

    function create_exp_lookup(min_val::Float64, max_val::Float64, n_points::Int=10_000)
        step = (max_val - min_val) / (n_points - 1)
        inv_step = 1.0 / step
        values = [exp(min_val + i * step) for i in 0:(n_points - 1)]
        return ExpLookupTable(values, min_val, max_val, step, inv_step)
    end

    function lookup_exp(table::ExpLookupTable, x::Float64)
        if x < table.min_val || x > table.max_val
            return exp(x)
        end
        idx = Int(floor((x - table.min_val) * table.inv_step)) + 1
        return table.values[idx]
    end

    mutable struct Param
        γ::Float64
        dims::Tuple{Vararg{Int}}
        ρ₀::Float64
        N::Int64
        D::Float64
        potential_type::String
        fluctuation_type::String
        potential_magnitude::Float64
        ffr::Union{Float64,Vector{Float64}}
        forcing_rate_scheme::String
    end

    const LEGACY_FORCING_RATE_SCHEME = "legacy_penalty"
    const SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME = "symmetric_normalized"

    function normalize_forcing_rate_scheme(raw_scheme)
        scheme = lowercase(strip(String(raw_scheme)))
        if scheme in ("legacy_penalty", "legacy", "current")
            return LEGACY_FORCING_RATE_SCHEME
        elseif scheme in ("symmetric_normalized", "symmetric", "normalized_symmetric")
            return SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
        end
        throw(ArgumentError("Unsupported forcing_rate_scheme: $raw_scheme. Use \"legacy_penalty\" (or \"current\") or \"symmetric_normalized\"."))
    end

    function setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffr=0.0;
                      forcing_rate_scheme=LEGACY_FORCING_RATE_SCHEME)
        ffrs = if ffr isa AbstractVector
            Float64.(collect(ffr))
        else
            [Float64(ffr)]
        end
        isempty(ffrs) && (ffrs = [0.0])

        num_sites = prod(dims)
        N = Int(round(Float64(ρ₀) * num_sites))
        if N < 0 || N > num_sites
            error("SSEP requires 0 <= N <= number of sites. Got N=$N for dims=$dims (ρ₀=$ρ₀).")
        end

        ffr_value = length(ffrs) == 1 ? ffrs[1] : ffrs
        scheme = normalize_forcing_rate_scheme(forcing_rate_scheme)
        return Param(Float64(γ), dims, Float64(ρ₀), Int64(N), Float64(D), String(potential_type), String(fluctuation_type), Float64(potential_magnitude), ffr_value, scheme)
    end

    @inline function forcing_rate_scheme(param)
        if hasfield(typeof(param), :forcing_rate_scheme)
            return normalize_forcing_rate_scheme(getfield(param, :forcing_rate_scheme))
        end
        return LEGACY_FORCING_RATE_SCHEME
    end

    mutable struct Particle{D}
        position::NTuple{D,Int64}
    end

    mutable struct State{N,C,B,D}
        t::Int64
        particles::Vector{Particle{D}}
        ρ::Array{Int64,N}
        ρ₊::Array{Int64,N}
        ρ₋::Array{Int64,N}
        ρ_avg::Array{Float64,N}
        ρ_matrix_avg_cuts::C
        bond_pass_stats::B
        max_site_occupancy::Int64
        T::Float64
        potential::AbstractPotential
        forcing::Union{BondForce,Vector{BondForce}}
        exp_table::ExpLookupTable
    end

    mutable struct RedundantSSEPViewStatus
        particles_stale::Bool
        directional_densities_stale::Bool
    end

    const REDUNDANT_SSEP_VIEW_STATUS = WeakKeyDict{Any,RedundantSSEPViewStatus}()

    const BOND_PASS_FORWARD_AVG_KEY = :bond_pass_forward_avg
    const BOND_PASS_REVERSE_AVG_KEY = :bond_pass_reverse_avg
    const BOND_PASS_TOTAL_AVG_KEY = :bond_pass_total_avg
    const BOND_PASS_TOTAL_SQ_AVG_KEY = :bond_pass_total_sq_avg
    const BOND_PASS_SAMPLE_COUNT_KEY = :bond_pass_sample_count
    const BOND_PASS_TRACK_MASK_KEY = :bond_pass_track_mask
    const BOND_PASS_SPATIAL_F_AVG_KEY = :bond_pass_spatial_f_avg
    const BOND_PASS_SPATIAL_F2_AVG_KEY = :bond_pass_spatial_f2_avg
    const BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY = :bond_pass_spatial_sample_count
    const SELECTED_SITE_CUTS_KEY = :selected_site_cuts
    const SELECTED_SITE_CUT_SITES_KEY = :selected_site_cut_sites
    const SELECTED_SITE_CUT_ORIGIN_SITE_KEY = :selected_site_cut_origin_site

    function bond_pass_track_mask_for_forcings(forcings::Vector{BondForce}, mode::AbstractString)
        if mode == "all_forcing_bonds"
            return ones(Float64, length(forcings))
        elseif mode == "nonzero_magnitude"
            return [abs(force.magnitude) > 0.0 ? 1.0 : 0.0 for force in forcings]
        end
        throw(ArgumentError("Unsupported bond_pass_count_mode: $mode"))
    end

    function initialize_bond_passage_stats!(bond_pass_stats, n_forces::Int; track_mask=nothing)
        bond_pass_stats[BOND_PASS_FORWARD_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_REVERSE_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_TOTAL_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_TOTAL_SQ_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_SAMPLE_COUNT_KEY] = [0.0]
        if track_mask === nothing
            bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = ones(Float64, n_forces)
        else
            bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = Float64.(track_mask)
        end
        return nothing
    end

    function initialize_spatial_bond_passage_stats!(bond_pass_stats, L::Int)
        bond_pass_stats[BOND_PASS_SPATIAL_F_AVG_KEY] = zeros(Float64, L)
        bond_pass_stats[BOND_PASS_SPATIAL_F2_AVG_KEY] = zeros(Float64, L)
        bond_pass_stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY] = [0.0]
        return nothing
    end

    function selected_site_cut_pair_matrix(ρ::AbstractVector{<:Number}, selected_sites::Vector{Int})
        L = length(ρ)
        cuts = zeros(Float64, length(selected_sites), L)
        for (cut_idx, site) in enumerate(selected_sites)
            occ_site = Float64(ρ[site])
            if occ_site == 0.0
                continue
            end
            @inbounds for other_site in 1:L
                cuts[cut_idx, other_site] = occ_site * Float64(ρ[other_site])
            end
        end
        return cuts
    end

    function initialize_selected_site_cut_stats!(
        ρ_matrix_avg_cuts,
        ρ::AbstractVector{<:Number},
        selected_sites::Vector{Int},
        origin_site::Int,
    )
        ρ_matrix_avg_cuts[SELECTED_SITE_CUTS_KEY] = selected_site_cut_pair_matrix(ρ, selected_sites)
        ρ_matrix_avg_cuts[SELECTED_SITE_CUT_SITES_KEY] = Float64.(selected_sites)
        ρ_matrix_avg_cuts[SELECTED_SITE_CUT_ORIGIN_SITE_KEY] = [Float64(origin_site)]
        return nothing
    end

    function ensure_bond_passage_stats!(state, n_forces::Int; forcings=nothing)
        stats = state.bond_pass_stats
        if !haskey(stats, BOND_PASS_FORWARD_AVG_KEY) || length(stats[BOND_PASS_FORWARD_AVG_KEY]) != n_forces
            stats[BOND_PASS_FORWARD_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_REVERSE_AVG_KEY) || length(stats[BOND_PASS_REVERSE_AVG_KEY]) != n_forces
            stats[BOND_PASS_REVERSE_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_TOTAL_AVG_KEY) || length(stats[BOND_PASS_TOTAL_AVG_KEY]) != n_forces
            stats[BOND_PASS_TOTAL_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_TOTAL_SQ_AVG_KEY) || length(stats[BOND_PASS_TOTAL_SQ_AVG_KEY]) != n_forces
            stats[BOND_PASS_TOTAL_SQ_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_SAMPLE_COUNT_KEY) || length(stats[BOND_PASS_SAMPLE_COUNT_KEY]) != 1
            stats[BOND_PASS_SAMPLE_COUNT_KEY] = [0.0]
        end
        if !haskey(stats, BOND_PASS_TRACK_MASK_KEY) || length(stats[BOND_PASS_TRACK_MASK_KEY]) != n_forces
            stats[BOND_PASS_TRACK_MASK_KEY] = forcings === nothing ? ones(Float64, n_forces) : bond_pass_track_mask_for_forcings(forcings, "nonzero_magnitude")
        end
        return nothing
    end

    function ensure_spatial_bond_passage_stats!(state, L::Int)
        stats = state.bond_pass_stats
        if !haskey(stats, BOND_PASS_SPATIAL_F_AVG_KEY) || length(stats[BOND_PASS_SPATIAL_F_AVG_KEY]) != L
            stats[BOND_PASS_SPATIAL_F_AVG_KEY] = zeros(Float64, L)
        end
        if !haskey(stats, BOND_PASS_SPATIAL_F2_AVG_KEY) || length(stats[BOND_PASS_SPATIAL_F2_AVG_KEY]) != L
            stats[BOND_PASS_SPATIAL_F2_AVG_KEY] = zeros(Float64, L)
        end
        if !haskey(stats, BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY) || length(stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY]) != 1
            stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY] = [0.0]
        end
        return nothing
    end

    function setDummyState(state_to_imitate, ρ_avg, ρ_matrix_avg_cuts, t, bond_pass_stats=nothing)
        stats_to_use = isnothing(bond_pass_stats) ? Dict{Symbol,Vector{Float64}}() : bond_pass_stats
        state = State(
            t,
            state_to_imitate.particles,
            state_to_imitate.ρ,
            state_to_imitate.ρ₊,
            state_to_imitate.ρ₋,
            ρ_avg,
            ρ_matrix_avg_cuts,
            stats_to_use,
            state_to_imitate.max_site_occupancy,
            state_to_imitate.T,
            state_to_imitate.potential,
            state_to_imitate.forcing,
            state_to_imitate.exp_table,
        )
        mark_redundant_ssep_views_stale!(state)
        return state
    end

    function populate_densities!(
        ρ::AbstractArray{<:Integer},
        ρ₊::AbstractArray{<:Integer},
        ρ₋::AbstractArray{<:Integer},
        particles::AbstractVector{<:Particle},
    )
        fill!(ρ, 0)
        fill!(ρ₊, 0)
        fill!(ρ₋, 0)
        for particle in particles
            idx = CartesianIndex(particle.position...)
            ρ[idx] += 1
            ρ₊[idx] += 1
            ρ₋[idx] += 1
        end
    end

    @inline function redundant_ssep_view_status(state)
        return get!(REDUNDANT_SSEP_VIEW_STATUS, state) do
            RedundantSSEPViewStatus(false, false)
        end
    end

    @inline function mark_redundant_ssep_views_stale!(state; particles::Bool=true, directional_densities::Bool=true)
        status = redundant_ssep_view_status(state)
        particles && (status.particles_stale = true)
        directional_densities && (status.directional_densities_stale = true)
        return nothing
    end

    @inline function clear_redundant_ssep_view_stale_flags!(state; particles::Bool=true, directional_densities::Bool=true)
        status = redundant_ssep_view_status(state)
        particles && (status.particles_stale = false)
        directional_densities && (status.directional_densities_stale = false)
        return nothing
    end

    @inline function cartesian_index_position(idx::CartesianIndex{D}) where {D}
        return ntuple(d -> Int64(idx[d]), D)
    end

    function sync_directional_densities_from_occupancy!(state; force::Bool=false)
        status = redundant_ssep_view_status(state)
        if !force && !status.directional_densities_stale
            return nothing
        end
        state.ρ₊ .= state.ρ
        state.ρ₋ .= state.ρ
        status.directional_densities_stale = false
        return nothing
    end

    function sync_particles_from_occupancy!(state; force::Bool=false)
        status = redundant_ssep_view_status(state)
        if !force && !status.particles_stale
            return nothing
        end

        D = ndims(state.ρ)
        occupied_count = Int(sum(state.ρ))
        positions = Vector{NTuple{D,Int64}}(undef, occupied_count)
        position_idx = 1
        @inbounds for idx in CartesianIndices(state.ρ)
            occupancy = Int(state.ρ[idx])
            for _ in 1:occupancy
                positions[position_idx] = cartesian_index_position(idx)
                position_idx += 1
            end
        end

        if length(state.particles) != occupied_count
            particles = Vector{Particle{D}}(undef, occupied_count)
            @inbounds for particle_idx in 1:occupied_count
                particles[particle_idx] = Particle{D}(positions[particle_idx])
            end
            state.particles = particles
        else
            @inbounds for particle_idx in 1:occupied_count
                state.particles[particle_idx].position = positions[particle_idx]
            end
        end

        status.particles_stale = false
        return nothing
    end

    function sync_redundant_ssep_views!(
        state;
        sync_particles::Bool=true,
        sync_directional_densities::Bool=true,
        force::Bool=false,
    )
        sync_particles && sync_particles_from_occupancy!(state; force=force)
        sync_directional_densities && sync_directional_densities_from_occupancy!(state; force=force)
        return state
    end

    function outer_density_2D(ρ::AbstractMatrix{T}) where {T<:Number}
        Nx, Ny = size(ρ)
        v = vec(ρ)
        M = v * v'
        return reshape(M, Nx, Ny, Nx, Ny)
    end

    function center_order(cart_inds, dims)
        centers = ntuple(i -> (dims[i] + 1) / 2, length(dims))
        return sort(collect(eachindex(cart_inds)); by=i -> begin
            pos = Tuple(cart_inds[i])
            sum((Float64(pos[d]) - centers[d])^2 for d in 1:length(dims))
        end)
    end

    function choose_initial_sites(param, rng; ic::AbstractString="random")
        num_sites = prod(param.dims)
        if param.N > num_sites
            error("SSEP requires at most one particle per site. Got N=$(param.N), sites=$num_sites.")
        end
        cart_inds = collect(CartesianIndices(param.dims))
        if ic == "random"
            return randperm(rng, num_sites)[1:param.N]
        elseif ic == "flat"
            return collect(1:param.N)
        elseif ic == "center"
            return center_order(cart_inds, param.dims)[1:param.N]
        elseif ic == "empty"
            return Int[]
        end
        throw(DomainError("Unsupported SSEP initial condition: $ic"))
    end

    function validate_exclusion_state(state)
        if any(state.ρ .< 0) || any(state.ρ .> 1)
            error("Loaded state is not a valid SSEP state: occupancies must stay in {0,1}.")
        end
        if sum(state.ρ) != length(state.particles)
            error("Loaded state is inconsistent: occupancy sum $(sum(state.ρ)) != particle count $(length(state.particles)).")
        end
        return true
    end

    function setState(
        t,
        rng,
        param,
        T,
        potential=setPotential(zeros(Float64, param.dims), zeros(Float64, param.dims)),
        bond_force=setBondForce(([1], [2]), true, 0.0);
        ic="random",
        full_corr_tensor=false,
        bond_pass_count_mode::AbstractString="nonzero_magnitude",
        correlation_observable_mode::AbstractString="full",
        selected_cut_sites::Vector{Int}=Int[],
        selected_cut_origin_site::Union{Nothing,Int}=nothing,
    )
        dim = length(param.dims)
        chosen_sites = choose_initial_sites(param, rng; ic=String(ic))
        cart_inds = collect(CartesianIndices(param.dims))
        particles = Vector{Particle{dim}}(undef, length(chosen_sites))
        for (idx, site_idx) in enumerate(chosen_sites)
            particles[idx] = Particle{dim}(Tuple(cart_inds[site_idx]))
        end

        ρ = zeros(Int64, param.dims...)
        ρ₊ = zeros(Int64, param.dims...)
        ρ₋ = zeros(Int64, param.dims...)
        populate_densities!(ρ, ρ₊, ρ₋, particles)

        ρ_avg = Float64.(ρ)
        if dim == 1
            ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}()
            if correlation_observable_mode == "selected_site_cuts"
                isempty(selected_cut_sites) && throw(ArgumentError("selected_cut_sites must be non-empty when correlation_observable_mode=\"selected_site_cuts\"."))
                origin_site = isnothing(selected_cut_origin_site) ? 1 : mod1(Int(selected_cut_origin_site), param.dims[1])
                initialize_selected_site_cut_stats!(ρ_matrix_avg_cuts, ρ_avg, unique(mod1.(selected_cut_sites, param.dims[1])), origin_site)
            else
                ρ_matrix_avg_cuts[:full] = ρ_avg * transpose(ρ_avg)
            end
        elseif dim == 2
            x_middle = clamp(div(param.dims[1], 2), 1, param.dims[1])
            y_middle = clamp(div(param.dims[2], 2), 1, param.dims[2])
            if full_corr_tensor
                ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                    :full => outer_density_2D(ρ_avg),
                )
            else
                ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                    :x_cut => ρ_avg[:, y_middle] * transpose(ρ_avg[:, y_middle]),
                    :y_cut => ρ_avg[x_middle, :] * transpose(ρ_avg[x_middle, :]),
                    :diag_cut => diag(ρ_avg) * transpose(diag(ρ_avg)),
                )
            end
        else
            throw(DomainError("Only 1D or 2D SSEP is supported"))
        end

        bond_pass_stats = Dict{Symbol,Vector{Float64}}()
        bond_forces = if bond_force isa BondForce
            [bond_force]
        elseif bond_force isa AbstractVector && all(force -> force isa BondForce, bond_force)
            BondForce[force for force in bond_force]
        else
            throw(ArgumentError("bond_force must be BondForce or Vector{BondForce}"))
        end
        track_mask = bond_pass_track_mask_for_forcings(bond_forces, String(bond_pass_count_mode))
        initialize_bond_passage_stats!(bond_pass_stats, length(bond_forces); track_mask=track_mask)
        if dim == 1
            initialize_spatial_bond_passage_stats!(bond_pass_stats, param.dims[1])
        end

        exp_table = create_exp_lookup(-20.0, 20.0, 10_000)
        state = State(
            Int64(t),
            particles,
            ρ,
            ρ₊,
            ρ₋,
            ρ_avg,
            ρ_matrix_avg_cuts,
            bond_pass_stats,
            isempty(particles) ? 0 : 1,
            Float64(T),
            potential,
            bond_forces,
            exp_table,
        )
        validate_exclusion_state(state)
        return state
    end

    function reset_statistics!(state)
        invalidate_ctmc_1d_workspace!(state)
        state.t = 0
        state.ρ_avg .= state.ρ
        state.max_site_occupancy = isempty(state.particles) ? 0 : 1

        dim = ndims(state.ρ)
        if dim == 1
            if haskey(state.ρ_matrix_avg_cuts, SELECTED_SITE_CUTS_KEY)
                selected_sites = Int.(round.(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUT_SITES_KEY]))
                origin_site = haskey(state.ρ_matrix_avg_cuts, SELECTED_SITE_CUT_ORIGIN_SITE_KEY) ?
                    mod1(Int(round(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUT_ORIGIN_SITE_KEY][1])), size(state.ρ, 1)) :
                    1
                initialize_selected_site_cut_stats!(state.ρ_matrix_avg_cuts, state.ρ, selected_sites, origin_site)
            else
                ρf = float(state.ρ)
                state.ρ_matrix_avg_cuts[:full] .= ρf * transpose(ρf)
            end
        elseif dim == 2
            ρf = float(state.ρ)
            if haskey(state.ρ_matrix_avg_cuts, :full)
                state.ρ_matrix_avg_cuts[:full] .= outer_density_2D(ρf)
            else
                x_middle = clamp(div(size(state.ρ, 1), 2), 1, size(state.ρ, 1))
                y_middle = clamp(div(size(state.ρ, 2), 2), 1, size(state.ρ, 2))
                state.ρ_matrix_avg_cuts[:x_cut] .= ρf[:, y_middle] * transpose(ρf[:, y_middle])
                state.ρ_matrix_avg_cuts[:y_cut] .= ρf[x_middle, :] * transpose(ρf[x_middle, :])
                state.ρ_matrix_avg_cuts[:diag_cut] .= diag(ρf) * transpose(diag(ρf))
            end
        else
            throw(DomainError("Only 1D or 2D SSEP is supported"))
        end

        forcings = get_state_forcings!(state)
        track_mask = if haskey(state.bond_pass_stats, BOND_PASS_TRACK_MASK_KEY) && length(state.bond_pass_stats[BOND_PASS_TRACK_MASK_KEY]) == length(forcings)
            Float64.(state.bond_pass_stats[BOND_PASS_TRACK_MASK_KEY])
        else
            bond_pass_track_mask_for_forcings(forcings, "nonzero_magnitude")
        end
        initialize_bond_passage_stats!(state.bond_pass_stats, length(forcings); track_mask=track_mask)
        if dim == 1
            initialize_spatial_bond_passage_stats!(state.bond_pass_stats, size(state.ρ, 1))
        end
        return state
    end

    @inline has_selected_site_cuts(state) = haskey(state.ρ_matrix_avg_cuts, SELECTED_SITE_CUTS_KEY)

    function selected_site_cut_sites_from_state(state)
        has_selected_site_cuts(state) || return Int[]
        return Int.(round.(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUT_SITES_KEY]))
    end

    function selected_site_cut_origin_site_from_state(state, L::Int)
        if has_selected_site_cuts(state) && haskey(state.ρ_matrix_avg_cuts, SELECTED_SITE_CUT_ORIGIN_SITE_KEY)
            return mod1(Int(round(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUT_ORIGIN_SITE_KEY][1])), L)
        end
        return 1
    end

    function param_ffrs(param)
        if param.ffr isa AbstractVector
            return Float64.(collect(param.ffr))
        end
        return [Float64(param.ffr)]
    end

    function get_state_forcings!(state)
        if state.forcing isa BondForce
            state.forcing = BondForce[state.forcing]
        elseif !(state.forcing isa Vector{BondForce})
            if state.forcing isa AbstractVector && all(force -> force isa BondForce, state.forcing)
                state.forcing = BondForce[force for force in state.forcing]
            else
                throw(ArgumentError("state.forcing must be BondForce or Vector{BondForce}"))
            end
        end
        return state.forcing
    end

    function tracked_force_indices_from_state(state, n_forces::Int)
        stats = state.bond_pass_stats
        if !haskey(stats, BOND_PASS_TRACK_MASK_KEY) || length(stats[BOND_PASS_TRACK_MASK_KEY]) != n_forces
            return collect(1:n_forces)
        end
        return [i for i in 1:n_forces if stats[BOND_PASS_TRACK_MASK_KEY][i] > 0.5]
    end

    function record_bond_passage_1d!(forward_counts::Vector{Int64}, reverse_counts::Vector{Int64},
                                     forcings::Vector{BondForce}, tracked_force_indices::Vector{Int},
                                     from_idx::Int, to_idx::Int)
        for force_idx in tracked_force_indices
            force = forcings[force_idx]
            b1 = force.bond_indices[1][1]
            b2 = force.bond_indices[2][1]
            if from_idx == b1 && to_idx == b2
                forward_counts[force_idx] += 1
            elseif from_idx == b2 && to_idx == b1
                reverse_counts[force_idx] += 1
            end
        end
        return nothing
    end

    function record_bond_passage_2d!(forward_counts::Vector{Int64}, reverse_counts::Vector{Int64},
                                     forcings::Vector{BondForce}, tracked_force_indices::Vector{Int},
                                     from_pos::NTuple{2,Int}, to_pos::NTuple{2,Int})
        for force_idx in tracked_force_indices
            force = forcings[force_idx]
            b1 = (force.bond_indices[1][1], force.bond_indices[1][2])
            b2 = (force.bond_indices[2][1], force.bond_indices[2][2])
            if from_pos == b1 && to_pos == b2
                forward_counts[force_idx] += 1
            elseif from_pos == b2 && to_pos == b1
                reverse_counts[force_idx] += 1
            end
        end
        return nothing
    end

    function update_bond_passage_averages!(state, sweep_forward_counts::Vector{Int64}, sweep_reverse_counts::Vector{Int64})
        n_forces = length(sweep_forward_counts)
        ensure_bond_passage_stats!(state, n_forces)
        stats = state.bond_pass_stats

        n_prev = stats[BOND_PASS_SAMPLE_COUNT_KEY][1]
        n_new = n_prev + 1.0

        forward = Float64.(sweep_forward_counts)
        reverse = -Float64.(sweep_reverse_counts)
        total = forward .+ reverse
        total_sq = total .^ 2

        @. stats[BOND_PASS_FORWARD_AVG_KEY] += (forward - stats[BOND_PASS_FORWARD_AVG_KEY]) / n_new
        @. stats[BOND_PASS_REVERSE_AVG_KEY] += (reverse - stats[BOND_PASS_REVERSE_AVG_KEY]) / n_new
        @. stats[BOND_PASS_TOTAL_AVG_KEY] += (total - stats[BOND_PASS_TOTAL_AVG_KEY]) / n_new
        @. stats[BOND_PASS_TOTAL_SQ_AVG_KEY] += (total_sq - stats[BOND_PASS_TOTAL_SQ_AVG_KEY]) / n_new
        stats[BOND_PASS_SAMPLE_COUNT_KEY][1] = n_new
        return nothing
    end

    function update_spatial_bond_passage_averages!(state, sweep_forward_counts::Vector{Int64}, sweep_reverse_counts::Vector{Int64})
        L = length(sweep_forward_counts)
        ensure_spatial_bond_passage_stats!(state, L)
        stats = state.bond_pass_stats

        n_prev = stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY][1]
        n_new = n_prev + 1.0
        f = Float64.(sweep_forward_counts) .- Float64.(sweep_reverse_counts)
        f2 = f .^ 2

        @. stats[BOND_PASS_SPATIAL_F_AVG_KEY] += (f - stats[BOND_PASS_SPATIAL_F_AVG_KEY]) / n_new
        @. stats[BOND_PASS_SPATIAL_F2_AVG_KEY] += (f2 - stats[BOND_PASS_SPATIAL_F2_AVG_KEY]) / n_new
        stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY][1] = n_new
        return nothing
    end

    averaged_sample_count(state) = haskey(state.bond_pass_stats, BOND_PASS_SAMPLE_COUNT_KEY) ? Int(round(state.bond_pass_stats[BOND_PASS_SAMPLE_COUNT_KEY][1])) : max(state.t, 0)

    @inline forcing_rate(ffrs::AbstractVector{<:Real}, force_idx::Int) = force_idx <= length(ffrs) ? Float64(ffrs[force_idx]) : 0.0
    @inline is_static_zero_potential(param) = param.potential_type == "zero" && param.fluctuation_type == "no-fluctuation"

    mutable struct FenwickTree
        tree::Vector{Float64}
        values::Vector{Float64}
        max_bit::Int
        total::Float64
    end

    FenwickTree(n::Int) = FenwickTree(
        zeros(Float64, n),
        zeros(Float64, n),
        largest_power_of_two_leq(max(n, 1)),
        0.0,
    )

    @inline function largest_power_of_two_leq(n::Int)
        bit = 1
        while (bit << 1) <= n
            bit <<= 1
        end
        return bit
    end

    @inline function fenwick_set!(tree::FenwickTree, idx::Int, value::Float64)
        delta = value - tree.values[idx]
        tree.values[idx] = value
        tree.total += delta
        i = idx
        n = length(tree.tree)
        while i <= n
            tree.tree[i] += delta
            i += i & -i
        end
        return value
    end

    @inline function fenwick_total(tree::FenwickTree)
        return tree.total
    end

    @inline function fenwick_sample(tree::FenwickTree, target::Float64)
        idx = 0
        cumulative = 0.0
        bit = tree.max_bit
        while bit != 0
            next_idx = idx + bit
            if next_idx <= length(tree.tree) && cumulative + tree.tree[next_idx] < target
                idx = next_idx
                cumulative += tree.tree[next_idx]
            end
            bit >>= 1
        end
        return idx + 1
    end

    @inline function fenwick_clear!(tree::FenwickTree)
        fill!(tree.tree, 0.0)
        fill!(tree.values, 0.0)
        tree.total = 0.0
        return nothing
    end

    mutable struct CTMC1DWorkspace
        L::Int
        force_signature::Vector{Tuple{Int,Int,Bool,Float64}}
        tracked_force_indices::Vector{Int}
        force_fluctuation_types::Vector{String}
        selected_cut_sites::Vector{Int}
        has_selected_site_cuts::Bool
        has_potential_event::Bool
        ffrs::Vector{Float64}
        param_signature::Tuple{Float64,Float64,Bool,String,Float64}
        left_neighbors::Vector{Int}
        right_neighbors::Vector{Int}
        force_bond_left_sites::Vector{Int}
        force_orientation_signs::Vector{Int8}
        tracked_forces_by_bond::Vector{Vector{Int}}
        right_hop_is_forward::BitVector
        fluctuating_force_indices::Vector{Int}
        force_right_contributions::Vector{Float64}
        bond_right_forcing::Vector{Float64}
        event_tree::FenwickTree
        site_to_particle::Vector{Int}
        sweep_forward_counts::Vector{Int64}
        sweep_reverse_counts::Vector{Int64}
        sweep_spatial_forward_counts::Vector{Int64}
        sweep_spatial_reverse_counts::Vector{Int64}
        density_accum::Vector{Float64}
        density_last_touch::Vector{Float64}
        cut_accum::Matrix{Float64}
        cut_last_touch::Matrix{Float64}
        cut_current::Matrix{Float64}
        corr_accum::Matrix{Float64}
        corr_last_touch::Matrix{Float64}
        corr_current::Matrix{Float64}
    end

    const CTMC_1D_WORKSPACES = WeakKeyDict{Any,CTMC1DWorkspace}()

    @inline function bond_left_site_and_orientation_1d(first_site::Int, second_site::Int, L::Int)
        if mod1(first_site + 1, L) == second_site
            return first_site, Int8(1)
        elseif mod1(second_site + 1, L) == first_site
            return second_site, Int8(-1)
        end
        throw(ArgumentError("CTMC 1D forcing bonds must connect nearest-neighbor sites on the ring. Got ($first_site, $second_site) for L=$L."))
    end

    @inline force_right_contribution(force::BondForce, orientation_sign::Int8) =
        force.magnitude * (force.direction_flag ? Float64(orientation_sign) : -Float64(orientation_sign))

    @inline function update_ctmc_site_observables!(
        density_accum::Vector{Float64},
        density_last_touch::Vector{Float64},
        corr_accum::Matrix{Float64},
        corr_last_touch::Matrix{Float64},
        corr_current::Matrix{Float64},
        ρ::AbstractVector{<:Integer},
        t_local::Float64,
        site::Int,
    )
        density_accum[site] += Float64(ρ[site]) * (t_local - density_last_touch[site])
        density_last_touch[site] = t_local

        L = length(ρ)
        @inbounds for other_site in 1:L
            corr_accum[site, other_site] += corr_current[site, other_site] * (t_local - corr_last_touch[site, other_site])
            corr_last_touch[site, other_site] = t_local
            if other_site != site
                corr_accum[other_site, site] += corr_current[other_site, site] * (t_local - corr_last_touch[other_site, site])
                corr_last_touch[other_site, site] = t_local
            end
        end
        return nothing
    end

    @inline function refresh_ctmc_site_correlations!(
        corr_current::Matrix{Float64},
        ρ::AbstractVector{<:Integer},
        site::Int,
    )
        occ_site = Float64(ρ[site])
        L = length(ρ)
        @inbounds for other_site in 1:L
            occ_other = Float64(ρ[other_site])
            corr_current[site, other_site] = occ_site * occ_other
            corr_current[other_site, site] = occ_other * occ_site
        end
        return nothing
    end

    @inline function update_selected_site_cut_observables!(
        cut_accum::Matrix{Float64},
        cut_last_touch::Matrix{Float64},
        cut_current::Matrix{Float64},
        selected_sites::Vector{Int},
        t_local::Float64,
        site_a::Int,
        site_b::Int,
    )
        L = size(cut_accum, 2)
        for (cut_idx, cut_site) in enumerate(selected_sites)
            if cut_site == site_a || cut_site == site_b
                @inbounds for other_site in 1:L
                    cut_accum[cut_idx, other_site] += cut_current[cut_idx, other_site] * (t_local - cut_last_touch[cut_idx, other_site])
                    cut_last_touch[cut_idx, other_site] = t_local
                end
            else
                cut_accum[cut_idx, site_a] += cut_current[cut_idx, site_a] * (t_local - cut_last_touch[cut_idx, site_a])
                cut_last_touch[cut_idx, site_a] = t_local
                cut_accum[cut_idx, site_b] += cut_current[cut_idx, site_b] * (t_local - cut_last_touch[cut_idx, site_b])
                cut_last_touch[cut_idx, site_b] = t_local
            end
        end
        return nothing
    end

    @inline function refresh_selected_site_cut_values!(
        cut_current::Matrix{Float64},
        ρ::AbstractVector{<:Integer},
        selected_sites::Vector{Int},
        site_a::Int,
        site_b::Int,
    )
        L = size(cut_current, 2)
        for (cut_idx, cut_site) in enumerate(selected_sites)
            occ_cut = Float64(ρ[cut_site])
            if cut_site == site_a || cut_site == site_b
                @inbounds for other_site in 1:L
                    cut_current[cut_idx, other_site] = occ_cut * Float64(ρ[other_site])
                end
            else
                cut_current[cut_idx, site_a] = occ_cut * Float64(ρ[site_a])
                cut_current[cut_idx, site_b] = occ_cut * Float64(ρ[site_b])
            end
        end
        return nothing
    end

    @inline function finalize_selected_site_cut_observables!(
        cut_accum::Matrix{Float64},
        cut_last_touch::Matrix{Float64},
        cut_current::Matrix{Float64},
    )
        n_cuts, L = size(cut_accum)
        @inbounds for cut_idx in 1:n_cuts
            for other_site in 1:L
                cut_accum[cut_idx, other_site] += cut_current[cut_idx, other_site] * (1.0 - cut_last_touch[cut_idx, other_site])
            end
        end
        return nothing
    end

    @inline function ctmc_rate_factor_1d(
        state,
        param,
        from_idx::Int,
        to_idx::Int,
        directed_bond_forcing::Float64,
        static_zero_potential::Bool,
        rate_normalization::Float64,
        scheme::AbstractString,
    )
        prefactor = bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme)
        prefactor <= 0.0 && return 0.0
        if static_zero_potential
            return prefactor
        end
        ΔV = Float64(state.potential.V[to_idx] - state.potential.V[from_idx])
        return prefactor * min(1.0, lookup_exp(state.exp_table, -ΔV / state.T))
    end

    @inline function recompute_ctmc_bond_rates_1d!(
        event_tree::FenwickTree,
        state,
        param,
        bond_left_site::Int,
        right_neighbors::Vector{Int},
        bond_right_forcing::Vector{Float64},
        static_zero_potential::Bool,
        rate_normalization::Float64,
        scheme::AbstractString,
    )
        L = length(state.ρ)
        right_site = right_neighbors[bond_left_site]

        right_rate = 0.0
        if state.ρ[bond_left_site] == 1 && state.ρ[right_site] == 0
            right_rate = ctmc_rate_factor_1d(
                state,
                param,
                bond_left_site,
                right_site,
                bond_right_forcing[bond_left_site],
                static_zero_potential,
                rate_normalization,
                scheme,
            )
        end

        left_rate = 0.0
        if state.ρ[right_site] == 1 && state.ρ[bond_left_site] == 0
            left_rate = ctmc_rate_factor_1d(
                state,
                param,
                right_site,
                bond_left_site,
                -bond_right_forcing[bond_left_site],
                static_zero_potential,
                rate_normalization,
                scheme,
            )
        end

        fenwick_set!(event_tree, bond_left_site, right_rate)
        fenwick_set!(event_tree, L + bond_left_site, left_rate)
        return nothing
    end

    @inline function recompute_ctmc_bond_neighborhood_1d!(
        event_tree::FenwickTree,
        state,
        param,
        site_a::Int,
        site_b::Int,
        left_neighbors::Vector{Int},
        right_neighbors::Vector{Int},
        bond_right_forcing::Vector{Float64},
        static_zero_potential::Bool,
        rate_normalization::Float64,
        scheme::AbstractString,
    )
        bond_1 = left_neighbors[site_a]
        bond_2 = site_a
        bond_3 = left_neighbors[site_b]
        bond_4 = site_b

        recompute_ctmc_bond_rates_1d!(event_tree, state, param, bond_1, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
        if bond_2 != bond_1
            recompute_ctmc_bond_rates_1d!(event_tree, state, param, bond_2, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
        end
        if bond_3 != bond_1 && bond_3 != bond_2
            recompute_ctmc_bond_rates_1d!(event_tree, state, param, bond_3, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
        end
        if bond_4 != bond_1 && bond_4 != bond_2 && bond_4 != bond_3
            recompute_ctmc_bond_rates_1d!(event_tree, state, param, bond_4, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
        end
        return nothing
    end

    @inline function record_ctmc_bond_passage_1d!(
        forward_counts::Vector{Int64},
        reverse_counts::Vector{Int64},
        tracked_forces_by_bond::Vector{Vector{Int}},
        right_hop_is_forward::BitVector,
        bond_left_site::Int,
        moved_right::Bool,
    )
        for force_idx in tracked_forces_by_bond[bond_left_site]
            if moved_right == right_hop_is_forward[force_idx]
                forward_counts[force_idx] += 1
            else
                reverse_counts[force_idx] += 1
            end
        end
        return nothing
    end

    @inline function resolve_force_fluctuation_types_1d(force_fluctuation_types::Vector{String}, n_forces::Int)
        if isempty(force_fluctuation_types)
            return fill("alternating_direction", n_forces)
        elseif length(force_fluctuation_types) != n_forces
            throw(ArgumentError("force_fluctuation_types must have length $n_forces, got $(length(force_fluctuation_types))."))
        end
        return copy(force_fluctuation_types)
    end

    @inline function force_signature_1d(force::BondForce, L::Int)
        first_site = mod1(force.bond_indices[1][1], L)
        second_site = mod1(force.bond_indices[2][1], L)
        return (first_site, second_site, Bool(force.direction_flag), Float64(force.magnitude))
    end

    @inline function ctmc_param_signature_1d(param, rate_normalization::Float64)
        return (
            Float64(param.D),
            Float64(param.γ),
            is_static_zero_potential(param),
            forcing_rate_scheme(param),
            rate_normalization,
        )
    end

    function refresh_all_selected_site_cut_values!(
        cut_current::Matrix{Float64},
        ρ::AbstractVector{<:Integer},
        selected_sites::Vector{Int},
    )
        n_cuts, L = size(cut_current)
        @inbounds for cut_idx in 1:n_cuts
            cut_site = selected_sites[cut_idx]
            occ_cut = Float64(ρ[cut_site])
            for other_site in 1:L
                cut_current[cut_idx, other_site] = occ_cut * Float64(ρ[other_site])
            end
        end
        return nothing
    end

    function refresh_all_ctmc_correlations!(
        corr_current::Matrix{Float64},
        ρ::AbstractVector{<:Integer},
    )
        L = length(ρ)
        @inbounds for site_i in 1:L
            occ_i = Float64(ρ[site_i])
            for site_j in 1:L
                corr_current[site_i, site_j] = occ_i * Float64(ρ[site_j])
            end
        end
        return nothing
    end

    function rebuild_site_to_particle_map_from_occupancy!(
        site_to_particle::Vector{Int},
        ρ::AbstractVector{<:Integer},
    )
        fill!(site_to_particle, 0)
        particle_idx = 1
        @inbounds for site_idx in eachindex(ρ)
            if ρ[site_idx] != 0
                site_to_particle[site_idx] = particle_idx
                particle_idx += 1
            end
        end
        return nothing
    end

    function reset_ctmc_sweep_counters!(workspace::CTMC1DWorkspace)
        fill!(workspace.sweep_forward_counts, 0)
        fill!(workspace.sweep_reverse_counts, 0)
        fill!(workspace.sweep_spatial_forward_counts, 0)
        fill!(workspace.sweep_spatial_reverse_counts, 0)
        return nothing
    end

    function prepare_ctmc_measurement_buffers!(workspace::CTMC1DWorkspace, state)
        fill!(workspace.density_accum, 0.0)
        fill!(workspace.density_last_touch, 0.0)
        if workspace.has_selected_site_cuts
            fill!(workspace.cut_accum, 0.0)
            fill!(workspace.cut_last_touch, 0.0)
            refresh_all_selected_site_cut_values!(workspace.cut_current, state.ρ, workspace.selected_cut_sites)
        else
            fill!(workspace.corr_accum, 0.0)
            fill!(workspace.corr_last_touch, 0.0)
            refresh_all_ctmc_correlations!(workspace.corr_current, state.ρ)
        end
        return nothing
    end

    function initialize_ctmc_1d_workspace_dynamic!(
        workspace::CTMC1DWorkspace,
        state,
        param,
        forcings::Vector{BondForce},
    )
        ffrs = workspace.ffrs
        static_zero_potential = is_static_zero_potential(param)
        scheme = forcing_rate_scheme(param)
        rate_normalization = workspace.param_signature[5]

        rebuild_site_to_particle_map_from_occupancy!(workspace.site_to_particle, state.ρ)
        fill!(workspace.bond_right_forcing, 0.0)
        fenwick_clear!(workspace.event_tree)

        @inbounds for force_idx in eachindex(forcings)
            contribution = force_right_contribution(forcings[force_idx], workspace.force_orientation_signs[force_idx])
            workspace.force_right_contributions[force_idx] = contribution
            workspace.bond_right_forcing[workspace.force_bond_left_sites[force_idx]] += contribution
        end

        @inbounds for bond_left_site in 1:workspace.L
            recompute_ctmc_bond_rates_1d!(
                workspace.event_tree,
                state,
                param,
                bond_left_site,
                workspace.right_neighbors,
                workspace.bond_right_forcing,
                static_zero_potential,
                rate_normalization,
                scheme,
            )
        end

        @inbounds for (event_offset, force_idx) in enumerate(workspace.fluctuating_force_indices)
            fenwick_set!(workspace.event_tree, 2 * workspace.L + event_offset, max(forcing_rate(ffrs, force_idx), 0.0))
        end
        if workspace.has_potential_event
            fenwick_set!(workspace.event_tree, length(workspace.event_tree.tree), max(Float64(param.γ), 0.0))
        end
        return nothing
    end

    function workspace_matches_ctmc_1d(
        workspace::CTMC1DWorkspace,
        state,
        param,
        forcings::Vector{BondForce},
        tracked_force_indices::Vector{Int},
        force_fluctuation_types::Vector{String},
        ffrs::Vector{Float64},
        rate_normalization::Float64,
    )
        workspace.L == length(state.ρ) || return false
        workspace.tracked_force_indices == tracked_force_indices || return false
        workspace.force_fluctuation_types == force_fluctuation_types || return false
        workspace.ffrs == ffrs || return false
        workspace.has_selected_site_cuts == has_selected_site_cuts(state) || return false
        workspace.selected_cut_sites == (workspace.has_selected_site_cuts ? selected_site_cut_sites_from_state(state) : Int[]) || return false
        workspace.has_potential_event == (Float64(param.γ) > 0.0) || return false
        workspace.param_signature == ctmc_param_signature_1d(param, rate_normalization) || return false
        length(workspace.force_signature) == length(forcings) || return false
        @inbounds for force_idx in eachindex(forcings)
            workspace.force_signature[force_idx] == force_signature_1d(forcings[force_idx], workspace.L) || return false
        end
        return true
    end

    function build_ctmc_1d_workspace(
        state,
        param,
        forcings::Vector{BondForce},
        tracked_force_indices::Vector{Int},
        force_fluctuation_types::Vector{String},
        ffrs::Vector{Float64},
        rate_normalization::Float64,
    )
        L = param.dims[1]
        left_neighbors, right_neighbors = precompute_periodic_neighbors_1d(L)
        n_forces = length(forcings)
        tracked_mask = falses(n_forces)
        @inbounds for force_idx in tracked_force_indices
            tracked_mask[force_idx] = true
        end

        force_signature = Vector{Tuple{Int,Int,Bool,Float64}}(undef, n_forces)
        force_bond_left_sites = zeros(Int, n_forces)
        force_orientation_signs = zeros(Int8, n_forces)
        tracked_forces_by_bond = [Int[] for _ in 1:L]
        right_hop_is_forward = falses(n_forces)
        fluctuating_force_indices = Int[]

        @inbounds for force_idx in 1:n_forces
            force = forcings[force_idx]
            force_signature[force_idx] = force_signature_1d(force, L)
            first_site, second_site, _, _ = force_signature[force_idx]
            bond_left_site, orientation_sign = bond_left_site_and_orientation_1d(first_site, second_site, L)
            force_bond_left_sites[force_idx] = bond_left_site
            force_orientation_signs[force_idx] = orientation_sign
            right_hop_is_forward[force_idx] = orientation_sign == 1

            tracked_mask[force_idx] && push!(tracked_forces_by_bond[bond_left_site], force_idx)

            fluctuation_type = force_fluctuation_types[force_idx]
            if fluctuation_type == "alternating_direction"
                if forcing_rate(ffrs, force_idx) > 0.0
                    push!(fluctuating_force_indices, force_idx)
                end
            elseif fluctuation_type != "static"
                throw(ArgumentError("Unsupported force fluctuation type for CTMC: $fluctuation_type"))
            end
        end

        selected_cut_sites = has_selected_site_cuts(state) ? selected_site_cut_sites_from_state(state) : Int[]
        has_selected = !isempty(selected_cut_sites)
        n_cut_sites = length(selected_cut_sites)
        event_count = 2 * L + length(fluctuating_force_indices) + (Float64(param.γ) > 0.0 ? 1 : 0)

        workspace = CTMC1DWorkspace(
            L,
            force_signature,
            copy(tracked_force_indices),
            copy(force_fluctuation_types),
            copy(selected_cut_sites),
            has_selected,
            Float64(param.γ) > 0.0,
            copy(ffrs),
            ctmc_param_signature_1d(param, rate_normalization),
            left_neighbors,
            right_neighbors,
            force_bond_left_sites,
            force_orientation_signs,
            tracked_forces_by_bond,
            right_hop_is_forward,
            fluctuating_force_indices,
            zeros(Float64, n_forces),
            zeros(Float64, L),
            FenwickTree(event_count),
            zeros(Int, L),
            zeros(Int64, n_forces),
            zeros(Int64, n_forces),
            zeros(Int64, L),
            zeros(Int64, L),
            zeros(Float64, L),
            zeros(Float64, L),
            zeros(Float64, n_cut_sites, L),
            zeros(Float64, n_cut_sites, L),
            zeros(Float64, n_cut_sites, L),
            has_selected ? zeros(Float64, 0, 0) : zeros(Float64, L, L),
            has_selected ? zeros(Float64, 0, 0) : zeros(Float64, L, L),
            has_selected ? zeros(Float64, 0, 0) : zeros(Float64, L, L),
        )
        initialize_ctmc_1d_workspace_dynamic!(workspace, state, param, forcings)
        return workspace
    end

    function get_ctmc_1d_workspace!(
        state,
        param,
        forcings::Vector{BondForce},
        tracked_force_indices::Vector{Int},
        force_fluctuation_types::Vector{String},
        ffrs::Vector{Float64},
        rate_normalization::Float64,
    )
        workspace = get(CTMC_1D_WORKSPACES, state, nothing)
        if workspace === nothing || !workspace_matches_ctmc_1d(workspace, state, param, forcings, tracked_force_indices, force_fluctuation_types, ffrs, rate_normalization)
            workspace = build_ctmc_1d_workspace(state, param, forcings, tracked_force_indices, force_fluctuation_types, ffrs, rate_normalization)
            CTMC_1D_WORKSPACES[state] = workspace
        end
        return workspace
    end

    function invalidate_ctmc_1d_workspace!(state)
        delete!(CTMC_1D_WORKSPACES, state)
        return nothing
    end

    function update_ctmc_1d!(
        param,
        state,
        rng;
        collect_statistics::Bool=true,
        force_fluctuation_types::Vector{String}=String[],
    )
        validate_exclusion_state(state)
        length(param.dims) == 1 || throw(ArgumentError("CTMC SSEP currently supports only 1D systems."))

        L = param.dims[1]
        L >= 2 || throw(ArgumentError("CTMC SSEP requires L >= 2."))

        static_zero_potential = is_static_zero_potential(param)
        ffrs = param_ffrs(param)
        forcings = get_state_forcings!(state)
        n_forces = length(forcings)
        scheme = forcing_rate_scheme(param)
        rate_normalization = hop_rate_normalization(param.D, max_forcing_magnitude_per_bond(forcings), scheme)

        force_fluctuation_types = resolve_force_fluctuation_types_1d(force_fluctuation_types, n_forces)

        ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
        tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
        workspace = get_ctmc_1d_workspace!(
            state,
            param,
            forcings,
            tracked_force_indices,
            force_fluctuation_types,
            ffrs,
            rate_normalization,
        )

        left_neighbors = workspace.left_neighbors
        right_neighbors = workspace.right_neighbors
        force_bond_left_sites = workspace.force_bond_left_sites
        force_orientation_signs = workspace.force_orientation_signs
        force_right_contributions = workspace.force_right_contributions
        tracked_forces_by_bond = workspace.tracked_forces_by_bond
        right_hop_is_forward = workspace.right_hop_is_forward
        fluctuating_force_indices = workspace.fluctuating_force_indices
        bond_right_forcing = workspace.bond_right_forcing
        event_tree = workspace.event_tree
        site_to_particle = workspace.site_to_particle
        sweep_forward_counts = workspace.sweep_forward_counts
        sweep_reverse_counts = workspace.sweep_reverse_counts
        sweep_spatial_forward_counts = workspace.sweep_spatial_forward_counts
        sweep_spatial_reverse_counts = workspace.sweep_spatial_reverse_counts
        reset_ctmc_sweep_counters!(workspace)

        flip_event_upper = 2 * L + length(fluctuating_force_indices)

        use_selected_site_cuts = collect_statistics && workspace.has_selected_site_cuts
        selected_cut_sites = workspace.selected_cut_sites
        density_accum = collect_statistics ? workspace.density_accum : nothing
        density_last_touch = collect_statistics ? workspace.density_last_touch : nothing
        cut_accum = use_selected_site_cuts ? workspace.cut_accum : nothing
        cut_last_touch = use_selected_site_cuts ? workspace.cut_last_touch : nothing
        cut_current = use_selected_site_cuts ? workspace.cut_current : nothing
        corr_accum = collect_statistics && !use_selected_site_cuts ? workspace.corr_accum : nothing
        corr_last_touch = collect_statistics && !use_selected_site_cuts ? workspace.corr_last_touch : nothing
        corr_current = collect_statistics && !use_selected_site_cuts ? workspace.corr_current : nothing
        collect_statistics && prepare_ctmc_measurement_buffers!(workspace, state)

        t_local = 0.0
        occupancy_changed = false
        while t_local < 1.0
            total_rate = fenwick_total(event_tree)
            total_rate <= 0.0 && break

            dt = -log(rand(rng)) / total_rate
            if t_local + dt >= 1.0
                break
            end
            t_local += dt

            target = max(rand(rng) * total_rate, nextfloat(0.0))
            event_idx = fenwick_sample(event_tree, target)

            if event_idx <= L
                bond_left_site = event_idx
                from_site = bond_left_site
                to_site = right_neighbors[bond_left_site]
                particle_idx = site_to_particle[from_site]
                particle_idx == 0 && error("CTMC rate table selected an empty source site on bond $bond_left_site.")
                site_to_particle[to_site] == 0 || error("CTMC rate table selected an occupied target site on bond $bond_left_site.")

                if collect_statistics
                    if use_selected_site_cuts
                        density_accum[from_site] += Float64(state.ρ[from_site]) * (t_local - density_last_touch[from_site])
                        density_last_touch[from_site] = t_local
                        density_accum[to_site] += Float64(state.ρ[to_site]) * (t_local - density_last_touch[to_site])
                        density_last_touch[to_site] = t_local
                        update_selected_site_cut_observables!(cut_accum, cut_last_touch, cut_current, selected_cut_sites, t_local, from_site, to_site)
                    else
                        update_ctmc_site_observables!(density_accum, density_last_touch, corr_accum, corr_last_touch, corr_current, state.ρ, t_local, from_site)
                        update_ctmc_site_observables!(density_accum, density_last_touch, corr_accum, corr_last_touch, corr_current, state.ρ, t_local, to_site)
                    end
                end

                site_to_particle[from_site] = 0
                site_to_particle[to_site] = particle_idx
                state.ρ[from_site] = 0
                state.ρ[to_site] = 1
                occupancy_changed = true

                if collect_statistics
                    if use_selected_site_cuts
                        refresh_selected_site_cut_values!(cut_current, state.ρ, selected_cut_sites, from_site, to_site)
                    else
                        refresh_ctmc_site_correlations!(corr_current, state.ρ, from_site)
                        refresh_ctmc_site_correlations!(corr_current, state.ρ, to_site)
                    end
                end

                record_ctmc_bond_passage_1d!(sweep_forward_counts, sweep_reverse_counts, tracked_forces_by_bond, right_hop_is_forward, bond_left_site, true)
                sweep_spatial_forward_counts[bond_left_site] += 1
                recompute_ctmc_bond_neighborhood_1d!(event_tree, state, param, from_site, to_site, left_neighbors, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
            elseif event_idx <= 2 * L
                bond_left_site = event_idx - L
                from_site = right_neighbors[bond_left_site]
                to_site = bond_left_site
                particle_idx = site_to_particle[from_site]
                particle_idx == 0 && error("CTMC rate table selected an empty source site on reverse bond $bond_left_site.")
                site_to_particle[to_site] == 0 || error("CTMC rate table selected an occupied target site on reverse bond $bond_left_site.")

                if collect_statistics
                    if use_selected_site_cuts
                        density_accum[from_site] += Float64(state.ρ[from_site]) * (t_local - density_last_touch[from_site])
                        density_last_touch[from_site] = t_local
                        density_accum[to_site] += Float64(state.ρ[to_site]) * (t_local - density_last_touch[to_site])
                        density_last_touch[to_site] = t_local
                        update_selected_site_cut_observables!(cut_accum, cut_last_touch, cut_current, selected_cut_sites, t_local, from_site, to_site)
                    else
                        update_ctmc_site_observables!(density_accum, density_last_touch, corr_accum, corr_last_touch, corr_current, state.ρ, t_local, from_site)
                        update_ctmc_site_observables!(density_accum, density_last_touch, corr_accum, corr_last_touch, corr_current, state.ρ, t_local, to_site)
                    end
                end

                site_to_particle[from_site] = 0
                site_to_particle[to_site] = particle_idx
                state.ρ[from_site] = 0
                state.ρ[to_site] = 1
                occupancy_changed = true

                if collect_statistics
                    if use_selected_site_cuts
                        refresh_selected_site_cut_values!(cut_current, state.ρ, selected_cut_sites, from_site, to_site)
                    else
                        refresh_ctmc_site_correlations!(corr_current, state.ρ, from_site)
                        refresh_ctmc_site_correlations!(corr_current, state.ρ, to_site)
                    end
                end

                record_ctmc_bond_passage_1d!(sweep_forward_counts, sweep_reverse_counts, tracked_forces_by_bond, right_hop_is_forward, bond_left_site, false)
                sweep_spatial_reverse_counts[bond_left_site] += 1
                recompute_ctmc_bond_neighborhood_1d!(event_tree, state, param, from_site, to_site, left_neighbors, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
            elseif event_idx <= flip_event_upper
                fluctuating_event_idx = event_idx - 2 * L
                force_idx = fluctuating_force_indices[fluctuating_event_idx]
                bond_left_site = force_bond_left_sites[force_idx]

                bond_right_forcing[bond_left_site] -= force_right_contributions[force_idx]
                bondforce_update!(forcings[force_idx])
                workspace.force_signature[force_idx] = force_signature_1d(forcings[force_idx], L)
                force_right_contributions[force_idx] = force_right_contribution(forcings[force_idx], force_orientation_signs[force_idx])
                bond_right_forcing[bond_left_site] += force_right_contributions[force_idx]
                recompute_ctmc_bond_rates_1d!(event_tree, state, param, bond_left_site, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
            else
                potential_update!(state.potential, rng)
                for bond_left_site in 1:L
                    recompute_ctmc_bond_rates_1d!(event_tree, state, param, bond_left_site, right_neighbors, bond_right_forcing, static_zero_potential, rate_normalization, scheme)
                end
            end
        end

        occupancy_changed && mark_redundant_ssep_views_stale!(state)

        interval_density = nothing
        interval_corr = nothing
        if collect_statistics
            @inbounds for site in 1:L
                density_accum[site] += Float64(state.ρ[site]) * (1.0 - density_last_touch[site])
            end
            interval_density = density_accum
            if use_selected_site_cuts
                finalize_selected_site_cut_observables!(cut_accum, cut_last_touch, cut_current)
                interval_corr = cut_accum
            else
                @inbounds for site_i in 1:L
                    for site_j in 1:L
                        corr_accum[site_i, site_j] += corr_current[site_i, site_j] * (1.0 - corr_last_touch[site_i, site_j])
                    end
                end
                interval_corr = corr_accum
            end
            update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
            update_spatial_bond_passage_averages!(state, sweep_spatial_forward_counts, sweep_spatial_reverse_counts)
        end

        state.max_site_occupancy = isempty(state.particles) ? 0 : 1
        state.t += 1
        validate_exclusion_state(state)
        return interval_density, interval_corr
    end

    @inline function precompute_periodic_neighbors_1d(L::Int)
        left_neighbors = [i == 1 ? L : i - 1 for i in 1:L]
        right_neighbors = [i == L ? 1 : i + 1 for i in 1:L]
        return left_neighbors, right_neighbors
    end

    @inline function precompute_periodic_neighbors_2d(Lx::Int, Ly::Int)
        left_x = [i == 1 ? Lx : i - 1 for i in 1:Lx]
        right_x = [i == Lx ? 1 : i + 1 for i in 1:Lx]
        down_y = [j == 1 ? Ly : j - 1 for j in 1:Ly]
        up_y = [j == Ly ? 1 : j + 1 for j in 1:Ly]
        return left_x, right_x, down_y, up_y
    end

    @inline endpoint_tuple(indices::AbstractVector{<:Integer}) = Tuple(Int(i) for i in indices)

    function max_forcing_magnitude_per_bond(forcings::Vector{BondForce})
        isempty(forcings) && return 0.0
        bond_totals = Dict{Any,Float64}()
        for force in forcings
            endpoint_1 = endpoint_tuple(force.bond_indices[1])
            endpoint_2 = endpoint_tuple(force.bond_indices[2])
            bond_key = endpoint_1 <= endpoint_2 ? (endpoint_1, endpoint_2) : (endpoint_2, endpoint_1)
            bond_totals[bond_key] = get(bond_totals, bond_key, 0.0) + abs(force.magnitude)
        end
        return maximum(values(bond_totals))
    end

    @inline function hop_rate_normalization(D::Real, max_bond_forcing::Real, scheme::AbstractString)
        if scheme == LEGACY_FORCING_RATE_SCHEME
            return 1.0
        elseif scheme == SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
            return max(Float64(D) + Float64(max_bond_forcing), eps(Float64))
        end
        throw(ArgumentError("Unsupported forcing_rate_scheme: $scheme"))
    end

    @inline function bond_rate_prefactor(D::Real, directed_bond_forcing::Real, rate_normalization::Real, scheme::AbstractString)
        base_rate = if scheme == LEGACY_FORCING_RATE_SCHEME
            Float64(D) - max(Float64(directed_bond_forcing), 0.0)
        elseif scheme == SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
            Float64(D) + Float64(directed_bond_forcing)
        else
            throw(ArgumentError("Unsupported forcing_rate_scheme: $scheme"))
        end
        return base_rate / rate_normalization
    end

    function directed_bond_forcing_1d(forcings::Vector{BondForce}, from_idx::Int, to_idx::Int)
        total_forcing = 0.0
        for force in forcings
            endpoint_1 = force.bond_indices[1][1]
            endpoint_2 = force.bond_indices[2][1]
            active_from = force.direction_flag ? endpoint_1 : endpoint_2
            active_to = force.direction_flag ? endpoint_2 : endpoint_1
            if from_idx == active_from && to_idx == active_to
                total_forcing += force.magnitude
            elseif from_idx == active_to && to_idx == active_from
                total_forcing -= force.magnitude
            end
        end
        return total_forcing
    end

    function directed_bond_forcing_2d(forcings::Vector{BondForce}, from_pos::NTuple{2,Int}, to_pos::NTuple{2,Int})
        total_forcing = 0.0
        for force in forcings
            endpoint_1 = (force.bond_indices[1][1], force.bond_indices[1][2])
            endpoint_2 = (force.bond_indices[2][1], force.bond_indices[2][2])
            active_from = force.direction_flag ? endpoint_1 : endpoint_2
            active_to = force.direction_flag ? endpoint_2 : endpoint_1
            if from_pos == active_from && to_pos == active_to
                total_forcing += force.magnitude
            elseif from_pos == active_to && to_pos == active_from
                total_forcing -= force.magnitude
            end
        end
        return total_forcing
    end

    function calculate_jump_probability(D, ΔV, T, exp_table::ExpLookupTable;
                                        directed_bond_forcing=0.0,
                                        rate_normalization=1.0,
                                        forcing_rate_scheme=LEGACY_FORCING_RATE_SCHEME)
        exp_val = lookup_exp(exp_table, -ΔV / T)
        return bond_rate_prefactor(D, directed_bond_forcing, rate_normalization, forcing_rate_scheme) * min(1.0, exp_val)
    end

    function update!(param, state, rng; collect_statistics::Bool=true)
        validate_exclusion_state(state)
        invalidate_ctmc_1d_workspace!(state)
        sync_redundant_ssep_views!(state; sync_particles=true, sync_directional_densities=false)

        if length(param.dims) == 1
            V = state.potential.V
            T = state.T
            γ = param.γ
            static_zero_potential = is_static_zero_potential(param)
            ffrs = param_ffrs(param)
            forcings = get_state_forcings!(state)
            n_forces = length(forcings)
            scheme = forcing_rate_scheme(param)
            rate_normalization = hop_rate_normalization(param.D, max_forcing_magnitude_per_bond(forcings), scheme)
            L = param.dims[1]
            left_neighbors, right_neighbors = precompute_periodic_neighbors_1d(L)
            ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
            tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
            sweep_forward_counts = zeros(Int64, n_forces)
            sweep_reverse_counts = zeros(Int64, n_forces)
            ensure_spatial_bond_passage_stats!(state, L)
            sweep_spatial_forward_counts = zeros(Int64, L)
            sweep_spatial_reverse_counts = zeros(Int64, L)

            micro_steps = max(param.N, 1)
            t = state.t
            t_end = state.t + 1
            occupancy_changed = false
            while t < t_end
                n_and_a = rand(rng, 1:2 * (param.N + 1))
                action_index = mod1(n_and_a, 2)
                n = (n_and_a - action_index) ÷ 2 + 1

                accepted = false
                if n <= param.N
                    particle = state.particles[n]
                    spot_index = particle.position[1]
                    candidate_spot_index = action_index == 1 ? left_neighbors[spot_index] : right_neighbors[spot_index]
                    vacancy = state.ρ[candidate_spot_index] == 0
                    if vacancy
                        directed_bond_forcing = directed_bond_forcing_1d(forcings, spot_index, candidate_spot_index)
                        p_candidate = static_zero_potential ?
                            bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme) :
                            calculate_jump_probability(param.D, V[candidate_spot_index] - V[spot_index], T, state.exp_table;
                                                       directed_bond_forcing=directed_bond_forcing,
                                                       rate_normalization=rate_normalization,
                                                       forcing_rate_scheme=scheme)
                        accepted = rand(rng) < clamp(p_candidate, 0.0, 1.0)
                    end

                    if accepted
                        if !isempty(tracked_force_indices)
                            record_bond_passage_1d!(sweep_forward_counts, sweep_reverse_counts, forcings, tracked_force_indices, spot_index, candidate_spot_index)
                        end
                        if action_index == 2
                            sweep_spatial_forward_counts[spot_index] += 1
                        else
                            sweep_spatial_reverse_counts[candidate_spot_index] += 1
                        end

                        particle.position = (candidate_spot_index,)
                        state.ρ[spot_index] = 0
                        state.ρ[candidate_spot_index] = 1
                        occupancy_changed = true
                    end
                else
                    accepted = rand(rng) < clamp(γ, 0.0, 1.0)
                    if accepted
                        potential_update!(state.potential, rng)
                    end
                end

                for force_idx in 1:n_forces
                    p_force = clamp(forcing_rate(ffrs, force_idx) / micro_steps, 0.0, 1.0)
                    rand(rng) < p_force && bondforce_update!(forcings[force_idx])
                end

                t += 1 / micro_steps
            end

            if collect_statistics
                update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
                update_spatial_bond_passage_averages!(state, sweep_spatial_forward_counts, sweep_spatial_reverse_counts)
            end
            occupancy_changed && mark_redundant_ssep_views_stale!(state; particles=false, directional_densities=true)
        elseif length(param.dims) == 2
            V = state.potential.V
            T = state.T
            γ = param.γ
            static_zero_potential = is_static_zero_potential(param)
            ffrs = param_ffrs(param)
            forcings = get_state_forcings!(state)
            n_forces = length(forcings)
            scheme = forcing_rate_scheme(param)
            rate_normalization = hop_rate_normalization(param.D, max_forcing_magnitude_per_bond(forcings), scheme)
            Lx, Ly = param.dims
            left_x, right_x, down_y, up_y = precompute_periodic_neighbors_2d(Lx, Ly)
            ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
            tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
            sweep_forward_counts = zeros(Int64, n_forces)
            sweep_reverse_counts = zeros(Int64, n_forces)

            micro_steps = max(param.N, 1)
            t = state.t
            t_end = state.t + 1
            occupancy_changed = false
            while t < t_end
                n_and_a = rand(rng, 1:4 * (param.N + 1))
                action_index = mod1(n_and_a, 4)
                n = (n_and_a - action_index) ÷ 4 + 1

                if n <= param.N
                    particle = state.particles[n]
                    i, j = particle.position
                    spot_index = (i, j)
                    cand = if action_index == 1
                        (left_x[i], j)
                    elseif action_index == 2
                        (right_x[i], j)
                    elseif action_index == 3
                        (i, down_y[j])
                    else
                        (i, up_y[j])
                    end

                    vacancy = state.ρ[cand...] == 0
                    if vacancy
                        directed_bond_forcing = directed_bond_forcing_2d(forcings, spot_index, cand)
                        p_candidate = static_zero_potential ?
                            bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme) :
                            calculate_jump_probability(param.D, V[cand...] - V[i, j], T, state.exp_table;
                                                       directed_bond_forcing=directed_bond_forcing,
                                                       rate_normalization=rate_normalization,
                                                       forcing_rate_scheme=scheme)
                        if rand(rng) < clamp(p_candidate, 0.0, 1.0)
                            if !isempty(tracked_force_indices)
                                record_bond_passage_2d!(sweep_forward_counts, sweep_reverse_counts, forcings, tracked_force_indices, spot_index, cand)
                            end
                            particle.position = cand
                            state.ρ[i, j] = 0
                            state.ρ[cand...] = 1
                            occupancy_changed = true
                        end
                    end
                else
                    if rand(rng) < clamp(γ, 0.0, 1.0)
                        potential_update!(state.potential, rng)
                    end
                end

                for force_idx in 1:n_forces
                    p_force = clamp(forcing_rate(ffrs, force_idx) / micro_steps, 0.0, 1.0)
                    rand(rng) < p_force && bondforce_update!(forcings[force_idx])
                end

                t += 1 / micro_steps
            end

            if collect_statistics
                update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
            end
            occupancy_changed && mark_redundant_ssep_views_stale!(state; particles=false, directional_densities=true)
        else
            throw(DomainError("Only 1D or 2D SSEP is supported"))
        end

        state.max_site_occupancy = isempty(state.particles) ? 0 : 1
        state.t += 1
        validate_exclusion_state(state)
        return nothing
    end
end

function update_and_compute_correlations!(
    state,
    param,
    rng;
    collect_statistics::Bool=true,
    simulation_mode::AbstractString="discrete_sweep",
    force_fluctuation_types::Vector{String}=String[],
)
    interval_density = nothing
    interval_corr = nothing
    if simulation_mode == "ctmc_1d"
        interval_density, interval_corr = FPSSEP.update_ctmc_1d!(
            param,
            state,
            rng;
            collect_statistics=collect_statistics,
            force_fluctuation_types=force_fluctuation_types,
        )
    elseif simulation_mode == "discrete_sweep"
        FPSSEP.update!(param, state, rng; collect_statistics=collect_statistics)
    else
        throw(ArgumentError("Unsupported SSEP simulation_mode: $simulation_mode"))
    end
    if !collect_statistics
        return nothing
    end

    n_eff = max(FPSSEP.averaged_sample_count(state), 1)
    weight_prev = max(n_eff - 1, 0)

    dim_num = length(param.dims)
    if dim_num == 1
        ρf = isnothing(interval_density) ? float(state.ρ) : interval_density
        state.ρ_avg .= (state.ρ_avg .* weight_prev .+ ρf) ./ n_eff
        if FPSSEP.has_selected_site_cuts(state)
            selected_cuts = isnothing(interval_corr) ?
                FPSSEP.selected_site_cut_pair_matrix(ρf, FPSSEP.selected_site_cut_sites_from_state(state)) :
                interval_corr
            state.ρ_matrix_avg_cuts[FPSSEP.SELECTED_SITE_CUTS_KEY] .=
                (state.ρ_matrix_avg_cuts[FPSSEP.SELECTED_SITE_CUTS_KEY] .* weight_prev .+ selected_cuts) ./ n_eff
        else
            ρ_matrix = isnothing(interval_corr) ? (ρf * transpose(ρf)) : interval_corr
            state.ρ_matrix_avg_cuts[:full] .= (state.ρ_matrix_avg_cuts[:full] .* weight_prev .+ ρ_matrix) ./ n_eff
        end
    elseif dim_num == 2
        ρf = float(state.ρ)
        state.ρ_avg .= (state.ρ_avg .* weight_prev .+ ρf) ./ n_eff
        if haskey(state.ρ_matrix_avg_cuts, :full)
            ρ_matrix = FPSSEP.outer_density_2D(ρf)
            state.ρ_matrix_avg_cuts[:full] .= (state.ρ_matrix_avg_cuts[:full] .* weight_prev .+ ρ_matrix) ./ n_eff
        else
            x_middle = clamp(div(param.dims[1], 2), 1, param.dims[1])
            y_middle = clamp(div(param.dims[2], 2), 1, param.dims[2])
            x_cut = ρf[:, y_middle] * transpose(ρf[:, y_middle])
            y_cut = ρf[x_middle, :] * transpose(ρf[x_middle, :])
            diag_cut = diag(ρf) * transpose(diag(ρf))
            state.ρ_matrix_avg_cuts[:x_cut] .= (state.ρ_matrix_avg_cuts[:x_cut] .* weight_prev .+ x_cut) ./ n_eff
            state.ρ_matrix_avg_cuts[:y_cut] .= (state.ρ_matrix_avg_cuts[:y_cut] .* weight_prev .+ y_cut) ./ n_eff
            state.ρ_matrix_avg_cuts[:diag_cut] .= (state.ρ_matrix_avg_cuts[:diag_cut] .* weight_prev .+ diag_cut) ./ n_eff
        end
    else
        throw(DomainError("Only 1D or 2D SSEP is supported"))
    end

    return nothing
end

function run_simulation!(
    state,
    param,
    n_sweeps,
    rng;
    show_times=Int[],
    save_times=Int[],
    save_dir="saved_states",
    plot_flag=false,
    plotter=nothing,
    plot_label="",
    save_description=nothing,
    warmup_sweeps::Int=0,
    show_progress::Bool=true,
    relaxed_ic::Bool=false,
    simulation_mode::AbstractString="discrete_sweep",
    force_fluctuation_types::Vector{String}=String[],
)
    println("Starting SSEP simulation")
    progress = show_progress ? Progress(n_sweeps) : nothing
    t_init = state.t + 1
    t_end = state.t + n_sweeps
    warmup_sweeps = max(warmup_sweeps, 0)

    for sweep in t_init:t_end
        sweep_since_start = sweep - t_init + 1
        collect_statistics = sweep_since_start > warmup_sweeps
        if warmup_sweeps > 0 && sweep_since_start == warmup_sweeps + 1
            println("Warmup complete at sweep $sweep. Starting statistics accumulation.")
        end

        update_and_compute_correlations!(
            state,
            param,
            rng;
            collect_statistics=collect_statistics,
            simulation_mode=simulation_mode,
            force_fluctuation_types=force_fluctuation_types,
        )

        if sweep in save_times
            FPSSEP.sync_redundant_ssep_views!(state; force=true)
            save_state(state, param, save_dir; relaxed_ic=relaxed_ic, description=save_description)
            println("State saved at sweep $sweep")
        end

        if plot_flag && !isnothing(plotter) && (sweep in show_times)
            FPSSEP.sync_redundant_ssep_views!(state; force=true)
            plotter(
                sweep,
                state,
                param;
                label=plot_label,
                prefer_multiforce_plots=false,
            )
        end

        if show_progress && !isnothing(progress)
            next!(progress)
        end
    end

    println("SSEP simulation complete")
    return state.ρ_avg, state.ρ_matrix_avg_cuts
end
