using LinearAlgebra
using Statistics
using ProgressMeter
include("plot_utils.jl")
using .PlotUtils
include("potentials.jl")

module FPSSEP
    using ..Potentials: AbstractPotential, BondForce, bondforce_update!, potential_update!, setPotential, setBondForce
    using LinearAlgebra
    using Random: rand, randperm

    export Param, Particle, State, setParam, setState, setDummyState, reset_statistics!
    export get_state_forcings!, averaged_sample_count, validate_exclusion_state

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

    const BOND_PASS_FORWARD_AVG_KEY = :bond_pass_forward_avg
    const BOND_PASS_REVERSE_AVG_KEY = :bond_pass_reverse_avg
    const BOND_PASS_TOTAL_AVG_KEY = :bond_pass_total_avg
    const BOND_PASS_TOTAL_SQ_AVG_KEY = :bond_pass_total_sq_avg
    const BOND_PASS_SAMPLE_COUNT_KEY = :bond_pass_sample_count
    const BOND_PASS_TRACK_MASK_KEY = :bond_pass_track_mask
    const BOND_PASS_SPATIAL_F_AVG_KEY = :bond_pass_spatial_f_avg
    const BOND_PASS_SPATIAL_F2_AVG_KEY = :bond_pass_spatial_f2_avg
    const BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY = :bond_pass_spatial_sample_count

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
        return State(
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
            ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                :full => ρ_avg * transpose(ρ_avg),
            )
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
        state.t = 0
        state.ρ_avg .= state.ρ
        state.max_site_occupancy = isempty(state.particles) ? 0 : 1

        dim = ndims(state.ρ)
        if dim == 1
            ρf = float(state.ρ)
            state.ρ_matrix_avg_cuts[:full] .= ρf * transpose(ρf)
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
                        state.ρ₊[spot_index] = 0
                        state.ρ₊[candidate_spot_index] = 1
                        state.ρ₋[spot_index] = 0
                        state.ρ₋[candidate_spot_index] = 1
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
                            state.ρ₊[i, j] = 0
                            state.ρ₊[cand...] = 1
                            state.ρ₋[i, j] = 0
                            state.ρ₋[cand...] = 1
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
        else
            throw(DomainError("Only 1D or 2D SSEP is supported"))
        end

        state.max_site_occupancy = isempty(state.particles) ? 0 : 1
        state.t += 1
        validate_exclusion_state(state)
        return nothing
    end
end

function update_and_compute_correlations!(state, param, rng; collect_statistics::Bool=true)
    FPSSEP.update!(param, state, rng; collect_statistics=collect_statistics)
    if !collect_statistics
        return nothing
    end

    n_eff = max(FPSSEP.averaged_sample_count(state), 1)
    weight_prev = max(n_eff - 1, 0)

    dim_num = length(param.dims)
    if dim_num == 1
        ρf = float(state.ρ)
        state.ρ_avg .= (state.ρ_avg .* weight_prev .+ ρf) ./ n_eff
        ρ_matrix = ρf * transpose(ρf)
        state.ρ_matrix_avg_cuts[:full] .= (state.ρ_matrix_avg_cuts[:full] .* weight_prev .+ ρ_matrix) ./ n_eff
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
    plot_label="",
    save_description=nothing,
    warmup_sweeps::Int=0,
    show_progress::Bool=true,
    relaxed_ic::Bool=false,
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

        update_and_compute_correlations!(state, param, rng; collect_statistics=collect_statistics)

        if sweep in save_times
            save_state(state, param, save_dir; relaxed_ic=relaxed_ic, description=save_description)
            println("State saved at sweep $sweep")
        end

        if plot_flag && (sweep in show_times)
            PlotUtils.plot_sweep(
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
