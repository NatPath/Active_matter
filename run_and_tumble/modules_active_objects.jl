module FPActiveObjects
    using Random: AbstractRNG, rand, randperm
    using ..Potentials: AbstractPotential, BondForce
    using ..FPDiffusive

    export Param, State, setParam, setState
    export normalize_object_motion_scheme, object_memory_alpha
    export apply_object_dynamics!, record_object_history!
    export reset_measurement_statistics_preserve_time!
    export object_left_sites, object_forward_distance, object_min_distance
    export object_forward_gap, object_min_gap, object_min_edge_distance

    const HARD_REFRESH_SCHEME = "hard_refresh"
    const EXPONENTIAL_MEMORY_SCHEME = "exponential_memory"

    function normalize_object_motion_scheme(raw_scheme)
        scheme = lowercase(strip(String(raw_scheme)))
        if scheme in ("hard_refresh", "hard", "refresh", "window")
            return HARD_REFRESH_SCHEME
        elseif scheme in ("exponential_memory", "exponential", "ema", "ewma", "memory")
            return EXPONENTIAL_MEMORY_SCHEME
        end
        throw(ArgumentError("Unsupported object_motion_scheme: $raw_scheme. Use \"hard_refresh\" or \"exponential_memory\"."))
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
        correlation_baseline_model::String
        object_motion_scheme::String
        object_refresh_sweeps::Int
        object_memory_sweeps::Float64
        object_kappa::Float64
        object_D0::Float64
        object_history_interval::Int
        object_history_on_move_only::Bool
    end

    mutable struct State{N,C,B,D}
        t::Int64
        particles::Vector{FPDiffusive.Particle{D}}
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
        exp_table::FPDiffusive.ExpLookupTable
        object_stats::Dict{Symbol,Any}
    end

    function setParam(
        γ,
        dims,
        ρ₀,
        D,
        potential_type,
        fluctuation_type,
        potential_magnitude,
        ffr=0.0;
        forcing_rate_scheme=FPDiffusive.LEGACY_FORCING_RATE_SCHEME,
        correlation_baseline_model="independent",
        object_motion_scheme=HARD_REFRESH_SCHEME,
        object_refresh_sweeps::Int=10,
        object_memory_sweeps::Real=max(object_refresh_sweeps, 1),
        object_kappa::Real=1.0,
        object_D0::Real=0.0,
        object_history_interval::Int=1,
        object_history_on_move_only::Bool=false,
    )
        base_param = FPDiffusive.setParam(
            γ,
            dims,
            ρ₀,
            D,
            potential_type,
            fluctuation_type,
            potential_magnitude,
            ffr;
            forcing_rate_scheme=forcing_rate_scheme,
        )
        refresh_sweeps = max(Int(object_refresh_sweeps), 1)
        memory_sweeps = max(Float64(object_memory_sweeps), 1.0)
        history_interval = max(Int(object_history_interval), 1)
        scheme = normalize_object_motion_scheme(object_motion_scheme)
        return Param(
            base_param.γ,
            base_param.dims,
            base_param.ρ₀,
            base_param.N,
            base_param.D,
            base_param.potential_type,
            base_param.fluctuation_type,
            base_param.potential_magnitude,
            base_param.ffr,
            base_param.forcing_rate_scheme,
            String(correlation_baseline_model),
            scheme,
            refresh_sweeps,
            memory_sweeps,
            Float64(object_kappa),
            Float64(object_D0),
            history_interval,
            Bool(object_history_on_move_only),
        )
    end

    object_memory_alpha(param::Param) = clamp(1.0 / max(param.object_memory_sweeps, 1.0), 0.0, 1.0)

    function bond_left_site_and_orientation_1d(first_site::Int, second_site::Int, L::Int)
        if mod1(first_site + 1, L) == second_site
            return first_site, Int8(1)
        elseif mod1(second_site + 1, L) == first_site
            return second_site, Int8(-1)
        end
        throw(ArgumentError("Active-object forcings must lie on nearest-neighbor bonds of the 1D ring. Got ($first_site, $second_site) for L=$L."))
    end

    function canonicalize_force_bond_1d!(force::BondForce, L::Int)
        first_site = mod1(force.bond_indices[1][1], L)
        second_site = mod1(force.bond_indices[2][1], L)
        left_site, orientation_sign = bond_left_site_and_orientation_1d(first_site, second_site, L)
        right_site = mod1(left_site + 1, L)
        force.bond_indices = ([left_site], [right_site])
        if orientation_sign == -1
            force.direction_flag = !force.direction_flag
        end
        return force
    end

    function current_object_left_sites(forcings::Vector{BondForce}, L::Int)
        left_sites = Int[]
        for force in forcings
            canonicalize_force_bond_1d!(force, L)
            push!(left_sites, force.bond_indices[1][1])
        end
        return left_sites
    end

    function object_forward_distance(left_sites::AbstractVector{<:Integer}, L::Int)
        length(left_sites) == 2 || return 0
        return mod(Int(left_sites[2]) - Int(left_sites[1]), L)
    end

    function object_min_distance(left_sites::AbstractVector{<:Integer}, L::Int)
        d_forward = object_forward_distance(left_sites, L)
        d_backward = mod(Int(left_sites[1]) - Int(left_sites[2]), L)
        return min(d_forward, d_backward)
    end

    function object_forward_gap(left_sites::AbstractVector{<:Integer}, L::Int)
        d_forward = object_forward_distance(left_sites, L)
        return max(d_forward - 1, 0)
    end

    function object_min_gap(left_sites::AbstractVector{<:Integer}, L::Int)
        d_min = object_min_distance(left_sites, L)
        return max(d_min - 1, 0)
    end

    @inline function circular_site_distance(a::Integer, b::Integer, L::Int)
        forward = mod(Int(b) - Int(a), L)
        backward = mod(Int(a) - Int(b), L)
        return min(forward, backward)
    end

    function object_min_edge_distance(left_sites::AbstractVector{<:Integer}, L::Int)
        length(left_sites) == 2 || return 0
        left_1 = Int(left_sites[1])
        left_2 = Int(left_sites[2])
        right_1 = mod1(left_1 + 1, L)
        right_2 = mod1(left_2 + 1, L)
        return min(
            circular_site_distance(left_1, right_2, L),
            circular_site_distance(right_1, left_2, L),
        )
    end

    function initialize_object_stats(param::Param, forcings::Vector{BondForce}, t::Int)
        n_objects = length(forcings)
        stats = Dict{Symbol,Any}()
        stats[:motion_scheme] = param.object_motion_scheme
        stats[:refresh_sweeps] = param.object_refresh_sweeps
        stats[:memory_sweeps] = param.object_memory_sweeps
        stats[:kappa] = param.object_kappa
        stats[:D0] = param.object_D0
        stats[:history_interval] = param.object_history_interval
        stats[:history_on_move_only] = param.object_history_on_move_only
        stats[:refresh_age] = 0
        stats[:window_forward] = zeros(Float64, n_objects)
        stats[:window_reverse] = zeros(Float64, n_objects)
        stats[:memory_forward] = zeros(Float64, n_objects)
        stats[:memory_reverse] = zeros(Float64, n_objects)
        stats[:last_rightward_counts] = zeros(Float64, n_objects)
        stats[:last_leftward_counts] = zeros(Float64, n_objects)
        stats[:last_move_deltas] = zeros(Int, n_objects)
        stats[:history_sweeps] = Int[]
        stats[:history_left_sites] = Vector{Vector{Int}}()
        stats[:history_move_deltas] = Vector{Vector{Int}}()
        stats[:history_rightward_counts] = Vector{Vector{Float64}}()
        stats[:history_leftward_counts] = Vector{Vector{Float64}}()
        stats[:history_forward_distance] = Int[]
        stats[:history_min_distance] = Int[]
        stats[:history_forward_gap] = Int[]
        stats[:history_min_gap] = Int[]
        stats[:history_min_edge_distance] = Int[]
        stats[:last_recorded_t] = -1
        return stats
    end

    function record_object_history!(state::State, param::Param; force::Bool=false)
        stats = state.object_stats
        interval = Int(stats[:history_interval])
        history_on_move_only = get(stats, :history_on_move_only, false) === true
        already_recorded = Int(stats[:last_recorded_t]) == state.t
        moved_this_sweep = any(!iszero, stats[:last_move_deltas])
        should_record = if force
            true
        elseif history_on_move_only
            !already_recorded && moved_this_sweep
        else
            !already_recorded && state.t % interval == 0
        end
        if !should_record
            return nothing
        end

        forcings = FPDiffusive.get_state_forcings!(state)
        left_sites = current_object_left_sites(forcings, param.dims[1])
        push!(stats[:history_sweeps], state.t)
        push!(stats[:history_left_sites], copy(left_sites))
        push!(stats[:history_move_deltas], Int.(copy(stats[:last_move_deltas])))
        push!(stats[:history_rightward_counts], Float64.(copy(stats[:last_rightward_counts])))
        push!(stats[:history_leftward_counts], Float64.(copy(stats[:last_leftward_counts])))
        if length(left_sites) == 2
            push!(stats[:history_forward_distance], object_forward_distance(left_sites, param.dims[1]))
            push!(stats[:history_min_distance], object_min_distance(left_sites, param.dims[1]))
            push!(stats[:history_forward_gap], object_forward_gap(left_sites, param.dims[1]))
            push!(stats[:history_min_gap], object_min_gap(left_sites, param.dims[1]))
            push!(stats[:history_min_edge_distance], object_min_edge_distance(left_sites, param.dims[1]))
        end
        stats[:last_recorded_t] = state.t
        return nothing
    end

    function setState(
        t,
        rng,
        param::Param,
        T,
        potential,
        bond_force;
        ic="random",
        full_corr_tensor=false,
        int_type::Type{<:Integer}=Int32,
        bond_pass_count_mode::AbstractString="all_forcing_bonds",
    )
        base_state = FPDiffusive.setState(
            t,
            rng,
            param,
            T,
            potential,
            bond_force;
            ic=ic,
            full_corr_tensor=full_corr_tensor,
            int_type=int_type,
            bond_pass_count_mode=bond_pass_count_mode,
        )
        forcings = FPDiffusive.get_state_forcings!(base_state)
        for force in forcings
            canonicalize_force_bond_1d!(force, param.dims[1])
        end
        object_stats = initialize_object_stats(param, forcings, t)
        state = State(
            base_state.t,
            base_state.particles,
            base_state.ρ,
            base_state.ρ₊,
            base_state.ρ₋,
            base_state.ρ_avg,
            base_state.ρ_matrix_avg_cuts,
            base_state.bond_pass_stats,
            base_state.max_site_occupancy,
            base_state.T,
            base_state.potential,
            base_state.forcing,
            base_state.exp_table,
            object_stats,
        )
        record_object_history!(state, param; force=true)
        return state
    end

    function object_left_sites(state::State, param::Param)
        forcings = FPDiffusive.get_state_forcings!(state)
        return current_object_left_sites(forcings, param.dims[1])
    end

    function sample_object_delta(left_rate::Real, right_rate::Real, rng::AbstractRNG)
        λ_left = max(Float64(left_rate), 0.0)
        λ_right = max(Float64(right_rate), 0.0)
        λ_total = λ_left + λ_right
        λ_total <= 0.0 && return 0

        p_move = 1.0 - exp(-λ_total)
        rand(rng) >= p_move && return 0
        rand(rng) < (λ_left / λ_total) ? -1 : 1
    end

    function apply_sampled_object_moves!(
        state::State,
        param::Param,
        rng::AbstractRNG,
        left_rates::AbstractVector{<:Real},
        right_rates::AbstractVector{<:Real},
    )
        forcings = FPDiffusive.get_state_forcings!(state)
        n_objects = length(forcings)
        n_objects == 0 && return Int[]

        L = param.dims[1]
        current_left_sites = current_object_left_sites(forcings, L)
        occupied_left_sites = Set(current_left_sites)
        proposed_deltas = [sample_object_delta(left_rates[idx], right_rates[idx], rng) for idx in 1:n_objects]
        accepted_deltas = zeros(Int, n_objects)

        for object_idx in randperm(rng, n_objects)
            delta = proposed_deltas[object_idx]
            delta == 0 && continue
            current_left_site = current_left_sites[object_idx]
            target_left_site = mod1(current_left_site + delta, L)
            if target_left_site in occupied_left_sites
                continue
            end
            delete!(occupied_left_sites, current_left_site)
            push!(occupied_left_sites, target_left_site)
            current_left_sites[object_idx] = target_left_site
            forcings[object_idx].bond_indices = ([target_left_site], [mod1(target_left_site + 1, L)])
            accepted_deltas[object_idx] = delta
        end

        return accepted_deltas
    end

    function apply_object_dynamics!(
        state::State,
        param::Param,
        rng::AbstractRNG,
        rightward_counts::AbstractVector{<:Real},
        leftward_counts::AbstractVector{<:Real},
    )
        stats = state.object_stats
        n_objects = length(FPDiffusive.get_state_forcings!(state))
        if n_objects == 0
            return zeros(Int, 0)
        end

        stats[:last_rightward_counts] .= rightward_counts
        stats[:last_leftward_counts] .= leftward_counts

        move_deltas = zeros(Int, n_objects)
        if param.object_motion_scheme == HARD_REFRESH_SCHEME
            stats[:window_forward] .+= rightward_counts
            stats[:window_reverse] .+= leftward_counts
            stats[:refresh_age] = Int(stats[:refresh_age]) + 1
            if Int(stats[:refresh_age]) >= param.object_refresh_sweeps
                left_rates = param.object_D0 * param.object_refresh_sweeps .+ param.object_kappa .* stats[:window_forward]
                right_rates = param.object_D0 * param.object_refresh_sweeps .+ param.object_kappa .* stats[:window_reverse]
                move_deltas .= apply_sampled_object_moves!(state, param, rng, left_rates, right_rates)
                fill!(stats[:window_forward], 0.0)
                fill!(stats[:window_reverse], 0.0)
                stats[:refresh_age] = 0
            end
        else
            α = object_memory_alpha(param)
            stats[:memory_forward] .= (1.0 - α) .* stats[:memory_forward] .+ α .* rightward_counts
            stats[:memory_reverse] .= (1.0 - α) .* stats[:memory_reverse] .+ α .* leftward_counts
            left_rates = param.object_D0 .+ param.object_kappa .* stats[:memory_forward]
            right_rates = param.object_D0 .+ param.object_kappa .* stats[:memory_reverse]
            move_deltas .= apply_sampled_object_moves!(state, param, rng, left_rates, right_rates)
        end

        stats[:last_move_deltas] .= move_deltas
        return move_deltas
    end

    function reset_measurement_statistics_preserve_time!(state::State)
        t_old = state.t
        FPDiffusive.reset_statistics!(state)
        state.t = t_old
        return state
    end
end
