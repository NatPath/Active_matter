const OCCUPANCY_SAMPLER_FENWICK = UInt8(1)
const OCCUPANCY_SAMPLER_REJECTION = UInt8(2)
const OCCUPANCY_REJECTION_THRESHOLD = 64.0

mutable struct OccupancySampler
    mode::UInt8
    tree::Vector{Int64}
    max_bit::Int
end

mutable struct OccupancyState{N, R, C, B, S}
    t::Int64
    particles::Nothing
    ρ::R
    ρ₊::Union{Nothing,R}
    ρ₋::Union{Nothing,R}
    ρ_avg::Array{Float64, N}
    ρ_matrix_avg_cuts::C
    bond_pass_stats::B
    max_site_occupancy::Int64
    T::Float64
    potential::AbstractPotential
    forcing::Union{BondForce, Vector{BondForce}}
    exp_table::ExpLookupTable
    sampler::S
end

@inline sampler_uses_fenwick(sampler::OccupancySampler) = sampler.mode == OCCUPANCY_SAMPLER_FENWICK
@inline sampler_uses_rejection(sampler::OccupancySampler) = sampler.mode == OCCUPANCY_SAMPLER_REJECTION

@inline function occupancy_sampler_mode_name(sampler::OccupancySampler)
    return sampler_uses_rejection(sampler) ? "rejection" : "fenwick"
end

@inline function occupancy_sampler_mode_name(state)
    return occupancy_sampler_mode_name(state.sampler)
end

@inline function highest_power_of_two_leq(n::Int)
    bit = 1
    while (bit << 1) <= n
        bit <<= 1
    end
    return bit
end

@inline function choose_occupancy_sampler_mode(ρ::AbstractArray{<:Integer})
    mean_occupancy = sum(Int64, ρ) / max(length(ρ), 1)
    if mean_occupancy >= OCCUPANCY_REJECTION_THRESHOLD
        return OCCUPANCY_SAMPLER_REJECTION
    end
    return OCCUPANCY_SAMPLER_FENWICK
end

function OccupancySampler(mode::UInt8, weights::Vector{Int64})
    n = length(weights)
    n == 0 && return OccupancySampler(mode, Int64[], 0)
    if mode == OCCUPANCY_SAMPLER_REJECTION
        return OccupancySampler(mode, Int64[], 0)
    end
    tree = copy(weights)
    for idx in 1:n
        parent = idx + (idx & -idx)
        if parent <= n
            tree[parent] += tree[idx]
        end
    end
    return OccupancySampler(mode, tree, highest_power_of_two_leq(n))
end

function occupancy_sampler_from_density(ρ::AbstractArray{<:Integer})
    weights = Int64.(vec(ρ))
    any(<(0), weights) && throw(ArgumentError("Occupancy density contains negative values."))
    return OccupancySampler(choose_occupancy_sampler_mode(ρ), weights)
end

@inline function fenwick_add!(sampler::OccupancySampler, idx::Int, delta::Int64)
    tree = sampler.tree
    n = length(tree)
    while idx <= n
        @inbounds tree[idx] += delta
        idx += idx & -idx
    end
    return nothing
end

@inline function fenwick_sample_index(sampler::OccupancySampler, target::Int64)
    idx = 0
    bit = sampler.max_bit
    while bit != 0
        next_idx = idx + bit
        if next_idx <= length(sampler.tree) && sampler.tree[next_idx] < target
            idx = next_idx
            target -= sampler.tree[next_idx]
        end
        bit >>= 1
    end
    return idx + 1
end

@inline function sample_occupied_site_index_fenwick(sampler::OccupancySampler, total_particles::Int, rng)
    total_particles > 0 || throw(ArgumentError("Cannot sample an occupied site when total_particles <= 0."))
    target = Int64(rand(rng, 1:total_particles))
    return fenwick_sample_index(sampler, target)
end

@inline function sample_occupied_site_index_rejection(state::OccupancyState, rng)
    ρ = state.ρ
    bound = max(state.max_site_occupancy, Int64(1))
    while true
        idx = rand(rng, 1:length(ρ))
        @inbounds occ = Int64(ρ[idx])
        if occ > 0 && rand(rng, 1:bound) <= occ
            return idx
        end
    end
end

@inline function sample_occupied_site_index(state::OccupancyState, total_particles::Int, rng)
    if sampler_uses_rejection(state.sampler)
        return sample_occupied_site_index_rejection(state, rng)
    end
    return sample_occupied_site_index_fenwick(state.sampler, total_particles, rng)
end

@inline function sample_occupied_site_coords_2d_rejection(state::OccupancyState, rng)
    ρ = state.ρ
    Lx, Ly = size(ρ)
    bound = max(state.max_site_occupancy, Int64(1))
    while true
        i = rand(rng, 1:Lx)
        j = rand(rng, 1:Ly)
        @inbounds occ = Int64(ρ[i, j])
        if occ > 0 && rand(rng, 1:bound) <= occ
            return i, j
        end
    end
end

@inline function sample_occupied_site_coords_2d(state::OccupancyState, total_particles::Int, rng)
    if sampler_uses_rejection(state.sampler)
        i, j = sample_occupied_site_coords_2d_rejection(state, rng)
        return i, j, 0
    end
    flat_idx = sample_occupied_site_index_fenwick(state.sampler, total_particles, rng)
    i, j = coords_from_flat_2d(flat_idx, size(state.ρ, 1))
    return i, j, flat_idx
end

@inline function refresh_occupancy_sampler_after_sweep!(state::OccupancyState)
    if sampler_uses_rejection(state.sampler)
        state.max_site_occupancy = Int64(maximum(state.ρ))
    end
    return nothing
end

@inline flat_index_1d(i::Int) = i
@inline flat_index_2d(i::Int, j::Int, Lx::Int) = i + (j - 1) * Lx

@inline function coords_from_flat_2d(flat_idx::Int, Lx::Int)
    j = (flat_idx - 1) ÷ Lx + 1
    i = flat_idx - (j - 1) * Lx
    return i, j
end

function diffusive_initial_corr_cuts(ρ_avg::AbstractArray{Float64}; full_corr_tensor::Bool=false)
    dim = ndims(ρ_avg)
    if dim == 1
        return Dict{Symbol,AbstractArray{Float64}}(
            :full => ρ_avg * transpose(ρ_avg),
        )
    elseif dim == 2
        x_middle = div(size(ρ_avg, 1), 2)
        y_middle = div(size(ρ_avg, 2), 2)
        if full_corr_tensor
            return Dict{Symbol,AbstractArray{Float64}}(
                :full => outer_density_2D(ρ_avg),
            )
        end
        return Dict{Symbol,AbstractArray{Float64}}(
            :x_cut => ρ_avg[:, y_middle] * transpose(ρ_avg[:, y_middle]),
            :y_cut => ρ_avg[x_middle, :] * transpose(ρ_avg[x_middle, :]),
            :diag_cut => diag(ρ_avg) * transpose(diag(ρ_avg)),
        )
    end
    throw(DomainError("Invalid input - dimension not supported yet"))
end

function initialize_occupancy_density!(
    ρ::AbstractArray{<:Integer},
    ρ₊,
    ρ₋,
    rng,
    param;
    ic::AbstractString="random",
    ic_specific=[]
)
    fill!(ρ, 0)
    if !(ρ₊ === nothing)
        fill!(ρ₊, 0)
    end
    if !(ρ₋ === nothing)
        fill!(ρ₋, 0)
    end

    dim_num = ndims(ρ)
    total_particles = param.N
    if ic == "flat"
        num_sites = length(ρ)
        base_count = fld(total_particles, num_sites)
        remainder = mod(total_particles, num_sites)
        linear_idx = 1
        for cart_idx in CartesianIndices(ρ)
            count = base_count + (linear_idx <= remainder ? 1 : 0)
            ρ[cart_idx] = convert(eltype(ρ), count)
            linear_idx += 1
        end
    elseif ic == "random"
        if dim_num == 1
            for _ in 1:total_particles
                ρ[rand(rng, 1:param.dims[1])] += one(eltype(ρ))
            end
        elseif dim_num == 2
            for _ in 1:total_particles
                i = rand(rng, 1:param.dims[1])
                j = rand(rng, 1:param.dims[2])
                ρ[i, j] += one(eltype(ρ))
            end
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end
    elseif ic == "center"
        center = ntuple(i -> div(param.dims[i], 2), dim_num)
        ρ[CartesianIndex(center...)] = convert(eltype(ρ), total_particles)
    elseif ic == "specific"
        if length(ic_specific) != dim_num
            throw(DomainError("Invalid input - specific initial condition must have length $(dim_num)"))
        end
        position = ntuple(i -> Int(ic_specific[i]), dim_num)
        ρ[CartesianIndex(position...)] = convert(eltype(ρ), total_particles)
    else
        throw(DomainError("Invalid input - initial condition not supported yet"))
    end

    if !(ρ₊ === nothing)
        copyto!(ρ₊, ρ)
    end
    if !(ρ₋ === nothing)
        copyto!(ρ₋, ρ)
    end
    return nothing
end

function occupancy_state_from_components(
    t,
    ρ,
    ρ₊,
    ρ₋,
    ρ_avg,
    ρ_matrix_avg_cuts,
    bond_pass_stats,
    T,
    potential,
    forcing,
    exp_table,
)
    sampler = occupancy_sampler_from_density(ρ)
    max_site_occupancy = Int64(maximum(ρ))
    return OccupancyState(
        t,
        nothing,
        ρ,
        ρ₊,
        ρ₋,
        ρ_avg,
        ρ_matrix_avg_cuts,
        bond_pass_stats,
        max_site_occupancy,
        T,
        potential,
        forcing,
        exp_table,
        sampler,
    )
end

function setOccupancyState(t, rng, param, T, potential=Potentials.setPotential(zeros(Float64, param.dims)), bond_force=Potentials.setBondForce(([1], [2]), true, 0.0);
                           ic="random", full_corr_tensor=false, int_type::Type{<:Signed}=Int32,
                           keep_directional_densities::Bool=false,
                           bond_pass_count_mode::AbstractString="nonzero_magnitude")
    N = param.N
    N > typemax(int_type) && throw(ArgumentError("Requested density int_type $(int_type) cannot represent max site occupancy N=$(N)."))

    ρ = zeros(int_type, param.dims...)
    ρ₊ = keep_directional_densities ? zeros(int_type, param.dims...) : nothing
    ρ₋ = keep_directional_densities ? zeros(int_type, param.dims...) : nothing
    initialize_occupancy_density!(ρ, ρ₊, ρ₋, rng, param; ic=String(ic))

    ρ_avg = Float64.(ρ)
    ρ_matrix_avg_cuts = diffusive_initial_corr_cuts(ρ_avg; full_corr_tensor=full_corr_tensor)
    exp_table = create_exp_lookup(-20.0, 20.0, 10000)
    bond_pass_stats = Dict{Symbol,Vector{Float64}}()
    bond_forces = if bond_force isa BondForce
        [bond_force]
    elseif bond_force isa AbstractVector && all(f -> f isa BondForce, bond_force)
        BondForce[f for f in bond_force]
    else
        throw(ArgumentError("bond_force must be BondForce or Vector{BondForce}"))
    end
    track_mask = bond_pass_track_mask_for_forcings(bond_forces, String(bond_pass_count_mode))
    initialize_bond_passage_stats!(bond_pass_stats, length(bond_forces); track_mask=track_mask)
    if ndims(ρ) == 1
        initialize_spatial_bond_passage_stats!(bond_pass_stats, param.dims[1])
    end

    return occupancy_state_from_components(
        t,
        ρ,
        ρ₊,
        ρ₋,
        ρ_avg,
        ρ_matrix_avg_cuts,
        bond_pass_stats,
        T,
        potential,
        bond_forces,
        exp_table,
    )
end

function reset_statistics!(state::OccupancyState)
    dim = ndims(state.ρ)
    state.t = 0
    state.ρ_avg .= state.ρ
    state.max_site_occupancy = Int64(maximum(state.ρ))
    if has_directional_densities(state)
        state.ρ₊ .= state.ρ
        state.ρ₋ .= state.ρ
    end

    if dim == 1
        ρf = float(state.ρ)
        state.ρ_matrix_avg_cuts[:full] .= ρf * transpose(ρf)
    elseif dim == 2
        x_middle = div(size(state.ρ, 1), 2)
        y_middle = div(size(state.ρ, 2), 2)
        if haskey(state.ρ_matrix_avg_cuts, :full)
            state.ρ_matrix_avg_cuts[:full] .= outer_density_2D(float(state.ρ))
        else
            ρf = float(state.ρ)
            state.ρ_matrix_avg_cuts[:x_cut] .= ρf[:, y_middle] * transpose(ρf[:, y_middle])
            state.ρ_matrix_avg_cuts[:y_cut] .= ρf[x_middle, :] * transpose(ρf[x_middle, :])
            state.ρ_matrix_avg_cuts[:diag_cut] .= diag(ρf) * transpose(diag(ρf))
        end
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end

    forcings = get_state_forcings!(state)
    existing_mask = if haskey(state.bond_pass_stats, BOND_PASS_TRACK_MASK_KEY) &&
                       length(state.bond_pass_stats[BOND_PASS_TRACK_MASK_KEY]) == length(forcings)
        Float64.(state.bond_pass_stats[BOND_PASS_TRACK_MASK_KEY])
    else
        bond_pass_track_mask_for_forcings(forcings, "nonzero_magnitude")
    end
    initialize_bond_passage_stats!(state.bond_pass_stats, length(forcings); track_mask=existing_mask)
    if dim == 1
        initialize_spatial_bond_passage_stats!(state.bond_pass_stats, size(state.ρ, 1))
    end
    state.sampler = occupancy_sampler_from_density(state.ρ)
    return state
end

function update!(param, state::OccupancyState, rng; benchmark=false, collect_statistics::Bool=true)
    bench_results = benchmark ? BenchmarkResults() : nothing

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
        scratch = runtime_scratch!(state)
        left_neighbors, right_neighbors = periodic_neighbors_1d!(state, L)
        ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
        tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
        sweep_forward_counts = cached_int_buffer!(scratch, :bond_forward_counts_1d, n_forces)
        sweep_reverse_counts = cached_int_buffer!(scratch, :bond_reverse_counts_1d, n_forces)
        ensure_spatial_bond_passage_stats!(state, L)
        sweep_spatial_forward_counts = cached_int_buffer!(scratch, :spatial_forward_counts_1d, L)
        sweep_spatial_reverse_counts = cached_int_buffer!(scratch, :spatial_reverse_counts_1d, L)
        t_end = state.t + 1
        t = state.t
        while t < t_end
            if benchmark
                t1 = time_ns()
            end

            n_and_a = rand(rng, 1:2 * (param.N + 1))
            action_index = mod1(n_and_a, 2)
            selected_channel = (n_and_a - action_index) ÷ 2 + 1
            particle_channel = selected_channel <= param.N
            if benchmark
                bench_results.action_selection_time += (time_ns() - t1) / 1e9
            end

            spot_index = 0
            candidate_spot_index = 0
            p_candidate = γ
            if particle_channel
                spot_index = sample_occupied_site_index(state, param.N, rng)
                if action_index == 1
                    candidate_spot_index = left_neighbors[spot_index]
                else
                    candidate_spot_index = right_neighbors[spot_index]
                end

                if benchmark
                    t2 = time_ns()
                end
                directed_bond_forcing = directed_bond_forcing_1d(forcings, spot_index, candidate_spot_index)
                if static_zero_potential
                    p_candidate = bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme)
                else
                    p_candidate = calculate_jump_probability(param.D, V[candidate_spot_index] - V[spot_index], T, state.exp_table;
                                                             directed_bond_forcing=directed_bond_forcing,
                                                             rate_normalization=rate_normalization,
                                                             forcing_rate_scheme=scheme)
                end
                if benchmark
                    bench_results.jump_probability_time += (time_ns() - t2) / 1e9
                end
            end

            if benchmark
                t3 = time_ns()
            end
            p_candidate = clamp(p_candidate, 0.0, 1.0)
            choice = tower_sampling(p_candidate, 1 - p_candidate, rng)
            if benchmark
                bench_results.tower_sampling_time += (time_ns() - t3) / 1e9
            end

            if benchmark
                t4 = time_ns()
            end
            if choice == 1
                if particle_channel
                    if !isempty(tracked_force_indices) && candidate_spot_index != spot_index
                        record_bond_passage_1d!(sweep_forward_counts, sweep_reverse_counts,
                                                forcings, tracked_force_indices,
                                                spot_index, candidate_spot_index)
                    end
                    if action_index == 2 && candidate_spot_index == right_neighbors[spot_index]
                        sweep_spatial_forward_counts[spot_index] += 1
                    elseif action_index == 1 && candidate_spot_index == left_neighbors[spot_index]
                        sweep_spatial_reverse_counts[candidate_spot_index] += 1
                    end

                    if benchmark
                        t5 = time_ns()
                    end
                    state.ρ[spot_index] -= 1
                    state.ρ[candidate_spot_index] += 1
                    if sampler_uses_fenwick(state.sampler)
                        fenwick_add!(state.sampler, spot_index, -1)
                        fenwick_add!(state.sampler, candidate_spot_index, 1)
                    end
                    if has_directional_densities(state)
                        state.ρ₊[spot_index] -= 1
                        state.ρ₊[candidate_spot_index] += 1
                        state.ρ₋[spot_index] -= 1
                        state.ρ₋[candidate_spot_index] += 1
                    end
                    if state.ρ[candidate_spot_index] > state.max_site_occupancy
                        state.max_site_occupancy = Int64(state.ρ[candidate_spot_index])
                    end
                    if benchmark
                        bench_results.density_update_time += (time_ns() - t5) / 1e9
                    end
                else
                    potential_update!(state.potential, rng)
                end
            end

            for force_idx in 1:n_forces
                p_force = forcing_rate(ffrs, force_idx) / max(param.N, 1)
                if p_force > 1.0
                    p_force = 1.0
                elseif p_force < 0.0
                    p_force = 0.0
                end
                if rand(rng) < p_force
                    bondforce_update!(forcings[force_idx])
                end
            end

            if benchmark
                bench_results.state_update_time += (time_ns() - t4) / 1e9
                bench_results.total_samples += 1
            end
            t += 1 / param.N
        end

        refresh_occupancy_sampler_after_sweep!(state)
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
        scratch = runtime_scratch!(state)
        left_x, right_x, down_y, up_y = periodic_neighbors_2d!(state, Lx, Ly)
        ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
        tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
        sweep_forward_counts = cached_int_buffer!(scratch, :bond_forward_counts_2d, n_forces)
        sweep_reverse_counts = cached_int_buffer!(scratch, :bond_reverse_counts_2d, n_forces)
        t_end = state.t + 1
        t = state.t
        while t < t_end
            if benchmark
                t1 = time_ns()
            end

            n_and_a = rand(rng, 1:4 * (param.N + 1))
            action_index = mod1(n_and_a, 4)
            selected_channel = (n_and_a - action_index) ÷ 4 + 1
            particle_channel = selected_channel <= param.N
            if benchmark
                bench_results.action_selection_time += (time_ns() - t1) / 1e9
            end

            source_flat = 0
            target_flat = 0
            source_i = 0
            source_j = 0
            cand_i = 0
            cand_j = 0
            p_cand = γ
            if particle_channel
                source_i, source_j, source_flat = sample_occupied_site_coords_2d(state, param.N, rng)
                if action_index == 1
                    cand_i = left_x[source_i]
                    cand_j = source_j
                elseif action_index == 2
                    cand_i = right_x[source_i]
                    cand_j = source_j
                elseif action_index == 3
                    cand_i = source_i
                    cand_j = down_y[source_j]
                else
                    cand_i = source_i
                    cand_j = up_y[source_j]
                end
                if sampler_uses_fenwick(state.sampler)
                    target_flat = flat_index_2d(cand_i, cand_j, Lx)
                end

                if benchmark
                    t2 = time_ns()
                end
                directed_bond_forcing = directed_bond_forcing_2d(forcings, (source_i, source_j), (cand_i, cand_j))
                if static_zero_potential
                    p_cand = bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme)
                else
                    p_cand = calculate_jump_probability(param.D, V[cand_i, cand_j] - V[source_i, source_j], T, state.exp_table;
                                                        directed_bond_forcing=directed_bond_forcing,
                                                        rate_normalization=rate_normalization,
                                                        forcing_rate_scheme=scheme)
                end
                if benchmark
                    bench_results.jump_probability_time += (time_ns() - t2) / 1e9
                end
            end

            if benchmark
                t3 = time_ns()
            end
            p_cand = clamp(p_cand, 0.0, 1.0)
            choice = tower_sampling(p_cand, 1 - p_cand, rng)
            if benchmark
                bench_results.tower_sampling_time += (time_ns() - t3) / 1e9
            end

            if benchmark
                t4 = time_ns()
            end
            if choice == 1
                if particle_channel
                    if !isempty(tracked_force_indices) && (cand_i != source_i || cand_j != source_j)
                        record_bond_passage_2d!(sweep_forward_counts, sweep_reverse_counts,
                                                forcings, tracked_force_indices,
                                                (source_i, source_j), (cand_i, cand_j))
                    end

                    if benchmark
                        t5 = time_ns()
                    end
                    state.ρ[source_i, source_j] -= 1
                    state.ρ[cand_i, cand_j] += 1
                    if sampler_uses_fenwick(state.sampler)
                        fenwick_add!(state.sampler, source_flat, -1)
                        fenwick_add!(state.sampler, target_flat, 1)
                    end
                    if has_directional_densities(state)
                        state.ρ₊[source_i, source_j] -= 1
                        state.ρ₊[cand_i, cand_j] += 1
                        state.ρ₋[source_i, source_j] -= 1
                        state.ρ₋[cand_i, cand_j] += 1
                    end
                    if state.ρ[cand_i, cand_j] > state.max_site_occupancy
                        state.max_site_occupancy = Int64(state.ρ[cand_i, cand_j])
                    end
                    if benchmark
                        bench_results.density_update_time += (time_ns() - t5) / 1e9
                    end
                else
                    potential_update!(state.potential, rng)
                end
            end

            for force_idx in 1:n_forces
                p_force = forcing_rate(ffrs, force_idx) / max(param.N, 1)
                if p_force > 1.0
                    p_force = 1.0
                elseif p_force < 0.0
                    p_force = 0.0
                end
                if rand(rng) < p_force
                    bondforce_update!(forcings[force_idx])
                end
            end

            if benchmark
                bench_results.state_update_time += (time_ns() - t4) / 1e9
                bench_results.total_samples += 1
            end
            t += 1 / param.N
        end

        refresh_occupancy_sampler_after_sweep!(state)
        if collect_statistics
            update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
        end
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end

    state.t += 1
    if benchmark
        print_benchmark_summary(bench_results)
        return bench_results
    end
end
