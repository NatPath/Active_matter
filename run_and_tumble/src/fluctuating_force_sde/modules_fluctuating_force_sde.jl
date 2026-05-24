module FPFluctuatingForceSDE

using Random: AbstractRNG, rand, randn

export FluctuatingForceParam, FluctuatingForceState, FluctuatingForceStats, ForceSDEWork
export normalize_dimension, wrap_centered, minimal_image, profile_value_r2
export initialize_state, step!, run!, result_dict, stability_recommendations

const GAUSSIAN_PROFILE = 1
const COMPACT_BUMP_PROFILE = 2

Base.@kwdef struct FluctuatingForceParam
    dims::Int = 1
    L::Float64 = 60.0
    N::Int = 8_000
    D_bath::Float64 = 1.0
    dt::Float64 = 0.05
    mu_bath::Float64 = 5.0
    f0::Float64 = 1.0
    sigma_f::Float64 = 0.5
    profile_type::String = "gaussian"
    force_centers::Matrix{Float64} = zeros(Float64, 1, 1)
    mobile_forces::Bool = false
    force_mobility::Float64 = 0.0
    force_diffusivity::Float64 = 0.0
    n_steps::Int = 10_000
    warmup_steps::Int = 1_000
    sample_interval::Int = 1
    n_bins::Int = 80
    n_radial_bins::Int = 20
    radial_min::Float64 = 0.8
    radial_max::Union{Nothing,Float64} = nothing
    edge_bins_for_offset::Int = 5
    variance_floor::Float64 = 1.0e-6
    history_interval::Int = 1_000
    max_history_records::Int = 20_000
    save_force_history::Bool = true
    seed::Int = 0
    description::String = ""
end

mutable struct FluctuatingForceState
    step::Int
    time::Float64
    positions::Matrix{Float64}
    force_centers::Matrix{Float64}
    force_centers_unwrapped::Matrix{Float64}
    last_force_sums::Vector{Float64}
    last_force_center_step::Matrix{Float64}
end

mutable struct ForceSDEWork
    force_values::Vector{Float64}
    force_sums::Vector{Float64}
    force_noise::Matrix{Float64}
    force_center_step::Matrix{Float64}
end

ForceSDEWork(dims::Integer, n_forces::Integer) = ForceSDEWork(
    zeros(Float64, n_forces),
    zeros(Float64, n_forces),
    zeros(Float64, dims, n_forces),
    zeros(Float64, dims, n_forces),
)

mutable struct FluctuatingForceStats
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    sum_counts::Vector{Float64}
    sum_counts2::Vector{Float64}
    scratch_counts::Vector{Int}
    sample_count::Int
    sum_force_sums::Vector{Float64}
    sum_force_sums2::Vector{Float64}
    history_time::Vector{Float64}
    history_force_centers_flat::Vector{Float64}
    history_force_sums_flat::Vector{Float64}
    history_records::Int
    history_truncated::Bool
    max_abs_bath_active_step::Float64
    max_abs_bath_thermal_step::Float64
    max_abs_force_center_step::Float64
end

function FluctuatingForceStats(param::FluctuatingForceParam)
    validate_param(param)
    edges = collect(range(-0.5 * param.L, 0.5 * param.L; length=param.n_bins + 1))
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:param.n_bins]
    n_cells = param.dims == 1 ? param.n_bins : param.n_bins * param.n_bins
    n_forces = size(param.force_centers, 2)
    return FluctuatingForceStats(
        edges,
        centers,
        zeros(Float64, n_cells),
        zeros(Float64, n_cells),
        zeros(Int, n_cells),
        0,
        zeros(Float64, n_forces),
        zeros(Float64, n_forces),
        Float64[],
        Float64[],
        Float64[],
        0,
        false,
        0.0,
        0.0,
        0.0,
    )
end

normalize_dimension(raw_dims) = begin
    dims = Int(raw_dims)
    dims in (1, 2) || throw(ArgumentError("dims must be 1 or 2. Got $(raw_dims)."))
    dims
end

@inline wrap_centered(x::Real, L::Real) = mod(Float64(x) + 0.5 * Float64(L), Float64(L)) - 0.5 * Float64(L)

@inline function minimal_image(dx::Real, L::Real)
    Lf = Float64(L)
    return mod(Float64(dx) + 0.5 * Lf, Lf) - 0.5 * Lf
end

@inline function profile_value_r2(r2::Real, param::FluctuatingForceParam)
    return profile_value_r2(r2, param, profile_kind(param.profile_type))
end

function profile_kind(profile_type::AbstractString)
    profile = lowercase(strip(profile_type))
    if profile == "gaussian"
        return GAUSSIAN_PROFILE
    elseif profile in ("bump", "compact_bump", "compact")
        return COMPACT_BUMP_PROFILE
    end
    throw(ArgumentError("Unsupported profile_type=$(profile_type). Use gaussian or compact_bump."))
end

@inline function profile_value_r2(r2::Real, param::FluctuatingForceParam, kind::Int)
    sigma = param.sigma_f
    sigma > 0 || throw(ArgumentError("sigma_f must be positive. Got $(param.sigma_f)."))
    if kind == GAUSSIAN_PROFILE
        return param.f0 * exp(-0.5 * Float64(r2) / (sigma * sigma))
    elseif kind == COMPACT_BUMP_PROFILE
        r = sqrt(max(Float64(r2), 0.0))
        u = r / sigma
        u >= 1.0 && return 0.0
        return param.f0 * exp(1.0 - 1.0 / (1.0 - u * u))
    end
    throw(ArgumentError("Unsupported profile kind code: $(kind)."))
end

function validate_param(param::FluctuatingForceParam)
    normalize_dimension(param.dims)
    param.L > 0 || throw(ArgumentError("L must be positive. Got $(param.L)."))
    param.N >= 0 || throw(ArgumentError("N must be nonnegative. Got $(param.N)."))
    param.dt > 0 || throw(ArgumentError("dt must be positive. Got $(param.dt)."))
    param.D_bath >= 0 || throw(ArgumentError("D_bath must be nonnegative. Got $(param.D_bath)."))
    param.sigma_f > 0 || throw(ArgumentError("sigma_f must be positive. Got $(param.sigma_f)."))
    profile_kind(param.profile_type)
    param.n_bins > 0 || throw(ArgumentError("n_bins must be positive. Got $(param.n_bins)."))
    param.sample_interval > 0 || throw(ArgumentError("sample_interval must be positive. Got $(param.sample_interval)."))
    size(param.force_centers, 1) == param.dims ||
        throw(ArgumentError("force_centers must have dims rows. Got $(size(param.force_centers, 1)) rows for dims=$(param.dims)."))
    size(param.force_centers, 2) > 0 || throw(ArgumentError("At least one fluctuating force is required."))
    param.mobile_forces || param.force_mobility == 0.0 ||
        throw(ArgumentError("force_mobility is nonzero but mobile_forces=false."))
    param.mobile_forces || param.force_diffusivity == 0.0 ||
        throw(ArgumentError("force_diffusivity is nonzero but mobile_forces=false."))
    param.force_diffusivity >= 0 || throw(ArgumentError("force_diffusivity must be nonnegative. Got $(param.force_diffusivity)."))
    return true
end

function initialize_uniform_positions!(positions::AbstractMatrix{Float64}, param::FluctuatingForceParam, rng::AbstractRNG)
    @inbounds for j in axes(positions, 2), d in axes(positions, 1)
        positions[d, j] = param.L * (rand(rng) - 0.5)
    end
    return positions
end

function initialize_state(param::FluctuatingForceParam, rng::AbstractRNG)
    validate_param(param)
    positions = zeros(Float64, param.dims, param.N)
    initialize_uniform_positions!(positions, param, rng)
    centers = copy(param.force_centers)
    centers_unwrapped = copy(param.force_centers)
    @inbounds for i in eachindex(centers)
        centers[i] = wrap_centered(centers[i], param.L)
    end
    n_forces = size(centers, 2)
    return FluctuatingForceState(
        0,
        0.0,
        positions,
        centers,
        centers_unwrapped,
        zeros(Float64, n_forces),
        zeros(Float64, param.dims, n_forces),
    )
end

@inline function distance_r2_to_force(state::FluctuatingForceState, param::FluctuatingForceParam, particle_idx::Int, force_idx::Int)
    r2 = 0.0
    @inbounds for d in 1:param.dims
        dx = minimal_image(state.positions[d, particle_idx] - state.force_centers[d, force_idx], param.L)
        r2 += dx * dx
    end
    return r2
end

function draw_force_noises!(work::ForceSDEWork, param::FluctuatingForceParam, rng::AbstractRNG)
    sqrt_dt = sqrt(param.dt)
    @inbounds for k in axes(work.force_noise, 2), d in axes(work.force_noise, 1)
        work.force_noise[d, k] = sqrt_dt * randn(rng)
    end
    return work.force_noise
end

function step!(state::FluctuatingForceState, param::FluctuatingForceParam, rng::AbstractRNG, work::ForceSDEWork)
    n_forces = size(param.force_centers, 2)
    kind = profile_kind(param.profile_type)

    fill!(work.force_sums, 0.0)
    fill!(work.force_center_step, 0.0)
    draw_force_noises!(work, param, rng)

    thermal_scale = sqrt(2.0 * param.D_bath * param.dt)
    max_active = 0.0
    max_thermal = 0.0
    @inbounds for i in 1:param.N
        for k in 1:n_forces
            val = profile_value_r2(distance_r2_to_force(state, param, i, k), param, kind)
            work.force_values[k] = val
            work.force_sums[k] += val
        end
        for d in 1:param.dims
            active_step = 0.0
            for k in 1:n_forces
                active_step += work.force_values[k] * work.force_noise[d, k]
            end
            active_step *= param.mu_bath
            thermal_step = thermal_scale * randn(rng)
            state.positions[d, i] = wrap_centered(state.positions[d, i] + active_step + thermal_step, param.L)
            max_active = max(max_active, abs(active_step))
            max_thermal = max(max_thermal, abs(thermal_step))
        end
    end

    max_center = 0.0
    if param.mobile_forces
        center_thermal_scale = sqrt(2.0 * param.force_diffusivity * param.dt)
        @inbounds for k in 1:n_forces, d in 1:param.dims
            center_step = -param.force_mobility * work.force_sums[k] * work.force_noise[d, k]
            if param.force_diffusivity > 0
                center_step += center_thermal_scale * randn(rng)
            end
            state.force_centers_unwrapped[d, k] += center_step
            state.force_centers[d, k] = wrap_centered(state.force_centers_unwrapped[d, k], param.L)
            work.force_center_step[d, k] = center_step
            max_center = max(max_center, abs(center_step))
        end
    end

    state.step += 1
    state.time += param.dt
    copyto!(state.last_force_sums, work.force_sums)
    copyto!(state.last_force_center_step, work.force_center_step)

    return (
        max_abs_bath_active_step=max_active,
        max_abs_bath_thermal_step=max_thermal,
        max_abs_force_center_step=max_center,
    )
end

function sample_counts!(counts::AbstractVector{Int}, state::FluctuatingForceState, param::FluctuatingForceParam)
    fill!(counts, 0)
    inv_dx = param.n_bins / param.L
    if param.dims == 1
        @inbounds for i in 1:param.N
            ix = clamp(floor(Int, (state.positions[1, i] + 0.5 * param.L) * inv_dx) + 1, 1, param.n_bins)
            counts[ix] += 1
        end
    else
        @inbounds for i in 1:param.N
            ix = clamp(floor(Int, (state.positions[1, i] + 0.5 * param.L) * inv_dx) + 1, 1, param.n_bins)
            iy = clamp(floor(Int, (state.positions[2, i] + 0.5 * param.L) * inv_dx) + 1, 1, param.n_bins)
            counts[ix + (iy - 1) * param.n_bins] += 1
        end
    end
    return counts
end

function maybe_record_history!(stats::FluctuatingForceStats, state::FluctuatingForceState, param::FluctuatingForceParam, production_step::Int)
    param.save_force_history || return stats
    param.history_interval > 0 || return stats
    production_step % param.history_interval == 0 || return stats
    if stats.history_records >= param.max_history_records
        stats.history_truncated = true
        return stats
    end

    push!(stats.history_time, state.time)
    append!(stats.history_force_centers_flat, vec(state.force_centers))
    append!(stats.history_force_sums_flat, state.last_force_sums)
    stats.history_records += 1
    return stats
end

function accumulate_sample!(stats::FluctuatingForceStats, state::FluctuatingForceState, param::FluctuatingForceParam)
    sample_counts!(stats.scratch_counts, state, param)
    @inbounds for i in eachindex(stats.scratch_counts)
        c = Float64(stats.scratch_counts[i])
        stats.sum_counts[i] += c
        stats.sum_counts2[i] += c * c
    end
    @inbounds for k in eachindex(stats.sum_force_sums)
        s = state.last_force_sums[k]
        stats.sum_force_sums[k] += s
        stats.sum_force_sums2[k] += s * s
    end
    stats.sample_count += 1
    return stats
end

function run!(state::FluctuatingForceState, param::FluctuatingForceParam, rng::AbstractRNG)
    validate_param(param)
    work = ForceSDEWork(param.dims, size(param.force_centers, 2))
    for _ in 1:max(param.warmup_steps, 0)
        step!(state, param, rng, work)
    end

    stats = FluctuatingForceStats(param)
    for production_step in 1:max(param.n_steps, 0)
        obs = step!(state, param, rng, work)
        stats.max_abs_bath_active_step = max(stats.max_abs_bath_active_step, obs.max_abs_bath_active_step)
        stats.max_abs_bath_thermal_step = max(stats.max_abs_bath_thermal_step, obs.max_abs_bath_thermal_step)
        stats.max_abs_force_center_step = max(stats.max_abs_force_center_step, obs.max_abs_force_center_step)
        if production_step % param.sample_interval == 0
            accumulate_sample!(stats, state, param)
        end
        maybe_record_history!(stats, state, param, production_step)
    end
    return stats
end

function count_variance(stats::FluctuatingForceStats)
    if stats.sample_count <= 0
        return zeros(Float64, length(stats.sum_counts)), zeros(Float64, length(stats.sum_counts))
    end
    mean_counts = stats.sum_counts ./ stats.sample_count
    variance = stats.sum_counts2 ./ stats.sample_count .- mean_counts .^ 2
    return mean_counts, variance
end

function one_dimensional_offset(variance::AbstractVector{Float64}, param::FluctuatingForceParam)
    n_edge = clamp(param.edge_bins_for_offset, 1, length(variance))
    total = 0.0
    count = 0
    @inbounds for i in 1:n_edge
        total += variance[i]
        count += 1
    end
    @inbounds for i in (length(variance) - n_edge + 1):length(variance)
        total += variance[i]
        count += 1
    end
    return total / count
end

function radial_edges(param::FluctuatingForceParam)
    rmax = isnothing(param.radial_max) ? param.L / 2.2 : Float64(param.radial_max)
    rmin = max(Float64(param.radial_min), eps(Float64))
    rmax > rmin || throw(ArgumentError("radial_max must be larger than radial_min. Got radial_min=$(rmin), radial_max=$(rmax)."))
    n = max(param.n_radial_bins, 1)
    logs = range(log(rmin), log(rmax); length=n + 1)
    return exp.(collect(logs))
end

function radial_summary(variance::AbstractVector{Float64}, param::FluctuatingForceParam)
    param.dims == 2 || return nothing
    edges = radial_edges(param)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:(length(edges)-1)]
    sums = zeros(Float64, length(centers))
    counts = zeros(Int, length(centers))
    @inbounds for iy in 1:param.n_bins, ix in 1:param.n_bins
        x = (ix - 0.5) * param.L / param.n_bins - 0.5 * param.L
        y = (iy - 0.5) * param.L / param.n_bins - 0.5 * param.L
        r = sqrt(x * x + y * y)
        for b in eachindex(centers)
            if edges[b] <= r < edges[b + 1]
                flat = ix + (iy - 1) * param.n_bins
                sums[b] += variance[flat]
                counts[b] += 1
                break
            end
        end
    end
    mean_total = similar(sums)
    @inbounds for i in eachindex(sums)
        mean_total[i] = counts[i] > 0 ? sums[i] / counts[i] : NaN
    end
    offset = begin
        finite_tail = [v for v in mean_total if isfinite(v)]
        isempty(finite_tail) ? 0.0 : finite_tail[end]
    end
    active = similar(mean_total)
    @inbounds for i in eachindex(mean_total)
        active[i] = isfinite(mean_total[i]) ? max(mean_total[i] - offset, param.variance_floor) : NaN
    end
    return Dict(
        "edges" => edges,
        "centers" => centers,
        "cell_counts" => counts,
        "variance_total" => mean_total,
        "thermal_offset" => offset,
        "variance_active" => active,
    )
end

function thermal_offset(variance::AbstractVector{Float64}, param::FluctuatingForceParam)
    if param.dims == 1
        return one_dimensional_offset(variance, param)
    end
    radial = radial_summary(variance, param)
    return Float64(radial["thermal_offset"])
end

function active_variance(variance::AbstractVector{Float64}, offset::Real, param::FluctuatingForceParam)
    out = similar(variance)
    @inbounds for i in eachindex(variance)
        out[i] = max(variance[i] - Float64(offset), param.variance_floor)
    end
    return out
end

function parameters_dict(param::FluctuatingForceParam)
    return Dict(
        "dims" => param.dims,
        "L" => param.L,
        "N" => param.N,
        "D_bath" => param.D_bath,
        "dt" => param.dt,
        "mu_bath" => param.mu_bath,
        "f0" => param.f0,
        "sigma_f" => param.sigma_f,
        "profile_type" => param.profile_type,
        "force_centers" => param.force_centers,
        "mobile_forces" => param.mobile_forces,
        "force_mobility" => param.force_mobility,
        "force_diffusivity" => param.force_diffusivity,
        "n_steps" => param.n_steps,
        "warmup_steps" => param.warmup_steps,
        "sample_interval" => param.sample_interval,
        "n_bins" => param.n_bins,
        "n_radial_bins" => param.n_radial_bins,
        "radial_min" => param.radial_min,
        "radial_max" => param.radial_max,
        "edge_bins_for_offset" => param.edge_bins_for_offset,
        "variance_floor" => param.variance_floor,
        "history_interval" => param.history_interval,
        "max_history_records" => param.max_history_records,
        "save_force_history" => param.save_force_history,
        "seed" => param.seed,
        "description" => param.description,
    )
end

function final_state_dict(state::FluctuatingForceState)
    return Dict(
        "step" => state.step,
        "time" => state.time,
        "force_centers" => state.force_centers,
        "force_centers_unwrapped" => state.force_centers_unwrapped,
        "last_force_sums" => state.last_force_sums,
        "last_force_center_step" => state.last_force_center_step,
    )
end

function stability_recommendations(param::FluctuatingForceParam, stats::FluctuatingForceStats)
    thermal_rms = sqrt(2.0 * param.D_bath * param.dt)
    bath_active_rms = abs(param.mu_bath * param.f0) * sqrt(param.dt)
    return Dict(
        "thermal_rms" => thermal_rms,
        "thermal_rms_over_sigma_f" => thermal_rms / param.sigma_f,
        "bath_active_single_profile_rms" => bath_active_rms,
        "bath_active_single_profile_rms_over_sigma_f" => bath_active_rms / param.sigma_f,
        "max_abs_bath_active_step" => stats.max_abs_bath_active_step,
        "max_abs_bath_thermal_step" => stats.max_abs_bath_thermal_step,
        "max_abs_force_center_step" => stats.max_abs_force_center_step,
        "max_abs_bath_active_step_over_sigma_f" => stats.max_abs_bath_active_step / param.sigma_f,
        "max_abs_bath_thermal_step_over_sigma_f" => stats.max_abs_bath_thermal_step / param.sigma_f,
        "max_abs_force_center_step_over_sigma_f" => stats.max_abs_force_center_step / param.sigma_f,
    )
end

function history_dict(stats::FluctuatingForceStats, param::FluctuatingForceParam)
    n_forces = size(param.force_centers, 2)
    return Dict(
        "time" => stats.history_time,
        "force_centers_flat" => stats.history_force_centers_flat,
        "force_sums_flat" => stats.history_force_sums_flat,
        "shape" => (stats.history_records, param.dims, n_forces),
        "truncated" => stats.history_truncated,
    )
end

function result_dict(param::FluctuatingForceParam, state::FluctuatingForceState, stats::FluctuatingForceStats)
    mean_counts, variance = count_variance(stats)
    offset = thermal_offset(variance, param)
    variance_active = active_variance(variance, offset, param)
    force_count = max(stats.sample_count, 1)
    mean_force_sums = stats.sum_force_sums ./ force_count
    variance_force_sums = stats.sum_force_sums2 ./ force_count .- mean_force_sums .^ 2
    radial = param.dims == 2 ? radial_summary(variance, param) : nothing
    return Dict(
        "result_type" => "fluctuating_force_sde_density_variance",
        "parameters" => parameters_dict(param),
        "final_state" => final_state_dict(state),
        "sample_count" => stats.sample_count,
        "bins" => Dict(
            "edges" => stats.bin_edges,
            "centers" => stats.bin_centers,
            "grid_shape" => param.dims == 1 ? (param.n_bins,) : (param.n_bins, param.n_bins),
            "sum_counts" => stats.sum_counts,
            "sum_counts2" => stats.sum_counts2,
            "mean_counts" => mean_counts,
            "variance_total" => variance,
            "thermal_offset" => offset,
            "variance_active" => variance_active,
        ),
        "radial" => radial,
        "forces" => Dict(
            "sum_profile_sums" => stats.sum_force_sums,
            "sum_profile_sums2" => stats.sum_force_sums2,
            "mean_profile_sums" => mean_force_sums,
            "variance_profile_sums" => variance_force_sums,
        ),
        "history" => history_dict(stats, param),
        "stability" => stability_recommendations(param, stats),
    )
end

end
