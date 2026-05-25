module FPCoupledSDEActiveObjects

using Random: AbstractRNG, rand, randn

export CoupledSDEParam, CoupledSDEState, SDEWork, StepObservables
export FIXED_SEPARATION_MODE, MOBILE_OBJECTS_MODE
export normalize_mode, initialize_state, initialize_uniform_bath!
export wrap_position, minimal_image, oriented_separation, minimum_separation
export profile_value, compute_profile_sums!, draw_sde_noises
export step_fixed_separation!, step_mobile_objects!
export run_fixed_separation!, run_mobile_objects!
export fixed_result_dict, mobile_result_dict, stability_recommendations

const FIXED_SEPARATION_MODE = "fixed_separation"
const MOBILE_OBJECTS_MODE = "mobile_objects"

Base.@kwdef struct CoupledSDEParam
    mode::String = MOBILE_OBJECTS_MODE
    L::Float64 = 128.0
    rho0::Float64 = 10.0
    N::Int = 1280
    D0::Float64 = 1.0
    dt::Float64 = 1.0e-3
    mu_bath::Float64 = 1.0
    mu_obj::Float64 = 1.0e-3
    f0::Float64 = 1.0
    sigma_f::Float64 = 1.0
    profile_type::String = "gaussian"
    separation::Float64 = 16.0
    initial_XA::Union{Nothing,Float64} = nothing
    initial_XB::Union{Nothing,Float64} = nothing
    random_initial_objects::Bool = false
    initial_min_separation::Float64 = 0.0
    initial_max_separation::Union{Nothing,Float64} = nothing
    n_steps::Int = 10_000
    warmup_steps::Int = 1_000
    sample_interval::Int = 1
    history_interval::Int = 100
    n_bins::Int = 64
    max_history_records::Int = 100_000
    save_raw_history::Bool = true
    seed::Int = 0
    description::String = ""
end

mutable struct CoupledSDEState
    step::Int
    time::Float64
    x::Vector{Float64}
    XA::Float64
    XB::Float64
    XA_unwrapped::Float64
    XB_unwrapped::Float64
    last_SA::Float64
    last_SB::Float64
    last_dXA::Float64
    last_dXB::Float64
    last_drel::Float64
end

mutable struct SDEWork
    fA::Vector{Float64}
    fB::Vector{Float64}
    dWi::Vector{Float64}
end

SDEWork(N::Integer) = SDEWork(zeros(Float64, N), zeros(Float64, N), zeros(Float64, N))

struct StepObservables
    step::Int
    time::Float64
    XA::Float64
    XB::Float64
    separation_oriented::Float64
    separation_min::Float64
    SA::Float64
    SB::Float64
    Ssum::Float64
    dWA::Float64
    dWB::Float64
    dXA::Float64
    dXB::Float64
    drel::Float64
    max_abs_bath_active_step::Float64
    max_abs_bath_thermal_step::Float64
    max_abs_object_step::Float64
end

mutable struct FixedStats
    sample_count::Int
    sum_SA::Float64
    sum_SB::Float64
    sum_SA2::Float64
    sum_SB2::Float64
    sum_SASB::Float64
    sum_Ssum::Float64
    sum_Ssum2::Float64
    max_abs_bath_active_step::Float64
    max_abs_bath_thermal_step::Float64
    max_abs_object_step::Float64
end

function FixedStats()
    return FixedStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

mutable struct MobileStats
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    histogram_counts::Vector{Float64}
    bin_counts::Vector{Float64}
    sum_delta_rel_sq::Vector{Float64}
    sum_Ssum::Vector{Float64}
    sum_separation_min::Vector{Float64}
    position_bin_edges::Vector{Float64}
    position_bin_centers::Vector{Float64}
    XA_histogram_counts::Vector{Float64}
    XB_histogram_counts::Vector{Float64}
    center_histogram_counts::Vector{Float64}
    sample_count::Int
    history_time::Vector{Float64}
    history_XA::Vector{Float64}
    history_XB::Vector{Float64}
    history_XA_unwrapped::Vector{Float64}
    history_XB_unwrapped::Vector{Float64}
    history_separation_oriented::Vector{Float64}
    history_separation_min::Vector{Float64}
    history_dXA::Vector{Float64}
    history_dXB::Vector{Float64}
    history_drel::Vector{Float64}
    history_SA::Vector{Float64}
    history_SB::Vector{Float64}
    history_Ssum::Vector{Float64}
    history_truncated::Bool
    max_abs_bath_active_step::Float64
    max_abs_bath_thermal_step::Float64
    max_abs_object_step::Float64
end

function MobileStats(param::CoupledSDEParam)
    max_dist = param.L / 2
    n_bins = max(param.n_bins, 1)
    edges = collect(range(0.0, max_dist; length=n_bins + 1))
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:n_bins]
    position_edges = collect(range(0.0, param.L; length=n_bins + 1))
    position_centers = [(position_edges[i] + position_edges[i + 1]) / 2 for i in 1:n_bins]
    return MobileStats(
        edges,
        centers,
        zeros(Float64, n_bins),
        zeros(Float64, n_bins),
        zeros(Float64, n_bins),
        zeros(Float64, n_bins),
        zeros(Float64, n_bins),
        position_edges,
        position_centers,
        zeros(Float64, n_bins),
        zeros(Float64, n_bins),
        zeros(Float64, n_bins),
        0,
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        false,
        0.0,
        0.0,
        0.0,
    )
end

function normalize_mode(raw_mode)
    mode = lowercase(strip(String(raw_mode)))
    if mode in ("fixed", "fixed_separation", "fixed-separation", "fixed_object", "fixed_objects")
        return FIXED_SEPARATION_MODE
    elseif mode in ("mobile", "mobile_objects", "mobile-objects", "moving", "moving_objects")
        return MOBILE_OBJECTS_MODE
    end
    throw(ArgumentError("Unsupported coupled-SDE mode: $(raw_mode). Use fixed_separation or mobile_objects."))
end

@inline wrap_position(x::Real, L::Real) = mod(Float64(x), Float64(L))

@inline function minimal_image(dx::Real, L::Real)
    Lf = Float64(L)
    return mod(Float64(dx) + 0.5 * Lf, Lf) - 0.5 * Lf
end

@inline oriented_separation(XA::Real, XB::Real, L::Real) = mod(Float64(XA) - Float64(XB), Float64(L))

@inline function minimum_separation(XA::Real, XB::Real, L::Real)
    d = oriented_separation(XA, XB, L)
    return min(d, Float64(L) - d)
end

@inline function profile_value(r::Real, param::CoupledSDEParam)
    sigma = param.sigma_f
    sigma > 0 || throw(ArgumentError("sigma_f must be positive. Got $(param.sigma_f)."))
    profile = lowercase(strip(param.profile_type))
    rr = Float64(r)
    if profile == "gaussian"
        return param.f0 * exp(-0.5 * (rr / sigma)^2)
    elseif profile in ("bump", "compact_bump", "compact")
        u = abs(rr) / sigma
        u >= 1.0 && return 0.0
        return param.f0 * exp(1.0 - 1.0 / (1.0 - u^2))
    end
    throw(ArgumentError("Unsupported profile_type=$(param.profile_type). Use gaussian or compact_bump."))
end

function default_object_positions(param::CoupledSDEParam, rng::AbstractRNG)
    if param.random_initial_objects
        max_sep = isnothing(param.initial_max_separation) ? param.L / 2 : Float64(param.initial_max_separation)
        min_sep = Float64(param.initial_min_separation)
        0.0 <= min_sep <= max_sep <= param.L / 2 ||
            throw(ArgumentError("Random initial separations require 0 <= initial_min_separation <= initial_max_separation <= L/2. Got min=$(min_sep), max=$(max_sep), L=$(param.L)."))
        sep_min = min_sep + (max_sep - min_sep) * rand(rng)
        oriented_sep = rand(rng) < 0.5 ? sep_min : param.L - sep_min
        XA_unwrapped = param.L * rand(rng)
        XB_unwrapped = XA_unwrapped - oriented_sep
        return wrap_position(XA_unwrapped, param.L), wrap_position(XB_unwrapped, param.L), XA_unwrapped, XB_unwrapped
    end

    if !isnothing(param.initial_XA) && !isnothing(param.initial_XB)
        XA = wrap_position(param.initial_XA, param.L)
        XB = wrap_position(param.initial_XB, param.L)
        return XA, XB, Float64(param.initial_XA), Float64(param.initial_XB)
    elseif !isnothing(param.initial_XA)
        XA_unwrapped = Float64(param.initial_XA)
        XB_unwrapped = XA_unwrapped + Float64(param.separation)
        return wrap_position(XA_unwrapped, param.L), wrap_position(XB_unwrapped, param.L), XA_unwrapped, XB_unwrapped
    elseif !isnothing(param.initial_XB)
        XB_unwrapped = Float64(param.initial_XB)
        XA_unwrapped = XB_unwrapped - Float64(param.separation)
        return wrap_position(XA_unwrapped, param.L), wrap_position(XB_unwrapped, param.L), XA_unwrapped, XB_unwrapped
    end

    sep = Float64(param.separation)
    0.0 <= sep <= param.L / 2 || throw(ArgumentError("separation must satisfy 0 <= separation <= L/2. Got separation=$(sep), L=$(param.L)."))
    center = 0.5 * param.L
    XA_unwrapped = center - 0.5 * sep
    XB_unwrapped = center + 0.5 * sep
    return wrap_position(XA_unwrapped, param.L), wrap_position(XB_unwrapped, param.L), XA_unwrapped, XB_unwrapped
end

function initialize_uniform_bath!(x::AbstractVector{Float64}, param::CoupledSDEParam, rng::AbstractRNG)
    for i in eachindex(x)
        x[i] = param.L * rand(rng)
    end
    return x
end

function initialize_state(param::CoupledSDEParam, rng::AbstractRNG)
    param.N >= 0 || throw(ArgumentError("N must be nonnegative. Got $(param.N)."))
    x = zeros(Float64, param.N)
    initialize_uniform_bath!(x, param, rng)
    XA, XB, XA_unwrapped, XB_unwrapped = default_object_positions(param, rng)
    return CoupledSDEState(0, 0.0, x, XA, XB, XA_unwrapped, XB_unwrapped, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function compute_profile_sums!(
    fA::AbstractVector{Float64},
    fB::AbstractVector{Float64},
    x::AbstractVector{Float64},
    XA::Real,
    XB::Real,
    param::CoupledSDEParam,
)
    length(fA) == length(x) || throw(ArgumentError("fA length does not match bath length."))
    length(fB) == length(x) || throw(ArgumentError("fB length does not match bath length."))
    SA = 0.0
    SB = 0.0
    @inbounds for i in eachindex(x)
        valA = profile_value(minimal_image(x[i] - XA, param.L), param)
        valB = profile_value(minimal_image(x[i] - XB, param.L), param)
        fA[i] = valA
        fB[i] = valB
        SA += valA
        SB += valB
    end
    return SA, SB
end

function draw_sde_noises!(dWi::AbstractVector{Float64}, rng::AbstractRNG, dt::Real)
    sqrt_dt = sqrt(Float64(dt))
    dWA = sqrt_dt * randn(rng)
    dWB = sqrt_dt * randn(rng)
    @inbounds for i in eachindex(dWi)
        dWi[i] = sqrt_dt * randn(rng)
    end
    return dWA, dWB, dWi
end

function draw_sde_noises(rng::AbstractRNG, N::Integer, dt::Real)
    dWi = zeros(Float64, Int(N))
    dWA, dWB, dWi = draw_sde_noises!(dWi, rng, dt)
    return dWA, dWB, dWi
end

function step_fixed_separation!(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    work::SDEWork,
    dWA::Real,
    dWB::Real,
    dWi::AbstractVector{Float64},
)
    length(dWi) == length(state.x) || throw(ArgumentError("Thermal-noise vector length does not match bath length."))

    old_step = state.step
    old_time = state.time
    old_XA = state.XA
    old_XB = state.XB
    sep_oriented = oriented_separation(old_XA, old_XB, param.L)
    sep_min = min(sep_oriented, param.L - sep_oriented)
    SA, SB = compute_profile_sums!(work.fA, work.fB, state.x, old_XA, old_XB, param)

    thermal_scale = sqrt(2.0 * param.D0)
    max_active = 0.0
    max_thermal = 0.0
    @inbounds for i in eachindex(state.x)
        active_step = param.mu_bath * (work.fA[i] * Float64(dWA) + work.fB[i] * Float64(dWB))
        thermal_step = thermal_scale * dWi[i]
        state.x[i] = wrap_position(state.x[i] + active_step + thermal_step, param.L)
        max_active = max(max_active, abs(active_step))
        max_thermal = max(max_thermal, abs(thermal_step))
    end

    state.step += 1
    state.time += param.dt
    state.last_SA = SA
    state.last_SB = SB
    state.last_dXA = 0.0
    state.last_dXB = 0.0
    state.last_drel = 0.0

    return StepObservables(
        old_step,
        old_time,
        old_XA,
        old_XB,
        sep_oriented,
        sep_min,
        SA,
        SB,
        SA^2 + SB^2,
        Float64(dWA),
        Float64(dWB),
        0.0,
        0.0,
        0.0,
        max_active,
        max_thermal,
        0.0,
    )
end

function step_fixed_separation!(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    rng::AbstractRNG,
    work::SDEWork,
)
    dWA, dWB, dWi = draw_sde_noises!(work.dWi, rng, param.dt)
    return step_fixed_separation!(state, param, work, dWA, dWB, dWi)
end

function step_mobile_objects!(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    work::SDEWork,
    dWA::Real,
    dWB::Real,
    dWi::AbstractVector{Float64},
)
    length(dWi) == length(state.x) || throw(ArgumentError("Thermal-noise vector length does not match bath length."))

    old_step = state.step
    old_time = state.time
    old_XA = state.XA
    old_XB = state.XB
    old_XA_unwrapped = state.XA_unwrapped
    old_XB_unwrapped = state.XB_unwrapped
    sep_oriented = oriented_separation(old_XA, old_XB, param.L)
    sep_min = min(sep_oriented, param.L - sep_oriented)
    SA, SB = compute_profile_sums!(work.fA, work.fB, state.x, old_XA, old_XB, param)

    thermal_scale = sqrt(2.0 * param.D0)
    max_active = 0.0
    max_thermal = 0.0
    @inbounds for i in eachindex(state.x)
        active_step = param.mu_bath * (work.fA[i] * Float64(dWA) + work.fB[i] * Float64(dWB))
        thermal_step = thermal_scale * dWi[i]
        state.x[i] = wrap_position(state.x[i] + active_step + thermal_step, param.L)
        max_active = max(max_active, abs(active_step))
        max_thermal = max(max_thermal, abs(thermal_step))
    end

    dXA = -param.mu_obj * SA * Float64(dWA)
    dXB = -param.mu_obj * SB * Float64(dWB)
    state.XA_unwrapped = old_XA_unwrapped + dXA
    state.XB_unwrapped = old_XB_unwrapped + dXB
    state.XA = wrap_position(state.XA_unwrapped, param.L)
    state.XB = wrap_position(state.XB_unwrapped, param.L)
    drel = dXA - dXB
    max_object = max(abs(dXA), abs(dXB))

    state.step += 1
    state.time += param.dt
    state.last_SA = SA
    state.last_SB = SB
    state.last_dXA = dXA
    state.last_dXB = dXB
    state.last_drel = drel

    return StepObservables(
        old_step,
        old_time,
        old_XA,
        old_XB,
        sep_oriented,
        sep_min,
        SA,
        SB,
        SA^2 + SB^2,
        Float64(dWA),
        Float64(dWB),
        dXA,
        dXB,
        drel,
        max_active,
        max_thermal,
        max_object,
    )
end

function step_mobile_objects!(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    rng::AbstractRNG,
    work::SDEWork,
)
    dWA, dWB, dWi = draw_sde_noises!(work.dWi, rng, param.dt)
    return step_mobile_objects!(state, param, work, dWA, dWB, dWi)
end

function accumulate!(stats::FixedStats, obs::StepObservables)
    stats.sample_count += 1
    stats.sum_SA += obs.SA
    stats.sum_SB += obs.SB
    stats.sum_SA2 += obs.SA^2
    stats.sum_SB2 += obs.SB^2
    stats.sum_SASB += obs.SA * obs.SB
    stats.sum_Ssum += obs.Ssum
    stats.sum_Ssum2 += obs.Ssum^2
    stats.max_abs_bath_active_step = max(stats.max_abs_bath_active_step, obs.max_abs_bath_active_step)
    stats.max_abs_bath_thermal_step = max(stats.max_abs_bath_thermal_step, obs.max_abs_bath_thermal_step)
    stats.max_abs_object_step = max(stats.max_abs_object_step, obs.max_abs_object_step)
    return stats
end

function distance_bin_index(stats::MobileStats, distance::Real)
    d = Float64(distance)
    d < stats.bin_edges[1] && return nothing
    d > stats.bin_edges[end] && return nothing
    d == stats.bin_edges[end] && return length(stats.bin_centers)
    width = stats.bin_edges[2] - stats.bin_edges[1]
    idx = floor(Int, (d - stats.bin_edges[1]) / width) + 1
    if 1 <= idx <= length(stats.bin_centers)
        return idx
    end
    return nothing
end

function position_bin_index(stats::MobileStats, position::Real, L::Real)
    x = wrap_position(position, L)
    x == stats.position_bin_edges[end] && return length(stats.position_bin_centers)
    width = stats.position_bin_edges[2] - stats.position_bin_edges[1]
    idx = floor(Int, (x - stats.position_bin_edges[1]) / width) + 1
    if 1 <= idx <= length(stats.position_bin_centers)
        return idx
    end
    return nothing
end

function maybe_record_history!(stats::MobileStats, obs::StepObservables, state::CoupledSDEState, param::CoupledSDEParam, production_step::Int)
    param.save_raw_history || return stats
    param.history_interval > 0 || return stats
    production_step % param.history_interval == 0 || return stats
    if length(stats.history_time) >= param.max_history_records
        stats.history_truncated = true
        return stats
    end

    push!(stats.history_time, obs.time)
    push!(stats.history_XA, obs.XA)
    push!(stats.history_XB, obs.XB)
    push!(stats.history_XA_unwrapped, state.XA_unwrapped - obs.dXA)
    push!(stats.history_XB_unwrapped, state.XB_unwrapped - obs.dXB)
    push!(stats.history_separation_oriented, obs.separation_oriented)
    push!(stats.history_separation_min, obs.separation_min)
    push!(stats.history_dXA, obs.dXA)
    push!(stats.history_dXB, obs.dXB)
    push!(stats.history_drel, obs.drel)
    push!(stats.history_SA, obs.SA)
    push!(stats.history_SB, obs.SB)
    push!(stats.history_Ssum, obs.Ssum)
    return stats
end

function accumulate!(stats::MobileStats, obs::StepObservables, state::CoupledSDEState, param::CoupledSDEParam, production_step::Int)
    stats.max_abs_bath_active_step = max(stats.max_abs_bath_active_step, obs.max_abs_bath_active_step)
    stats.max_abs_bath_thermal_step = max(stats.max_abs_bath_thermal_step, obs.max_abs_bath_thermal_step)
    stats.max_abs_object_step = max(stats.max_abs_object_step, obs.max_abs_object_step)

    if production_step % param.sample_interval == 0
        idx = distance_bin_index(stats, obs.separation_min)
        if !isnothing(idx)
            stats.histogram_counts[idx] += 1.0
            stats.bin_counts[idx] += 1.0
            stats.sum_delta_rel_sq[idx] += obs.drel^2
            stats.sum_Ssum[idx] += obs.Ssum
            stats.sum_separation_min[idx] += obs.separation_min
            stats.sample_count += 1
        end

        idxA = position_bin_index(stats, obs.XA, param.L)
        idxB = position_bin_index(stats, obs.XB, param.L)
        center = wrap_position(0.5 * (state.XA_unwrapped + state.XB_unwrapped), param.L)
        idxC = position_bin_index(stats, center, param.L)
        isnothing(idxA) || (stats.XA_histogram_counts[idxA] += 1.0)
        isnothing(idxB) || (stats.XB_histogram_counts[idxB] += 1.0)
        isnothing(idxC) || (stats.center_histogram_counts[idxC] += 1.0)
    end

    maybe_record_history!(stats, obs, state, param, production_step)
    return stats
end

function run_fixed_separation!(state::CoupledSDEState, param::CoupledSDEParam, rng::AbstractRNG)
    work = SDEWork(param.N)
    for _ in 1:max(param.warmup_steps, 0)
        step_fixed_separation!(state, param, rng, work)
    end

    stats = FixedStats()
    sample_interval = max(param.sample_interval, 1)
    for production_step in 1:max(param.n_steps, 0)
        obs = step_fixed_separation!(state, param, rng, work)
        if production_step % sample_interval == 0
            accumulate!(stats, obs)
        else
            stats.max_abs_bath_active_step = max(stats.max_abs_bath_active_step, obs.max_abs_bath_active_step)
            stats.max_abs_bath_thermal_step = max(stats.max_abs_bath_thermal_step, obs.max_abs_bath_thermal_step)
        end
    end
    return stats
end

function run_mobile_objects!(state::CoupledSDEState, param::CoupledSDEParam, rng::AbstractRNG)
    work = SDEWork(param.N)
    for _ in 1:max(param.warmup_steps, 0)
        step_mobile_objects!(state, param, rng, work)
    end

    stats = MobileStats(param)
    for production_step in 1:max(param.n_steps, 0)
        obs = step_mobile_objects!(state, param, rng, work)
        accumulate!(stats, obs, state, param, production_step)
    end
    return stats
end

function parameters_dict(param::CoupledSDEParam)
    return Dict(
        "mode" => param.mode,
        "L" => param.L,
        "rho0" => param.rho0,
        "N" => param.N,
        "D0" => param.D0,
        "dt" => param.dt,
        "mu_bath" => param.mu_bath,
        "mu_obj" => param.mu_obj,
        "f0" => param.f0,
        "sigma_f" => param.sigma_f,
        "profile_type" => param.profile_type,
        "separation" => param.separation,
        "initial_XA" => param.initial_XA,
        "initial_XB" => param.initial_XB,
        "random_initial_objects" => param.random_initial_objects,
        "initial_min_separation" => param.initial_min_separation,
        "initial_max_separation" => param.initial_max_separation,
        "n_steps" => param.n_steps,
        "warmup_steps" => param.warmup_steps,
        "sample_interval" => param.sample_interval,
        "history_interval" => param.history_interval,
        "n_bins" => param.n_bins,
        "max_history_records" => param.max_history_records,
        "save_raw_history" => param.save_raw_history,
        "seed" => param.seed,
        "description" => param.description,
    )
end

function final_state_dict(state::CoupledSDEState)
    return Dict(
        "step" => state.step,
        "time" => state.time,
        "XA" => state.XA,
        "XB" => state.XB,
        "XA_unwrapped" => state.XA_unwrapped,
        "XB_unwrapped" => state.XB_unwrapped,
        "last_SA" => state.last_SA,
        "last_SB" => state.last_SB,
        "last_dXA" => state.last_dXA,
        "last_dXB" => state.last_dXB,
        "last_drel" => state.last_drel,
    )
end

function stability_recommendations(param::CoupledSDEParam, stats)
    thermal_rms = sqrt(2.0 * param.D0 * param.dt)
    bath_active_rms = abs(param.mu_bath * param.f0) * sqrt(param.dt)
    object_max = hasproperty(stats, :max_abs_object_step) ? getproperty(stats, :max_abs_object_step) : 0.0
    bath_active_max = hasproperty(stats, :max_abs_bath_active_step) ? getproperty(stats, :max_abs_bath_active_step) : 0.0
    bath_thermal_max = hasproperty(stats, :max_abs_bath_thermal_step) ? getproperty(stats, :max_abs_bath_thermal_step) : 0.0
    sigma = param.sigma_f
    return Dict(
        "thermal_rms_over_sigma_f" => thermal_rms / sigma,
        "bath_active_single_profile_rms_over_sigma_f" => bath_active_rms / sigma,
        "max_abs_bath_active_step_over_sigma_f" => bath_active_max / sigma,
        "max_abs_bath_thermal_step_over_sigma_f" => bath_thermal_max / sigma,
        "max_abs_object_step_over_sigma_f" => object_max / sigma,
        "thermal_rms" => thermal_rms,
        "bath_active_single_profile_rms" => bath_active_rms,
        "max_abs_bath_active_step" => bath_active_max,
        "max_abs_bath_thermal_step" => bath_thermal_max,
        "max_abs_object_step" => object_max,
    )
end

function fixed_result_dict(param::CoupledSDEParam, state::CoupledSDEState, stats::FixedStats)
    count = stats.sample_count
    mean_or_nan(x) = count > 0 ? x / count : NaN
    mean_Ssum = mean_or_nan(stats.sum_Ssum)
    D_proxy = 0.5 * param.mu_obj^2 * mean_Ssum
    sep_oriented = oriented_separation(state.XA, state.XB, param.L)
    sep_min = min(sep_oriented, param.L - sep_oriented)
    return Dict(
        "result_type" => "coupled_sde_fixed_separation",
        "mode" => FIXED_SEPARATION_MODE,
        "parameters" => parameters_dict(param),
        "final_state" => final_state_dict(state),
        "separation_oriented" => sep_oriented,
        "separation_min" => sep_min,
        "sample_count" => count,
        "sums" => Dict(
            "SA" => stats.sum_SA,
            "SB" => stats.sum_SB,
            "SA2" => stats.sum_SA2,
            "SB2" => stats.sum_SB2,
            "SASB" => stats.sum_SASB,
            "Ssum" => stats.sum_Ssum,
            "Ssum2" => stats.sum_Ssum2,
        ),
        "means" => Dict(
            "SA" => mean_or_nan(stats.sum_SA),
            "SB" => mean_or_nan(stats.sum_SB),
            "SA2" => mean_or_nan(stats.sum_SA2),
            "SB2" => mean_or_nan(stats.sum_SB2),
            "SASB" => mean_or_nan(stats.sum_SASB),
            "Ssum" => mean_Ssum,
        ),
        "D_rel_proxy" => D_proxy,
        "stability" => stability_recommendations(param, stats),
    )
end

function vector_divide_or_nan(numer::AbstractVector{Float64}, denom::AbstractVector{Float64})
    out = similar(numer)
    @inbounds for i in eachindex(numer)
        out[i] = denom[i] > 0 ? numer[i] / denom[i] : NaN
    end
    return out
end

function mobile_result_dict(param::CoupledSDEParam, state::CoupledSDEState, stats::MobileStats)
    bin_counts = stats.bin_counts
    mean_delta_rel_sq = vector_divide_or_nan(stats.sum_delta_rel_sq, bin_counts)
    mean_Ssum = vector_divide_or_nan(stats.sum_Ssum, bin_counts)
    D_traj = mean_delta_rel_sq ./ (2.0 * param.dt)
    D_proxy = 0.5 * param.mu_obj^2 .* mean_Ssum
    total_hist = sum(stats.histogram_counts)
    bin_widths = diff(stats.bin_edges)
    P_mass = total_hist > 0 ? stats.histogram_counts ./ total_hist : fill(NaN, length(stats.histogram_counts))
    P_density = similar(P_mass)
    @inbounds for i in eachindex(P_mass)
        P_density[i] = isfinite(P_mass[i]) ? P_mass[i] / bin_widths[i] : NaN
    end
    position_widths = diff(stats.position_bin_edges)
    XA_total = sum(stats.XA_histogram_counts)
    XB_total = sum(stats.XB_histogram_counts)
    center_total = sum(stats.center_histogram_counts)
    XA_mass = XA_total > 0 ? stats.XA_histogram_counts ./ XA_total : fill(NaN, length(stats.XA_histogram_counts))
    XB_mass = XB_total > 0 ? stats.XB_histogram_counts ./ XB_total : fill(NaN, length(stats.XB_histogram_counts))
    center_mass = center_total > 0 ? stats.center_histogram_counts ./ center_total : fill(NaN, length(stats.center_histogram_counts))
    XA_density = XA_mass ./ position_widths
    XB_density = XB_mass ./ position_widths
    center_density = center_mass ./ position_widths

    return Dict(
        "result_type" => "coupled_sde_mobile_objects",
        "mode" => MOBILE_OBJECTS_MODE,
        "parameters" => parameters_dict(param),
        "final_state" => final_state_dict(state),
        "sample_count" => stats.sample_count,
        "bins" => Dict(
            "edges" => stats.bin_edges,
            "centers" => stats.bin_centers,
            "histogram_counts" => stats.histogram_counts,
            "bin_counts" => stats.bin_counts,
            "sum_delta_rel_sq" => stats.sum_delta_rel_sq,
            "sum_Ssum" => stats.sum_Ssum,
            "sum_separation_min" => stats.sum_separation_min,
            "mean_delta_rel_sq" => mean_delta_rel_sq,
            "mean_Ssum" => mean_Ssum,
            "D_rel_traj" => D_traj,
            "D_rel_proxy" => D_proxy,
            "P_mass" => P_mass,
            "P_density" => P_density,
        ),
        "locations" => Dict(
            "edges" => stats.position_bin_edges,
            "centers" => stats.position_bin_centers,
            "XA_histogram_counts" => stats.XA_histogram_counts,
            "XB_histogram_counts" => stats.XB_histogram_counts,
            "center_histogram_counts" => stats.center_histogram_counts,
            "XA_P_mass" => XA_mass,
            "XB_P_mass" => XB_mass,
            "center_P_mass" => center_mass,
            "XA_P_density" => XA_density,
            "XB_P_density" => XB_density,
            "center_P_density" => center_density,
        ),
        "history" => Dict(
            "time" => stats.history_time,
            "XA" => stats.history_XA,
            "XB" => stats.history_XB,
            "XA_unwrapped" => stats.history_XA_unwrapped,
            "XB_unwrapped" => stats.history_XB_unwrapped,
            "separation_oriented" => stats.history_separation_oriented,
            "separation_min" => stats.history_separation_min,
            "dXA" => stats.history_dXA,
            "dXB" => stats.history_dXB,
            "drel" => stats.history_drel,
            "SA" => stats.history_SA,
            "SB" => stats.history_SB,
            "Ssum" => stats.history_Ssum,
            "truncated" => stats.history_truncated,
        ),
        "stability" => stability_recommendations(param, stats),
    )
end

end
