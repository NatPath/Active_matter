#!/usr/bin/env julia

using ArgParse
using Dates
using JLD2
using Printf
using Random

const REPO_ROOT = @__DIR__
include(joinpath(REPO_ROOT, "src", "radial_diffusion_sde", "modules_radial_diffusion_sde.jl"))
using .RadialDiffusionSDE

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--L"
            help = "Finite system size; the domain is [-L/2, L/2]"
            arg_type = Float64
            default = 16.0
        "--boundary"
            help = "Boundary condition: reflecting or periodic"
            arg_type = String
            default = "periodic"
        "--D_inf"
            help = "Far-field diffusion coefficient"
            arg_type = Float64
            default = 1.0
        "--A"
            help = "Amplitude in D(x) = D_inf * (1 + A / (x^2 + core_radius^2))"
            arg_type = Float64
            default = 1.0
        "--core_radius"
            help = "Positive regularization radius for the singularity at x=0"
            arg_type = Float64
            default = 0.5
        "--interpretation"
            help = "Stochastic model: ito or fickian"
            arg_type = String
            default = "ito"
        "--x0"
            help = "Initial location; a scalar in the default 1D model"
            arg_type = String
            default = "1"
        "--initial_distribution"
            help = "Initial walker distribution: point, uniform, or stationary"
            arg_type = String
            default = "point"
        "--sample_initial"
            help = "Include the initial walker distribution in the accumulated histogram"
            action = :store_true
        "--dt"
            help = "Euler-Maruyama time step"
            arg_type = Float64
            default = 0.01
        "--steps"
            help = "Number of time steps; when using --continue, number of additional steps"
            arg_type = Int
            default = 5_000
        "--walkers"
            help = "Independent walkers used to estimate the current-time distribution"
            arg_type = Int
            default = 100_000
        "--bins"
            help = "Histogram bins per spatial direction"
            arg_type = Int
            default = 100
        "--sample_start_step"
            help = "First step eligible for accumulated steady-state histogram sampling"
            arg_type = Int
            default = 0
        "--sample_interval"
            help = "Steps between accumulated steady-state histogram samples"
            arg_type = Int
            default = 100
        "--tail_min"
            help = "Lower distance for the power-law diagnostic; default is 3*sqrt(core_radius^2+A)"
            arg_type = Float64
        "--tail_max"
            help = "Upper distance for the power-law diagnostic; default is L/4"
            arg_type = Float64
        "--tail_log_bins"
            help = "Approximately logarithmic bins used for tail plotting and fitting"
            arg_type = Int
            default = 16
        "--plot_interval_steps"
            help = "Refresh live_latest.png every this many steps"
            arg_type = Int
            default = 10000
        "--seed"
            help = "Random seed; zero chooses a random seed"
            arg_type = Int
            default = 12345
        "--continue"
            help = "Continue from a checkpoint_latest.jld2 or checkpoint_final.jld2 file; --steps is additional steps"
            arg_type = String
        "--checkpoint_interval_steps"
            help = "Write checkpoint_latest.jld2 every this many steps"
            arg_type = Int
            default = 100000
        "--no_checkpoint"
            help = "Disable periodic and final checkpoint writing"
            action = :store_true
        "--output_dir"
            help = "Output directory; default is a timestamped directory under analysis_outputs"
            arg_type = String
        "--display"
            help = "Also display the updating plot in a graphical window"
            action = :store_true
        "--no_plot"
            help = "Skip plot generation and write only final CSV/text outputs"
            action = :store_true
    end
    return parse_args(settings)
end

const CLI_ARGS = parse_commandline()

if !Bool(CLI_ARGS["display"]) && get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

if !Bool(CLI_ARGS["no_plot"])
    using Plots
end

function parse_interpretation(raw::AbstractString)
    value = Symbol(lowercase(strip(raw)))
    value in (:ito, :fickian) || error("--interpretation must be ito or fickian.")
    return value
end

function parse_boundary(raw::AbstractString)
    value = Symbol(lowercase(strip(raw)))
    value in (:reflecting, :periodic) || error("--boundary must be reflecting or periodic.")
    return value
end

function parse_initial_distribution(raw::AbstractString)
    value = Symbol(lowercase(strip(raw)))
    value in (:point, :uniform, :stationary) ||
        error("--initial_distribution must be point, uniform, or stationary.")
    return value
end

function parse_x0(raw::AbstractString, dims::Integer)
    values = [parse(Float64, strip(token)) for token in split(raw, ",") if !isempty(strip(token))]
    if dims == 1 && length(values) == 2 && values[2] == 0.0
        values = values[1:1]
    end
    length(values) == dims || error("--x0 must contain exactly $dims comma-separated value(s).")
    all(isfinite, values) || error("--x0 values must be finite.")
    return values
end

function build_param(args)
    param = RadialDiffusionParam(
        dims=1,
        D_inf=Float64(args["D_inf"]),
        A=Float64(args["A"]),
        core_radius=Float64(args["core_radius"]),
        L=Float64(args["L"]),
        dt=Float64(args["dt"]),
        interpretation=parse_interpretation(String(args["interpretation"])),
        boundary=parse_boundary(String(args["boundary"])),
    )
    return validate_param(param)
end

function default_output_dir(param::RadialDiffusionParam)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    return joinpath(
        REPO_ROOT,
        "analysis_outputs",
        "radial_diffusion_sde",
        "1d_$(param.boundary)_$(param.interpretation)_$(stamp)",
    )
end

function validate_args(args)
    Int(args["steps"]) >= 0 || error("--steps must be nonnegative.")
    Int(args["walkers"]) > 0 || error("--walkers must be positive.")
    Int(args["bins"]) >= 4 || error("--bins must be at least 4.")
    Int(args["sample_start_step"]) >= 0 || error("--sample_start_step must be nonnegative.")
    Int(args["sample_interval"]) > 0 || error("--sample_interval must be positive.")
    isnothing(args["tail_min"]) || Float64(args["tail_min"]) > 0.0 ||
        error("--tail_min must be positive.")
    isnothing(args["tail_max"]) || Float64(args["tail_max"]) > 0.0 ||
        error("--tail_max must be positive.")
    Int(args["tail_log_bins"]) >= 2 || error("--tail_log_bins must be at least 2.")
    Int(args["plot_interval_steps"]) > 0 || error("--plot_interval_steps must be positive.")
    Int(args["checkpoint_interval_steps"]) > 0 || error("--checkpoint_interval_steps must be positive.")
    return args
end

function initialize_simulation_state(
    param::RadialDiffusionParam,
    n_walkers::Integer,
    x0::AbstractVector{<:Real},
    rng::AbstractRNG,
    initial_distribution::Symbol,
)
    state = initialize_state(param, n_walkers, x0)
    if initial_distribution === :uniform
        half_L = 0.5 * param.L
        rand!(rng, state.positions)
        @. state.positions = param.L * state.positions - half_L
    elseif initial_distribution === :stationary
        initialize_stationary_positions!(state, param, rng)
    end
    return state
end

@inline function stationary_shape(x::Real, param::RadialDiffusionParam)
    param.interpretation === :fickian && return 1.0
    x2 = Float64(x)^2
    return (x2 + param.core_radius^2) / (x2 + param.core_radius^2 + param.A)
end

function initialize_stationary_positions!(
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    rng::AbstractRNG,
)
    half_L = 0.5 * param.L
    max_shape = max(stationary_shape(0.0, param), stationary_shape(half_L, param))
    @inbounds for walker in axes(state.positions, 2)
        while true
            x = param.L * rand(rng) - half_L
            if max_shape * rand(rng) <= stationary_shape(x, param)
                state.positions[1, walker] = x
                break
            end
        end
    end
    return state
end

mutable struct DistributionAccumulator
    counts::Vector{Int64}
    absolute_counts::Vector{Int64}
    sample_count::Int
end

DistributionAccumulator(bins::Integer) = DistributionAccumulator(
    zeros(Int64, Int(bins)),
    zeros(Int64, max(fld(Int(bins), 2), 2)),
    0,
)

function accumulate_distribution!(
    accumulator::DistributionAccumulator,
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
)
    bins = length(accumulator.counts)
    absolute_bins = length(accumulator.absolute_counts)
    inv_dx = bins / param.L
    inv_dd = absolute_bins / (0.5 * param.L)
    half_L = 0.5 * param.L
    @inbounds for x in @view state.positions[1, :]
        x_idx = clamp(floor(Int, (x + half_L) * inv_dx) + 1, 1, bins)
        d_idx = clamp(floor(Int, abs(x) * inv_dd) + 1, 1, absolute_bins)
        accumulator.counts[x_idx] += 1
        accumulator.absolute_counts[d_idx] += 1
    end
    accumulator.sample_count += 1
    return accumulator
end

function tail_window(args, param::RadialDiffusionParam)
    crossover = sqrt(param.core_radius^2 + param.A)
    tail_min = isnothing(args["tail_min"]) ? 3.0 * crossover : Float64(args["tail_min"])
    tail_max = isnothing(args["tail_max"]) ? 0.25 * param.L : Float64(args["tail_max"])
    tail_min < tail_max ||
        error("Power-law window is empty: tail_min=$tail_min must be below tail_max=$tail_max.")
    tail_max <= 0.5 * param.L ||
        error("--tail_max must not exceed L/2.")
    return tail_min, tail_max
end

function rng_from_seed(seed::Integer)
    actual_seed = seed == 0 ? rand(1:typemax(Int32)) : Int(seed)
    return MersenneTwister(actual_seed), actual_seed
end

function save_checkpoint(
    path::AbstractString;
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    rng::AbstractRNG,
    accumulator::DistributionAccumulator,
    actual_seed::Integer,
    initial_distribution::Symbol,
    x0::AbstractVector{<:Real},
    sample_start_step::Integer,
    sample_interval::Integer,
)
    mkpath(dirname(path))
    tmp_path = string(path, ".tmp")
    jldsave(
        tmp_path;
        schema_version=2,
        param=param,
        positions=Matrix{Float64}(state.positions),
        step=Int(state.step),
        rng=rng,
        accumulator_counts=Vector{Int64}(accumulator.counts),
        accumulator_absolute_counts=Vector{Int64}(accumulator.absolute_counts),
        accumulator_sample_count=Int(accumulator.sample_count),
        actual_seed=Int(actual_seed),
        initial_distribution=String(initial_distribution),
        x0=Vector{Float64}(x0),
        sample_start_step=Int(sample_start_step),
        sample_interval=Int(sample_interval),
    )
    mv(tmp_path, path; force=true)
    return path
end

function load_checkpoint(path::AbstractString)
    isfile(path) || error("Checkpoint not found: $path")
    data = load(path)
    schema_version = Int(data["schema_version"])
    schema_version in (1, 2) ||
        error("Unsupported checkpoint schema version: $schema_version")
    param = data["param"]
    validate_param(param)
    positions = Matrix{Float64}(data["positions"])
    state = RadialDiffusionState(positions, similar(positions), Int(data["step"]))
    rng = data["rng"]
    rng isa AbstractRNG || error("Checkpoint RNG is not an AbstractRNG.")

    counts = Vector{Int64}(data["accumulator_counts"])
    absolute_counts = Vector{Int64}(data["accumulator_absolute_counts"])
    accumulator = DistributionAccumulator(length(counts))
    length(accumulator.absolute_counts) == length(absolute_counts) ||
        error("Checkpoint histogram shapes are incompatible.")
    accumulator.counts .= counts
    accumulator.absolute_counts .= absolute_counts
    accumulator.sample_count = Int(data["accumulator_sample_count"])

    actual_seed = Int(data["actual_seed"])
    initial_distribution = Symbol(String(data["initial_distribution"]))
    x0 = Vector{Float64}(data["x0"])

    return (
        param=param,
        state=state,
        rng=rng,
        accumulator=accumulator,
        actual_seed=actual_seed,
        initial_distribution=initial_distribution,
        x0=x0,
        sample_start_step=Int(data["sample_start_step"]),
        sample_interval=Int(data["sample_interval"]),
    )
end

function distribution_1d(state::RadialDiffusionState, radius::Real, bins::Integer)
    counts = zeros(Int, bins)
    inv_dx = bins / (2.0 * radius)
    outside = 0
    @inbounds for x in @view state.positions[1, :]
        if -radius <= x < radius
            counts[clamp(floor(Int, (x + radius) * inv_dx) + 1, 1, bins)] += 1
        else
            outside += 1
        end
    end
    dx = 2.0 * radius / bins
    centers = collect(range(-radius + 0.5 * dx, radius - 0.5 * dx; length=bins))
    density = counts ./ (size(state.positions, 2) * dx)
    return centers, density, counts, outside
end

function accumulated_distribution_1d(
    accumulator::DistributionAccumulator,
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
)
    accumulator.sample_count > 0 ||
        return distribution_1d(state, 0.5 * param.L, length(accumulator.counts))
    bins = length(accumulator.counts)
    dx = param.L / bins
    centers = collect(range(-0.5 * param.L + 0.5 * dx, 0.5 * param.L - 0.5 * dx; length=bins))
    normalization = accumulator.sample_count * size(state.positions, 2) * dx
    density = accumulator.counts ./ normalization
    return centers, density, accumulator.counts, 0
end

function distribution_2d(state::RadialDiffusionState, radius::Real, bins::Integer)
    counts = zeros(Int, bins, bins)
    inv_dx = bins / (2.0 * radius)
    outside = 0
    @inbounds for walker in axes(state.positions, 2)
        x = state.positions[1, walker]
        y = state.positions[2, walker]
        if -radius <= x < radius && -radius <= y < radius
            ix = clamp(floor(Int, (x + radius) * inv_dx) + 1, 1, bins)
            iy = clamp(floor(Int, (y + radius) * inv_dx) + 1, 1, bins)
            counts[ix, iy] += 1
        else
            outside += 1
        end
    end
    dx = 2.0 * radius / bins
    centers = collect(range(-radius + 0.5 * dx, radius - 0.5 * dx; length=bins))
    density = counts ./ (size(state.positions, 2) * dx * dx)
    return centers, density, counts, outside
end

function radial_distribution(state::RadialDiffusionState, radius::Real, bins::Integer)
    counts = zeros(Int, bins)
    inv_dr = bins / radius
    outside = 0
    @inbounds for walker in axes(state.positions, 2)
        r2 = 0.0
        for dim in axes(state.positions, 1)
            x = state.positions[dim, walker]
            r2 += x * x
        end
        r = sqrt(r2)
        if r < radius
            counts[clamp(floor(Int, r * inv_dr) + 1, 1, bins)] += 1
        else
            outside += 1
        end
    end
    dr = radius / bins
    centers = collect(range(0.5 * dr, radius - 0.5 * dr; length=bins))
    density = counts ./ (size(state.positions, 2) * dr)
    return centers, density, counts, outside
end

function symmetric_location_distribution(
    state::RadialDiffusionState,
    radius::Real,
    bins::Integer,
)
    n_distance_bins = max(fld(Int(bins), 2), 2)
    counts = zeros(Int, n_distance_bins)
    inv_dd = n_distance_bins / radius
    @inbounds for x in @view state.positions[1, :]
        distance = abs(x)
        idx = clamp(floor(Int, distance * inv_dd) + 1, 1, n_distance_bins)
        counts[idx] += 1
    end
    dd = radius / n_distance_bins
    distances = collect(range(0.5 * dd, radius - 0.5 * dd; length=n_distance_bins))
    density = counts ./ (2.0 * size(state.positions, 2) * dd)
    return distances, density, counts
end

function accumulated_symmetric_location_distribution(
    accumulator::DistributionAccumulator,
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
)
    accumulator.sample_count > 0 ||
        return symmetric_location_distribution(state, 0.5 * param.L, length(accumulator.counts))
    bins = length(accumulator.absolute_counts)
    dd = 0.5 * param.L / bins
    distances = collect(range(0.5 * dd, 0.5 * param.L - 0.5 * dd; length=bins))
    normalization = 2.0 * accumulator.sample_count * size(state.positions, 2) * dd
    density = accumulator.absolute_counts ./ normalization
    return distances, density, accumulator.absolute_counts
end

function log_binned_tail_distribution(
    accumulator::DistributionAccumulator,
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    tail_min::Real,
    tail_max::Real,
    requested_bins::Integer,
)
    radius = 0.5 * param.L
    if accumulator.sample_count > 0
        raw_counts = accumulator.absolute_counts
        snapshots = accumulator.sample_count
    else
        _, _, raw_counts = symmetric_location_distribution(
            state,
            radius,
            length(accumulator.counts),
        )
        snapshots = 1
    end

    raw_bins = length(raw_counts)
    dd = radius / raw_bins
    lower_index = clamp(ceil(Int, Float64(tail_min) / dd), 1, raw_bins - 1)
    upper_index = clamp(floor(Int, Float64(tail_max) / dd), lower_index + 1, raw_bins)
    target_edges = exp.(range(
        log(lower_index * dd),
        log(upper_index * dd);
        length=Int(requested_bins) + 1,
    ))
    edge_indices = sort(unique(clamp.(round.(Int, target_edges ./ dd), lower_index, upper_index)))
    first(edge_indices) == lower_index || pushfirst!(edge_indices, lower_index)
    last(edge_indices) == upper_index || push!(edge_indices, upper_index)

    n_bins = length(edge_indices) - 1
    lower_edges = zeros(Float64, n_bins)
    upper_edges = zeros(Float64, n_bins)
    distances = zeros(Float64, n_bins)
    density = zeros(Float64, n_bins)
    counts = zeros(Int64, n_bins)
    density_stderr = zeros(Float64, n_bins)
    exact_density = zeros(Float64, n_bins)
    exact_deficit = zeros(Float64, n_bins)
    plateau = steady_state_plateau(param)
    b = sqrt(param.core_radius^2 + param.A)
    observations = snapshots * size(state.positions, 2)

    @inbounds for bin in 1:n_bins
        lower_index_bin = edge_indices[bin]
        upper_index_bin = edge_indices[bin + 1]
        lower = lower_index_bin * dd
        upper = upper_index_bin * dd
        width = upper - lower
        count = sum(@view raw_counts[(lower_index_bin + 1):upper_index_bin])
        normalization = 2.0 * observations * width
        averaged_exact_deficit = param.interpretation === :fickian ? 0.0 :
            plateau * param.A * (atan(upper / b) - atan(lower / b)) / (b * width)

        lower_edges[bin] = lower
        upper_edges[bin] = upper
        distances[bin] = sqrt(lower * upper)
        counts[bin] = count
        density[bin] = count / normalization
        density_stderr[bin] = sqrt(count) / normalization
        exact_deficit[bin] = averaged_exact_deficit
        exact_density[bin] = plateau - averaged_exact_deficit
    end

    return (
        lower_edges=lower_edges,
        upper_edges=upper_edges,
        distances=distances,
        density=density,
        counts=counts,
        density_stderr=density_stderr,
        exact_density=exact_density,
        exact_deficit=exact_deficit,
    )
end

function fit_minus2_amplitude(
    distances::AbstractVector{<:Real},
    deficits::AbstractVector{<:Real},
    tail_min::Real,
    tail_max::Real,
)
    sum_log_amplitude = 0.0
    sum_log_deficit = 0.0
    sum_squared_log_residual = 0.0
    sum_squared_log_total = 0.0
    count = 0
    @inbounds for i in eachindex(distances, deficits)
        d = Float64(distances[i])
        deficit = Float64(deficits[i])
        if tail_min <= d <= tail_max && deficit > 0.0 && isfinite(deficit)
            log_deficit = log(deficit)
            sum_log_amplitude += log_deficit + 2.0 * log(d)
            sum_log_deficit += log_deficit
            count += 1
        end
    end
    count > 0 || return NaN, 0, NaN, NaN

    log_amplitude = sum_log_amplitude / count
    mean_log_deficit = sum_log_deficit / count
    @inbounds for i in eachindex(distances, deficits)
        d = Float64(distances[i])
        deficit = Float64(deficits[i])
        if tail_min <= d <= tail_max && deficit > 0.0 && isfinite(deficit)
            log_deficit = log(deficit)
            residual = log_deficit - (log_amplitude - 2.0 * log(d))
            sum_squared_log_residual += residual * residual
            centered = log_deficit - mean_log_deficit
            sum_squared_log_total += centered * centered
        end
    end
    log_r_squared = sum_squared_log_total > 0.0 ?
        1.0 - sum_squared_log_residual / sum_squared_log_total :
        NaN
    return exp(log_amplitude), count, sqrt(sum_squared_log_residual / count), log_r_squared
end

function ensemble_stats(state::RadialDiffusionState, radius::Real)
    sum_r = 0.0
    sum_r2 = 0.0
    max_r = 0.0
    outside = 0
    @inbounds for walker in axes(state.positions, 2)
        r2 = 0.0
        for dim in axes(state.positions, 1)
            x = state.positions[dim, walker]
            r2 += x * x
        end
        r = sqrt(r2)
        sum_r += r
        sum_r2 += r2
        max_r = max(max_r, r)
        outside += r >= radius
    end
    n = size(state.positions, 2)
    return (mean_r=sum_r / n, mean_r2=sum_r2 / n, max_r=max_r, outside_fraction=outside / n)
end

function tracked_radius(state::RadialDiffusionState)
    r2 = 0.0
    @inbounds for dim in axes(state.positions, 1)
        x = state.positions[dim, 1]
        r2 += x * x
    end
    return sqrt(r2)
end

function diffusivity_curve(param::RadialDiffusionParam, radius::Real)
    rs = collect(range(0.0, radius; length=500))
    values = [diffusivity(r * r, param) for r in rs]
    return rs, values
end

function make_live_plot(
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    args,
    accumulator::DistributionAccumulator,
)
    radius = 0.5 * param.L
    bins = length(accumulator.counts)
    t = state.step * param.dt
    n_walkers = size(state.positions, 2)
    stats = ensemble_stats(state, radius)
    tail_min, tail_max = tail_window(args, param)
    tail = log_binned_tail_distribution(
        accumulator,
        state,
        param,
        tail_min,
        tail_max,
        Int(args["tail_log_bins"]),
    )
    plateau = steady_state_plateau(param)
    empirical_deficit = plateau .- tail.density
    fitted_amplitude, fit_points, fit_log_rmse, fit_log_r_squared = fit_minus2_amplitude(
        tail.distances,
        empirical_deficit,
        tail_min,
        tail_max,
    )
    fixed_fit_distances = exp.(range(log(tail_min), log(tail_max); length=200))
    fixed_fit = fitted_amplitude ./ fixed_fit_distances .^ 2
    tail_mask = [
        tail_min <= tail.distances[i] <= tail_max &&
        empirical_deficit[i] > 0.0 &&
        tail.exact_deficit[i] > 0.0
        for i in eachindex(tail.distances)
    ]
    tail_edge_min = first(tail.lower_edges)
    tail_edge_max = last(tail.upper_edges)
    tail_point_min = first(tail.distances)
    tail_point_max = last(tail.distances)
    tail_midpoint = sqrt(tail_edge_min * tail_edge_max)
    tail_ticks = (
        [tail_edge_min, tail_midpoint, tail_edge_max],
        [
            @sprintf("%.4g", tail_edge_min),
            @sprintf("%.4g", tail_midpoint),
            @sprintf("%.4g", tail_edge_max),
        ],
    )
    current_r = tracked_radius(state)
    rs, diffusion_values = diffusivity_curve(param, radius)

    if param.interpretation === :ito && param.A > 0.0
        p_radial = plot(
            tail.distances[tail_mask],
            empirical_deficit[tail_mask];
            xlabel="d = |x|",
            ylabel="P_inf - P_ss(d)",
            label=@sprintf(
                "accumulated; centers %.4g to %.4g",
                tail_point_min,
                tail_point_max,
            ),
            linewidth=2,
            marker=:circle,
            xscale=:log10,
            yscale=:log10,
            xlims=(tail_min, tail_max),
            xticks=tail_ticks,
            title=@sprintf(
                "Fixed -2 tail fit: edges %.4g to %.4g (%.3f decade)",
                tail_edge_min,
                tail_edge_max,
                log10(tail_edge_max / tail_edge_min),
            ),
        )
        plot!(
            p_radial,
            fixed_fit_distances,
            fixed_fit;
            label=@sprintf(
                "fixed d^-2: C=%.4g, log R²=%.5f (%d points)",
                fitted_amplitude,
                fit_log_r_squared,
                fit_points,
            ),
            linewidth=2.5,
            linestyle=:dot,
        )
        plot!(
            p_radial,
            tail.distances,
            tail.exact_deficit;
            label="exact finite-bin average",
            linewidth=2,
            linestyle=:dash,
        )
    else
        p_radial = plot(
            tail.distances,
            empirical_deficit;
            xlabel="d = |x|",
            ylabel="P_inf - P_ss(d)",
            label="accumulated",
            linewidth=2,
            title="No positive d^-2 deficit for this model",
        )
    end

    p_diffusion = plot(
        rs,
        diffusion_values;
        xlabel="|x|",
        ylabel="D(|x|)",
        label="D(|x|)",
        linewidth=2,
        title="Spatial diffusion coefficient on [-L/2,L/2]",
    )
    vline!(p_diffusion, [current_r]; label="tracked |x|", linewidth=2, color=:red)

    if param.dims == 1
        centers, density, _, _ = accumulated_distribution_1d(accumulator, state, param)
        current_x = state.positions[1, 1]
        p_distribution = plot(
            centers,
            density;
            seriestype=:steppost,
            xlabel="x",
            ylabel="probability density",
            label=@sprintf("accumulated (%d samples)", accumulator.sample_count),
            linewidth=2,
            xlims=(-radius, radius),
            title="P(x,t); $(param.boundary) boundaries",
        )
        plot!(
            p_distribution,
            centers,
            [steady_state_density(x, param) for x in centers];
            label="exact steady state",
            linewidth=2,
            linestyle=:dash,
        )
        vline!(p_distribution, [current_x]; label="tracked x", linewidth=2, color=:red)
        p_location = scatter(
            [current_x],
            [0.0];
            xlabel="x",
            label=@sprintf("walker 1: x=%.4g", current_x),
            markersize=8,
            color=:red,
            markerstrokecolor=:white,
            xlims=(-radius, radius),
            ylims=(-1.0, 1.0),
            yticks=false,
            title="Tracked particle current location",
        )
    else
        centers, density, _, outside = distribution_2d(state, radius, bins)
        current_x = state.positions[1, 1]
        current_y = state.positions[2, 1]
        p_distribution = heatmap(
            centers,
            centers,
            permutedims(density);
            xlabel="x",
            ylabel="y",
            color=:viridis,
            colorbar_title="p(x,y,t)",
            aspect_ratio=:equal,
            xlims=(-radius, radius),
            ylims=(-radius, radius),
            title=@sprintf("P(x,y,t); outside square %.2f%%", 100.0 * outside / n_walkers),
        )
        scatter!(
            p_distribution,
            [current_x],
            [current_y];
            label="tracked",
            color=:red,
            markerstrokecolor=:white,
            markersize=5,
        )
        p_location = scatter(
            [current_x],
            [current_y];
            xlabel="x",
            ylabel="y",
            label="walker 1",
            color=:red,
            markerstrokecolor=:white,
            markersize=8,
            aspect_ratio=:equal,
            xlims=(-radius, radius),
            ylims=(-radius, radius),
            title="Tracked particle current location",
        )
    end

    moment_label = param.dims == 1 ? "E[x²]" : "E[r²]"
    title = @sprintf(
        "%s SDE, %s, L=%.4g, t=%.4g, walkers=%d, %s=%.4g",
        uppercasefirst(String(param.interpretation)),
        String(param.boundary),
        param.L,
        t,
        n_walkers,
        moment_label,
        stats.mean_r2,
    )
    return plot(
        p_distribution,
        p_radial,
        p_location,
        p_diffusion;
        layout=(2, 2),
        size=(1150, 850),
        plot_title=title,
    )
end

function write_distribution_csv(
    out_dir::AbstractString,
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    args,
    accumulator::DistributionAccumulator,
)
    radius = 0.5 * param.L
    bins = length(accumulator.counts)
    n_walkers = size(state.positions, 2)
    if param.dims == 1
        centers, density, counts, _ = accumulated_distribution_1d(accumulator, state, param)
        plateau = steady_state_plateau(param)
        open(joinpath(out_dir, "final_distribution.csv"), "w") do io
            println(io, "x,empirical_density,count,exact_steady_density,P_inf,exact_P_inf_minus_P_ss")
            for i in eachindex(centers)
                exact_density = steady_state_density(centers[i], param)
                @printf(
                    io,
                    "%.17g,%.17g,%d,%.17g,%.17g,%.17g\n",
                    centers[i],
                    density[i],
                    counts[i],
                    exact_density,
                    plateau,
                    plateau - exact_density,
                )
            end
        end
    else
        centers, density, counts, _ = distribution_2d(state, radius, bins)
        open(joinpath(out_dir, "final_distribution.csv"), "w") do io
            println(io, "x,y,density,count")
            for iy in eachindex(centers), ix in eachindex(centers)
                @printf(
                    io,
                    "%.17g,%.17g,%.17g,%d\n",
                    centers[ix],
                    centers[iy],
                    density[ix, iy],
                    counts[ix, iy],
                )
            end
        end
    end

    radial_centers, radial_density, radial_counts, _ = radial_distribution(state, radius, bins)
    distance_filename = param.dims == 1 ?
        "final_absolute_position_distribution.csv" :
        "final_radial_distribution.csv"
    open(joinpath(out_dir, distance_filename), "w") do io
        println(io, param.dims == 1 ? "abs_x,absolute_position_density,count" : "r,radial_density,count")
        for i in eachindex(radial_centers)
            @printf(io, "%.17g,%.17g,%d\n", radial_centers[i], radial_density[i], radial_counts[i])
        end
    end

    if param.dims == 1
        distances, symmetric_density, symmetric_counts =
            accumulated_symmetric_location_distribution(accumulator, state, param)
        plateau = steady_state_plateau(param)
        open(joinpath(out_dir, "final_steady_state_tail.csv"), "w") do io
            println(io, "distance,empirical_symmetric_density,count,P_inf,empirical_deficit,exact_steady_density,exact_deficit")
            for i in eachindex(distances)
                exact_density = steady_state_density(distances[i], param)
                @printf(
                    io,
                    "%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g\n",
                    distances[i],
                    symmetric_density[i],
                    symmetric_counts[i],
                    plateau,
                    plateau - symmetric_density[i],
                    exact_density,
                    plateau - exact_density,
                )
            end
        end

        tail_min, tail_max = tail_window(args, param)
        log_tail = log_binned_tail_distribution(
            accumulator,
            state,
            param,
            tail_min,
            tail_max,
            Int(args["tail_log_bins"]),
        )
        log_tail_empirical_deficit = plateau .- log_tail.density
        fitted_amplitude, _, _, _ = fit_minus2_amplitude(
            log_tail.distances,
            log_tail_empirical_deficit,
            tail_min,
            tail_max,
        )
        open(joinpath(out_dir, "final_steady_state_tail_logbins.csv"), "w") do io
            println(
                io,
                "lower_distance,upper_distance,distance,empirical_symmetric_density,count,naive_density_stderr,P_inf,empirical_deficit,fixed_minus2_fit,exact_bin_average_density,exact_bin_average_deficit",
            )
            for i in eachindex(log_tail.distances)
                @printf(
                    io,
                    "%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    log_tail.lower_edges[i],
                    log_tail.upper_edges[i],
                    log_tail.distances[i],
                    log_tail.density[i],
                    log_tail.counts[i],
                    log_tail.density_stderr[i],
                    plateau,
                    log_tail_empirical_deficit[i],
                    fitted_amplitude / log_tail.distances[i]^2,
                    log_tail.exact_density[i],
                    log_tail.exact_deficit[i],
                )
            end
        end
    end
    return n_walkers
end

function write_summary(
    out_dir::AbstractString,
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    args,
    x0::AbstractVector{<:Real},
    actual_seed::Integer,
    initial_distribution::Symbol,
    accumulator::DistributionAccumulator,
    sample_start_step::Integer,
    sample_interval::Integer,
)
    stats = ensemble_stats(state, 0.5 * param.L)
    tail_min, tail_max = tail_window(args, param)
    tail = log_binned_tail_distribution(
        accumulator,
        state,
        param,
        tail_min,
        tail_max,
        Int(args["tail_log_bins"]),
    )
    fitted_amplitude, fit_points, fit_log_rmse, fit_log_r_squared = fit_minus2_amplitude(
        tail.distances,
        steady_state_plateau(param) .- tail.density,
        tail_min,
        tail_max,
    )
    checkpoint_interval = Int(args["checkpoint_interval_steps"])
    continue_path = isnothing(args["continue"]) ? "" : abspath(String(args["continue"]))
    checkpoint_latest_summary = joinpath(out_dir, "checkpoint_latest.jld2")
    checkpoint_final_summary = joinpath(out_dir, "checkpoint_final.jld2")
    crossover = sqrt(param.core_radius^2 + param.A)
    max_D = param.A >= 0.0 ? diffusivity(0.0, param) : param.D_inf
    core_resolution_ratio = param.dt * max_D / param.core_radius^2
    far_relative_deficit = steady_state_deficit(tail_max, param) / steady_state_plateau(param)
    open(joinpath(out_dir, "run_info.txt"), "w") do io
        println(io, param.dims == 1 ? "model: 1D spatially varying diffusion" : "model: radial spatially varying diffusion")
        println(io, "interpretation: $(param.interpretation)")
        println(io, "dims: $(param.dims)")
        println(io, "D_inf: $(param.D_inf)")
        println(io, "A: $(param.A)")
        println(io, "core_radius: $(param.core_radius)")
        println(io, "L: $(param.L)")
        println(io, "domain: [-L/2, L/2]")
        println(io, "boundary: $(param.boundary)")
        println(io, "initial_distribution: $initial_distribution")
        println(io, "sample_start_step: $sample_start_step")
        println(io, "sample_interval: $sample_interval")
        println(io, "sample_initial: $(Bool(args["sample_initial"]))")
        println(io, "tracked_trajectory_history_saved: false")
        println(io, "continue_from_checkpoint: $continue_path")
        println(io, "checkpoint_interval_steps: $checkpoint_interval")
        println(io, "checkpoint_latest: $checkpoint_latest_summary")
        println(io, "checkpoint_final: $checkpoint_final_summary")
        println(io, "accumulated_histogram_samples: $(accumulator.sample_count)")
        println(io, "accumulated_walker_observations: $(accumulator.sample_count * size(state.positions, 2))")
        println(io, "steady_state_plateau_P_inf: $(steady_state_plateau(param))")
        println(io, "crossover_b: $crossover")
        println(io, "core_resolution_ratio_dt_maxD_over_core_radius2: $core_resolution_ratio")
        println(io, "tail_fit_min: $tail_min")
        println(io, "tail_fit_max: $tail_max")
        println(io, "tail_fit_decades: $(log10(tail_max / tail_min))")
        println(io, "tail_first_bin_lower_edge: $(first(tail.lower_edges))")
        println(io, "tail_first_point_distance: $(first(tail.distances))")
        println(io, "tail_last_point_distance: $(last(tail.distances))")
        println(io, "tail_last_bin_upper_edge: $(last(tail.upper_edges))")
        println(io, "tail_point_span_decades: $(log10(last(tail.distances) / first(tail.distances)))")
        println(io, "exact_relative_deficit_at_tail_max: $far_relative_deficit")
        println(io, "tail_log_bins_requested: $(Int(args["tail_log_bins"]))")
        println(io, "tail_log_bins_used: $(length(tail.distances))")
        println(io, "fixed_tail_fit_exponent: -2")
        println(io, "fixed_tail_fit_amplitude_C: $fitted_amplitude")
        println(io, "fixed_tail_fit_points: $fit_points")
        println(io, "fixed_tail_fit_log_rmse: $fit_log_rmse")
        println(io, "fixed_tail_fit_log_r_squared: $fit_log_r_squared")
        println(io, "exact_asymptotic_amplitude_P_inf_times_A: $(steady_state_plateau(param) * param.A)")
        if param.dims == 1
            println(io, "D(x): D_inf * (1 + A / (x^2 + core_radius^2))")
        else
            println(io, "D(r): D_inf * (1 + A / (r^2 + core_radius^2))")
        end
        println(io, "x0: $(join(x0, ','))")
        println(io, "dt: $(param.dt)")
        println(io, "steps: $(state.step)")
        println(io, "final_time: $(state.step * param.dt)")
        println(io, "walkers: $(size(state.positions, 2))")
        println(io, "seed: $actual_seed")
        if param.dims == 1
            println(io, "mean_abs_x: $(stats.mean_r)")
            println(io, "mean_x2: $(stats.mean_r2)")
            println(io, "max_abs_x: $(stats.max_r)")
        else
            println(io, "mean_r: $(stats.mean_r)")
            println(io, "mean_r2: $(stats.mean_r2)")
            println(io, "max_r: $(stats.max_r)")
            println(io, "outside_plot_radius_fraction: $(stats.outside_fraction)")
        end
    end
end

function main(args)
    validate_args(args)
    additional_steps = Int(args["steps"])
    continue_path = args["continue"]

    if isnothing(continue_path)
        param = build_param(args)
        x0 = parse_x0(String(args["x0"]), param.dims)
        initial_distribution = parse_initial_distribution(String(args["initial_distribution"]))
        sample_start_step = Int(args["sample_start_step"])
        sample_interval = Int(args["sample_interval"])
        rng, actual_seed = rng_from_seed(Int(args["seed"]))
        state = initialize_simulation_state(
            param,
            Int(args["walkers"]),
            x0,
            rng,
            initial_distribution,
        )
        accumulator = DistributionAccumulator(Int(args["bins"]))
        Bool(args["sample_initial"]) && accumulate_distribution!(accumulator, state, param)
        if initial_distribution === :stationary
            @warn "Stationary initialization removes equilibration time but uses the known steady-state density; use uniform initialization to demonstrate relaxation from the SDE."
        end
    else
        Bool(args["sample_initial"]) &&
            @warn "--sample_initial is ignored when continuing an existing checkpoint"
        restored = load_checkpoint(String(continue_path))
        param = restored.param
        state = restored.state
        rng = restored.rng
        accumulator = restored.accumulator
        actual_seed = restored.actual_seed
        initial_distribution = restored.initial_distribution
        x0 = restored.x0
        sample_start_step = restored.sample_start_step
        sample_interval = restored.sample_interval
    end

    out_dir = if isnothing(args["output_dir"])
        isnothing(continue_path) ? default_output_dir(param) : dirname(abspath(String(continue_path)))
    else
        abspath(String(args["output_dir"]))
    end
    mkpath(out_dir)
    live_path = joinpath(out_dir, "live_latest.png")
    checkpoint_latest_path = joinpath(out_dir, "checkpoint_latest.jld2")
    checkpoint_final_path = joinpath(out_dir, "checkpoint_final.jld2")

    max_D = param.A >= 0.0 ? diffusivity(0.0, param) : param.D_inf
    resolution_ratio = param.dt * max_D / (param.core_radius^2)
    if resolution_ratio > 0.05
        @warn "The time step may underresolve the core region" dt=param.dt max_D resolution_ratio
    end

    @printf(
        "Running %dD %s SDE on L=%.6g with %s boundaries: %d walkers, %d additional steps, dt=%.6g, seed=%d\n",
        param.dims,
        String(param.interpretation),
        param.L,
        String(param.boundary),
        size(state.positions, 2),
        additional_steps,
        param.dt,
        actual_seed,
    )
    if !isnothing(continue_path)
        @printf("Continuing from checkpoint at global step %d: %s\n", state.step, abspath(String(continue_path)))
    end
    @printf("Outputs: %s\n", out_dir)

    plotting = !Bool(args["no_plot"])
    plot_interval = Int(args["plot_interval_steps"])
    checkpointing = !Bool(args["no_checkpoint"])
    checkpoint_interval = Int(args["checkpoint_interval_steps"])
    start_step = state.step
    if checkpointing
        save_checkpoint(
            checkpoint_latest_path;
            state=state,
            param=param,
            rng=rng,
            accumulator=accumulator,
            actual_seed=actual_seed,
            initial_distribution=initial_distribution,
            x0=x0,
            sample_start_step=sample_start_step,
            sample_interval=sample_interval,
        )
    end
    for local_step in 1:additional_steps
        step!(state, param, rng)
        global_step = state.step
        if global_step >= sample_start_step && global_step % sample_interval == 0
            accumulate_distribution!(accumulator, state, param)
        end
        if checkpointing && (global_step % checkpoint_interval == 0 || local_step == additional_steps)
            save_checkpoint(
                checkpoint_latest_path;
                state=state,
                param=param,
                rng=rng,
                accumulator=accumulator,
                actual_seed=actual_seed,
                initial_distribution=initial_distribution,
                x0=x0,
                sample_start_step=sample_start_step,
                sample_interval=sample_interval,
            )
        end
        if plotting && (local_step % plot_interval == 0 || local_step == additional_steps)
            fig = make_live_plot(state, param, args, accumulator)
            savefig(fig, live_path)
            Bool(args["display"]) && display(fig)
            stats = ensemble_stats(state, 0.5 * param.L)
            if param.dims == 1
                @printf(
                    "local step %d/%d, global step %d, t=%.5g, E[x^2]=%.5g, accumulated samples=%d\n",
                    local_step,
                    additional_steps,
                    global_step,
                    state.step * param.dt,
                    stats.mean_r2,
                    accumulator.sample_count,
                )
            else
                @printf(
                    "local step %d/%d, global step %d, t=%.5g, E[r^2]=%.5g, outside radius=%.3f%%\n",
                    local_step,
                    additional_steps,
                    global_step,
                    state.step * param.dt,
                    stats.mean_r2,
                    100.0 * stats.outside_fraction,
                )
            end
        end
    end
    if plotting
        fig = make_live_plot(state, param, args, accumulator)
        savefig(fig, live_path)
        savefig(fig, joinpath(out_dir, "final_distribution.png"))
    end
    write_distribution_csv(out_dir, state, param, args, accumulator)
    write_summary(
        out_dir,
        state,
        param,
        args,
        x0,
        actual_seed,
        initial_distribution,
        accumulator,
        sample_start_step,
        sample_interval,
    )
    if checkpointing
        save_checkpoint(
            checkpoint_final_path;
            state=state,
            param=param,
            rng=rng,
            accumulator=accumulator,
            actual_seed=actual_seed,
            initial_distribution=initial_distribution,
            x0=x0,
            sample_start_step=sample_start_step,
            sample_interval=sample_interval,
        )
        if checkpoint_latest_path != checkpoint_final_path
            cp(checkpoint_final_path, checkpoint_latest_path; force=true)
        end
    end
    @printf(
        "Finished. Advanced from step %d to %d. Final time %.6g; results written to %s\n",
        start_step,
        state.step,
        state.step * param.dt,
        out_dir,
    )
    return nothing
end

main(CLI_ARGS)
