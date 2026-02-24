using ArgParse
using JLD2
using LinearAlgebra
using Plots
using Printf
using Statistics

include("potentials.jl")
include("modules_diffusive_no_activity.jl")
include("plot_utils.jl")

using .PlotUtils

const BOND_PASS_TOTAL_SQ_AVG_KEY = :bond_pass_total_sq_avg
const BOND_PASS_TRACK_MASK_KEY = :bond_pass_track_mask

function wildcard_to_regex(pattern::String)
    special = Set(['.', '^', '$', '+', '(', ')', '[', ']', '{', '}', '|', '\\'])
    io = IOBuffer()
    for c in pattern
        if c == '*'
            print(io, ".*")
        elseif c == '?'
            print(io, ".")
        elseif c in special
            print(io, '\\', c)
        else
            print(io, c)
        end
    end
    return Regex("^" * String(take!(io)) * "\$")
end

function collect_matching_files(dir::String, pattern::String; recursive::Bool=false)
    matcher = wildcard_to_regex(pattern)
    files = String[]

    if recursive
        for (root, _, names) in walkdir(dir)
            for name in names
                if occursin(matcher, name)
                    push!(files, joinpath(root, name))
                end
            end
        end
    else
        for name in readdir(dir)
            path = joinpath(dir, name)
            if isfile(path) && occursin(matcher, name)
                push!(files, path)
            end
        end
    end

    sort!(files)
    return files
end

function add_unique!(paths::Vector{String}, seen::Set{String}, candidate::String)
    full_path = abspath(candidate)
    if !(full_path in seen)
        push!(paths, full_path)
        push!(seen, full_path)
    end
end

function collect_input_files(inputs::Vector{String}, default_glob::String; recursive::Bool=false)
    files = String[]
    seen = Set{String}()

    for input in inputs
        if isfile(input)
            if endswith(lowercase(input), ".jld2")
                add_unique!(files, seen, input)
            else
                println("Skipping non-JLD2 file: ", input)
            end
            continue
        end

        if isdir(input)
            for match in collect_matching_files(input, default_glob; recursive=recursive)
                add_unique!(files, seen, match)
            end
            continue
        end

        if occursin('*', input) || occursin('?', input)
            dir = dirname(input)
            dir = isempty(dir) ? "." : dir
            pattern = basename(input)
            if isdir(dir)
                for match in collect_matching_files(dir, pattern; recursive=recursive)
                    add_unique!(files, seen, match)
                end
            else
                println("Skipping glob with missing directory: ", input)
            end
            continue
        end

        println("Skipping missing path: ", input)
    end

    sort!(files)
    return files
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "inputs"
            help = "Input path(s): JLD2 files, directories, or globs (optional when --run_id is used)"
            nargs = '*'
        "--mode"
            help = "Mode: single, two_force_d, or run_id (run_id is an alias for two_force_d and expects --run_id)"
            default = "single"
        "--run_id"
            help = "Fetched run_id under cluster_results/runs/two_force_d/<warmup|production>/<run_id>"
            default = ""
        "--run_result_mode"
            help = "When using --run_id: warmup, production, or auto"
            default = "auto"
        "--cluster_results_root"
            help = "Root directory containing fetched cluster run folders and fitting outputs"
            default = "cluster_results"
        "--glob"
            help = "Filename wildcard for directory inputs"
            default = "*.jld2"
        "--recursive"
            help = "Recursively scan directory inputs"
            action = :store_true
        "--out_dir"
            help = "Output directory"
            default = "results_figures/fitting"
        "--include_abs_mean_in_spatial_f_plot"
            help = "Include |<J>| in spatial J statistics panel"
            action = :store_true
        "--keep_diagonal_in_multiforce_cut"
            help = "Do not smooth diagonal points in bond-centered cuts"
            action = :store_true
        "--skip_per_state_sweep"
            help = "Skip per-state sweep/component plots"
            action = :store_true
        "--baseline_j2"
            help = "Absolute baseline override for (<J^2>-baseline) vs d analysis (default: use baseline_rho_factor*ρ₀^2)"
            arg_type = Float64
            default = NaN
        "--baseline_rho_factor"
            help = "Baseline multiplier A in (<J^2>-A*ρ₀^2) vs d analysis"
            arg_type = Float64
            default = 0.088
        "--corr_model_c"
            help = "Fixed constant offset C in bond-correlation model fit (default 0)"
            arg_type = Float64
            default = 0.0
    end
    return parse_args(s)
end

function resolve_run_id_dir(run_id::AbstractString, cluster_results_root::AbstractString, run_result_mode::AbstractString)
    run_id_str = String(run_id)
    run_result_mode_str = String(run_result_mode)
    cluster_results_root_str = String(cluster_results_root)
    root = abspath(cluster_results_root_str)
    base = joinpath(root, "runs", "two_force_d")
    warmup_dir = joinpath(base, "warmup", run_id_str)
    production_dir = joinpath(base, "production", run_id_str)

    if run_result_mode_str == "warmup"
        isdir(warmup_dir) || error("run_id not found at $(warmup_dir)")
        return warmup_dir, "warmup"
    elseif run_result_mode_str == "production"
        isdir(production_dir) || error("run_id not found at $(production_dir)")
        return production_dir, "production"
    elseif run_result_mode_str == "auto"
        has_warmup = isdir(warmup_dir)
        has_production = isdir(production_dir)
        if has_warmup && !has_production
            return warmup_dir, "warmup"
        elseif has_production && !has_warmup
            return production_dir, "production"
        elseif has_warmup && has_production
            error("run_id exists in both warmup and production. Use --run_result_mode to disambiguate.")
        else
            error("run_id not found under $(base) for mode=auto.")
        end
    else
        error("--run_result_mode must be warmup, production, or auto. Got '$run_result_mode_str'.")
    end
end

function collect_files_from_run_id(run_id::AbstractString, cluster_results_root::AbstractString, run_result_mode::AbstractString)
    run_id_str = String(run_id)
    cluster_results_root_str = String(cluster_results_root)
    run_result_mode_str = String(run_result_mode)
    run_dir, resolved_mode = resolve_run_id_dir(run_id, cluster_results_root, run_result_mode)
    states_dir = joinpath(run_dir, "states")
    isdir(states_dir) || error("States directory not found for run_id: $(states_dir)")

    files = sort(filter(path -> isfile(path) && endswith(lowercase(path), ".jld2"),
                        readdir(states_dir; join=true)))
    isempty(files) && error("No JLD2 files found in run states directory: $(states_dir)")

    default_out_dir = joinpath(abspath(cluster_results_root_str), "fitting", "two_force_d", resolved_mode, run_id_str)
    return files, default_out_dir, resolved_mode, run_dir
end

function ensure_state_potential!(state, potential)
    if isnothing(potential)
        return
    end
    try
        state.potential = potential
    catch
    end
end

has_prop(obj, name::Symbol) = hasproperty(obj, name)
get_prop(obj, name::Symbol, default=nothing) = hasproperty(obj, name) ? getproperty(obj, name) : default

function has_loaded_key(data, key::AbstractString)
    return haskey(data, key) || haskey(data, Symbol(key))
end

function get_loaded_value(data, key::AbstractString, default=nothing)
    if haskey(data, key)
        return data[key]
    end
    sym = Symbol(key)
    if haskey(data, sym)
        return data[sym]
    end
    return default
end

function load_state_bundle(saved_state::String)
    data = JLD2.load(saved_state)

    state = get_loaded_value(data, "state", nothing)
    if state === nothing
        state = get_loaded_value(data, "dummy_state", nothing)
    end
    if state === nothing && has_loaded_key(data, "states")
        states = get_loaded_value(data, "states", nothing)
        if states isa AbstractVector && !isempty(states)
            state = states[1]
        end
    end

    param = get_loaded_value(data, "param", nothing)
    if param === nothing && has_loaded_key(data, "params")
        params = get_loaded_value(data, "params", nothing)
        if params isa AbstractVector && !isempty(params)
            param = params[1]
        end
    end

    potential = get_loaded_value(data, "potential", nothing)
    if state !== nothing && potential === nothing && has_prop(state, :potential)
        potential = get_prop(state, :potential)
    end

    if state === nothing || param === nothing
        available_keys = join(sort!(String.(collect(keys(data)))), ", ")
        error("Missing state/param payload (keys: $available_keys)")
    end

    return state, param, potential
end

function is_common_diffusive_state(state, param)
    has_prop(state, :ρ_avg) || return false
    has_prop(state, :ρ_matrix_avg_cuts) || return false
    has_prop(state, :forcing) || return false
    has_prop(param, :dims) || return false
    dims = get_prop(param, :dims)
    return dims isa Tuple || dims isa AbstractVector
end

function bond_pass_stats_dict(state)
    if has_prop(state, :bond_pass_stats)
        stats = get_prop(state, :bond_pass_stats)
        if stats isa AbstractDict
            return stats
        end
    end
    if has_prop(state, :ρ_matrix_avg_cuts)
        cuts = get_prop(state, :ρ_matrix_avg_cuts)
        if cuts isa AbstractDict
            legacy = Dict{Symbol,Vector{Float64}}()
            for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                        :bond_pass_total_sq_avg, :bond_pass_sample_count, :bond_pass_track_mask,
                        :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
                if haskey(cuts, key)
                    legacy[key] = Float64.(cuts[key])
                end
            end
            return legacy
        end
    end
    return Dict{Symbol,Vector{Float64}}()
end

function forcing_bonds_1d(state, L::Int)
    if !has_prop(state, :forcing)
        return Tuple{Int,Int}[], Float64[]
    end
    forcing_raw = get_prop(state, :forcing)
    forcings = forcing_raw isa AbstractVector ? forcing_raw : [forcing_raw]
    bonds = Tuple{Int,Int}[]
    magnitudes = Float64[]
    for force in forcings
        if length(force.bond_indices[1]) == 1 && length(force.bond_indices[2]) == 1
            b1 = mod1(force.bond_indices[1][1], L)
            b2 = mod1(force.bond_indices[2][1], L)
            push!(bonds, (b1, b2))
            push!(magnitudes, Float64(force.magnitude))
        end
    end
    return bonds, magnitudes
end

function tracked_force_indices(state, magnitudes::AbstractVector{<:Real})
    n_forces = length(magnitudes)
    stats = bond_pass_stats_dict(state)
    if haskey(stats, BOND_PASS_TRACK_MASK_KEY)
        mask = stats[BOND_PASS_TRACK_MASK_KEY]
        if length(mask) == n_forces
            return [i for i in 1:n_forces if mask[i] > 0.5]
        end
    end
    return [i for i in 1:n_forces if abs(Float64(magnitudes[i])) > 0]
end

function connected_corr_mat_1d(state)
    outer_prod_ρ = state.ρ_avg * transpose(state.ρ_avg)
    return state.ρ_matrix_avg_cuts[:full] .- outer_prod_ρ
end

function bond_centered_cut_1d(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    L = size(corr_mat, 2)
    cut = 0.5 .* (Float64.(corr_mat[b1, :]) .+ Float64.(corr_mat[b2, :]))
    if smooth_diagonal
        b1_left = mod1(b1 - 1, L)
        b1_right = mod1(b1 + 1, L)
        b2_left = mod1(b2 - 1, L)
        b2_right = mod1(b2 + 1, L)
        cut[b1] = 0.5 * (cut[b1_left] + cut[b1_right])
        cut[b2] = 0.5 * (cut[b2_left] + cut[b2_right])
    end
    return cut
end

function bond_center_value(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    cut = bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
    return 0.5 * (cut[b1] + cut[b2])
end

function pair_distance_d(bonds::Vector{Tuple{Int,Int}}, L::Int)
    if length(bonds) < 2
        return nothing
    end
    (a1, a2) = bonds[1]
    (b1, b2) = bonds[2]
    d1 = mod(b1 - a2, L)
    d2 = mod(a1 - b2, L)
    d = min(d1, d2)
    if d == 0
        d = L
    end
    return d
end

function finite_mean(values::AbstractVector{<:Real})
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    return isempty(vals) ? NaN : mean(vals)
end

function rho0_from_param(param)
    if has_prop(param, :ρ₀)
        value = get_prop(param, :ρ₀)
        return value isa Number ? Float64(value) : NaN
    elseif has_prop(param, :rho0)
        value = get_prop(param, :rho0)
        return value isa Number ? Float64(value) : NaN
    elseif param isa AbstractDict
        for key in ("ρ₀", "rho0")
            value = get_loaded_value(param, key, nothing)
            if value isa Number
                return Float64(value)
            end
        end
    end
    return NaN
end

function periodic_displacement_1d(dx::Float64, L::Int)
    return mod(dx + 0.5 * L, L) - 0.5 * L
end

function bond_centered_axis_1d(L::Int, b1::Int, b2::Int)
    center = bond_center_coordinate_1d(L, b1, b2)
    x_rel = [periodic_displacement_1d(Float64(i) - center, L) for i in 1:L]
    perm = sortperm(x_rel)
    return x_rel, perm
end

function bond_center_coordinate_1d(L::Int, b1::Int, b2::Int)
    if b2 == mod1(b1 + 1, L)
        return Float64(b1) + 0.5
    elseif b1 == mod1(b2 + 1, L)
        return Float64(b2) + 0.5
    else
        return 0.5 * (Float64(b1) + Float64(b2))
    end
end

function bond_cut_model(x::AbstractVector{<:Real}, K1::Float64, K2::Float64; x0::Float64=0.0, orientation_sign::Float64=1.0, C::Float64=0.0)
    x_vals = Float64.(x)
    x_shifted = x_vals .- x0
    return C .+ orientation_sign .* (K1 * K2) .* x_shifted ./ (x_shifted .^ 2 .+ K2^2) .^ 2
end

function fit_bond_cut_profile(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; x0::Float64=0.0, orientation_sign::Float64=1.0, C_fixed::Float64=0.0)
    x_data = Float64.(x)
    y_data = Float64.(y)
    mask = isfinite.(x_data) .& isfinite.(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]

    if length(x_data) < 6
        return nothing
    end

    abs_x = abs.(x_data)
    nonzero = abs_x[abs_x .> 0]
    if isempty(nonzero)
        return nothing
    end

    k2_min = max(0.5 * minimum(nonzero), 1e-3)
    k2_max = max(maximum(abs_x), 1.2 * k2_min)
    k2_grid = exp.(range(log(k2_min), log(k2_max), length=220))

    best_sse = Inf
    best_k2 = NaN
    best_A = NaN
    for k2 in k2_grid
        basis_x = x_data .- x0
        basis = orientation_sign .* basis_x ./ (basis_x .^ 2 .+ k2^2) .^ 2
        denom = dot(basis, basis)
        if !(isfinite(denom) && denom > eps(Float64))
            continue
        end
        centered_y = y_data .- C_fixed
        A = dot(basis, centered_y) / denom
        residual = y_data .- (A .* basis .+ C_fixed)
        sse = dot(residual, residual)
        if isfinite(sse) && sse < best_sse
            best_sse = sse
            best_k2 = k2
            best_A = A
        end
    end

    if !isfinite(best_sse)
        return nothing
    end

    y_mean = mean(y_data)
    ss_tot = sum((y_data .- y_mean) .^ 2)
    r2 = ss_tot > 0 ? 1 - best_sse / ss_tot : NaN
    K1 = best_A / best_k2
    return (K1=K1, K2=best_k2, x0=x0, orientation_sign=orientation_sign, A=best_A, C=C_fixed, r2=r2)
end

function fit_loglog_powerlaw(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    x_data = Float64.(x)
    y_data = Float64.(y)
    mask = isfinite.(x_data) .& isfinite.(y_data) .& (x_data .> 0) .& (y_data .> 0)
    if count(mask) < 2
        return nothing
    end

    x_fit = x_data[mask]
    y_fit = y_data[mask]
    lx = log10.(x_fit)
    ly = log10.(y_fit)

    x_mean = mean(lx)
    y_mean = mean(ly)
    denom = sum((lx .- x_mean) .^ 2)
    if denom <= eps(Float64)
        return nothing
    end

    slope = sum((lx .- x_mean) .* (ly .- y_mean)) / denom
    intercept = y_mean - slope * x_mean
    pred = intercept .+ slope .* lx
    ss_tot = sum((ly .- y_mean) .^ 2)
    ss_res = sum((ly .- pred) .^ 2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    return (slope=slope, intercept=intercept, r2=r2, x=x_fit)
end

function add_reference_slopes!(p, x::Vector{Float64}, anchor_x::Float64, anchor_y::Float64)
    if !(isfinite(anchor_x) && isfinite(anchor_y) && anchor_x > 0 && anchor_y > 0)
        return
    end
    x_ref = sort([xi for xi in x if isfinite(xi) && xi > 0])
    isempty(x_ref) && return
    x_max = maximum(x_ref)
    for n in 1:4
        y_ref = anchor_y .* (x_ref ./ anchor_x) .^ (-n)
        mask = isfinite.(y_ref) .& (y_ref .> 0)
        if !any(mask)
            continue
        end
        plot!(p, x_ref[mask], y_ref[mask], color=:gray40, lw=1.0, linestyle=:dash, alpha=0.2, label=false)
        y_lab = anchor_y * (x_max / anchor_x)^(-n)
        if isfinite(y_lab) && y_lab > 0
            annotate!(p, x_max, y_lab, text("d^(-$(n))", 7, :gray40))
        end
    end
end

function plot_bond_cut_model_fits_1d(state, param; smooth_diagonal::Bool=true, corr_model_c::Float64=0.0)
    L = param.dims[1]
    bonds, magnitudes = forcing_bonds_1d(state, L)
    tracked = tracked_force_indices(state, magnitudes)
    if isempty(tracked)
        tracked = collect(1:length(bonds))
    end
    if isempty(tracked)
        return plot(title="Bond cut fit unavailable", axis=false, legend=false)
    end

    corr_mat = connected_corr_mat_1d(state)
    colors = [:red, :orange, :green, :blue, :magenta, :cyan, :black]
    centers = Dict{Int,Float64}()
    for idx in tracked
        if idx <= length(bonds)
            b1, b2 = bonds[idx]
            centers[idx] = bond_center_coordinate_1d(L, b1, b2)
        end
    end

    p = plot(title="Bond Correlation Model Fit",
             xlabel="Δx = x' - x_bond",
             ylabel="C(x_bond, x')",
             framestyle=:box,
             legend=:outerright,
             titlefontsize=9,
             legendfontsize=7,
             guidefontsize=9,
             tickfontsize=8)
    model_label = iszero(corr_model_c) ?
        "Model: s*K1*K2*(Δx-x0)/((Δx-x0)^2+K2^2)^2" :
        @sprintf("Model: C + s*K1*K2*(Δx-x0)/((Δx-x0)^2+K2^2)^2, C=%.3g", corr_model_c)
    plot!(p, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=model_label)
    hline!(p, [0.0], color=:gray, linestyle=:dot, label=false)
    vline!(p, [0.0], color=:gray, linestyle=:dash, label=false)

    for (draw_i, idx) in enumerate(tracked)
        if idx > length(bonds)
            continue
        end
        b1, b2 = bonds[idx]
        color = colors[mod1(draw_i, length(colors))]
        cut = bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
        x_rel, perm = bond_centered_axis_1d(L, b1, b2)
        x_plot = x_rel[perm]
        y_plot = cut[perm]

        bond_short = "b$(idx)"
        bond_label = "$(bond_short) ($(b1),$(b2))"
        plot!(p, x_plot, y_plot, lw=1.8, color=color, alpha=0.35, label="$(bond_label) data")

        x0_fixed = 0.0
        orientation_sign = 1.0
        if haskey(centers, idx)
            center_i = centers[idx]
            best_disp = NaN
            best_abs = Inf
            for jdx in tracked
                if jdx == idx || !haskey(centers, jdx)
                    continue
                end
                disp = periodic_displacement_1d(centers[jdx] - center_i, L)
                abs_disp = abs(disp)
                if abs_disp > 1e-9 && abs_disp < best_abs
                    best_abs = abs_disp
                    best_disp = disp
                end
            end
            if isfinite(best_disp)
                x0_fixed = best_disp
                orientation_sign = best_disp >= 0 ? 1.0 : -1.0
            end
        end

        fit = fit_bond_cut_profile(x_plot, y_plot; x0=x0_fixed, orientation_sign=orientation_sign, C_fixed=corr_model_c)
        if isnothing(fit)
            plot!(p, [NaN], [NaN], color=color, linestyle=:dash, label="$(bond_short) fit: failed")
            continue
        end

        y_fit = bond_cut_model(x_plot, fit.K1, fit.K2; x0=fit.x0, orientation_sign=fit.orientation_sign, C=fit.C)
        fit_label = @sprintf("%s fit: K1=%.3g, K2=%.3g, x0=%.2f, s=%.0f, R²=%.2f",
                             bond_short, fit.K1, fit.K2, fit.x0, fit.orientation_sign, fit.r2)
        if !iszero(fit.C)
            fit_label = string(fit_label, @sprintf(", C=%.3g", fit.C))
        end
        plot!(p, x_plot, y_fit, lw=2.2, color=color, linestyle=:dash, alpha=0.95, label=fit_label)
    end

    return p
end

function save_diffusive_sweep_components(saved_state::String, state, param, out_dir::String;
                                         include_abs_mean_in_spatial_f_plot::Bool=false,
                                         keep_diagonal_in_multiforce_cut::Bool=false,
                                         corr_model_c::Float64=0.0)
    base_name = replace(basename(saved_state), ".jld2" => "")
    state_dir = joinpath(out_dir, base_name, "current_sweep_statistics")
    mkpath(state_dir)

    if length(param.dims) == 1
        components = PlotUtils.plot_sweep_1d_multiforce(
            state.t,
            state,
            param;
            remove_diagonal_for_cuts=!keep_diagonal_in_multiforce_cut,
            include_abs_mean_in_spatial_f_plot=include_abs_mean_in_spatial_f_plot,
            return_components=true,
        )
        p_bond_model_fit = plot_bond_cut_model_fits_1d(
            state,
            param;
            smooth_diagonal=!keep_diagonal_in_multiforce_cut,
            corr_model_c=corr_model_c,
        )

        panel_map = [
            ("00_composite", components.final_plot),
            ("01_avg_density", components.avg_density),
            ("02_inst_density", components.inst_density),
            ("03_force_averages", components.force_averages),
            ("04_spatial_j_stats", components.spatial_f_stats),
            ("05_corr_origin_bond", components.corr_origin_cut),
            ("06_corr_fluctuating_bonds", components.corr_fluctuating_bond_cuts),
            ("07_corr_heat", components.corr_heat),
            ("08_bond_center_variance", components.density_variance_bond_sites),
            ("09_corr_fluctuating_bonds_model_fit", p_bond_model_fit),
        ]

        for (name, plt) in panel_map
            output_file = joinpath(state_dir, string(name, ".png"))
            savefig(plt, output_file)
            println("Saved ", output_file)
        end
    else
        PlotUtils.plot_sweep(state.t, state, param)
        output_file = joinpath(state_dir, "00_composite.png")
        savefig(output_file)
        println("Saved ", output_file)
    end
end

function save_legacy_sweep_plot(saved_state::String, state, param, out_dir::String)
    base_name = replace(basename(saved_state), ".jld2" => "")
    state_dir = joinpath(out_dir, base_name, "current_sweep_statistics")
    mkpath(state_dir)
    output_file = joinpath(state_dir, "00_composite.png")
    try
        PlotUtils.plot_sweep(state.t, state, param)
        savefig(output_file)
        println("Saved ", output_file)
    catch e
        p = if has_prop(state, :ρ_avg)
            ρ_avg = Float64.(get_prop(state, :ρ_avg))
            if ndims(ρ_avg) == 1
                plot(1:length(ρ_avg), ρ_avg,
                     lw=2.2,
                     color=:black,
                     title="Legacy fallback: time-averaged density",
                     xlabel="Site",
                     ylabel="⟨ρ⟩",
                     framestyle=:box)
            else
                heatmap(ρ_avg',
                        title="Legacy fallback: time-averaged density",
                        xlabel="x",
                        ylabel="y",
                        aspect_ratio=1,
                        framestyle=:box)
            end
        else
            plot(title="Legacy fallback: unsupported state format", axis=false, legend=false)
        end
        savefig(p, output_file)
        println("Saved ", output_file, " (legacy fallback)")
        println("Legacy sweep fallback used for ", saved_state, " due to: ", e)
    end
end

function analyze_two_force_d(files::Vector{String}, out_dir::String;
                             baseline_rho_factor::Float64=0.088,
                             baseline_j2_override::Float64=NaN,
                             smooth_diagonal::Bool=true)
    rows = NamedTuple[]

    for saved_state in files
        try
            state, param, potential = load_state_bundle(saved_state)
            ensure_state_potential!(state, potential)

            if !is_common_diffusive_state(state, param)
                println("Skipping non-diffusive state for d-analysis: ", saved_state)
                continue
            end

            if length(param.dims) != 1
                println("Skipping non-1D state for d-analysis: ", saved_state)
                continue
            end

            L = param.dims[1]
            bonds, magnitudes = forcing_bonds_1d(state, L)
            if length(bonds) < 2
                println("Skipping state without two 1D force bonds: ", saved_state)
                continue
            end

            d = pair_distance_d(bonds, L)
            if isnothing(d)
                println("Skipping state with undefined d: ", saved_state)
                continue
            end

            tracked = tracked_force_indices(state, magnitudes)
            if isempty(tracked)
                tracked = collect(1:length(bonds))
            end

            corr_mat = connected_corr_mat_1d(state)
            var_vals = [bond_center_value(corr_mat, bonds[i][1], bonds[i][2]; smooth_diagonal=smooth_diagonal) for i in tracked]

            stats = bond_pass_stats_dict(state)
            j2_all = haskey(stats, BOND_PASS_TOTAL_SQ_AVG_KEY) ? Float64.(stats[BOND_PASS_TOTAL_SQ_AVG_KEY]) : Float64[]
            j2_vals = Float64[]
            for idx in tracked
                if idx <= length(j2_all)
                    push!(j2_vals, j2_all[idx])
                else
                    push!(j2_vals, NaN)
                end
            end

            rho0 = rho0_from_param(param)
            baseline = if isfinite(baseline_j2_override)
                baseline_j2_override
            elseif isfinite(rho0)
                baseline_rho_factor * (rho0^2)
            else
                println("Skipping state without finite ρ₀ for baseline formula: ", saved_state)
                continue
            end
            j2_vals_shifted = [isfinite(v) ? (v - baseline) : NaN for v in j2_vals]

            push!(rows, (
                file=saved_state,
                d=Int(d),
                rho0=rho0,
                baseline=baseline,
                var_vals=var_vals,
                var_mean=finite_mean(var_vals),
                j2_vals=j2_vals,
                j2_mean=finite_mean(j2_vals),
                j2_vals_shifted=j2_vals_shifted,
                j2_mean_shifted=finite_mean(j2_vals_shifted),
            ))
        catch e
            println("Failed in d-analysis for ", saved_state, ": ", e)
        end
    end

    if isempty(rows)
        println("No valid two-force 1D states found for d-analysis.")
        return
    end

    d_values = sort(unique(row.d for row in rows))
    max_slots = maximum(length(row.var_vals) for row in rows)
    grouped = Dict(d => [row for row in rows if row.d == d] for d in d_values)

    x = Float64.(d_values)
    y_var_mean = [finite_mean([row.var_mean for row in grouped[d]]) for d in d_values]
    y_j2_mean = [finite_mean([row.j2_mean for row in grouped[d]]) for d in d_values]
    y_j2_mean_shifted = [finite_mean([row.j2_mean_shifted for row in grouped[d]]) for d in d_values]
    y_baseline_mean = [finite_mean([row.baseline for row in grouped[d]]) for d in d_values]

    y_var_slots = [fill(NaN, length(d_values)) for _ in 1:max_slots]
    y_j2_slots = [fill(NaN, length(d_values)) for _ in 1:max_slots]
    y_j2_slots_shifted = [fill(NaN, length(d_values)) for _ in 1:max_slots]
    for (di, d) in enumerate(d_values)
        bucket = grouped[d]
        for slot in 1:max_slots
            vals_var = Float64[]
            vals_j2 = Float64[]
            vals_j2_shifted = Float64[]
            for row in bucket
                if slot <= length(row.var_vals)
                    push!(vals_var, row.var_vals[slot])
                end
                if slot <= length(row.j2_vals)
                    push!(vals_j2, row.j2_vals[slot])
                end
                if slot <= length(row.j2_vals_shifted)
                    push!(vals_j2_shifted, row.j2_vals_shifted[slot])
                end
            end
            y_var_slots[slot][di] = finite_mean(vals_var)
            y_j2_slots[slot][di] = finite_mean(vals_j2)
            y_j2_slots_shifted[slot][di] = finite_mean(vals_j2_shifted)
        end
    end

    analysis_dir = joinpath(out_dir, "two_force_d_analysis")
    mkpath(analysis_dir)

    p_var = plot(title="Bond-center variance vs d",
                 xlabel="d",
                 ylabel="C_bond(0)",
                 xscale=:log10,
                 yscale=:log10,
                 framestyle=:box,
                 legend=:outerright)
    for slot in 1:max_slots
        y = y_var_slots[slot]
        mask = isfinite.(y) .& (y .> 0)
        if any(mask)
            plot!(p_var, x[mask], y[mask], marker=:circle, lw=2, label="bond $(slot)")
        end
    end
    mask_var_mean = isfinite.(y_var_mean) .& (y_var_mean .> 0)
    if any(mask_var_mean)
        plot!(p_var, x[mask_var_mean], y_var_mean[mask_var_mean], marker=:diamond, lw=2.8, color=:black, label="mean")
    end
    fit_var = fit_loglog_powerlaw(x, y_var_mean)
    if !isnothing(fit_var)
        x_fit = fit_var.x
        y_fit = 10 .^ (fit_var.intercept .+ fit_var.slope .* log10.(x_fit))
        plot!(p_var, x_fit, y_fit, lw=2.2, color=:black, linestyle=:dashdot, alpha=0.9,
              label=@sprintf("mean fit slope=%.3f (R²=%.3f)", fit_var.slope, fit_var.r2))
        anchor_x = exp(mean(log.(x_fit)))
        anchor_y = 10^(fit_var.intercept + fit_var.slope * log10(anchor_x))
        add_reference_slopes!(p_var, x_fit, anchor_x, anchor_y)
    elseif any(mask_var_mean)
        x_ref = x[mask_var_mean]
        y_ref = y_var_mean[mask_var_mean]
        anchor_x = exp(mean(log.(x_ref)))
        anchor_y = exp(mean(log.(y_ref)))
        add_reference_slopes!(p_var, x_ref, anchor_x, anchor_y)
    end
    savefig(p_var, joinpath(analysis_dir, "00_bond_center_variance_vs_d_loglog.png"))
    println("Saved ", joinpath(analysis_dir, "00_bond_center_variance_vs_d_loglog.png"))

    p_var_linear = plot(title="Bond-center variance vs d",
                        xlabel="d",
                        ylabel="C_bond(0)",
                        framestyle=:box,
                        legend=:outerright)
    for slot in 1:max_slots
        y = y_var_slots[slot]
        mask = isfinite.(y)
        if any(mask)
            plot!(p_var_linear, x[mask], y[mask], marker=:circle, lw=2, label="bond $(slot)")
        end
    end
    mask_var_mean_linear = isfinite.(y_var_mean)
    if any(mask_var_mean_linear)
        plot!(p_var_linear, x[mask_var_mean_linear], y_var_mean[mask_var_mean_linear], marker=:diamond, lw=2.8, color=:black, label="mean")
    end
    savefig(p_var_linear, joinpath(analysis_dir, "01_bond_center_variance_vs_d_linear.png"))
    println("Saved ", joinpath(analysis_dir, "01_bond_center_variance_vs_d_linear.png"))

    baseline_title = isfinite(baseline_j2_override) ?
        @sprintf("baseline=%.6g", baseline_j2_override) :
        @sprintf("baseline=%.6g*ρ₀^2", baseline_rho_factor)

    function save_series_plot(file_name::String, title::String, ylabel::String,
                              y_slots::Vector{Vector{Float64}}, y_mean::Vector{Float64};
                              loglog::Bool=false, fit_mean_loglog::Bool=false)
        p = plot(title=title,
                 xlabel="d",
                 ylabel=ylabel,
                 framestyle=:box,
                 legend=:outerright)
        if loglog
            plot!(p; xscale=:log10, yscale=:log10)
        end
        for slot in 1:max_slots
            y = y_slots[slot]
            mask = isfinite.(y)
            if loglog
                mask .&= (y .> 0)
            end
            if any(mask)
                plot!(p, x[mask], y[mask], marker=:circle, lw=2, label="bond $(slot)")
            end
        end
        mask_mean = isfinite.(y_mean)
        if loglog
            mask_mean .&= (y_mean .> 0)
        end
        if any(mask_mean)
            plot!(p, x[mask_mean], y_mean[mask_mean], marker=:diamond, lw=2.8, color=:black, label="mean")
        end
        if fit_mean_loglog && loglog
            fit = fit_loglog_powerlaw(x, y_mean)
            if !isnothing(fit)
                x_fit = fit.x
                y_fit = 10 .^ (fit.intercept .+ fit.slope .* log10.(x_fit))
                plot!(p, x_fit, y_fit, lw=2.2, color=:gray20, linestyle=:dashdot,
                      label=@sprintf("mean fit slope=%.3f (R²=%.3f)", fit.slope, fit.r2))
            end
        end
        out_path = joinpath(analysis_dir, file_name)
        savefig(p, out_path)
        println("Saved ", out_path)
    end

    save_series_plot("02_j2_vs_d_linear.png",
                     "⟨J²⟩ vs d",
                     "⟨J²⟩",
                     y_j2_slots,
                     y_j2_mean;
                     loglog=false)
    save_series_plot("03_j2_vs_d_loglog.png",
                     "⟨J²⟩ vs d (log-log)",
                     "⟨J²⟩",
                     y_j2_slots,
                     y_j2_mean;
                     loglog=true)
    save_series_plot("04_j2_minus_baseline_vs_d_linear.png",
                     "⟨J²⟩-baseline vs d, $(baseline_title)",
                     "⟨J²⟩ - baseline",
                     y_j2_slots_shifted,
                     y_j2_mean_shifted;
                     loglog=false)
    save_series_plot("05_j2_minus_baseline_vs_d_loglog.png",
                     "⟨J²⟩-baseline vs d (log-log), $(baseline_title)",
                     "⟨J²⟩ - baseline",
                     y_j2_slots_shifted,
                     y_j2_mean_shifted;
                     loglog=true,
                     fit_mean_loglog=true)

    summary_file = joinpath(analysis_dir, "summary_vs_d.csv")
    open(summary_file, "w") do io
        println(io, "d,var_mean,j2_mean,baseline_mean,j2_minus_baseline")
        for (i, d) in enumerate(d_values)
            println(io, @sprintf("%d,%.10g,%.10g,%.10g,%.10g", d, y_var_mean[i], y_j2_mean[i], y_baseline_mean[i], y_j2_mean_shifted[i]))
        end
    end
    println("Saved ", summary_file)
end

function main()
    args = parse_commandline()
    recursive = get(args, "recursive", false)
    include_abs_mean = get(args, "include_abs_mean_in_spatial_f_plot", false)
    keep_diag = get(args, "keep_diagonal_in_multiforce_cut", false)
    skip_per_state = get(args, "skip_per_state_sweep", false)
    corr_model_c = Float64(get(args, "corr_model_c", 0.0))
    mode_raw = String(args["mode"])
    mode = mode_raw == "run_id" ? "two_force_d" : mode_raw
    run_id = strip(String(get(args, "run_id", "")))
    run_result_mode = String(get(args, "run_result_mode", "auto"))
    cluster_results_root = String(get(args, "cluster_results_root", "cluster_results"))
    out_dir_arg = String(args["out_dir"])
    out_dir_default = "results_figures/fitting"

    inputs_raw = get(args, "inputs", Any[])
    inputs = String.(inputs_raw isa AbstractVector ? inputs_raw : Any[])
    default_glob = String(args["glob"])
    files = String[]
    resolved_run_mode = ""
    resolved_run_dir = ""
    out_dir = out_dir_arg

    if mode_raw == "run_id" && isempty(run_id)
        error("--mode run_id requires --run_id.")
    end

    if !isempty(run_id)
        files, run_out_dir, resolved_run_mode, resolved_run_dir = collect_files_from_run_id(
            run_id,
            cluster_results_root,
            run_result_mode,
        )
        if out_dir_arg == out_dir_default
            out_dir = run_out_dir
        end
        println("Using run_id='", run_id, "' (mode=", resolved_run_mode, ") from ", resolved_run_dir)
    end

    if !isempty(inputs)
        file_inputs = collect_input_files(inputs, default_glob; recursive=recursive)
        if isempty(files)
            files = file_inputs
        else
            files = sort(unique(vcat(files, file_inputs)))
        end
    end

    println("Found ", length(files), " matching file(s)")
    println("Output directory: ", out_dir)

    if isempty(files)
        if isempty(run_id) && isempty(inputs)
            error("No inputs were provided. Pass input file/dir/glob paths or use --run_id.")
        end
        return
    end

    if !skip_per_state
        for saved_state in files
            try
                println("Processing sweep plots for ", saved_state)
                state, param, potential = load_state_bundle(saved_state)
                ensure_state_potential!(state, potential)
                if is_common_diffusive_state(state, param)
                    save_diffusive_sweep_components(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        include_abs_mean_in_spatial_f_plot=include_abs_mean,
                        keep_diagonal_in_multiforce_cut=keep_diag,
                        corr_model_c=corr_model_c,
                    )
                else
                    save_legacy_sweep_plot(saved_state, state, param, out_dir)
                end
            catch e
                println("Failed to export sweep plots for ", saved_state, ": ", e)
            end
        end
    end

    if mode == "two_force_d"
        analyze_two_force_d(
            files,
            out_dir;
            baseline_rho_factor=Float64(args["baseline_rho_factor"]),
            baseline_j2_override=Float64(args["baseline_j2"]),
            smooth_diagonal=!keep_diag,
        )
    elseif mode != "single"
        error("Unsupported mode: $mode_raw. Use --mode single, --mode two_force_d, or --mode run_id.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
