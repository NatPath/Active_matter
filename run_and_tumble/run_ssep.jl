using Distributed
using Dates
using JLD2
using YAML
using ArgParse
using Random
if haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
    # Under WSLg, Qt/GR can pick Wayland and then fail EGL initialization.
    # Force the stable XWayland path before Plots/GR initialize.
    ENV["QT_QPA_PLATFORM"] = "xcb"
end
using Plots
using Statistics

include("modules_ssep.jl")
include("save_utils.jl")

using .Potentials
using .FPSSEP
using .PlotUtils
using .SaveUtils

@everywhere begin
    using JLD2
    using YAML
    using Random
    using Statistics
    if haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
        ENV["QT_QPA_PLATFORM"] = "xcb"
    end
    using Plots
    if myid() != 1
        include("modules_ssep.jl")
        using .Potentials
        using .FPSSEP
        using .PlotUtils
    end
end

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--config"
            help = "Configuration file path"
            required = false
        "--continue"
            help = "Path to a saved SSEP state to continue from"
            required = false
        "--continue_sweeps"
            help = "Number of sweeps to continue for (overrides config)"
            arg_type = Int
            required = false
        "--initial_state"
            help = "Path to a saved SSEP state to use as a fresh initial condition"
            required = false
        "--num_runs"
            help = "Number of independent runs to execute before aggregation"
            arg_type = Int
            default = 1
        "--save_tag"
            help = "Optional save tag override"
            arg_type = String
            required = false
        "--aggregate_state_list"
            help = "Path to a newline-delimited list of saved states to aggregate"
            arg_type = String
            required = false
    end
    return parse_args(settings)
end

@everywhere function get_default_params()
    return Dict(
        "dim_num" => 1,
        "L" => 64,
        "ρ₀" => 0.25,
        "D" => 1.0,
        "T" => 1.0,
        "γ" => 0.0,
        "n_sweeps" => 100000,
        "warmup_sweeps" => 10000,
        "cluster_mode" => false,
        "description" => "",
        "potential_type" => "zero",
        "fluctuation_type" => "no-fluctuation",
        "potential_magnitude" => 0.0,
        "save_dir" => "saved_states_ssep",
        "show_times" => [j*10^i for i in 0:12 for j in 1:9],
        "save_times" => Int[],
        "forcing_type" => "center_bond_x",
        "forcing_types" => String[],
        "forcing_magnitude" => 1.0,
        "forcing_magnitudes" => Float64[],
        "ffr" => 1.0,
        "ffrs" => Float64[],
        "forcing_direction_flags" => Bool[],
        "forcing_rate_scheme" => "legacy_penalty",
        "bond_pass_count_mode" => "all_forcing_bonds",
        "ic" => "random",
        "plot_final" => true,
        "save_final_plot" => true,
        "full_corr_tensor" => false,
    )
end

@everywhere function load_params(args)
    defaults = get_default_params()
    params = if haskey(args, "config") && !isnothing(args["config"])
        YAML.load_file(args["config"])
    else
        deepcopy(defaults)
    end
    return params, defaults
end

@everywhere function to_int(value, key_name::String)
    value isa Number || error("$key_name must be numeric.")
    return Int(round(Float64(value)))
end

@everywhere function to_float(value, key_name::String)
    value isa Number || error("$key_name must be numeric.")
    return Float64(value)
end

@everywhere function to_bool(value, key_name::String)
    if value isa Bool
        return value
    elseif value isa Number
        return value != 0
    elseif value isa AbstractString
        lowered = lowercase(strip(value))
        if lowered in ("true", "t", "1", "yes", "y")
            return true
        elseif lowered in ("false", "f", "0", "no", "n")
            return false
        end
    end
    error("$key_name must be boolean-like.")
end

@everywhere function to_string_vector(value, key_name::String)
    if value isa AbstractString
        return [String(value)]
    elseif value isa AbstractVector
        return [String(v) for v in value]
    end
    error("$key_name must be a string or list of strings.")
end

@everywhere function to_float_vector(value, key_name::String)
    if value isa Number
        return [Float64(value)]
    elseif value isa AbstractVector
        return [Float64(v) for v in value]
    end
    error("$key_name must be numeric or a list of numerics.")
end

@everywhere function to_int_vector(value, key_name::String)
    if value isa Number
        return [Int(round(Float64(value)))]
    elseif value isa AbstractVector
        return [Int(round(Float64(v))) for v in value]
    end
    error("$key_name must be numeric or a list of numerics.")
end

@everywhere function to_bool_vector(value, key_name::String)
    if value isa Bool || value isa Number || value isa AbstractString
        return [to_bool(value, key_name)]
    elseif value isa AbstractVector
        return [to_bool(v, key_name) for v in value]
    end
    error("$key_name must be boolean-like or a list of boolean-like values.")
end

@everywhere function expand_to_length(values::Vector{T}, n::Int, key_name::String) where {T}
    if length(values) == n
        return values
    elseif length(values) == 1 && n > 1
        return fill(values[1], n)
    end
    error("$key_name must have length 1 or length $n.")
end

@everywhere function get_warmup_sweeps(params, defaults)
    return max(to_int(get(params, "warmup_sweeps", defaults["warmup_sweeps"]), "warmup_sweeps"), 0)
end

@everywhere function get_cluster_mode(params, defaults)
    return to_bool(get(params, "cluster_mode", defaults["cluster_mode"]), "cluster_mode")
end

@everywhere function get_description(params, defaults)
    return String(strip(String(get(params, "description", defaults["description"]))))
end

@everywhere function get_n_sweeps(params, defaults, args)
    if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
        return Int(args["continue_sweeps"])
    end
    return to_int(get(params, "n_sweeps", defaults["n_sweeps"]), "n_sweeps")
end

@everywhere function get_show_times(params, defaults, warmup_sweeps::Int; has_explicit_show_times::Bool=false)
    raw_show_times = if has_explicit_show_times
        params["show_times"]
    else
        to_int_vector(defaults["show_times"], "show_times") .+ warmup_sweeps
    end
    return sort(unique(to_int_vector(raw_show_times, "show_times")))
end

function configure_interactive_plots!(cluster_mode::Bool, params, defaults)
    return nothing
end

@everywhere function get_save_times(params, defaults)
    return sort(unique(to_int_vector(get(params, "save_times", defaults["save_times"]), "save_times")))
end

@everywhere function param_ffr_input(param)
    if hasfield(typeof(param), :ffr)
        return getfield(param, :ffr)
    elseif hasfield(typeof(param), :ffrs)
        return getfield(param, :ffrs)
    end
    return 0.0
end

@everywhere function maybe_override_forcing_rate_scheme(param, args, params)
    has_explicit_config = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "forcing_rate_scheme")
    if !has_explicit_config
        return param
    end
    return FPSSEP.setParam(param.γ, param.dims, param.ρ₀, param.D,
                           param.potential_type, param.fluctuation_type, param.potential_magnitude,
                           param_ffr_input(param);
                           forcing_rate_scheme=String(params["forcing_rate_scheme"]))
end

@everywhere function forcing_bond_indices_from_type(forcing_type::AbstractString, dim_num::Int, L::Int)
    if forcing_type == "center_bond_x"
        if dim_num == 1
            return ([L ÷ 2], [L ÷ 2 + 1])
        elseif dim_num == 2
            return ([L ÷ 2, L ÷ 2], [L ÷ 2 + 1, L ÷ 2])
        end
    elseif forcing_type == "center_bond_y"
        if dim_num == 1
            return ([L ÷ 2], [L ÷ 2 + 1])
        elseif dim_num == 2
            return ([L ÷ 2, L ÷ 2], [L ÷ 2, L ÷ 2 + 1])
        end
    end
    error("Unsupported forcing_type: $forcing_type")
end

@everywhere function parse_forcing_bond_pair(raw_pair, dim_num::Int, L::Int)
    if !(raw_pair isa AbstractVector) || length(raw_pair) != 2
        error("Each forcing_bond_pairs entry must contain exactly two endpoints.")
    end
    if dim_num == 1
        return ([mod1(Int(raw_pair[1]), L)], [mod1(Int(raw_pair[2]), L)])
    end

    first_endpoint = raw_pair[1]
    second_endpoint = raw_pair[2]
    if !(first_endpoint isa AbstractVector) || !(second_endpoint isa AbstractVector)
        error("For dim_num=$dim_num, forcing_bond_pairs entries must look like [[...],[...]].")
    end
    length(first_endpoint) == dim_num || error("First endpoint must have $dim_num coordinates.")
    length(second_endpoint) == dim_num || error("Second endpoint must have $dim_num coordinates.")

    b1 = [mod1(Int(coord), L) for coord in first_endpoint]
    b2 = [mod1(Int(coord), L) for coord in second_endpoint]
    return (b1, b2)
end

@everywhere function infer_requested_force_count(params, defaults)
    candidate_lengths = Int[]

    if haskey(params, "forcing_magnitudes")
        push!(candidate_lengths, length(to_float_vector(params["forcing_magnitudes"], "forcing_magnitudes")))
    elseif haskey(params, "forcing_magnitude")
        push!(candidate_lengths, 1)
    end

    if haskey(params, "ffrs")
        push!(candidate_lengths, length(to_float_vector(params["ffrs"], "ffrs")))
    elseif haskey(params, "ffr")
        push!(candidate_lengths, 1)
    end

    if haskey(params, "forcing_direction_flags")
        push!(candidate_lengths, length(to_bool_vector(params["forcing_direction_flags"], "forcing_direction_flags")))
    end

    return isempty(candidate_lengths) ? 1 : maximum(candidate_lengths)
end

@everywhere function inferred_bond_indices_list(params, defaults, dim_num::Int, L::Int)
    requested_force_count = infer_requested_force_count(params, defaults)
    forcing_type = String(get(params, "forcing_type", defaults["forcing_type"]))

    if requested_force_count == 1
        return [forcing_bond_indices_from_type(forcing_type, dim_num, L)]
    end

    if requested_force_count == 2 && (haskey(params, "distance_between_forces") || haskey(params, "forcing_distance_d"))
        dim_num == 1 || error("distance_between_forces is currently supported only for dim_num=1.")
        if haskey(params, "distance_between_forces") && haskey(params, "forcing_distance_d")
            error("Use either distance_between_forces or forcing_distance_d, not both.")
        end

        d_raw = haskey(params, "distance_between_forces") ? params["distance_between_forces"] : params["forcing_distance_d"]
        d_raw isa Number || error("distance_between_forces must be numeric.")
        d = mod(Int(round(Float64(d_raw))), L)

        left_site = if haskey(params, "forcing_base_site")
            base_site_raw = params["forcing_base_site"]
            base_site_raw isa Number || error("forcing_base_site must be numeric.")
            mod1(Int(round(Float64(base_site_raw))), L)
        else
            # Center the midpoint of the two bond midpoints on the middle bond of the system.
            # Here d is the gap between the right site of the left bond and the left site of the right bond.
            mod1((L ÷ 2) - fld(d + 1, 2), L)
        end
        right_site = mod1(left_site + d + 1, L)

        return [
            ([left_site], [mod1(left_site + 1, L)]),
            ([right_site], [mod1(right_site + 1, L)]),
        ]
    end

    error("Unable to infer forcing bond locations. Provide forcing_bond_pairs, explicit forcing_types, or for two 1D forces set distance_between_forces.")
end

@everywhere function build_forcings_and_ffrs(params, defaults, dim_num::Int, L::Int)
    bond_indices_list = Tuple{Vector{Int},Vector{Int}}[]

    if haskey(params, "forcing_bond_pairs")
        raw_pairs = params["forcing_bond_pairs"]
        raw_pairs isa AbstractVector || error("forcing_bond_pairs must be a list.")
        for raw_pair in raw_pairs
            push!(bond_indices_list, parse_forcing_bond_pair(raw_pair, dim_num, L))
        end
    elseif haskey(params, "forcing_types") || haskey(params, "forcing_type")
        forcing_types_raw = haskey(params, "forcing_types") ? params["forcing_types"] : get(params, "forcing_type", defaults["forcing_type"])
        forcing_types = to_string_vector(forcing_types_raw, "forcing_types")
        for forcing_type in forcing_types
            push!(bond_indices_list, forcing_bond_indices_from_type(forcing_type, dim_num, L))
        end
    else
        append!(bond_indices_list, inferred_bond_indices_list(params, defaults, dim_num, L))
    end

    n_forces = length(bond_indices_list)
    n_forces == 0 && return Potentials.BondForce[], Float64[]

    forcing_magnitudes_raw = haskey(params, "forcing_magnitudes") ? params["forcing_magnitudes"] : get(params, "forcing_magnitude", defaults["forcing_magnitude"])
    ffrs_raw = haskey(params, "ffrs") ? params["ffrs"] : get(params, "ffr", defaults["ffr"])
    direction_flags_raw = get(params, "forcing_direction_flags", defaults["forcing_direction_flags"])

    forcing_magnitudes = expand_to_length(to_float_vector(forcing_magnitudes_raw, "forcing_magnitudes"), n_forces, "forcing_magnitudes")
    ffrs = expand_to_length(to_float_vector(ffrs_raw, "ffrs"), n_forces, "ffrs")
    direction_flags = expand_to_length(to_bool_vector(direction_flags_raw, "forcing_direction_flags"), n_forces, "forcing_direction_flags")

    forcings = [Potentials.setBondForce(bond_indices_list[i], direction_flags[i], forcing_magnitudes[i]) for i in 1:n_forces]
    return forcings, ffrs
end

@everywhere function create_state_from_params(params, defaults, rng)
    dim_num = to_int(get(params, "dim_num", defaults["dim_num"]), "dim_num")
    L = to_int(get(params, "L", defaults["L"]), "L")
    ρ₀ = to_float(get(params, "ρ₀", defaults["ρ₀"]), "ρ₀")
    D = to_float(get(params, "D", defaults["D"]), "D")
    T = to_float(get(params, "T", defaults["T"]), "T")
    γ = to_float(get(params, "γ", defaults["γ"]), "γ")
    potential_type = String(get(params, "potential_type", defaults["potential_type"]))
    fluctuation_type = String(get(params, "fluctuation_type", defaults["fluctuation_type"]))
    potential_magnitude = to_float(get(params, "potential_magnitude", defaults["potential_magnitude"]), "potential_magnitude")
    forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
    bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))
    ic = String(get(params, "ic", defaults["ic"]))
    full_corr_tensor = to_bool(get(params, "full_corr_tensor", defaults["full_corr_tensor"]), "full_corr_tensor")

    dims = ntuple(_ -> L, dim_num)
    forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)

    param = FPSSEP.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs;
                            forcing_rate_scheme=forcing_rate_scheme)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=false)
    state = FPSSEP.setState(0, rng, param, T, potential, forcings; ic=ic, full_corr_tensor=full_corr_tensor, bond_pass_count_mode=bond_pass_count_mode)
    return state, param, ic
end

@everywhere function run_one_simulation_from_config(args, seed)
    rng = MersenneTwister(seed)
    params, defaults = load_params(args)
    state, param, _ = create_state_from_params(params, defaults, rng)
    n_sweeps = get_n_sweeps(params, defaults, args)
    warmup_sweeps = get_warmup_sweeps(params, defaults)
    description = get_description(params, defaults)

    run_simulation!(state, param, n_sweeps, rng;
                    show_times=Int[],
                    save_times=Int[],
                    plot_flag=false,
                    plot_label=description,
                    save_description=description,
                    warmup_sweeps=warmup_sweeps,
                    show_progress=false)
    return state, param
end

@everywhere function run_one_simulation_from_state(param, state, seed, n_sweeps; warmup_sweeps::Int=0, description::String="")
    rng = MersenneTwister(seed)
    run_simulation!(state, param, n_sweeps, rng;
                    show_times=Int[],
                    save_times=Int[],
                    plot_flag=false,
                    plot_label=description,
                    save_description=description,
                    warmup_sweeps=warmup_sweeps,
                    show_progress=false)
    return state, param
end

function state_sample_weight(state)
    if hasfield(typeof(state), :bond_pass_stats)
        stats = getfield(state, :bond_pass_stats)
        if haskey(stats, FPSSEP.BOND_PASS_SAMPLE_COUNT_KEY) && !isempty(stats[FPSSEP.BOND_PASS_SAMPLE_COUNT_KEY])
            return Float64(stats[FPSSEP.BOND_PASS_SAMPLE_COUNT_KEY][1])
        end
    end
    return Float64(max(getfield(state, :t), 0))
end

function copy_bond_pass_stats(state)
    stats = Dict{Symbol,Vector{Float64}}()
    if hasfield(typeof(state), :bond_pass_stats)
        for (key, value) in state.bond_pass_stats
            stats[key] = Float64.(value)
        end
    end
    return stats
end

function average_bond_pass_stats(states, weights)
    stats_list = [copy_bond_pass_stats(state) for state in states]
    nonempty_indices = [i for i in eachindex(stats_list) if !isempty(stats_list[i])]
    isempty(nonempty_indices) && return Dict{Symbol,Vector{Float64}}()

    shared_keys = Set(keys(stats_list[nonempty_indices[1]]))
    for idx in nonempty_indices[2:end]
        intersect!(shared_keys, Set(keys(stats_list[idx])))
    end

    averaged = Dict{Symbol,Vector{Float64}}()
    for key in shared_keys
        vectors = [stats_list[idx][key] for idx in nonempty_indices]
        if key == FPSSEP.BOND_PASS_SAMPLE_COUNT_KEY || key == FPSSEP.BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY
            averaged[key] = [sum(!isempty(vec) ? Float64(vec[1]) : 0.0 for vec in vectors)]
        elseif key == FPSSEP.BOND_PASS_TRACK_MASK_KEY
            averaged[key] = Float64.(vectors[1])
        else
            total_weight = sum(weights[idx] for idx in nonempty_indices)
            if total_weight <= 0
                averaged[key] = mean(reduce(hcat, vectors), dims=2)[:, 1]
            else
                weighted_sum = zeros(Float64, size(vectors[1]))
                for idx in nonempty_indices
                    weighted_sum .+= stats_list[idx][key] .* weights[idx]
                end
                averaged[key] = weighted_sum ./ total_weight
            end
        end
    end
    return averaged
end

function load_saved_state(path; reset_statistics::Bool=false)
    println("Loading state from $path")
    @load path state param potential
    state.potential = potential
    FPSSEP.validate_exclusion_state(state)
    if reset_statistics
        FPSSEP.reset_statistics!(state)
    end
    return state, param
end

function read_state_list(path::String)
    files = String[]
    for raw_line in eachline(path)
        line = strip(raw_line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        push!(files, line)
    end
    return files
end

function aggregate_states(states, result_param, save_dir; save_tag=nothing, description="", relaxed_ic::Bool=false)
    isempty(states) && error("No states were provided for aggregation.")
    weights = [state_sample_weight(state) for state in states]
    total_weight = sum(weights)
    total_weight > 0 || error("Cannot aggregate states with zero effective sample weight.")

    first_state = states[1]
    avg_density = zeros(Float64, size(first_state.ρ_avg))
    for (state, weight) in zip(states, weights)
        size(state.ρ_avg) == size(avg_density) || error("All states must have matching density shapes.")
        avg_density .+= state.ρ_avg .* weight
    end
    avg_density ./= total_weight

    shared_corr_keys = Set(keys(first_state.ρ_matrix_avg_cuts))
    for state in states[2:end]
        intersect!(shared_corr_keys, Set(keys(state.ρ_matrix_avg_cuts)))
    end
    averaged_cuts = Dict{Symbol,AbstractArray{Float64}}()
    for key in shared_corr_keys
        weighted_sum = zeros(Float64, size(first_state.ρ_matrix_avg_cuts[key]))
        for (state, weight) in zip(states, weights)
            size(state.ρ_matrix_avg_cuts[key]) == size(weighted_sum) || error("All states must have matching correlation shapes for key $key.")
            weighted_sum .+= state.ρ_matrix_avg_cuts[key] .* weight
        end
        averaged_cuts[key] = weighted_sum ./ total_weight
    end

    averaged_stats = average_bond_pass_stats(states, weights)
    aggregated_state = FPSSEP.setDummyState(first_state, avg_density, averaged_cuts, Int(round(total_weight)), averaged_stats)

    resolved_tag = isnothing(save_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : String(save_tag)
    if !startswith(resolved_tag, "aggregated_")
        resolved_tag = "aggregated_" * resolved_tag
    end
    filename = save_state(aggregated_state, result_param, save_dir; tag=resolved_tag, ic="aggregated", relaxed_ic=relaxed_ic, description=description)
    return aggregated_state, filename
end

function aggregate_state_list_and_save(args)
    params, defaults = load_params(args)
    description = get_description(params, defaults)
    save_dir = String(get(params, "save_dir", defaults["save_dir"]))
    state_files = read_state_list(String(args["aggregate_state_list"]))
    isempty(state_files) && error("No state files found in $(args["aggregate_state_list"]).")

    states = Any[]
    first_param = nothing
    for state_file in state_files
        state, param = load_saved_state(state_file; reset_statistics=false)
        push!(states, state)
        if isnothing(first_param)
            first_param = param
        end
    end

    aggregated_state, filename = aggregate_states(states, first_param, save_dir;
                                                  save_tag=get(args, "save_tag", nothing),
                                                  description=description,
                                                  relaxed_ic=false)
    return aggregated_state, first_param, filename, params, defaults
end

function maybe_render_summary(state, param, params, defaults, filename, description)
    cluster_mode = get_cluster_mode(params, defaults)
    plot_final = to_bool(get(params, "plot_final", defaults["plot_final"]), "plot_final")
    save_final_plot = to_bool(get(params, "save_final_plot", defaults["save_final_plot"]), "save_final_plot")
    if !plot_final && !save_final_plot
        return nothing
    end

    plot_obj = PlotUtils.plot_average_density_and_correlation(state, param; label=description)
    if plot_final && !cluster_mode
        configure_interactive_plots!(cluster_mode, params, defaults)
        PlotUtils.present_plot!(plot_obj)
    end
    if save_final_plot
        plot_dir = joinpath(dirname(filename), "plots")
        mkpath(plot_dir)
        plot_name = splitext(basename(filename))[1] * "_summary.png"
        plot_path = joinpath(plot_dir, plot_name)
        savefig(plot_obj, plot_path)
        println("Saved summary plot to $plot_path")
    end
    return nothing
end

function run_single_simulation(args)
    params, defaults = load_params(args)
    description = get_description(params, defaults)
    save_dir = String(get(params, "save_dir", defaults["save_dir"]))
    warmup_sweeps = get_warmup_sweeps(params, defaults)
    n_sweeps = get_n_sweeps(params, defaults, args)
    has_explicit_show_times = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "show_times")
    show_times = get_show_times(params, defaults, warmup_sweeps; has_explicit_show_times=has_explicit_show_times)
    save_times = get_save_times(params, defaults)
    cluster_mode = get_cluster_mode(params, defaults)
    configure_interactive_plots!(cluster_mode, params, defaults)

    using_initial_state = haskey(args, "initial_state") && !isnothing(args["initial_state"])
    continuing = haskey(args, "continue") && !isnothing(args["continue"])

    rng = MersenneTwister(rand(1:2^30))
    if continuing
        state, param = load_saved_state(String(args["continue"]); reset_statistics=false)
        ic = "continued"
    elseif using_initial_state
        state, param = load_saved_state(String(args["initial_state"]); reset_statistics=true)
        ic = "initial_state"
    else
        state, param, ic = create_state_from_params(params, defaults, rng)
    end
    param = maybe_override_forcing_rate_scheme(param, args, params)

    final_state_saved = Ref(false)
    atexit() do
        if final_state_saved[]
            return
        end
        try
            save_state(state, param, save_dir; ic=ic, relaxed_ic=using_initial_state, description=description)
        catch err
            println("Failed to save final SSEP state during shutdown: $err")
        end
    end

    run_simulation!(state, param, n_sweeps, rng;
                    show_times=cluster_mode ? Int[] : show_times,
                    save_times=save_times,
                    save_dir=save_dir,
                    plot_flag=!cluster_mode,
                    plot_label=description,
                    save_description=description,
                    warmup_sweeps=warmup_sweeps,
                    show_progress=!cluster_mode,
                    relaxed_ic=using_initial_state)

    filename = save_state(state, param, save_dir;
                          tag=get(args, "save_tag", nothing),
                          ic=ic,
                          relaxed_ic=using_initial_state,
                          description=description)
    final_state_saved[] = true
    maybe_render_summary(state, param, params, defaults, filename, description)
    return state, param, filename
end

function run_multiple_simulations(args)
    params, defaults = load_params(args)
    description = get_description(params, defaults)
    save_dir = String(get(params, "save_dir", defaults["save_dir"]))
    warmup_sweeps = get_warmup_sweeps(params, defaults)
    n_sweeps = get_n_sweeps(params, defaults, args)
    num_runs = Int(get(args, "num_runs", 1))
    seeds = rand(1:2^30, num_runs)

    using_initial_state = haskey(args, "initial_state") && !isnothing(args["initial_state"])
    continuing = haskey(args, "continue") && !isnothing(args["continue"])

    if continuing
        base_state, base_param = load_saved_state(String(args["continue"]); reset_statistics=false)
        base_param = maybe_override_forcing_rate_scheme(base_param, args, params)
        results = pmap(seed -> begin
            state = deepcopy(base_state)
            param = deepcopy(base_param)
            run_one_simulation_from_state(param, state, seed, n_sweeps; warmup_sweeps=warmup_sweeps, description=description)
        end, seeds)
    elseif using_initial_state
        base_state, base_param = load_saved_state(String(args["initial_state"]); reset_statistics=true)
        base_param = maybe_override_forcing_rate_scheme(base_param, args, params)
        results = pmap(seed -> begin
            state = deepcopy(base_state)
            param = deepcopy(base_param)
            run_one_simulation_from_state(param, state, seed, n_sweeps; warmup_sweeps=warmup_sweeps, description=description)
        end, seeds)
    else
        results = pmap(seed -> run_one_simulation_from_config(args, seed), seeds)
    end

    states = [result[1] for result in results]
    first_param = results[1][2]
    aggregated_state, filename = aggregate_states(states, first_param, save_dir;
                                                  save_tag=get(args, "save_tag", nothing),
                                                  description=description,
                                                  relaxed_ic=using_initial_state)
    maybe_render_summary(aggregated_state, first_param, params, defaults, filename, description)
    return aggregated_state, first_param, filename
end

function main()
    args = parse_commandline()

    if haskey(args, "aggregate_state_list") && !isnothing(args["aggregate_state_list"])
        state, param, filename, params, defaults = aggregate_state_list_and_save(args)
        maybe_render_summary(state, param, params, defaults, filename, get_description(params, defaults))
        return
    end

    num_runs = Int(get(args, "num_runs", 1))
    if num_runs > 1
        run_multiple_simulations(args)
    else
        run_single_simulation(args)
    end
end

main()
