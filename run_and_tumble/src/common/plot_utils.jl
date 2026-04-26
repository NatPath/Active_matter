module PlotUtils
using Plots
using LsqFit
using Printf
using LinearAlgebra
export plot_sweep, plot_density, plot_data_colapse, plot_spatial_correlation, plot_average_density_and_correlation

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
const AGG_CONNECTED_FULL_EXACT_KEY = :agg_connected_corr_full_exact

# Exact (Float64) value of exp(-1)*(I0(1)+I1(1)), where I0/I1 are modified Bessel functions.
const SINGLE_ORIGIN_VARJ_BASELINE_FACTOR = 0.6736700229433489
# Exact (Float64) value of 0.5*exp(-0.5)*(I0(0.5)+I1(0.5)).
const SINGLE_ORIGIN_VARJ_BASELINE_FACTOR_SYMMETRIC_NORMALIZED = 0.4007280368170109

const LEGACY_FORCING_RATE_SCHEME = "legacy_penalty"
const SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME = "symmetric_normalized"
const INDEPENDENT_CORRELATION_BASELINE_MODEL = :independent
const SSEP_CORRELATION_BASELINE_MODEL = :ssep

@inline function directional_density_pair(state)
    if !hasfield(typeof(state), :ρ₊) || !hasfield(typeof(state), :ρ₋)
        return nothing
    end
    ρ₊ = getfield(state, :ρ₊)
    ρ₋ = getfield(state, :ρ₋)
    if isnothing(ρ₊) || isnothing(ρ₋)
        return nothing
    end
    return ρ₊, ρ₋
end

@inline function has_directional_densities(state)
    return directional_density_pair(state) !== nothing
end

@inline function normalized_forcing_rate_scheme_for_plot(param)
    if !hasfield(typeof(param), :forcing_rate_scheme)
        return LEGACY_FORCING_RATE_SCHEME
    end
    raw_scheme = lowercase(strip(String(getfield(param, :forcing_rate_scheme))))
    if raw_scheme in ("legacy_penalty", "legacy", "current")
        return LEGACY_FORCING_RATE_SCHEME
    elseif raw_scheme == SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
        return SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
    end
    return LEGACY_FORCING_RATE_SCHEME
end

@inline function single_origin_varj_baseline_factor(param)
    if normalized_forcing_rate_scheme_for_plot(param) == SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
        return SINGLE_ORIGIN_VARJ_BASELINE_FACTOR_SYMMETRIC_NORMALIZED
    end
    return SINGLE_ORIGIN_VARJ_BASELINE_FACTOR
end

@inline function single_origin_varj_baseline(param)
    return single_origin_varj_baseline_factor(param) * param.ρ₀
end

@inline function normalize_correlation_baseline_model(raw_model)
    model = lowercase(strip(String(raw_model)))
    if model in ("independent", "noninteracting", "non_interacting", "multinomial")
        return INDEPENDENT_CORRELATION_BASELINE_MODEL
    elseif model in ("ssep", "exclusion", "hardcore_exclusion", "hard_core_exclusion")
        return SSEP_CORRELATION_BASELINE_MODEL
    end
    throw(ArgumentError("Unsupported correlation baseline model: $raw_model"))
end

@inline function correlation_baseline_model(param)
    if hasfield(typeof(param), :correlation_baseline_model)
        return normalize_correlation_baseline_model(getfield(param, :correlation_baseline_model))
    end

    param_module_name = String(nameof(parentmodule(typeof(param))))
    if param_module_name == "FPSSEP"
        return SSEP_CORRELATION_BASELINE_MODEL
    elseif param_module_name in ("FP", "FPDiffusive")
        return INDEPENDENT_CORRELATION_BASELINE_MODEL
    end

    throw(ArgumentError("Could not infer correlation baseline model for parameter type $(typeof(param)). Add a correlation_baseline_model field or extend correlation_baseline_model(param)."))
end

@inline function independent_connected_correlation_fix_term(num_sites::Integer, num_particles::Integer)
    num_sites <= 0 && return 0.0
    return Float64(num_particles) / (Float64(num_sites)^2)
end

@inline function ssep_connected_correlation_fix_term(num_sites::Integer, num_particles::Integer)
    num_sites <= 1 && return 0.0
    return Float64(num_particles * (num_sites - num_particles)) / (Float64(num_sites)^2 * Float64(num_sites - 1))
end

@inline function connected_correlation_fix_term(param; statistics_model=nothing)
    num_sites = prod(param.dims)
    num_particles = Int(param.N)
    baseline_model = isnothing(statistics_model) ? correlation_baseline_model(param) : normalize_correlation_baseline_model(statistics_model)
    if baseline_model == INDEPENDENT_CORRELATION_BASELINE_MODEL
        return independent_connected_correlation_fix_term(num_sites, num_particles)
    elseif baseline_model == SSEP_CORRELATION_BASELINE_MODEL
        return ssep_connected_correlation_fix_term(num_sites, num_particles)
    end
    throw(ArgumentError("Unsupported connected-correlation statistics_model: $baseline_model"))
end

function connected_correlation_matrix_1d(state, param)
    if haskey(state.ρ_matrix_avg_cuts, :full)
        rho_avg = Float64.(state.ρ_avg)
        derived_connected = Float64.(state.ρ_matrix_avg_cuts[:full]) .- (rho_avg * transpose(rho_avg))
        if haskey(state.ρ_matrix_avg_cuts, AGG_CONNECTED_FULL_EXACT_KEY)
            stored_exact = Float64.(state.ρ_matrix_avg_cuts[AGG_CONNECTED_FULL_EXACT_KEY])
            if maximum(abs, stored_exact) <= 1e-12 && maximum(abs, derived_connected) > 1e-12
                return derived_connected
            end
            return stored_exact
        end
        return derived_connected
    end
    return Float64.(state.ρ_matrix_avg_cuts[AGG_CONNECTED_FULL_EXACT_KEY])
end

function finite_curve_ylims(curves...; pad_fraction::Float64=0.08, min_pad::Float64=1e-6)
    values = Float64[]
    for curve in curves
        for value in curve
            if isfinite(value)
                push!(values, Float64(value))
            end
        end
    end
    isempty(values) && return nothing
    ymin, ymax = extrema(values)
    span = ymax - ymin
    pad = span > 0 ? pad_fraction * span : max(abs(ymin), 1.0) * pad_fraction + min_pad
    return (ymin - pad, ymax + pad)
end

function remove_antisymmetric_part_reflection(matrix, x0)
    n = size(matrix, 1)
    indices = mod1.(2 * x0 .- (1:n), n)  # Compute the reflected indices for each row
    symmetric_matrix = (matrix .+ matrix[:, indices]) ./ 2  # Vectorized computation
    return symmetric_matrix
end
function remove_symmetric_part_reflection(matrix, x0)
    n = size(matrix, 1)
    indices = mod1.(2 * x0 .- (1:n), n)  # Compute the reflected indices for each row
    antisymmetric_matrix = (matrix .- matrix[:, indices]) ./ 2  # Vectorized computation
    return antisymmetric_matrix
end
function smooth_diagonal!(matrix)
    n = size(matrix, 1)
    for i in 1:n
        left_idx = i == 1 ? n : i - 1
        right_idx = i == n ? 1 : i + 1
        matrix[i, i] = (matrix[i, left_idx] + matrix[i, right_idx]) / 2
    end
    return matrix
end

function zero_middle(data, indices)
    result = copy(data)
    result[indices] .= 0
    return result
end

function positive_half(data, center)
    positive_range = center:length(data)
    positions = positive_range .- center .+ 1
    return positions, data[center:end]
end

function scatter_zero_marks!(p, indices; color=:red)
    scatter!(p, indices, zeros(length(indices)),
             color=color, markersize=6, markershape=:x,
             label="Zeroed middle")
end

function add_powerlaw_series!(p, distances, values, n; label="", add_reference=false, extra_reference_exponent=nothing, add_extra_reference=false)
    mask = (distances .> 0) .& isfinite.(values) .& (abs.(values) .> 0)
    if !any(mask)
        return false
    end
    d = distances[mask]
    v = abs.(values[mask])
    order = sortperm(d)
    d = d[order]
    v = v[order]
    plot!(p, d, v; label=label, lw=2)
    if add_reference || (add_extra_reference && extra_reference_exponent !== nothing)
        x_ref = [minimum(d), maximum(d)]
        ref_anchor = v[end]
        if add_reference
            y_ref = ref_anchor .* (x_ref ./ x_ref[end]).^(-n)
            plot!(p, x_ref, y_ref; label="slope -$n", linestyle=:dash, color=:black)
        end
        if add_extra_reference && extra_reference_exponent !== nothing
            y_ref_extra = ref_anchor .* (x_ref ./ x_ref[end]).^(-extra_reference_exponent)
            plot!(p, x_ref, y_ref_extra; label="slope -$(extra_reference_exponent)", linestyle=:dot, color=:gray)
        end
    end
    return true
end

function add_dimension_reference!(p, dim, x_vals, data_vals)
    mask = isfinite.(x_vals) .& isfinite.(data_vals)
    if !any(mask)
        return false
    end
    x_ref = x_vals[mask]
    y_ref = x_ref ./ (((x_ref .^ 2) .+ 1) .^ (dim .+ 1))
    max_ref = maximum(abs.(y_ref))
    max_data = maximum(abs.(data_vals[mask]))
    if max_ref == 0 || max_data == 0
        return false
    end
    scale = max_data / max_ref
    plot!(p, x_ref, y_ref .* scale; label="reference", color=:gray, linestyle=:dashdot, lw=2)
    return true
end

function data_collapse_maximum(x_vals, y_vals)
    finite_mask = isfinite.(x_vals) .& isfinite.(y_vals)
    any(finite_mask) || return nothing
    x_finite = Float64.(x_vals[finite_mask])
    y_finite = Float64.(y_vals[finite_mask])
    max_idx = argmax(y_finite)
    return (
        x = x_finite[max_idx],
        y = y_finite[max_idx],
        label = @sprintf("max(data) = %.6g at x/y = %.6g", y_finite[max_idx], x_finite[max_idx]),
    )
end

function apply_data_collapse_max_note!(p, base_title::AbstractString, x_vals, y_vals; mark_point::Bool=false)
    max_info = data_collapse_maximum(x_vals, y_vals)
    plot!(p; title=base_title)
    if !isnothing(max_info) && mark_point
        scatter!(p, [max_info.x], [max_info.y];
                 markershape=:star5,
                 markersize=7,
                 markercolor=:black,
                 markerstrokecolor=:black,
                 label=false)
    end
    return p
end

function data_collapse_axis_labels(n::Real)
    return "scaled separation Δ / offset", @sprintf("collapsed correlation C · offset^%.3g", Float64(n))
end

function compact_run_label(label::AbstractString; max_len::Int=18)
    cleaned = replace(basename(String(label)), ".jld2" => "")
    if length(cleaned) <= max_len
        return cleaned
    end
    head_len = max(6, min(10, max_len - 7))
    tail_len = max(4, max_len - head_len - 1)
    return string(first(cleaned, head_len), "…", last(cleaned, tail_len))
end

function collapse_curve_label(label::AbstractString, offset::Int, state_count::Int)
    offset_label = "offset $(offset)"
    if state_count <= 1
        return offset_label
    end
    return string(compact_run_label(label), ", ", offset_label)
end

function data_collapse_plot(title::AbstractString, xlabel::AbstractString, ylabel::AbstractString;
                            powerlaw::Bool=false, legend_position=:outerright)
    size = powerlaw ? (1180, 680) : (1240, 700)
    right_margin = powerlaw ? 26Plots.mm : 32Plots.mm
    return plot(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend_position,
        framestyle=:box,
        size=size,
        dpi=180,
        titlefontsize=12,
        guidefontsize=11,
        tickfontsize=9,
        legendfontsize=8,
        top_margin=10Plots.mm,
        bottom_margin=8Plots.mm,
        left_margin=8Plots.mm,
        right_margin=right_margin,
        gridalpha=0.18,
    )
end

function apply_robust_data_collapse_limits!(p, y_vals; trim_quantile::Float64=0.98, trim_threshold::Float64=1.75)
    finite_vals = Float64[]
    for value in y_vals
        isfinite(value) || continue
        push!(finite_vals, Float64(value))
    end
    length(finite_vals) < 24 && return p

    abs_vals = sort!(abs.(finite_vals))
    peak = abs_vals[end]
    peak > 0 || return p

    q_index = clamp(ceil(Int, trim_quantile * length(abs_vals)), 1, length(abs_vals))
    robust_peak = abs_vals[q_index]
    robust_peak > 0 || return p
    peak > trim_threshold * robust_peak || return p

    y_limit = 1.12 * robust_peak
    plot!(p; ylims=(-y_limit, y_limit))
    return p
end

function antisymmetric_with_smoothing(matrix, center)
    antisym = remove_symmetric_part_reflection(matrix, center)
    smooth_diagonal!(antisym)
    return antisym
end

function correlation_slice_x(state, param, y0, fix_term)
    slice2d_x = haskey(state.ρ_matrix_avg_cuts, :full) ? state.ρ_matrix_avg_cuts[:full][:, y0, :, y0] : state.ρ_matrix_avg_cuts[:x_cut]
    mean_vec = state.ρ_avg[:, y0]
    corr_mat2 = slice2d_x .- (mean_vec * transpose(mean_vec)) .+ fix_term
    smooth_diagonal!(corr_mat2)
    return corr_mat2
end

function correlation_slice_y(state, param, x0, fix_term)
    slice2d_y = haskey(state.ρ_matrix_avg_cuts, :full) ? state.ρ_matrix_avg_cuts[:full][x0, :, x0, :] : state.ρ_matrix_avg_cuts[:y_cut]
    mean_vec_y = state.ρ_avg[x0, :]
    corr_mat_y = slice2d_y .- (mean_vec_y * transpose(mean_vec_y)) .+ fix_term
    smooth_diagonal!(corr_mat_y)
    return corr_mat_y
end

function correlation_diag(state, param, fix_term)
    dims = param.dims
    if haskey(state.ρ_matrix_avg_cuts, :full)
        corr_diag = zeros(dims[1], dims[1])
        for i in 1:dims[1], j in 1:dims[1]
            corr_diag[i, j] = state.ρ_matrix_avg_cuts[:full][i, i, j, j] - state.ρ_avg[i, i] * state.ρ_avg[j, j] + fix_term
        end
    else
        diag_mean = diag(state.ρ_avg)
        corr_diag = state.ρ_matrix_avg_cuts[:diag_cut] .- (diag_mean * transpose(diag_mean)) .+ fix_term
    end
    smooth_diagonal!(corr_diag)
    return corr_diag
end

function get_forcing_list(state)
    if !hasfield(typeof(state), :forcing)
        return Any[]
    end
    if state.forcing isa AbstractVector
        return state.forcing
    end
    return [state.forcing]
end

function annotate_forcing_1d!(p, state)
    forcings = get_forcing_list(state)
    if isempty(forcings)
        return
    end
    colors = [:red, :orange, :yellow, :cyan, :magenta, :white]
    y_max = maximum(state.ρ_avg)
    for (idx, forcing) in enumerate(forcings)
        b1 = forcing.bond_indices[1][1]
        b2 = forcing.bond_indices[2][1]
        dir_label = forcing.direction_flag ? "f$(idx) ->" : "f$(idx) <-"
        color = colors[mod1(idx, length(colors))]
        y_pos = y_max * (0.9 - 0.08 * (idx - 1))
        vline!(p, [b1, b2], color=color, linestyle=:dash, label=false)
        start_x, end_x = forcing.direction_flag ? (b1, b2) : (b2, b1)
        plot!(p, [start_x, end_x], [y_pos, y_pos],
              arrow=:arrow, color=color, lw=3, label=false)
        annotate!(p, ((b1 + b2) / 2, y_pos * 1.05, text(dir_label, color, 8)))
    end
end

function annotate_forcing_2d!(p_current_density, state)
    forcings = get_forcing_list(state)
    if isempty(forcings)
        return
    end
    colors = [:white, :yellow, :orange, :cyan, :magenta]
    for (idx, forcing) in enumerate(forcings)
        b1 = forcing.bond_indices[1]
        b2 = forcing.bond_indices[2]
        if length(b1) == 2 && length(b2) == 2
            color = colors[mod1(idx, length(colors))]
            start = forcing.direction_flag ? b1 : b2
            dx = forcing.direction_flag ? (b2[1] - b1[1]) : (b1[1] - b2[1])
            dy = forcing.direction_flag ? (b2[2] - b1[2]) : (b1[2] - b2[2])
            quiver!(p_current_density,
                    [start[1]], [start[2]],
                    quiver=([dx], [dy]),
                    color=color,
                    lw=3,
                    arrow=:arrow,
                    label=false)
            annotate!(p_current_density,
                      start[1] + 0.3*dx,
                      start[2] + 0.3*dy,
                      text("bond force $(idx)", color, 8))
        end
    end
end

function force_bond_sites_2d(state, dims)
    forcings = get_forcing_list(state)
    bonds = Tuple{NTuple{2, Int}, NTuple{2, Int}}[]
    direction_flags = Bool[]
    magnitudes = Float64[]
    for force in forcings
        if length(force.bond_indices[1]) == 2 && length(force.bond_indices[2]) == 2
            b1 = (
                mod1(Int(force.bond_indices[1][1]), dims[1]),
                mod1(Int(force.bond_indices[1][2]), dims[2]),
            )
            b2 = (
                mod1(Int(force.bond_indices[2][1]), dims[1]),
                mod1(Int(force.bond_indices[2][2]), dims[2]),
            )
            push!(bonds, (b1, b2))
            push!(direction_flags, force.direction_flag)
            push!(magnitudes, Float64(force.magnitude))
        end
    end
    return bonds, direction_flags, magnitudes
end

function preferred_x_axis_reference_bond_2d(state, dims, x0::Int, y0::Int)
    bonds, _, _ = force_bond_sites_2d(state, dims)
    for (b1, b2) in bonds
        if b1[2] == y0 && b2[2] == y0
            return b1[1], b2[1]
        end
    end
    return x0, mod1(x0 + 1, dims[1])
end

function plot_instantaneous_state_2d(state, param)
    dims = param.dims
    occupancy = Float64.(state.ρ)
    p = heatmap(
        1:dims[1],
        1:dims[2],
        occupancy',
        title="Instantaneous particles and forces",
        xlabel="x",
        ylabel="y",
        aspect_ratio=1,
        colorbar=true,
        c=cgrad([:white, :black]),
        clims=(0.0, max(1.0, maximum(occupancy))),
    )
    if hasfield(typeof(state), :forcing)
        annotate_forcing_2d!(p, state)
    end
    return p
end

function plot_sweep(
    sweep,
    state,
    param;
    label="",
    plot_directional=false,
    remove_diagonal_for_multiforce_cuts=true,
    include_abs_mean_in_spatial_f_plot=false,
    prefer_multiforce_plots=true,
)
    dim_num = length(param.dims)
    if dim_num == 1
        if has_selected_site_cuts_for_plot(state)
            p = plot_selected_site_cuts_1d(state, param; sweep=sweep, label=label, include_instantaneous=true)
            present_plot!(p)
            return p
        end
        if prefer_multiforce_plots && (length(get_forcing_list(state)) > 1 || has_tracked_force_bonds_for_plot(state, param))
            return plot_sweep_1d_multiforce(
                sweep,
                state,
                param;
                label=label,
                plot_directional=plot_directional,
                remove_diagonal_for_cuts=remove_diagonal_for_multiforce_cuts,
                include_abs_mean_in_spatial_f_plot=include_abs_mean_in_spatial_f_plot,
            )
        end
        return plot_sweep_1d(sweep, state, param; label=label, plot_directional=plot_directional)
    elseif dim_num == 2
        return plot_sweep_2d(sweep, state, param; label=label, plot_directional=plot_directional)
    else
        throw(DomainError("Only 1D or 2D plotting supported"))
    end
end

function sweep_title_with_label(base_title::AbstractString, sweep, label)
    if isnothing(label)
        return "$(base_title) $(sweep)"
    end
    label_str = strip(String(label))
    if isempty(label_str)
        return "$(base_title) $(sweep)"
    end
    return "$(base_title) $(sweep) | $(label_str)"
end

function force_bond_sites_1d(state, L)
    forcings = get_forcing_list(state)
    bonds = Tuple{Int, Int}[]
    direction_flags = Bool[]
    magnitudes = Float64[]
    for force in forcings
        if length(force.bond_indices[1]) == 1 && length(force.bond_indices[2]) == 1
            b1 = mod1(force.bond_indices[1][1], L)
            b2 = mod1(force.bond_indices[2][1], L)
            push!(bonds, (b1, b2))
            push!(direction_flags, force.direction_flag)
            push!(magnitudes, force.magnitude)
        end
    end
    return bonds, direction_flags, magnitudes
end

function has_selected_site_cuts_for_plot(state)
    return haskey(state.ρ_matrix_avg_cuts, SELECTED_SITE_CUTS_KEY)
end

function selected_site_cut_metadata_for_plot(state, L::Int)
    has_selected_site_cuts_for_plot(state) || return Int[], 1, zeros(Float64, 0, L)
    pair_avgs = Float64.(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUTS_KEY])
    selected_sites = Int.(round.(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUT_SITES_KEY]))
    origin_site = haskey(state.ρ_matrix_avg_cuts, SELECTED_SITE_CUT_ORIGIN_SITE_KEY) ?
        mod1(Int(round(state.ρ_matrix_avg_cuts[SELECTED_SITE_CUT_ORIGIN_SITE_KEY][1])), L) :
        1
    return selected_sites, origin_site, pair_avgs
end

function smooth_periodic_cut_site!(cut::AbstractVector{<:Real}, site::Int)
    L = length(cut)
    ref_site = mod1(Int(site), L)
    left_site = mod1(ref_site - 1, L)
    right_site = mod1(ref_site + 1, L)
    cut[ref_site] = 0.5 * (cut[left_site] + cut[right_site])
    return cut
end

@inline function reflected_site_about_center_1d(site::Int, center_site::Int, L::Int)
    return mod1(2 * center_site - site, L)
end

function connected_selected_site_cut_1d(
    state,
    param,
    cut_site::Int,
    pair_avg_row::AbstractVector{<:Real};
    smooth_diagonal::Bool=true,
)
    L = length(pair_avg_row)
    ref_site = mod1(Int(cut_site), L)
    rho_avg = Float64.(state.ρ_avg)
    cut = Float64.(pair_avg_row) .- rho_avg[ref_site] .* rho_avg
    smooth_diagonal && smooth_periodic_cut_site!(cut, ref_site)
    return cut
end

function connected_correlation_cut_1d(
    state,
    param,
    site::Int;
    smooth_diagonal::Bool=true,
)
    L = param.dims[1]
    ref_site = mod1(Int(site), L)
    if has_selected_site_cuts_for_plot(state)
        selected_sites, _, pair_avgs = selected_site_cut_metadata_for_plot(state, L)
        selected_idx = findfirst(==(ref_site), selected_sites)
        if isnothing(selected_idx)
            available_sites = isempty(selected_sites) ? "none" : join(sort(selected_sites), ", ")
            throw(ArgumentError("Selected-site cut data for site $ref_site is not available. Available cut sites: [$available_sites]."))
        end
        return connected_selected_site_cut_1d(
            state,
            param,
            ref_site,
            vec(pair_avgs[selected_idx, :]);
            smooth_diagonal=smooth_diagonal,
        )
    end
    corr_mat = connected_correlation_matrix_1d(state, param)
    return site_cut_1d(corr_mat, ref_site; smooth_diagonal=smooth_diagonal)
end

function reflected_cut_1d_about_center(
    cut::AbstractVector{<:Real},
    center_site::Int,
)
    L = length(cut)
    reflected = Vector{Float64}(undef, L)
    @inbounds for site in 1:L
        reflected[site] = Float64(cut[reflected_site_about_center_1d(site, center_site, L)])
    end
    return reflected
end

function reflection_pair_averaged_cut_1d(
    state,
    param,
    site::Int;
    origin_site::Union{Nothing,Int}=nothing,
    smooth_diagonal::Bool=true,
)
    L = param.dims[1]
    center_site = if isnothing(origin_site)
        if has_selected_site_cuts_for_plot(state)
            _, inferred_origin_site, _ = selected_site_cut_metadata_for_plot(state, L)
            inferred_origin_site
        else
            max(1, div(L, 2))
        end
    else
        mod1(Int(origin_site), L)
    end

    ref_site = mod1(Int(site), L)
    cut = connected_correlation_cut_1d(state, param, ref_site; smooth_diagonal=false)
    partner_site = reflected_site_about_center_1d(ref_site, center_site, L)
    pair_used = false

    if partner_site != ref_site
        partner_available = true
        if has_selected_site_cuts_for_plot(state)
            selected_sites, _, _ = selected_site_cut_metadata_for_plot(state, L)
            partner_available = partner_site in selected_sites
        end

        if partner_available
            partner_cut = connected_correlation_cut_1d(state, param, partner_site; smooth_diagonal=false)
            cut = 0.5 .* (cut .+ reflected_cut_1d_about_center(partner_cut, center_site))
            pair_used = true
        end
    end

    smooth_diagonal && smooth_periodic_cut_site!(cut, ref_site)
    return cut, pair_used
end

function reflection_decomposed_cut_1d(
    cut::AbstractVector{<:Real},
    center_site::Int,
)
    L = length(cut)
    cut_vals = Float64.(cut)
    reflected = similar(cut_vals)
    @inbounds for site in 1:L
        reflected[site] = cut_vals[reflected_site_about_center_1d(site, center_site, L)]
    end
    sym = 0.5 .* (cut_vals .+ reflected)
    antisym = 0.5 .* (cut_vals .- reflected)
    return sym, antisym
end

function centered_selected_site_cut_1d(
    state,
    param,
    cut_site::Int,
    pair_avg_row::AbstractVector{<:Real},
    origin_site::Int;
    smooth_diagonal::Bool=true,
)
    L = length(pair_avg_row)
    cut = connected_selected_site_cut_1d(state, param, cut_site, pair_avg_row; smooth_diagonal=smooth_diagonal)
    x_rel, perm = centered_periodic_axis(L, origin_site)
    return x_rel, cut[perm]
end

function centered_reflection_pair_averaged_cut_1d(
    state,
    param,
    cut_site::Int,
    origin_site::Int;
    smooth_diagonal::Bool=true,
)
    L = param.dims[1]
    cut, pair_used = reflection_pair_averaged_cut_1d(
        state,
        param,
        cut_site;
        origin_site=origin_site,
        smooth_diagonal=smooth_diagonal,
    )
    x_rel, perm = centered_periodic_axis(L, origin_site)
    return x_rel, cut[perm], pair_used
end

function plot_selected_site_cuts_1d(
    state,
    param;
    sweep=state.t,
    label="",
    include_instantaneous::Bool=true,
)
    L = param.dims[1]
    x_range = 1:L
    colors = [:red, :orange, :yellow, :cyan, :magenta, :green, :blue]
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    bonds, direction_flags, _ = force_bond_sites_1d(state, L)
    selected_sites, origin_site, pair_avgs = selected_site_cut_metadata_for_plot(state, L)

    p_avg_density = plot(x_range, Float64.(state.ρ_avg),
                         title="Time-averaged density",
                         xlabel="Site",
                         ylabel="⟨ρ(x)⟩",
                         lw=2.5,
                         color=:black,
                         legend=false,
                         framestyle=:box,
                         grid=:y)
    if hasfield(typeof(state), :forcing)
        annotate_forcing_1d!(p_avg_density, state)
    end
    for (cut_idx, cut_site) in enumerate(selected_sites)
        vline!(p_avg_density, [cut_site], color=colors[mod1(cut_idx, length(colors))], linestyle=:dash, alpha=0.4, lw=1.3, label=false)
    end

    p_inst_density = if hasfield(typeof(state), :particles) && !isnothing(getfield(state, :particles))
        particles = getfield(state, :particles)
        positions = sort!(Int[mod1(Int(getfield(particle, :position)[1]), L) for particle in particles])
        site_counts = zeros(Int, L)
        stack_levels = Vector{Int}(undef, length(positions))
        @inbounds for idx in eachindex(positions)
            site = positions[idx]
            site_counts[site] += 1
            stack_levels[idx] = site_counts[site]
        end
        scatter(
            positions,
            stack_levels;
            title="Instantaneous particle positions",
            xlabel="Site",
            ylabel="Particle stack",
            xlim=(1, L),
            ylim=(0, isempty(stack_levels) ? 1 : maximum(stack_levels) + 1),
            legend=false,
            color=:black,
            markersize=1.8,
            markerstrokewidth=0,
            alpha=0.5,
            framestyle=:box,
            grid=:y,
        )
    else
        plot(x_range, Float64.(state.ρ),
             title="Instantaneous density",
             xlabel="Site",
             ylabel="ρ(x,t)",
             lw=2.0,
             color=:steelblue,
             legend=false,
             framestyle=:box,
             grid=:y)
    end
    annotate_force_markers_minimal_1d!(p_inst_density, bonds; direction_flags=direction_flags, colors=colors)
    for (cut_idx, cut_site) in enumerate(selected_sites)
        vline!(p_inst_density, [cut_site], color=colors[mod1(cut_idx, length(colors))], linestyle=:dash, alpha=0.4, lw=1.3, label=false)
    end

    p_force_averages = plot_force_passage_averages_1d(state, param)

    p_selected_cuts = plot(title="Selected connected cuts",
                           xlabel="x' - x_center",
                           ylabel="C(x_ref,x')",
                           framestyle=:box,
                           grid=:y)
    centered_cuts = Vector{Vector{Float64}}()
    for (cut_idx, cut_site) in enumerate(selected_sites)
        x_rel, centered_cut = centered_selected_site_cut_1d(state, param, cut_site, pair_avgs[cut_idx, :], origin_site)
        relative_site = periodic_relative_coordinate(L, origin_site, cut_site)
        color = colors[mod1(cut_idx, length(colors))]
        linestyle = linestyles[mod1(cut_idx, length(linestyles))]
        push!(centered_cuts, centered_cut)
        plot!(p_selected_cuts, x_rel, centered_cut,
              lw=2.2,
              color=color,
              linestyle=linestyle,
              label=@sprintf("x_ref = center %+d", relative_site))
    end
    if !isempty(centered_cuts)
        cut_ylims = finite_curve_ylims(centered_cuts...; pad_fraction=0.12)
        if cut_ylims !== nothing
            ylims!(p_selected_cuts, cut_ylims)
        end
    end

    p_positive_raw = plot(title="Positive selected cuts",
                          xlabel="x' - x_center",
                          ylabel="C(x_ref,x')",
                          framestyle=:box,
                          grid=:y)
    p_positive_avg = plot(title="Reflection-averaged ± cuts",
                          xlabel="x' - x_center",
                          ylabel="C_avg(x_ref,x')",
                          framestyle=:box,
                          grid=:y)
    positive_raw_curves = Vector{Vector{Float64}}()
    positive_avg_curves = Vector{Vector{Float64}}()
    positive_pair_count = 0
    for (cut_idx, cut_site) in enumerate(selected_sites)
        relative_site = periodic_relative_coordinate(L, origin_site, cut_site)
        relative_site > 0 || continue

        raw_x_rel, raw_centered_cut = centered_selected_site_cut_1d(state, param, cut_site, pair_avgs[cut_idx, :], origin_site)
        avg_x_rel, avg_centered_cut, pair_used = centered_reflection_pair_averaged_cut_1d(state, param, cut_site, origin_site)
        pair_used || continue

        color = colors[mod1(cut_idx, length(colors))]
        raw_style = linestyles[mod1(cut_idx, length(linestyles))]
        push!(positive_raw_curves, raw_centered_cut)
        push!(positive_avg_curves, avg_centered_cut)
        plot!(p_positive_raw, raw_x_rel, raw_centered_cut,
              lw=2.0,
              color=color,
              linestyle=raw_style,
              label=@sprintf("%+d", relative_site))
        plot!(p_positive_avg, avg_x_rel, avg_centered_cut,
              lw=2.6,
              color=color,
              linestyle=:solid,
              label=@sprintf("avg ±%d", relative_site))
        positive_pair_count += 1
    end
    if positive_pair_count == 0
        annotate!(p_positive_raw, 0.5, 0.5, text("No positive cut pairs available", 10, :gray40, :center))
        annotate!(p_positive_avg, 0.5, 0.5, text("No positive/reflected cut pairs available", 10, :gray40, :center))
        plot!(p_positive_raw, legend=false)
        plot!(p_positive_avg, legend=false)
    else
        if !isempty(positive_raw_curves)
            raw_ylims = finite_curve_ylims(positive_raw_curves...; pad_fraction=0.12)
            if raw_ylims !== nothing
                ylims!(p_positive_raw, raw_ylims)
            end
        end
        if !isempty(positive_avg_curves)
            avg_ylims = finite_curve_ylims(positive_avg_curves...; pad_fraction=0.12)
            if avg_ylims !== nothing
                ylims!(p_positive_avg, avg_ylims)
            end
        end
    end

    if include_instantaneous
        return plot(p_avg_density, p_inst_density, p_force_averages, p_selected_cuts, p_positive_raw, p_positive_avg,
                    layout=(2, 3),
                    size=(2400, 980),
                    plot_title=sweep_title_with_label("1D selected-cut sweep", sweep, label))
    end
    return plot(p_avg_density, p_force_averages, p_selected_cuts,
                layout=(1, 3),
                size=(1800, 520),
                plot_title=sweep_title_with_label("1D selected-cut observables", sweep, label))
end

function centered_periodic_axis(L, ref_site)
    half = L ÷ 2
    x_rel = [mod(i - ref_site + half, L) - half for i in 1:L]
    perm = sortperm(x_rel)
    return x_rel[perm], perm
end

function periodic_relative_coordinate(L::Int, origin_site::Int, target_site::Int)
    half = L ÷ 2
    return mod(target_site - origin_site + half, L) - half
end

function site_cut_1d(
    corr_mat::AbstractMatrix{<:Real},
    site::Int;
    smooth_diagonal::Bool=true,
)
    L = size(corr_mat, 2)
    ref_site = mod1(Int(site), L)
    cut = Float64.(corr_mat[ref_site, :])
    if smooth_diagonal
        left_site = mod1(ref_site - 1, L)
        right_site = mod1(ref_site + 1, L)
        cut[ref_site] = 0.5 * (cut[left_site] + cut[right_site])
    end
    return cut
end

function centered_site_cut_1d(
    corr_mat::AbstractMatrix{<:Real},
    site::Int,
    origin_site::Int;
    smooth_diagonal::Bool=true,
)
    L = size(corr_mat, 2)
    cut = site_cut_1d(corr_mat, site; smooth_diagonal=smooth_diagonal)
    x_rel, perm = centered_periodic_axis(L, origin_site)
    return x_rel, Float64.(cut[perm])
end

function centered_bond_cut_1d(
    corr_mat::AbstractMatrix{<:Real},
    b1::Int,
    b2::Int,
    origin_site::Int;
    smooth_diagonal::Bool=true,
)
    L = size(corr_mat, 2)
    cut = bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
    x_rel, perm = centered_periodic_axis(L, origin_site)
    return x_rel, Float64.(cut[perm])
end

function diagonal_smoothed_cut_like_plot_sweep_1d(corr_mat::AbstractMatrix{<:Real}, point_to_look_at::Int)
    return site_cut_1d(corr_mat, point_to_look_at; smooth_diagonal=true)
end

function bond_centered_cut_for_plot_sweep_1d(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int)
    return bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=true)
end

function present_plot!(p)
    display(p)
    return p
end

function bond_pass_stats_dict(state)
    if hasfield(typeof(state), :bond_pass_stats)
        stats = getfield(state, :bond_pass_stats)
        if stats isa AbstractDict
            return stats
        end
    end
    if hasfield(typeof(state), :ρ_matrix_avg_cuts)
        cuts = getfield(state, :ρ_matrix_avg_cuts)
        if cuts isa AbstractDict
            return cuts
        end
    end
    return Dict{Symbol,Vector{Float64}}()
end

function average_by_abs_distance(values::AbstractVector{<:Real}, ref_site::Int)
    L = length(values)
    x_rel_sorted, perm = centered_periodic_axis(L, ref_site)
    vals_sorted = Float64.(values[perm])
    abs_dist = abs.(Int.(x_rel_sorted))
    max_dist = maximum(abs_dist)

    sums = zeros(Float64, max_dist + 1)
    counts = zeros(Int, max_dist + 1)
    for i in eachindex(vals_sorted)
        d_idx = abs_dist[i] + 1
        sums[d_idx] += vals_sorted[i]
        counts[d_idx] += 1
    end

    avg_vals = fill(NaN, max_dist + 1)
    for i in eachindex(avg_vals)
        if counts[i] > 0
            avg_vals[i] = sums[i] / counts[i]
        end
    end
    return collect(0:max_dist), avg_vals
end

function average_by_abs_distance_with_sem(values::AbstractVector{<:Real}, ref_site::Int)
    L = length(values)
    x_rel_sorted, perm = centered_periodic_axis(L, ref_site)
    vals_sorted = Float64.(values[perm])
    abs_dist = abs.(Int.(x_rel_sorted))
    max_dist = maximum(abs_dist)

    grouped = [Float64[] for _ in 0:max_dist]
    for i in eachindex(vals_sorted)
        v = vals_sorted[i]
        if isfinite(v)
            push!(grouped[abs_dist[i] + 1], v)
        end
    end

    avg_vals = fill(NaN, max_dist + 1)
    sem_vals = fill(NaN, max_dist + 1)
    counts = zeros(Int, max_dist + 1)
    for i in eachindex(grouped)
        vals = grouped[i]
        n = length(vals)
        counts[i] = n
        if n == 0
            continue
        end
        μ = sum(vals) / n
        avg_vals[i] = μ
        if n >= 2
            s2 = sum((v - μ)^2 for v in vals) / (n - 1)
            sem_vals[i] = sqrt(max(s2, 0.0)) / sqrt(n)
        end
    end

    return collect(0:max_dist), avg_vals, sem_vals, counts
end

function fluctuating_bond_indices_for_plot(bonds, tracked_indices::AbstractVector{Int})
    if isempty(bonds)
        return Int[]
    end
    if isempty(tracked_indices)
        return collect(1:length(bonds))
    end
    return collect(tracked_indices)
end

function density_variance_profile_1d(state, L::Int)
    rho_avg = Float64.(state.ρ_avg)
    if length(rho_avg) != L
        rho_avg = zeros(Float64, L)
    end

    rho2_diag = zeros(Float64, L)
    if haskey(state.ρ_matrix_avg_cuts, :full)
        rho2_diag = Float64.(diag(state.ρ_matrix_avg_cuts[:full]))
        if length(rho2_diag) != L
            rho2_diag = zeros(Float64, L)
        end
    end

    return max.(0.0, rho2_diag .- rho_avg .^ 2)
end

function plot_centered_corr_cuts_at_fluctuating_bonds_1d(
    corr_mat::AbstractMatrix{<:Real},
    bonds,
    fluctuating_indices::AbstractVector{Int};
    colors=[:red, :orange, :yellow, :cyan, :magenta, :green, :blue],
    remove_diagonal_for_cuts::Bool=true,
)
    if isempty(fluctuating_indices)
        return plot(title="Bond-centered correlation cuts (no fluctuating bonds)", axis=false, legend=false)
    end

    L = size(corr_mat, 1)
    p = plot(title="C(x_bond,x') cuts at fluctuating bonds (centered)",
             xlabel="x' - x_bond",
             ylabel="C(x_bond,x')",
             framestyle=:box,
             grid=:y)

    for force_idx in fluctuating_indices
        b1, b2 = bonds[force_idx]
        x_rel, centered_cut = centered_bond_cut_1d(corr_mat, b1, b2, b1; smooth_diagonal=remove_diagonal_for_cuts)
        color = colors[mod1(force_idx, length(colors))]
        plot!(p, x_rel, centered_cut,
              lw=2.2,
              color=color,
              label=@sprintf("j%d bond (%d,%d)", force_idx, b1, b2))
    end
    return p
end

function bond_centered_cut_1d(
    corr_mat::AbstractMatrix{<:Real},
    b1::Int,
    b2::Int;
    smooth_diagonal::Bool=true,
)
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

function bond_center_value_from_cut(cut::AbstractVector{<:Real}, b1::Int, b2::Int)
    return 0.5 * (Float64(cut[b1]) + Float64(cut[b2]))
end

function plot_centered_corr_cut_at_origin_1d(
    corr_mat::AbstractMatrix{<:Real},
    L::Int;
    remove_diagonal_for_cuts::Bool=true,
)
    origin_site = max(1, div(L, 2))
    origin_partner = mod1(origin_site + 1, L)
    x_rel, centered_cut = centered_bond_cut_1d(corr_mat, origin_site, origin_partner, origin_site; smooth_diagonal=remove_diagonal_for_cuts)

    p = plot(x_rel, centered_cut,
             title="Connected correlation at origin bond",
             xlabel="x' - x_origin",
             ylabel="C_origin(x')",
             lw=2.4,
             color=:black,
             label=@sprintf("origin bond (%d,%d)", origin_site, origin_partner),
             framestyle=:box,
             grid=:y)
    return p
end

function plot_centered_corr_cut_at_bond_1d(
    corr_mat::AbstractMatrix{<:Real},
    L::Int,
    bond_site::Int;
    remove_diagonal_for_cuts::Bool=true,
    title::AbstractString="Connected correlation cut",
    color=:darkgreen,
    origin_site::Int=mod1(Int(bond_site), L),
)
    ref_site = mod1(Int(bond_site), L)
    ref_partner = mod1(ref_site + 1, L)
    axis_origin_site = mod1(Int(origin_site), L)
    x_rel, centered_cut = centered_bond_cut_1d(corr_mat, ref_site, ref_partner, axis_origin_site; smooth_diagonal=remove_diagonal_for_cuts)
    ref_position = periodic_relative_coordinate(L, axis_origin_site, ref_site)

    p = plot(x_rel, centered_cut,
             title=title,
             xlabel=axis_origin_site == ref_site ? "x' - x_ref" : "x' - x_center",
             ylabel="C_ref(x')",
             lw=2.4,
             color=color,
             label=@sprintf("bond (%d,%d)", ref_site, ref_partner),
             framestyle=:box,
             grid=:y)
    if ref_position != 0
        vline!(p, [ref_position], color=color, linestyle=:dot, lw=1.4, alpha=0.8, label=false)
    end
    return p
end

function plot_centered_corr_cut_at_site_1d(
    corr_mat::AbstractMatrix{<:Real},
    L::Int,
    site::Int;
    origin_site::Int,
    remove_diagonal_for_cuts::Bool=true,
    title::AbstractString="Connected correlation cut",
    color=:darkgreen,
)
    ref_site = mod1(Int(site), L)
    axis_origin_site = mod1(Int(origin_site), L)
    x_rel, centered_cut = centered_site_cut_1d(corr_mat, ref_site, axis_origin_site; smooth_diagonal=remove_diagonal_for_cuts)
    ref_position = periodic_relative_coordinate(L, axis_origin_site, ref_site)

    p = plot(x_rel, centered_cut,
             title=title,
             xlabel="x' - x_center",
             ylabel="C(x_ref,x')",
             lw=2.4,
             color=color,
             label=@sprintf("x_ref = %d", ref_site),
             framestyle=:box,
             grid=:y)
    if ref_position != 0
        vline!(p, [ref_position], color=color, linestyle=:dot, lw=1.4, alpha=0.8, label=false)
    end
    return p
end

function plot_density_variance_at_fluctuating_bonds_1d(
    corr_mat::AbstractMatrix{<:Real},
    bonds,
    fluctuating_indices::AbstractVector{Int};
    colors=[:red, :orange, :yellow, :cyan, :magenta, :green, :blue],
    remove_diagonal_for_cuts::Bool=true,
)
    if isempty(fluctuating_indices)
        return plot(title="Bond-center variance at fluctuating bonds (none)", axis=false, legend=false)
    end

    n_lines = length(fluctuating_indices) + (length(fluctuating_indices) > 1 ? 1 : 0)

    p = plot(axis=false,
             showaxis=false,
             framestyle=:none,
             legend=false,
             xlim=(0.0, 1.0),
             ylim=(0.0, 1.0),
             title="Bond-center variance from C_bond(Δx=0)")

    dy = 1.0 / (n_lines + 2)
    line_id = 1
    bond_center_values = Float64[]
    for force_idx in fluctuating_indices
        b1, b2 = bonds[force_idx]
        cut = bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=remove_diagonal_for_cuts)
        bond_center_value = bond_center_value_from_cut(cut, b1, b2)
        push!(bond_center_values, bond_center_value)
        color = colors[mod1(force_idx, length(colors))]
        y = 1.0 - line_id * dy
        text_line = @sprintf("j%d (%d,%d): C_bond(0) = %.6g", force_idx, b1, b2, bond_center_value)
        annotate!(p, 0.02, y, text(text_line, color, 9, :left))
        line_id += 1
    end

    if length(fluctuating_indices) > 1
        mean_var = sum(bond_center_values) / length(bond_center_values)
        y = 1.0 - line_id * dy
        annotate!(p, 0.02, y, text(@sprintf("Mean over fluctuating bonds: %.6g", mean_var), :black, 9, :left))
    end

    return p
end

function tracked_force_indices_for_plot(state, magnitudes::AbstractVector{<:Real}; tol::Float64=0.0)
    n_forces = length(magnitudes)
    stats = bond_pass_stats_dict(state)
    if haskey(stats, BOND_PASS_TRACK_MASK_KEY)
        mask = stats[BOND_PASS_TRACK_MASK_KEY]
        if length(mask) == n_forces
            return [i for i in 1:n_forces if mask[i] > 0.5]
        end
    end
    return [i for i in 1:n_forces if abs(Float64(magnitudes[i])) > tol]
end

function has_tracked_force_bonds_for_plot(state, param)
    if length(param.dims) != 1
        return false
    end
    L = param.dims[1]
    _, _, magnitudes = force_bond_sites_1d(state, L)
    tracked_indices = tracked_force_indices_for_plot(state, magnitudes)
    return !isempty(tracked_indices)
end

function force_passage_averages(state, n_forces::Int)
    stats = bond_pass_stats_dict(state)
    forward_avg = haskey(stats, BOND_PASS_FORWARD_AVG_KEY) ? Float64.(stats[BOND_PASS_FORWARD_AVG_KEY]) : zeros(Float64, n_forces)
    reverse_avg = haskey(stats, BOND_PASS_REVERSE_AVG_KEY) ? Float64.(stats[BOND_PASS_REVERSE_AVG_KEY]) : zeros(Float64, n_forces)
    total_avg = haskey(stats, BOND_PASS_TOTAL_AVG_KEY) ? Float64.(stats[BOND_PASS_TOTAL_AVG_KEY]) : forward_avg .+ reverse_avg
    total_sq_avg = haskey(stats, BOND_PASS_TOTAL_SQ_AVG_KEY) ? Float64.(stats[BOND_PASS_TOTAL_SQ_AVG_KEY]) : total_avg .^ 2

    if length(forward_avg) != n_forces
        forward_avg = zeros(Float64, n_forces)
    end
    if length(reverse_avg) != n_forces
        reverse_avg = zeros(Float64, n_forces)
    end
    if length(total_avg) != n_forces
        total_avg = forward_avg .+ reverse_avg
    end
    if length(total_sq_avg) != n_forces
        total_sq_avg = total_avg .^ 2
    end

    samples = if haskey(stats, BOND_PASS_SAMPLE_COUNT_KEY) && !isempty(stats[BOND_PASS_SAMPLE_COUNT_KEY])
        Int(round(stats[BOND_PASS_SAMPLE_COUNT_KEY][1]))
    else
        0
    end
    return forward_avg, reverse_avg, total_avg, total_sq_avg, samples
end

function spatial_force_moments_1d(state, L::Int)
    stats = bond_pass_stats_dict(state)
    f_avg = haskey(stats, BOND_PASS_SPATIAL_F_AVG_KEY) ? Float64.(stats[BOND_PASS_SPATIAL_F_AVG_KEY]) : zeros(Float64, L)
    f2_avg = haskey(stats, BOND_PASS_SPATIAL_F2_AVG_KEY) ? Float64.(stats[BOND_PASS_SPATIAL_F2_AVG_KEY]) : zeros(Float64, L)
    if length(f_avg) != L
        f_avg = zeros(Float64, L)
    end
    if length(f2_avg) != L
        f2_avg = zeros(Float64, L)
    end
    samples = if haskey(stats, BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY) && !isempty(stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY])
        Int(round(stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY][1]))
    else
        0
    end
    return f_avg, f2_avg, samples
end

function spatial_force_second_moment_1d(state, L::Int)
    _, f2_avg, samples = spatial_force_moments_1d(state, L)
    return f2_avg, samples
end

function plot_spatial_force_statistics_1d(
    state,
    param,
    bonds;
    colors=[:red, :orange, :yellow, :cyan, :magenta, :green, :blue],
    include_abs_mean=false,
    var_baseline=nothing,
    show_reference=false,
    ref_site=nothing,
)
    L = param.dims[1]
    f_avg, f2_avg, samples = spatial_force_moments_1d(state, L)
    var_f = max.(0.0, f2_avg .- f_avg .^ 2)
    log_floor = 1e-12
    baseline_value = isnothing(var_baseline) ? single_origin_varj_baseline(param) : Float64(var_baseline)
    baseline_label = @sprintf("%.4g", baseline_value)
    ref_bond_site = if ref_site === nothing
        isempty(bonds) ? max(1, div(L, 2)) : bonds[1][1]
    else
        mod1(Int(ref_site), L)
    end

    dist_var, var_sym, var_sym_sem, var_sym_counts = average_by_abs_distance_with_sem(var_f, ref_bond_site)
    var_sym_shifted = var_sym .- baseline_value
    # Baseline is treated as fixed here, so shifted uncertainty equals the original SEM.
    var_sym_shifted_sem = var_sym_sem

    yerror_sanitized = [isfinite(v) && v > 0 ? v : 0.0 for v in Float64.(var_sym_shifted_sem)]

    mask_var_linear = (dist_var .> 0) .& isfinite.(var_sym_shifted)
    x_var_linear = Float64.(dist_var[mask_var_linear])
    y_var_linear = Float64.(var_sym_shifted[mask_var_linear])
    yerr_var_linear = Float64.(yerror_sanitized[mask_var_linear])
    if isempty(x_var_linear)
        x_var_linear = [1.0]
        y_var_linear = [0.0]
        yerr_var_linear = [0.0]
    end

    p_linear = plot(x_var_linear, y_var_linear;
                    yerror=yerr_var_linear,
                    title="Symmetrized Var(J)-$(baseline_label) vs |Δx| over $(samples) sweeps",
                    xlabel="|Δx| from reference bond",
                    ylabel="Symmetrized Var(J)-baseline",
                    lw=2.2,
                    color=:purple,
                    marker=:circle,
                    markersize=3.5,
                    label="Var(J)-$(baseline_label), L/R averaged",
                    framestyle=:box,
                    grid=:y)
    hline!(p_linear, [0.0], color=:gray55, linestyle=:dash, lw=1.2, label=false)

    mask_var = (dist_var .> 0) .& isfinite.(var_sym_shifted) .& (var_sym_shifted .> 0)
    x_var = Float64.(dist_var[mask_var])
    y_var = max.(Float64.(var_sym_shifted[mask_var]), log_floor)
    yerr_var = Float64.(yerror_sanitized[mask_var])
    if isempty(x_var)
        x_var = [1.0]
        y_var = [log_floor]
        yerr_var = [0.0]
    end

    p_loglog = plot(x_var, y_var;
                    yerror=yerr_var,
                    title="Symmetrized Var(J)-$(baseline_label) vs |Δx| over $(samples) sweeps",
                    xlabel="|Δx| from reference bond",
                    ylabel="Symmetrized Var(J)-baseline (log-log)",
                    lw=2.2,
                    color=:purple,
                    marker=:circle,
                    markersize=3.5,
                    label="Var(J)-$(baseline_label), L/R averaged",
                    xscale=:log10,
                    yscale=:log10,
                    framestyle=:box,
                    grid=:both)

    if include_abs_mean
        dist_mean, abs_mean_sym = average_by_abs_distance(abs.(f_avg), ref_bond_site)
        mask_mean = (dist_mean .> 0) .& isfinite.(abs_mean_sym)
        if any(mask_mean)
            x_mean = Float64.(dist_mean[mask_mean])
            y_mean = max.(Float64.(abs_mean_sym[mask_mean]), log_floor)
            plot!(p_loglog, x_mean, y_mean, lw=2.0, color=:navy, linestyle=:dash, label="|⟨J⟩|, L/R averaged")
        end
    end

    if show_reference && !isempty(x_var)
        x_anchor = x_var[end]
        y_anchor = y_var[end]
        y_ref_m2 = y_anchor .* (x_var ./ x_anchor) .^ (-2.0)
        y_ref_m3 = y_anchor .* (x_var ./ x_anchor) .^ (-3.0)
        plot!(p_loglog, x_var, y_ref_m2, color=:black, linestyle=:dash, lw=1.8, label="reference r^-2")
        plot!(p_loglog, x_var, y_ref_m3, color=:gray35, linestyle=:dash, lw=1.8, label="reference r^-3")
    end

    finite_counts = [c for c in var_sym_counts if c > 0]
    if !isempty(finite_counts)
        mean_count = sum(finite_counts) / length(finite_counts)
        count_label = @sprintf("SEM from L/R spatial symmetry; mean n=%.2f", mean_count)
        plot!(p_linear, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=count_label)
        plot!(p_loglog, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=count_label)
    end

    return (loglog=p_loglog, linear=p_linear)
end

function plot_spatial_force_second_moment_1d(state, param, bonds; colors=[:red, :orange, :yellow, :cyan, :magenta, :green, :blue])
    L = param.dims[1]
    x = 1:L
    f2_avg, samples = spatial_force_second_moment_1d(state, L)
    p = plot(x, f2_avg,
             title="Spatial second moment ⟨J(x)^2⟩ over $(samples) sweeps",
             xlabel="Bond index x for bond (x,x+1)",
             ylabel="⟨J(x)^2⟩",
             lw=2.2,
             color=:darkgreen,
             legend=false,
             framestyle=:box,
             grid=:y)
    for (idx, (b1, b2)) in enumerate(bonds)
        color = colors[mod1(idx, length(colors))]
        vline!(p, [b1], color=color, linestyle=:dash, alpha=0.45, lw=1.5, label=false)
        annotate!(p, (b1, maximum(f2_avg) * 0.95 + eps(Float64), text("f$(idx)", color, 8)))
    end
    return p
end

function annotate_force_markers_minimal_1d!(p, bonds; direction_flags=nothing, colors=[:red, :orange, :yellow, :cyan, :magenta, :green, :blue])
    if isempty(bonds)
        return
    end
    y_low, y_high = ylims(p)
    y = y_low + 0.95 * (y_high - y_low)
    y_arrow = y_low + 0.985 * (y_high - y_low)
    show_direction_hints = direction_flags !== nothing && length(direction_flags) == length(bonds)
    x_low, x_high = xlims(p)
    L_plot = max(1, Int(round(x_high - x_low + 1)))
    for (idx, (b1, b2)) in enumerate(bonds)
        color = colors[mod1(idx, length(colors))]
        vline!(p, [b1, b2], color=color, linestyle=:dash, lw=1.0, alpha=0.45, label=false)
        scatter!(p, [b1, b2], [y, y], color=color, markersize=2.5, alpha=0.7, label=false)
        if show_direction_hints
            # Keep the direction cue subtle: a tiny arrow glyph close to the top of the panel.
            if abs(b2 - b1) != L_plot - 1
                x_mid = 0.5 * (b1 + b2)
                arrow_txt = direction_flags[idx] ? "→" : "←"
                annotate!(p, (x_mid, y_arrow, text(arrow_txt, color, 7)))
            end
        end
    end
end

function annotate_force_directions_1d!(p, bonds, direction_flags; colors=[:red, :orange, :yellow, :cyan, :magenta, :green, :blue], label_prefix="f")
    if isempty(bonds)
        return
    end
    y_low, y_high = ylims(p)
    y_range = y_high - y_low
    if y_range <= 0
        return
    end

    y_top = y_low + 0.90 * y_range
    y_step = min(0.12 * y_range, 0.8 * y_range / max(length(bonds), 1))
    x_low, x_high = xlims(p)
    L_plot = max(1, Int(round(x_high - x_low + 1)))

    for (idx, (b1, b2)) in enumerate(bonds)
        color = colors[mod1(idx, length(colors))]
        dir_flag = direction_flags[idx]
        y = y_top - (idx - 1) * y_step

        vline!(p, [b1, b2], color=color, linestyle=:dash, lw=1.5, alpha=0.35, label=false)
        if abs(b2 - b1) == L_plot - 1
            scatter!(p, [b1, b2], [y, y], color=color, markersize=4, label=false)
            dir_txt = dir_flag ? "$(label_prefix)$(idx): wrap ->" : "$(label_prefix)$(idx): <- wrap"
            annotate!(p, (min(b1, b2), y + 0.04 * y_range, text(dir_txt, color, 7)))
        else
            start_x, end_x = dir_flag ? (b1, b2) : (b2, b1)
            plot!(p, [start_x, end_x], [y, y], arrow=:arrow, color=color, lw=3, label=false)
            annotate!(p, ((b1 + b2) / 2, y + 0.04 * y_range, text("$(label_prefix)$(idx)", color, 8)))
        end
    end
end

function plot_force_realization_1d(state, param)
    L = param.dims[1]
    bonds, direction_flags, magnitudes = force_bond_sites_1d(state, L)
    n_forces = length(bonds)
    if n_forces == 0
        return plot(title="Instantaneous force realization", axis=false, legend=false)
    end

    colors = [:red, :orange, :yellow, :cyan, :magenta, :green, :blue]
    p = plot(1:L, zeros(L),
             title="Instantaneous force realization",
             xlabel="Site",
             ylabel="Force index",
             yticks=(1:n_forces, ["f$(i)" for i in 1:n_forces]),
             ylim=(0.5, n_forces + 0.9),
             xlim=(1, L),
             legend=false,
             color=:black,
             lw=1,
             alpha=0.25,
             framestyle=:box,
             grid=:y)

    for idx in 1:n_forces
        b1, b2 = bonds[idx]
        dir_flag = direction_flags[idx]
        color = colors[mod1(idx, length(colors))]
        y = idx
        start_x, end_x = dir_flag ? (b1, b2) : (b2, b1)

        if abs(b2 - b1) == L - 1
            scatter!(p, [b1, b2], [y, y], color=color, markersize=6, label=false)
            dir_txt = dir_flag ? "wrap ->" : "<- wrap"
            annotate!(p, (min(b1, b2), y + 0.22, text(dir_txt, color, 8)))
        else
            plot!(p, [start_x, end_x], [y, y], arrow=:arrow, color=color, lw=3, label=false)
            scatter!(p, [b1, b2], [y, y], color=color, markersize=5, label=false)
        end

        dir_char = dir_flag ? "->" : "<-"
        annotate!(p, (L * 0.03, y + 0.22, text(@sprintf("f%d: %d %s %d |F|=%.2f", idx, b1, dir_char, b2, magnitudes[idx]), color, 8)))
    end
    return p
end

function plot_instantaneous_state_1d(state, param)
    L = param.dims[1]
    bonds, direction_flags, _ = force_bond_sites_1d(state, L)
    if hasfield(typeof(state), :particles) && !isnothing(getfield(state, :particles))
        particles = getfield(state, :particles)
        positions = sort!(Int[mod1(Int(getfield(particle, :position)[1]), L) for particle in particles])
        site_counts = zeros(Int, L)
        stack_levels = Vector{Int}(undef, length(positions))
        @inbounds for idx in eachindex(positions)
            site = positions[idx]
            site_counts[site] += 1
            stack_levels[idx] = site_counts[site]
        end

        p = scatter(
            positions,
            stack_levels;
            title="Instantaneous particle positions and forces",
            xlabel="Site",
            ylabel="Particle stack",
            xlim=(1, L),
            ylim=(0, isempty(stack_levels) ? 1 : maximum(stack_levels) + 1),
            legend=false,
            color=:black,
            markersize=1.8,
            markerstrokewidth=0,
            alpha=0.5,
            framestyle=:box,
            grid=:y,
        )
    else
        occupied_sites = findall(!iszero, vec(state.ρ))
        p = plot(1:L, zeros(L),
                 title="Instantaneous particles and forces",
                 xlabel="Site",
                 ylabel="Occupation",
                 xlim=(1, L),
                 ylim=(-0.05, 1.35),
                 yticks=([0.0, 1.0], ["empty", "occupied"]),
                 legend=false,
                 color=:black,
                 lw=1,
                 alpha=0.15,
                 framestyle=:box,
                 grid=:y)
        if !isempty(occupied_sites)
            scatter!(p, occupied_sites, ones(length(occupied_sites)),
                     color=:black,
                     markersize=5,
                     markerstrokewidth=0.5,
                     label=false)
        end
    end

    if !isempty(bonds)
        annotate_force_markers_minimal_1d!(p, bonds; direction_flags=direction_flags)
    end
    return p
end

function plot_force_passage_averages_1d(state, param)
    L = param.dims[1]
    bonds_all, _, magnitudes_all = force_bond_sites_1d(state, L)
    n_forces = length(bonds_all)
    if n_forces == 0
        return plot(title="Bond flux", axis=false, legend=false)
    end

    tracked_indices = tracked_force_indices_for_plot(state, magnitudes_all)
    if isempty(tracked_indices)
        return plot(title="Bond flux (no tracked bonds)", axis=false, legend=false)
    end

    forward_avg_all, reverse_avg_all, total_avg_all, total_sq_avg_all, samples = force_passage_averages(state, n_forces)
    n_rows = length(tracked_indices)
    headers = ["Bond", "⟨J_left⟩", "⟨J_right⟩", "⟨J⟩", "⟨J²⟩"]
    n_cols = length(headers)

    p = plot(xlim=(0.5, n_cols + 0.5),
             ylim=(0.5, n_rows + 1.5),
             legend=false,
             axis=false,
             framestyle=:none,
             title="Bond flux over $(samples) sweeps")

    for x in 0.5:1.0:(n_cols + 0.5)
        vline!(p, [x], color=:gray70, lw=1.1, label=false)
    end
    for y in 0.5:1.0:(n_rows + 1.5)
        hline!(p, [y], color=:gray70, lw=1.1, label=false)
    end

    for col in 1:n_cols
        annotate!(p, (col, n_rows + 1, text(headers[col], :black, 9)))
    end

    for (row_idx, force_idx) in enumerate(tracked_indices)
        y = n_rows - row_idx + 1
        b1, b2 = bonds_all[force_idx]
        row_values = [
            @sprintf("j%d (%d,%d)", force_idx, b1, b2),
            @sprintf("%.5g", forward_avg_all[force_idx]),
            @sprintf("%.5g", reverse_avg_all[force_idx]),
            @sprintf("%.5g", total_avg_all[force_idx]),
            @sprintf("%.5g", total_sq_avg_all[force_idx]),
        ]
        for col in 1:n_cols
            annotate!(p, (col, y, text(row_values[col], :black, 8)))
        end
    end

    return p
end

function plot_sweep_1d_multiforce(
    sweep,
    state,
    param;
    label="",
    plot_directional=false,
    remove_diagonal_for_cuts=true,
    include_abs_mean_in_spatial_f_plot=false,
    return_components=false,
)
    L = param.dims[1]
    x_range = 1:L
    colors = [:red, :orange, :yellow, :cyan, :magenta, :green, :blue]

    corr_mat = connected_correlation_matrix_1d(state, param)
    corr_scale = maximum(abs.(corr_mat))
    if !isfinite(corr_scale) || corr_scale == 0
        corr_scale = 1.0
    end

    bonds, direction_flags, magnitudes = force_bond_sites_1d(state, L)

    p_avg_density = plot(x_range, state.ρ_avg,
                         title="Time-averaged density (sweep $(sweep))",
                         xlabel="Site",
                         ylabel="⟨ρ(x)⟩",
                         lw=2.5,
                         color=:black,
                         legend=false,
                         framestyle=:box,
                         grid=:y)
    for (idx, (b1, b2)) in enumerate(bonds)
        color = colors[mod1(idx, length(colors))]
        vline!(p_avg_density, [b1, b2], color=color, linestyle=:dash, alpha=0.25, lw=1.2, label=false)
    end

    p_inst_density = if hasfield(typeof(state), :particles) && !isnothing(getfield(state, :particles))
        particles = getfield(state, :particles)
        positions = sort!(Int[mod1(Int(getfield(particle, :position)[1]), L) for particle in particles])
        site_counts = zeros(Int, L)
        stack_levels = Vector{Int}(undef, length(positions))
        @inbounds for idx in eachindex(positions)
            site = positions[idx]
            site_counts[site] += 1
            stack_levels[idx] = site_counts[site]
        end
        scatter(
            positions,
            stack_levels;
            title="Instantaneous particle positions (bond markers)",
            xlabel="Site",
            ylabel="Particle stack",
            xlim=(1, L),
            ylim=(0, isempty(stack_levels) ? 1 : maximum(stack_levels) + 1),
            legend=false,
            color=:black,
            markersize=1.8,
            markerstrokewidth=0,
            alpha=0.5,
            framestyle=:box,
            grid=:y,
        )
    else
        plot(x_range, Float64.(state.ρ),
             title="Instantaneous density (bond markers)",
             xlabel="Site",
             ylabel="ρ(x,t)",
             lw=2.0,
             color=:steelblue,
             legend=false,
             framestyle=:box,
             grid=:y)
    end
    annotate_force_markers_minimal_1d!(p_inst_density, bonds; direction_flags=direction_flags, colors=colors)

    p_force_averages = plot_force_passage_averages_1d(state, param)

    p_corr_heat = heatmap(corr_mat,
                          xlabel="x",
                          ylabel="x'",
                          title="Connected correlation matrix C(x,x')",
                          color=:balance,
                          clims=(-corr_scale, corr_scale),
                          framestyle=:box)

    tracked_indices = tracked_force_indices_for_plot(state, magnitudes)
    fluctuating_indices = fluctuating_bond_indices_for_plot(bonds, tracked_indices)
    origin_site = max(1, div(L, 2))
    ref_bond_site = if isempty(tracked_indices)
        isempty(bonds) ? origin_site : bonds[1][1]
    else
        bonds[first(tracked_indices)][1]
    end

    for (idx, (b1, b2)) in enumerate(bonds)
        color = colors[mod1(idx, length(colors))]
        vline!(p_corr_heat, [b1, b2], color=color, linestyle=:dot, lw=2, label=false)
        hline!(p_corr_heat, [b1, b2], color=color, linestyle=:dot, lw=2, label=false)
    end

    p_corr_fluctuating_bond_cuts = plot_centered_corr_cuts_at_fluctuating_bonds_1d(
        corr_mat,
        bonds,
        fluctuating_indices;
        colors=colors,
        remove_diagonal_for_cuts=remove_diagonal_for_cuts,
    )
    p_corr_origin_centered = plot_centered_corr_cut_at_origin_1d(
        corr_mat,
        L;
        remove_diagonal_for_cuts=remove_diagonal_for_cuts,
    )
    quarter_site = mod1(origin_site + fld(L, 4), L)
    p_corr_quarter_centered = plot_centered_corr_cut_at_site_1d(
        corr_mat,
        L,
        quarter_site;
        origin_site=origin_site,
        remove_diagonal_for_cuts=remove_diagonal_for_cuts,
        title="Connected correlation at x = center + L/4",
        color=:darkgreen,
    )

    p_density_var_bond_sites = plot_density_variance_at_fluctuating_bonds_1d(
        corr_mat,
        bonds,
        fluctuating_indices;
        colors=colors,
        remove_diagonal_for_cuts=remove_diagonal_for_cuts,
    )

    spatial_varj_plots = plot_spatial_force_statistics_1d(
        state,
        param,
        bonds;
        colors=colors,
        include_abs_mean=include_abs_mean_in_spatial_f_plot,
        ref_site=ref_bond_site,
    )
    p_spatial_f2 = spatial_varj_plots.loglog
    p_spatial_f2_linear = spatial_varj_plots.linear

    vline!(p_corr_heat, [quarter_site], color=:darkgreen, linestyle=:dash, lw=1.5, label=false)
    hline!(p_corr_heat, [quarter_site], color=:darkgreen, linestyle=:dash, lw=1.5, label=false)

    p_final = plot(p_avg_density, p_inst_density, p_force_averages,
                   p_spatial_f2_linear, p_spatial_f2, p_density_var_bond_sites,
                   p_corr_origin_centered, p_corr_quarter_centered, p_corr_fluctuating_bond_cuts, p_corr_heat,
                   layout=(2, 5),
                   size=(2550, 1250),
                   plot_title=sweep_title_with_label("1D multi-force sweep", sweep, label))
    if return_components
        return (
            corr_mat=corr_mat,
            final_plot=p_final,
            avg_density=p_avg_density,
            inst_density=p_inst_density,
            force_averages=p_force_averages,
            spatial_f_stats=p_spatial_f2,
            spatial_f_stats_linear=p_spatial_f2_linear,
            corr_heat=p_corr_heat,
            corr_origin_cut=p_corr_origin_centered,
            corr_quarter_cut=p_corr_quarter_centered,
            corr_fluctuating_bond_cuts=p_corr_fluctuating_bond_cuts,
            corr_origin_centered=p_corr_origin_centered,
            corr_quarter_centered=p_corr_quarter_centered,
            density_variance_bond_sites=p_density_var_bond_sites,
        )
    end
    present_plot!(p_final)
    return corr_mat
end

function plot_sweep_1d(sweep, state, param; label="", plot_directional=false)
    corr_mat = connected_correlation_matrix_1d(state, param)

    p0 = plot_density(state.ρ_avg, param, state; title="Time averaged density")
    p1 = plot_instantaneous_state_1d(state, param)

    p4 = heatmap(corr_mat, xlabel="x", ylabel="y",
                 title="Correlation Matrix Heatmap", color=:viridis)
    L = param.dims[1]
    middle_spot = max(1, L ÷ 2)

    bonds, _, _ = force_bond_sites_1d(state, L)
    center_cut = if isempty(bonds)
        diagonal_smoothed_cut_like_plot_sweep_1d(corr_mat, middle_spot)
    else
        b1, b2 = bonds[1]
        bond_centered_cut_for_plot_sweep_1d(corr_mat, b1, b2)
    end
    p5 = plot(center_cut,
              title="Correlation cut at center bond",
              xlabel="Site",
              ylabel="C",
              lw=2,
              color=:black)
    point_to_look_at = mod1(middle_spot + fld(L, 4), L)
    vline!(p4, [point_to_look_at], label="x=$(point_to_look_at)")
    x_rel_quarter, corr_mat_cut = centered_site_cut_1d(corr_mat, point_to_look_at, middle_spot; smooth_diagonal=true)
    quarter_offset = periodic_relative_coordinate(L, middle_spot, point_to_look_at)

    p6 = plot(x_rel_quarter, corr_mat_cut,
              title="Correlation cut at x=$(point_to_look_at)",
              xlabel="x' - x_center",
              ylabel="C",
              lw=2,
              color=:darkgreen)
    vline!(p6, [quarter_offset], label="x=center+L/4", color=:darkgreen, linestyle=:dot)

    p_final = plot(p0, p1, p4, p5, p6,
                   size=(2100, 950),
                   plot_title=sweep_title_with_label("sweep", sweep, label),
                   layout=(2, 3))
    present_plot!(p_final)
    return corr_mat
end

function plot_sweep_2d(sweep, state, param; label="", plot_directional=false)
    dims = param.dims
    fix_term = connected_correlation_fix_term(param)
    y0 = clamp(div(dims[2], 2), 1, dims[2])
    x0 = clamp(div(dims[1], 2), 1, dims[1])
    x_range = 1:dims[1]
    point_to_look_at = clamp(Int(floor(3 * dims[1] / 4)), 1, dims[1])

    p_avg_density = heatmap(
        state.ρ_avg',
        title="Time averaged density",
        xlabel="x",
        ylabel="y",
        aspect_ratio=1,
        colorbar=true,
        color=:inferno,
    )
    if hasfield(typeof(state), :forcing)
        annotate_forcing_2d!(p_avg_density, state)
    end

    p_instantaneous = plot_instantaneous_state_2d(state, param)

    corr_mat_x = correlation_slice_x(state, param, y0, fix_term)
    p_corr_x_axis = heatmap(
        corr_mat_x,
        title="Correlation tensor x-cut at y=$(y0)",
        xlabel="x₁",
        ylabel="x₂",
        aspect_ratio=1,
        colorbar=true,
        color=:viridis,
    )
    vline!(p_corr_x_axis, [point_to_look_at], label="x=$(point_to_look_at)", color=:orange, linestyle=:dash)

    center_b1, center_b2 = preferred_x_axis_reference_bond_2d(state, dims, x0, y0)
    center_cut = bond_centered_cut_for_plot_sweep_1d(corr_mat_x, center_b1, center_b2)
    p_center_cut = plot(
        x_range,
        center_cut,
        title="Correlation cut at center bond",
        xlabel="x₂",
        ylabel="C",
        lw=2,
        color=:black,
        legend=false,
    )

    x_rel_quarter, corr_mat_cut = centered_site_cut_1d(corr_mat_x, point_to_look_at, x0; smooth_diagonal=true)
    quarter_offset = periodic_relative_coordinate(dims[1], x0, point_to_look_at)
    p_quarter_cut = plot(
        x_rel_quarter,
        corr_mat_cut,
        title="Correlation cut at x=$(point_to_look_at)",
        xlabel="x₂ - x_center",
        ylabel="C",
        lw=2,
        color=:darkgreen,
        legend=false,
    )
    vline!(p_quarter_cut, [quarter_offset], label="x=center+L/4", color=:darkgreen, linestyle=:dot)

    p_empty = plot(axis=false, showaxis=false, grid=false, title="")
    present_plot!(plot(
        p_avg_density,
        p_instantaneous,
        p_corr_x_axis,
        p_center_cut,
        p_quarter_cut,
        p_empty,
        layout=(2, 3),
        size=(2200, 1200),
        plot_title=sweep_title_with_label("2D sweep", sweep, label),
    ))
    return corr_mat_x
end
function plot_density(density, param, state; title="Density", show_directions=false)
    dim_num = length(size(density))
    if dim_num==1
        directional_pair = directional_density_pair(state)
        if !show_directions || directional_pair === nothing
            # Original plotting code
            x_range = 1:param.dims[1]
            # p = plot(x_range, density, 
            #      title=title, 
            #      xlabel="Position", ylabel=title, 
            #      legend=false, lw=2, seriestype=:scatter)
            p = plot(x_range, density,
                title=title,
                xlabel="Position",
                ylabel="Density",
                label="Density",
                lw=2,
                seriestype=:scatter,
                color=:blue,
                legend=:outerright)
            
            # Secondary axis for potential
            plot!(twinx(), x_range, state.potential.V,
                  ylabel="Potential",
                  label="Potential",
                  color=:red,
                  alpha=0.3,
                  linestyle=:dash,
                  legend=:outerright)
        else
            ρ₊, ρ₋ = directional_pair
            # Create subplot with total density, right-moving and left-moving particles
            x_range = 1:param.dims[1]
            p1 = plot(x_range, density, 
                     title="Total Density", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter)
            
            p2 = plot(x_range, ρ₊,
                     title="Right-moving (ρ₊)", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter,
                     color=:red)
            
            p3 = plot(x_range, ρ₋,
                     title="Left-moving (ρ₋)", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter,
                     color=:blue)
            
            p = plot(p1, p2, p3, layout=(3,1), size=(600,600))
        end
    elseif dim_num == 2
        Lx= param.dims[1]
        Ly= param.dims[2]
        x_range = range(1, Lx, length = Lx)
        y_range = range(1, Ly, length = Ly)
        p = heatmap(x_range, y_range, transpose(density),
                    title=title,
                    c=cgrad(:inferno), xlims=(1, Lx), ylims=(1, Ly),
                    clims=(0,3), aspect_ratio=1, xlabel="x", ylabel="y")
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
    return p
end

function plot_average_density_and_correlation(state, param; label="")
    dim_num = length(param.dims)

    if dim_num == 1
        if has_selected_site_cuts_for_plot(state)
            return plot_selected_site_cuts_1d(state, param; sweep=state.t, label=label, include_instantaneous=false)
        end
        corr_mat = connected_correlation_matrix_1d(state, param)
        L = param.dims[1]
        center = clamp(div(L, 2), 1, L)
        quarter_site = mod1(center + fld(L, 4), L)
        bonds, _, _ = force_bond_sites_1d(state, L)

        p_density = plot_density(state.ρ_avg, param, state; title="Time averaged density")
        if hasfield(typeof(state), :forcing)
            annotate_forcing_1d!(p_density, state)
        end
        p_corr = heatmap(corr_mat,
                         xlabel="x", ylabel="x'",
                         title="Connected correlation",
                         color=:viridis)
        center_cut = if isempty(bonds)
            diagonal_smoothed_cut_like_plot_sweep_1d(corr_mat, center)
        else
            b1, b2 = bonds[1]
            bond_centered_cut_for_plot_sweep_1d(corr_mat, b1, b2)
        end
        p_center_cut = plot(center_cut,
                            xlabel="x'",
                            ylabel="C",
                            title="Connected correlation cut at center bond",
                            lw=2,
                            color=:black)
        x_rel_quarter, centered_quarter_cut = centered_site_cut_1d(corr_mat, quarter_site, center; smooth_diagonal=true)
        quarter_offset = periodic_relative_coordinate(L, center, quarter_site)
        p_quarter_cut = plot(x_rel_quarter, centered_quarter_cut,
                             xlabel="x' - x_center",
                             ylabel="C",
                             title="Connected correlation cut at x=$(quarter_site)",
                             lw=2,
                             color=:darkgreen)
        vline!(p_quarter_cut, [quarter_offset], label="x=center+L/4", color=:darkgreen, linestyle=:dot)
        vline!(p_corr, [quarter_site], label=false, color=:darkgreen, linestyle=:dash)
        return plot(p_density, p_corr, p_center_cut, p_quarter_cut,
                    layout=(2, 2),
                    size=(1800, 950),
                    plot_title=sweep_title_with_label("Average observables", state.t, label))
    elseif dim_num == 2
        x0 = clamp(div(param.dims[1] + 1, 2), 1, param.dims[1])
        y0 = clamp(div(param.dims[2] + 1, 2), 1, param.dims[2])
        fix_term = connected_correlation_fix_term(param)

        p_density = plot_density(state.ρ_avg, param, state; title="Time averaged density")
        if hasfield(typeof(state), :forcing)
            annotate_forcing_2d!(p_density, state)
        end
        corr_x = correlation_slice_x(state, param, y0, fix_term)
        corr_y = correlation_slice_y(state, param, x0, fix_term)
        p_corr_x = heatmap(corr_x,
                           xlabel="x₁", ylabel="x₂",
                           title="Connected x-cut at y=$(y0)",
                           aspect_ratio=1,
                           color=:viridis)
        p_corr_y = heatmap(corr_y,
                           xlabel="y₁", ylabel="y₂",
                           title="Connected y-cut at x=$(x0)",
                           aspect_ratio=1,
                           color=:viridis)
        return plot(p_density, p_corr_x, p_corr_y,
                    layout=(1, 3),
                    size=(1850, 560),
                    plot_title=sweep_title_with_label("Average observables", state.t, label))
    end

    throw(DomainError("Only 1D or 2D plotting supported"))
end

function plot_data_colapse(states_params_names, power_n, indices, results_dir = "results_figures", do_fit=true;
                           show_powerlaw=true, use_reflection_pair_average=true)
    n = Float64(power_n)
    extra_power_exp = 2.0  # reference slope -2
    x_label, y_label = data_collapse_axis_labels(n)
    combined_title = "Data collapse"
    all_x = Float64[]
    all_y = Float64[]

    p_combined = data_collapse_plot(combined_title, x_label, y_label)
    state_count = length(states_params_names)

    for (idx, (state, param, label)) in enumerate(states_params_names)
        dim_num = length(param.dims)

        if dim_num == 1
            L = param.dims[1]
            N = param.N
            fix_term = connected_correlation_fix_term(param)
            # Setup output directories
            full_dir = "$(results_dir)/full_data"
            antisym_dir = "$(results_dir)/antisymmetric"
            sym_dir = "$(results_dir)/symmetric"
            mkpath(full_dir)
            mkpath(antisym_dir)
            mkpath(sym_dir)

            full_title = "Full-cut collapse"
            antisym_title = "Antisymmetric collapse"
            sym_title = "Symmetric collapse"
            p_full_combined = data_collapse_plot(full_title, x_label, y_label)
            p_antisym_combined = data_collapse_plot(antisym_title, x_label, y_label)
            p_sym_combined = data_collapse_plot(sym_title, x_label, y_label)
            plot_x = Dict(:full => Float64[], :antisym => Float64[], :sym => Float64[])
            plot_y = Dict(:full => Float64[], :antisym => Float64[], :sym => Float64[])
            ref_dim_added = Dict(:full=>false, :antisym=>false, :sym=>false, :combined=>false)
            if show_powerlaw
                p_full_powerlaw = data_collapse_plot("Power-law check: full cut", "|Δ|", "|C|"; powerlaw=true)
                p_antisym_powerlaw = data_collapse_plot("Power-law check: antisymmetric cut", "|Δ|", "|C|"; powerlaw=true)
                p_sym_powerlaw = data_collapse_plot("Power-law check: symmetric cut", "|Δ|", "|C|"; powerlaw=true)
                ref_added_full = false
                ref_added_antisym = false
                ref_added_sym = false
                ref_added_full_extra = false
                ref_added_antisym_extra = false
                ref_added_sym_extra = false
            end

            for i in indices
                middle_spot = if has_selected_site_cuts_for_plot(state)
                    _, selected_origin_site, _ = selected_site_cut_metadata_for_plot(state, L)
                    selected_origin_site
                else
                    L ÷ 2
                end
                point_to_look_at = mod1(Int(middle_spot + i), L)

                full_cut_raw = if use_reflection_pair_average
                    reflection_pair_averaged_cut_1d(
                        state,
                        param,
                        point_to_look_at;
                        origin_site=middle_spot,
                        smooth_diagonal=false,
                    )[1]
                else
                    connected_correlation_cut_1d(state, param, point_to_look_at; smooth_diagonal=false)
                end
                full_cut_raw = Float64.(full_cut_raw) .+ fix_term
                full_data = copy(full_cut_raw)
                smooth_periodic_cut_site!(full_data, point_to_look_at)

                sym_data, antisym_data = reflection_decomposed_cut_1d(full_cut_raw, middle_spot)
                reflected_site = reflected_site_about_center_1d(point_to_look_at, middle_spot, L)
                smooth_periodic_cut_site!(antisym_data, point_to_look_at)
                smooth_periodic_cut_site!(antisym_data, reflected_site)
                smooth_periodic_cut_site!(sym_data, point_to_look_at)
                smooth_periodic_cut_site!(sym_data, reflected_site)

                x_rel, perm = centered_periodic_axis(L, middle_spot)
                x_scaled = Float64.(x_rel) ./ Float64(i)
                full_data_centered = full_data[perm]
                antisym_data_centered = antisym_data[perm]
                sym_data_centered = sym_data[perm]

                for (data, p_combined_plot, key) in zip((full_data_centered, antisym_data_centered, sym_data_centered), (p_full_combined, p_antisym_combined, p_sym_combined), (:full, :antisym, :sym))
                    y_scaled = data .* i^n
                    mask = (-5 .<= x_scaled .<= 5)
                    x_filtered = x_scaled[mask]
                    y_filtered = y_scaled[mask]
                    append!(plot_x[key], x_filtered)
                    append!(plot_y[key], y_filtered)
                    plot!(p_combined_plot, x_filtered, y_filtered, label=collapse_curve_label(label, i, state_count), lw=2.2)
                    if !ref_dim_added[key]
                        ref_dim_added[key] = add_dimension_reference!(p_combined_plot, dim_num, x_filtered, y_filtered)
                    end
                end

                y_scaled = full_data_centered .* i^n
                mask = (-5 .<= x_scaled .<= 5)
                x_filtered = x_scaled[mask]
                y_filtered = y_scaled[mask]
                append!(all_x, x_filtered)
                append!(all_y, y_filtered)
                plot!(p_combined, x_filtered, y_filtered, label=collapse_curve_label(label, i, state_count), linewidth=2.2)
                if !ref_dim_added[:combined]
                    ref_dim_added[:combined] = add_dimension_reference!(p_combined, dim_num, x_filtered, y_filtered)
                end
                if show_powerlaw
                    distances = abs.(Float64.(x_rel))
                    curve_label = collapse_curve_label(label, i, state_count)
                    ref_added_full |= add_powerlaw_series!(p_full_powerlaw, distances, full_data_centered, n; label=curve_label, add_reference=!ref_added_full, extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_full_extra)
                    ref_added_antisym |= add_powerlaw_series!(p_antisym_powerlaw, distances, antisym_data_centered, n; label=curve_label, add_reference=!ref_added_antisym, extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_antisym_extra)
                    ref_added_sym |= add_powerlaw_series!(p_sym_powerlaw, distances, sym_data_centered, n; label=curve_label, add_reference=!ref_added_sym, extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_sym_extra)
                    ref_added_full_extra = true
                    ref_added_antisym_extra = true
                    ref_added_sym_extra = true
                end
            end

            apply_data_collapse_max_note!(p_full_combined, full_title, plot_x[:full], plot_y[:full])
            apply_data_collapse_max_note!(p_antisym_combined, antisym_title, plot_x[:antisym], plot_y[:antisym])
            apply_data_collapse_max_note!(p_sym_combined, sym_title, plot_x[:sym], plot_y[:sym])
            apply_robust_data_collapse_limits!(p_full_combined, plot_y[:full])
            apply_robust_data_collapse_limits!(p_antisym_combined, plot_y[:antisym])
            apply_robust_data_collapse_limits!(p_sym_combined, plot_y[:sym])
            savefig(p_full_combined, "$(full_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_antisym_combined, "$(antisym_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_sym_combined, "$(sym_dir)/data_collapse_$(n)_indices-$(indices).png")
            if show_powerlaw
                savefig(p_full_powerlaw, "$(full_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_antisym_powerlaw, "$(antisym_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_sym_powerlaw, "$(sym_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
            end
        elseif dim_num == 2
            dims = param.dims
            fix_term = connected_correlation_fix_term(param)
            
            # Setup output directories for 2D
            x_axis_dir = "$(results_dir)/x_axis_cut"
            diag_dir = "$(results_dir)/diagonal_cut"
            x_axis_pos_dir = "$(results_dir)/x_axis_positive_cut"
            diag_pos_dir = "$(results_dir)/diagonal_positive_cut"
            x_axis_antisym_dir = "$(results_dir)/x_axis_cut/antisymmetric"
            diag_antisym_dir = "$(results_dir)/diagonal_cut/antisymmetric"
            mkpath(x_axis_dir)
            mkpath(diag_dir)
            mkpath(x_axis_pos_dir)
            mkpath(diag_pos_dir)
            mkpath(x_axis_antisym_dir)
            mkpath(diag_antisym_dir)

            x_title = "X-cut collapse"
            diag_title = "Diagonal-cut collapse"
            x_pos_title = "X-cut collapse (positive branch)"
            diag_pos_title = "Diagonal-cut collapse (positive branch)"
            x_antisym_title = "X-cut collapse (antisymmetric)"
            diag_antisym_title = "Diagonal-cut collapse (antisymmetric)"
            p_x_combined = data_collapse_plot(x_title, x_label, y_label)
            p_diag_combined = data_collapse_plot(diag_title, x_label, y_label)
            p_x_pos_combined = data_collapse_plot(x_pos_title, x_label, y_label)
            p_diag_pos_combined = data_collapse_plot(diag_pos_title, x_label, y_label)
            p_x_antisym_combined = data_collapse_plot(x_antisym_title, x_label, y_label)
            p_diag_antisym_combined = data_collapse_plot(diag_antisym_title, x_label, y_label)
            plot_x = Dict(
                :x => Float64[], :diag => Float64[], :x_pos => Float64[],
                :diag_pos => Float64[], :x_antisym => Float64[], :diag_antisym => Float64[]
            )
            plot_y = Dict(
                :x => Float64[], :diag => Float64[], :x_pos => Float64[],
                :diag_pos => Float64[], :x_antisym => Float64[], :diag_antisym => Float64[]
            )
            ref_dim_added = Dict(
                :x => false, :diag => false, :x_pos => false, :diag_pos => false,
                :x_antisym => false, :diag_antisym => false, :combined => false
            )
            if show_powerlaw
                p_x_powerlaw = data_collapse_plot("Power-law check: x cut", "|Δ|", "|C|"; powerlaw=true)
                p_diag_powerlaw = data_collapse_plot("Power-law check: diagonal cut", "|Δ|", "|C|"; powerlaw=true)
                p_x_pos_powerlaw = data_collapse_plot("Power-law check: x cut (positive)", "|Δ|", "|C|"; powerlaw=true)
                p_diag_pos_powerlaw = data_collapse_plot("Power-law check: diagonal cut (positive)", "|Δ|", "|C|"; powerlaw=true)
                p_x_antisym_powerlaw = data_collapse_plot("Power-law check: x cut (antisymmetric)", "|Δ|", "|C|"; powerlaw=true)
                p_diag_antisym_powerlaw = data_collapse_plot("Power-law check: diagonal cut (antisymmetric)", "|Δ|", "|C|"; powerlaw=true)
                plot!(p_x_powerlaw; xscale=:log10, yscale=:log10)
                plot!(p_diag_powerlaw; xscale=:log10, yscale=:log10)
                plot!(p_x_pos_powerlaw; xscale=:log10, yscale=:log10)
                plot!(p_diag_pos_powerlaw; xscale=:log10, yscale=:log10)
                plot!(p_x_antisym_powerlaw; xscale=:log10, yscale=:log10)
                plot!(p_diag_antisym_powerlaw; xscale=:log10, yscale=:log10)
                ref_added = Dict(
                    :x => false, :diag => false, :x_pos => false, :diag_pos => false,
                    :x_antisym => false, :diag_antisym => false
                )
                ref_added_extra = Dict(
                    :x => false, :diag => false, :x_pos => false, :diag_pos => false,
                    :x_antisym => false, :diag_antisym => false
                )
            end

            y0 = div(dims[2] + 1, 2)  # middle y index
            
            for i in indices
                # Check for potential overflow before scaling
                scaling_factor = Float64(i)^n
                if scaling_factor > 1e10 || !isfinite(scaling_factor)
                    println("Warning: Skipping i=$i due to overflow (scaling factor: $scaling_factor)")
                    continue
                end
                
                # X-axis cut: C(x1,y0; x2,y0)
                # slice2d_x = state.ρ_matrix_avg[:, y0, :, y0]
                if haskey(state.ρ_matrix_avg_cuts, :full)
                    slice2d_x = state.ρ_matrix_avg_cuts[:full][:, y0, :, y0]
                else
                    slice2d_x = state.ρ_matrix_avg_cuts[:x_cut]
                end
                mean_vec_x = state.ρ_avg[:, y0]
                corr_mat_x = slice2d_x .- mean_vec_x * transpose(mean_vec_x) .+ fix_term
                
                # Apply diagonal smoothing
                for j in 1:dims[1]
                    left_idx = j == 1 ? dims[1] : j - 1
                    right_idx = j == dims[1] ? 1 : j + 1
                    corr_mat_x[j, j] = (corr_mat_x[j, left_idx] + corr_mat_x[j, right_idx]) / 2
                end
                
                middle_spot = dims[1] ÷ 2
                point_to_look_at = Int(middle_spot + i)
                
                # Handle boundary wrapping for point_to_look_at
                if point_to_look_at > dims[1]
                    point_to_look_at = point_to_look_at - dims[1]
                elseif point_to_look_at < 1
                    point_to_look_at = point_to_look_at + dims[1]
                end
                
                x_axis_data = corr_mat_x[point_to_look_at, :]
                
                # Extract positive half of x-axis data
                middle_x = div(dims[1], 2) + 1
                x_axis_positive_data = x_axis_data[middle_x:end]
                
                # Diagonal cut: C(x,x; x',x')
                corr_diag = zeros(dims[1], dims[1])
                # for j in 1:dims[1], k in 1:dims[1]
                #     corr_diag[j,k] = state.ρ_matrix_avg[j, j, k, k] - state.ρ_avg[j,j] * state.ρ_avg[k,k] + fix_term
                # end
                if haskey(state.ρ_matrix_avg_cuts, :full)
                    for j in 1:dims[1], k in 1:dims[1]
                        corr_diag[j,k] = state.ρ_matrix_avg_cuts[:full][j, j, k, k] - state.ρ_avg[j,j] * state.ρ_avg[k,k] + fix_term
                    end
                else
                    diag_mean = diag(state.ρ_avg)
                    corr_diag = state.ρ_matrix_avg_cuts[:diag_cut] .- (diag_mean * transpose(diag_mean)) .+ fix_term
                end
                
                # Apply diagonal smoothing to diagonal correlation
                for j in 1:dims[1]
                    left_idx = j == 1 ? dims[1] : j - 1
                    right_idx = j == dims[1] ? 1 : j + 1
                    corr_diag[j, j] = (corr_diag[j, left_idx] + corr_diag[j, right_idx]) / 2
                end
                
                diag_data = corr_diag[point_to_look_at, :]
                
                # Extract positive half of diagonal data
                diag_positive_data = diag_data[middle_x:end]
                
                # Extract antisymmetric parts
                corr_mat_x_antisym = remove_symmetric_part_reflection(corr_mat_x, middle_x)
                # Apply diagonal smoothing to antisymmetric x-axis
                for j in 1:dims[1]
                    left_idx = j == 1 ? dims[1] : j - 1
                    right_idx = j == dims[1] ? 1 : j + 1
                    corr_mat_x_antisym[j, j] = (corr_mat_x_antisym[j, left_idx] + corr_mat_x_antisym[j, right_idx]) / 2
                end
                x_axis_antisym_data = corr_mat_x_antisym[point_to_look_at, :]
                
                corr_diag_antisym = remove_symmetric_part_reflection(corr_diag, middle_x)
                # Apply diagonal smoothing to antisymmetric diagonal
                for j in 1:dims[1]
                    left_idx = j == 1 ? dims[1] : j - 1
                    right_idx = j == dims[1] ? 1 : j + 1
                    corr_diag_antisym[j, j] = (corr_diag_antisym[j, left_idx] + corr_diag_antisym[j, right_idx]) / 2
                end
                diag_antisym_data = corr_diag_antisym[point_to_look_at, :]

                # Scale and plot all six cuts
                x_positions = 1:length(x_axis_data)
                x_scaled = (x_positions .- middle_spot) ./ Float64(i)
                curve_label = collapse_curve_label(label, i, state_count)
                
                # For positive x-axis cut
                x_positions_pos = middle_x:dims[1]
                x_scaled_pos = (x_positions_pos .- middle_spot) ./ Float64(i)
                
                for (data, p_plot, cut_type, x_scale, key) in zip((x_axis_data, diag_data, x_axis_positive_data, diag_positive_data, x_axis_antisym_data, diag_antisym_data), 
                                                           (p_x_combined, p_diag_combined, p_x_pos_combined, p_diag_pos_combined, p_x_antisym_combined, p_diag_antisym_combined),
                                                           ("x-axis", "diagonal", "x-axis-positive", "diagonal-positive", "x-axis-antisymmetric", "diagonal-antisymmetric"),
                                                           (x_scaled, x_scaled, x_scaled_pos, x_scaled_pos, x_scaled, x_scaled),
                                                           (:x, :diag, :x_pos, :diag_pos, :x_antisym, :diag_antisym))
                    y_scaled = data .* scaling_factor
                    
                    # Filter out non-finite values and extreme outliers
                    finite_mask = isfinite.(y_scaled)
                    range_mask = (-5 .<= x_scale .<= 5)
                    final_mask = finite_mask .& range_mask
                    
                    if sum(final_mask) > 0  # Only plot if we have valid data points
                        x_filtered = x_scale[final_mask]
                        y_filtered = y_scaled[final_mask]
                        
                        # Additional outlier filtering based on reasonable y-range
                        y_max = maximum(abs.(y_filtered))
                        if y_max < 1e8  # Reasonable threshold
                            append!(plot_x[key], x_filtered)
                            append!(plot_y[key], y_filtered)
                            plot!(p_plot, x_filtered, y_filtered, label=curve_label, lw=2.2)
                            if !ref_dim_added[key]
                                ref_dim_added[key] = add_dimension_reference!(p_plot, dim_num, x_filtered, y_filtered)
                            end
                        else
                            println("Warning: Skipping $(cut_type) plot for i=$i due to extreme values (max: $y_max)")
                        end
                    else
                        println("Warning: No valid data points for $(cut_type) plot at i=$i")
                    end
                end
                
                # Add to combined plot (using x-axis cut)
                y_scaled = x_axis_data .* scaling_factor
                finite_mask = isfinite.(y_scaled)
                range_mask = (-5 .<= x_scaled .<= 5)
                final_mask = finite_mask .& range_mask
                
                if sum(final_mask) > 0
                    x_filtered = x_scaled[final_mask]
                    y_filtered = y_scaled[final_mask]
                    
                    y_max = maximum(abs.(y_filtered))
                    if y_max < 1e8
                        append!(all_x, x_filtered)
                        append!(all_y, y_filtered)
                        plot!(p_combined, x_filtered, y_filtered, label=curve_label, linewidth=2.2)
                        if !ref_dim_added[:combined]
                            ref_dim_added[:combined] = add_dimension_reference!(p_combined, dim_num, x_filtered, y_filtered)
                        end
                    else
                        println("Warning: Skipping combined plot for i=$i due to extreme values")
                    end
                end

                if show_powerlaw
                    distances = abs.(x_positions .- middle_spot)
                    ref_added[:x] |= add_powerlaw_series!(p_x_powerlaw, distances, x_axis_data, n; label=curve_label, add_reference=!ref_added[:x], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:x])
                    ref_added[:diag] |= add_powerlaw_series!(p_diag_powerlaw, distances, diag_data, n; label=curve_label, add_reference=!ref_added[:diag], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:diag])
                    distances_pos = x_positions_pos .- middle_spot
                    ref_added[:x_pos] |= add_powerlaw_series!(p_x_pos_powerlaw, distances_pos, x_axis_positive_data, n; label=curve_label, add_reference=!ref_added[:x_pos], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:x_pos])
                    ref_added[:diag_pos] |= add_powerlaw_series!(p_diag_pos_powerlaw, distances_pos, diag_positive_data, n; label=curve_label, add_reference=!ref_added[:diag_pos], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:diag_pos])
                    ref_added[:x_antisym] |= add_powerlaw_series!(p_x_antisym_powerlaw, distances, x_axis_antisym_data, n; label=curve_label, add_reference=!ref_added[:x_antisym], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:x_antisym])
                    ref_added[:diag_antisym] |= add_powerlaw_series!(p_diag_antisym_powerlaw, distances, diag_antisym_data, n; label=curve_label, add_reference=!ref_added[:diag_antisym], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:diag_antisym])
                    ref_added_extra[:x] = true
                    ref_added_extra[:diag] = true
                    ref_added_extra[:x_pos] = true
                    ref_added_extra[:diag_pos] = true
                    ref_added_extra[:x_antisym] = true
                    ref_added_extra[:diag_antisym] = true
                end
            end
            
            apply_data_collapse_max_note!(p_x_combined, x_title, plot_x[:x], plot_y[:x])
            apply_data_collapse_max_note!(p_diag_combined, diag_title, plot_x[:diag], plot_y[:diag])
            apply_data_collapse_max_note!(p_x_pos_combined, x_pos_title, plot_x[:x_pos], plot_y[:x_pos])
            apply_data_collapse_max_note!(p_diag_pos_combined, diag_pos_title, plot_x[:diag_pos], plot_y[:diag_pos])
            apply_data_collapse_max_note!(p_x_antisym_combined, x_antisym_title, plot_x[:x_antisym], plot_y[:x_antisym])
            apply_data_collapse_max_note!(p_diag_antisym_combined, diag_antisym_title, plot_x[:diag_antisym], plot_y[:diag_antisym])
            apply_robust_data_collapse_limits!(p_x_combined, plot_y[:x])
            apply_robust_data_collapse_limits!(p_diag_combined, plot_y[:diag])
            apply_robust_data_collapse_limits!(p_x_pos_combined, plot_y[:x_pos])
            apply_robust_data_collapse_limits!(p_diag_pos_combined, plot_y[:diag_pos])
            apply_robust_data_collapse_limits!(p_x_antisym_combined, plot_y[:x_antisym])
            apply_robust_data_collapse_limits!(p_diag_antisym_combined, plot_y[:diag_antisym])
            savefig(p_x_combined, "$(x_axis_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_diag_combined, "$(diag_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_x_pos_combined, "$(x_axis_pos_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_diag_pos_combined, "$(diag_pos_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_x_antisym_combined, "$(x_axis_antisym_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_diag_antisym_combined, "$(diag_antisym_dir)/data_collapse_$(n)_indices-$(indices).png")
            if show_powerlaw
                savefig(p_x_powerlaw, "$(x_axis_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_diag_powerlaw, "$(diag_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_x_pos_powerlaw, "$(x_axis_pos_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_diag_pos_powerlaw, "$(diag_pos_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_x_antisym_powerlaw, "$(x_axis_antisym_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
                savefig(p_diag_antisym_powerlaw, "$(diag_antisym_dir)/powerlaw_loglog_$(n)_indices-$(indices).png")
            end
        end
    end

    if do_fit
        f(x, p) = (p[1] * x ./ ((1 .+ p[2]*x.^2).^2)).+p[3]
        p0 = [1.0, 0.0, 0.0]
        fit_combined = curve_fit(f, all_x, all_y, p0)
        x_theory = range(-5, 5, length=1000)
        plot!(p_combined, x_theory, f(x_theory, fit_combined.param), 
              label="Theoretical Fit", color=:black, linewidth=3, linestyle=:dash)
    end

    apply_data_collapse_max_note!(p_combined, combined_title, all_x, all_y; mark_point=true)
    apply_robust_data_collapse_limits!(p_combined, all_y)

    savefig(p_combined, joinpath(results_dir, "data_collapse_combined_y^$(n).png"))
    display(p_combined)
    return p_combined
end
"""
    corr_slice(corr4, ref)

Extract a 2D slice from a 4D correlation tensor at the reference
Cartesian index `ref::CartesianIndex{2}`: returns `corr4[i0,j0,:,:]`.
"""
function corr_slice(corr4::AbstractArray{T,4}, ref::CartesianIndex{2}) where T
    i0, j0 = Tuple(ref)
    return corr4[i0, j0, :, :]
end
function plot_spatial_correlation(spatial_corr, param)
    dim_num = length(param.dims)
    if dim_num == 1
        # Adjust dx_range to match the length of spatial_corr
        dx_range = range(-param.dims[1] ÷ 2, length=length(spatial_corr))
        plot(dx_range, spatial_corr,
             title="1D Spatial Correlation",
             xlabel="Δx", ylabel="Correlation",
             legend=false, lw=2)
        # Define the range for Δx
        # dx_range = range(-param.dims[1] �� 2, param.dims[1] ÷ 2 - 1, length=param.dims[1])
        # # Plot the 1D spatial correlation
        # plot(dx_range, spatial_corr,
        #      title="1D Spatial Correlation",
        #      xlabel="Δx", ylabel="Correlation",
        #      legend=false, lw=2)
    elseif dim_num == 2
        # Define the ranges for Δx and Δy
        dx_range = range(-param.dims[1] ÷ 2, param.dims[1] ÷ 2 - 1, length=param.dims[1])
        dy_range = range(-param.dims[2] ÷ 2, param.dims[2] ÷ 2 - 1, length=param.dims[2])
        # Plot the 2D spatial correlation as a heatmap
        heatmap(dx_range, dy_range, transpose(spatial_corr),
                title="2D Spatial Correlation",
                c=cgrad(:viridis),
                aspect_ratio=1, xlabel="Δx", ylabel="Δy")
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
end

function plot_directional_densities(state, param; title="Average Directional Densities")
    x_range = 1:param.dims[1]
    directional_pair = directional_density_pair(state)
    if directional_pair === nothing
        p = plot(
            title="$(title) (unavailable)",
            xlabel="Position",
            ylabel="Density",
            legend=:outerright,
            size=(1000,400)
        )
        plot!(p, x_range, state.ρ_avg, label="Density", color=:blue, lw=2, marker=:circle, markersize=4)
        return p
    end
    ρ₊, ρ₋ = directional_pair
    
    # Create the plot with both densities
    p = plot(
        title=title,
        xlabel="Position",
        ylabel="Density",
        legend=:outerright,
        size=(1000,400)
    )
    
    # Normalize the densities
    ρ₊_avg = ρ₊ / sum(ρ₊)
    ρ₋_avg = ρ₋ / sum(ρ₋)
    
    # Plot right-moving particles
    plot!(p, x_range, ρ₊_avg,
        label="Right-moving (ρ₊)",
        color=:red,
        lw=2,
        marker=:circle,
        markersize=4
    )
    
    # Plot left-moving particles
    plot!(p, x_range, ρ₋_avg,
        label="Left-moving (ρ₋)",
        color=:blue,
        lw=2,
        marker=:circle,
        markersize=4
    )
    
    # Add potential on secondary y-axis
    plot!(twinx(), x_range, state.potential.V,
        ylabel="Potential",
        label="Potential",
        color=:black,
        alpha=0.3,
        linestyle=:dash,
        legend=:outerright
    )
    
    return p
end
function plot_magnetization(state, param; title="Average Magnetization")
    x_range = 1:param.dims[1]
    directional_pair = directional_density_pair(state)
    if directional_pair === nothing
        return plot(
            x_range,
            zeros(length(x_range)),
            title="$(title) (unavailable)",
            xlabel="Position",
            ylabel="Magnetization",
            legend=false,
            size=(1000,400)
        )
    end
    ρ₊, ρ₋ = directional_pair
    
    # Calculate magnetization (ρ₊ - ρ₋)/(ρ₊ + ρ₋)
    magnetization = (ρ₊ - ρ₋) ./ (ρ₊ + ρ₋)
    
    # Create the plot
    p = plot(
        title=title,
        xlabel="Position",
        ylabel="Magnetization",
        legend=:outerright,
        size=(1000,400)
    )
    
    # Plot magnetization
    plot!(p, x_range, magnetization,
        label="m = (ρ₊ - ρ₋)/(ρ₊ + ρ₋)",
        color=:purple,
        lw=2,
        marker=:circle,
        markersize=4
    )
    
    # Add potential on secondary y-axis
    plot!(twinx(), x_range, state.potential.V,
        ylabel="Potential",
        label="Potential",
        color=:black,
        alpha=0.3,
        linestyle=:dash,
        legend=:outerright
    )
    
    return p
end

function make_movie!(state, param, n_frame, rng, file_name, in_fps; 
                    show_directions = false,
                    show_times = [],
                    save_times = [])
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_frame, true)
    
    # Initialize the animation
    anim = @animate for frame in 1:n_frame
        update_and_compute_correlations!(state, param, ρ_history, frame, rng)
        
        # Save state at specified times
        if frame in save_times
            save_dir = "saved_states"
            save_state(state,param,save_dir)
            println("State saved at sweep $frame to: ", filename)
        end

        # Show visualization at specified times
        if frame in show_times
            if show_directions && has_directional_densities(state)
                # Create two subplots side by side
                p1 = plot(title="Particle Densities by Direction",
                        xlabel="Position", ylabel="Density")
                
                # Plot right-moving particles
                ρ₊, ρ₋ = directional_density_pair(state)
                plot!(p1, 1:param.dims[1], ρ₊,
                    label="Right-moving", color=:red, 
                    marker=:circle, markersize=4)
                
                # Plot left-moving particles on the same graph
                plot!(p1, 1:param.dims[1], ρ₋,
                    label="Left-moving", color=:blue, 
                    marker=:circle, markersize=4)
                
                # Plot total density
                p2 = plot(1:param.dims[1], state.ρ,
                        title="Total Density",
                        xlabel="Position", ylabel="Density",
                        label="Total", color=:black,
                        marker=:circle, markersize=4)
                
                # Combine plots
                plot(p1, p2, layout=(2,1), size=(800,800))
            else
                p0 = plot_density(state.ρ_avg, param, state; title="Time averaged density")
                outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
                p4 = heatmap(state.ρ_matrix_avg - outer_prod_ρ, xlabel="x", ylabel="y", 
                            title="Correlation Matrix Heatmap", color=:viridis)

                p_show = plot(p0, p4, size=(1200,600), plot_title="frame $(frame)")
                display(p_show)
            end
        end
        
        # For the animation frame
        if show_directions && has_directional_densities(state)
            # ... existing show_directions plotting code ...
        else
            p0 = plot_density(state.ρ_avg, param, state; title="Time averaged density")
            outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
            p4 = heatmap(state.ρ_matrix_avg - outer_prod_ρ, xlabel="x", ylabel="y", 
                        title="Correlation Matrix Heatmap", color=:viridis)

            plot(p0, p4, size=(1200,600))
        end
        
        next!(prg)
    end
    
    println("Simulation complete, producing movie")
    name = @sprintf("%s.gif", file_name)
    gif(anim, name, fps = in_fps)

    # After movie is complete, show final statistics
    println("Generating final statistics...")
    
    # Calculate and display final statistics
    p0 = plot_density(state.ρ_avg, param, state; title="Time averaged density")
    outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
    p4 = heatmap(state.ρ_matrix_avg - outer_prod_ρ, xlabel="x", ylabel="y", 
                 title="Correlation Matrix Heatmap", color=:viridis)

    # Display final plots
    final_plots = plot(p0, p4, layout=(1,2), size=(1200,600))
    display(final_plots)
    
    # Save final statistics plot
    savefig(final_plots, replace(file_name, ".gif" => "_final_stats.png"))
end

function plot_decay_time_evolution(decay_times)
    p_decay = plot(decay_times, 
                   title="Evolution of Decay Time", 
                   xlabel="Frame", ylabel="τ", 
                   legend=false, lw=2)
    savefig(p_decay, "decay_time_evolution.png")
end

end
