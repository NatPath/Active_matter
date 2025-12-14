module PlotUtils
using Plots
using LsqFit
using Printf
using LinearAlgebra
export plot_sweep, plot_density, plot_data_colapse, plot_spatial_correlation 
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
    plot!(p, x_ref, y_ref .* scale; label="(x)/((x^2+1)^$(dim+1))", color=:gray, linestyle=:dashdot, lw=2)
    return true
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

function annotate_forcing_1d!(p, state)
    b1 = state.forcing.bond_indices[1][1]
    b2 = state.forcing.bond_indices[2][1]
    dir_label = state.forcing.direction_flag ? "force ->" : "force <-"
    vline!(p, [b1, b2], color=:red, linestyle=:dash, label=false)
    y_max = maximum(state.ρ_avg)
    y_pos = y_max * 0.9
    start_x, end_x = state.forcing.direction_flag ? (b1, b2) : (b2, b1)
    plot!(p, [start_x, end_x], [y_pos, y_pos],
          arrow=:arrow, color=:red, lw=3, label=false)
    annotate!(p, ((b1 + b2) / 2, y_pos * 1.05, text(dir_label, :red, 8)))
end

function annotate_forcing_2d!(p_current_density, state)
    b1 = state.forcing.bond_indices[1]
    b2 = state.forcing.bond_indices[2]
    if length(b1) == 2 && length(b2) == 2
        start = state.forcing.direction_flag ? b1 : b2
        dx = state.forcing.direction_flag ? (b2[1] - b1[1]) : (b1[1] - b2[1])
        dy = state.forcing.direction_flag ? (b2[2] - b1[2]) : (b1[2] - b2[2])
        quiver!(p_current_density,
                [start[1]], [start[2]],
                quiver=([dx], [dy]),
                color=:white,
                lw=3,
                arrow=:arrow,
                label=false)
        annotate!(p_current_density,
                  start[1] + 0.3*dx,
                  start[2] + 0.3*dy,
                  text("bond force", :white, 8))
    end
end

function plot_sweep(sweep, state, param; label="", plot_directional=false)
    dim_num = length(param.dims)
    if dim_num == 1
        return plot_sweep_1d(sweep, state, param; label=label, plot_directional=plot_directional)
    elseif dim_num == 2
        return plot_sweep_2d(sweep, state, param; label=label, plot_directional=plot_directional)
    else
        throw(DomainError("Only 1D or 2D plotting supported"))
    end
end

function plot_sweep_1d(sweep, state, param; label="", plot_directional=false)
    outer_prod_ρ = state.ρ_avg * transpose(state.ρ_avg)
    corr_mat = state.ρ_matrix_avg_cuts[:full] - outer_prod_ρ

    p0 = plot_density(state.ρ_avg, param, state; title="Time averaged density")
    if hasfield(typeof(state), :forcing)
        annotate_forcing_1d!(p0, state)
    end
    p1 = plot_magnetization(state, param)

    p4 = heatmap(corr_mat, xlabel="x", ylabel="y",
                 title="Correlation Matrix Heatmap", color=:viridis)
    L = param.dims[1]
    middle_spot = L ÷ 2

    p5 = plot(corr_mat[middle_spot, :], title="correlation matrix cut for x=$(middle_spot)")
    point_to_look_at = middle_spot + middle_spot ÷ 4
    vline!(p4, [point_to_look_at], label="x=$(point_to_look_at)")
    left_value = corr_mat[point_to_look_at, point_to_look_at-1]
    right_value = corr_mat[point_to_look_at, point_to_look_at+1]
    left_side = corr_mat[point_to_look_at, 1:point_to_look_at-1]
    right_side = corr_mat[point_to_look_at, point_to_look_at+1:end]
    corr_mat_cut = vcat(left_side, [(left_value + right_value)/2], right_side)

    p6 = plot(corr_mat_cut, title="correlation matrix cut for x=$(point_to_look_at) ")
    vline!(p6, [point_to_look_at], label="x=$(point_to_look_at)")

    corr_mat_antisym = remove_symmetric_part_reflection(corr_mat, middle_spot)
    corr_mat_antisym[point_to_look_at, point_to_look_at] = (corr_mat_antisym[point_to_look_at, point_to_look_at+1] + corr_mat_antisym[point_to_look_at, point_to_look_at-1]) / 2
    corr_mat_antisym[point_to_look_at, L - point_to_look_at] = (corr_mat_antisym[point_to_look_at, L - (point_to_look_at+1)] + corr_mat_antisym[point_to_look_at, L - (point_to_look_at-1)]) / 2

    p7 = plot(corr_mat_antisym[point_to_look_at, 1:end], title="anti-symmetric part of corr_mat cut for x=$(point_to_look_at) ")
    p_final = plot(p0, p1, p4, p5, p6, p7, size=(2100, 1000), plot_title="sweep $(sweep)", layout=grid(2, 3))
    display(p_final)
    return corr_mat
end

function plot_sweep_2d(sweep, state, param; label="", plot_directional=false)
    dims = param.dims
    fix_term = param.N / (prod(param.dims)^2)

    offset_middle = 0
    y0 = div(dims[2], 2) + offset_middle
    x0 = div(dims[1], 2) + offset_middle
    x_range = 1:dims[1]
    y_range = 1:dims[2]

    x_idx = Int(floor(3 * dims[1] / 4))
    y_idx = Int(floor(3 * dims[2] / 4))
    zero_indices = [x0-1, x0, x0+1]
    zero_indices_y = [y0-1, y0, y0+1]
    zero_indices_pos = [1, 2]
    zero_indices_y_pos = [1, 2]

    # Densities and potentials
    p_avg_density = heatmap(state.ρ_avg',
                            title="⟨ρ⟩ (t=$(sweep))",
                            xlabel="x", ylabel="y",
                            aspect_ratio=1, colorbar=true)

    density_x_cut = state.ρ_avg[:, y0]
    p_density_x_cut = plot(x_range, density_x_cut,
                           title="⟨ρ⟩ x-cut at y=$(y0)",
                           xlabel="x", ylabel="Density",
                           legend=false, lw=2, color=:blue)

    density_y_cut = state.ρ_avg[x0, :]
    p_density_y_cut = plot(y_range, density_y_cut,
                           title="⟨ρ⟩ y-cut at x=$(x0)",
                           xlabel="y", ylabel="Density",
                           legend=false, lw=2, color=:green)

    x_right_range = x0+1:dims[1]
    y_right_range = y0+1:dims[2]

    p_density_x_cut_log = plot(x_right_range, log10.(density_x_cut[x0+1:end]),
                               title="⟨ρ⟩ x-cut at y=$(y0) (log-log scale)",
                               xlabel="x (log)", ylabel="Density (log)",
                               legend=false, lw=2, color=:blue,
                               xscale=:log10)

    p_density_y_cut_log = plot(y_right_range, log10.(density_y_cut[y0+1:end]),
                               title="⟨ρ⟩ y-cut at x=$(x0) (log-log scale)",
                               xlabel="y (log)", ylabel="Density (log)",
                               legend=false, lw=2, color=:green,
                               xscale=:log10)

    p_potential = heatmap(state.potential.V',
                          title="Potential V(x,y)",
                          xlabel="x", ylabel="y",
                          aspect_ratio=1, colorbar=true,
                          color=:reds)

    current_density = state.ρ ./ sum(state.ρ)
    p_current_density = heatmap(current_density',
                                title="Current ρ(x,y) (t=$(sweep))",
                                xlabel="x", ylabel="y",
                                aspect_ratio=1, colorbar=true,
                                color=:inferno)

    if hasfield(typeof(state), :forcing)
        annotate_forcing_2d!(p_current_density, state)
    end

    # X-axis correlations
    corr_mat2 = correlation_slice_x(state, param, y0, fix_term)
    p_corr_x_axis = heatmap(corr_mat2,
                            title="C(x₁,y=$offset_middle; x₂,y=$offset_middle)",
                            xlabel="x₁", ylabel="x₂",
                            aspect_ratio=1, colorbar=true)

    line_data = corr_mat2[x_idx, :]
    p_x_cut_full = plot(x_range, line_data,
                        title="C at x₁=3/4·L₁ (idx=$(x_idx))",
                        xlabel="x₂", ylabel="C",
                        legend=false, lw=2)

    line_data_zeroed = zero_middle(line_data, zero_indices)
    p_x_cut_zeroed = plot(x_range, line_data_zeroed,
                          title="C at x₁=3/4·L₁ (middle zeroed)",
                          xlabel="x₂", ylabel="C",
                          legend=false, lw=2, color=:blue)
    scatter_zero_marks!(p_x_cut_zeroed, zero_indices)

    positive_x_positions, positive_line_data = positive_half(line_data, x0)
    p_x_cut_positive = plot(positive_x_positions, positive_line_data,
                            title="C at x₁=3/4·L₁ (Positive Half)",
                            xlabel="Distance from center", ylabel="C",
                            legend=false, lw=2, color=:orange)

    positive_line_data_zeroed = zero_middle(positive_line_data, zero_indices_pos)
    p_x_cut_positive_zeroed = plot(positive_x_positions, positive_line_data_zeroed,
                                   title="C at x₁=3/4·L₁ (Positive Half, middle zeroed)",
                                   xlabel="Distance from center", ylabel="C",
                                   legend=false, lw=2, color=:blue)
    scatter_zero_marks!(p_x_cut_positive_zeroed, positive_x_positions[zero_indices_pos])

    corr_mat2_antisym = antisymmetric_with_smoothing(corr_mat2, x0)
    line_data_antisym = corr_mat2_antisym[x_idx, :]
    p_x_cut_antisymmetric = plot(x_range, line_data_antisym,
                                 title="C at x₁=3/4·L₁ (Antisymmetric)",
                                 xlabel="x₂", ylabel="C",
                                 legend=false, lw=2, color=:red)

    line_data_antisym_zeroed = zero_middle(line_data_antisym, zero_indices)
    p_x_cut_antisymmetric_zeroed = plot(x_range, line_data_antisym_zeroed,
                                        title="C at x₁=3/4·L₁ (Antisymmetric, middle zeroed)",
                                        xlabel="x₂", ylabel="C",
                                        legend=false, lw=2, color=:red)
    scatter_zero_marks!(p_x_cut_antisymmetric_zeroed, zero_indices; color=:darkred)

    # Diagonal correlations
    corr_diag = correlation_diag(state, param, fix_term)
    p_corr_diag = heatmap(corr_diag,
                          title="C(x,x; x',x') - Diagonal",
                          xlabel="x", ylabel="x'",
                          aspect_ratio=1, colorbar=true)

    diag_line_data = corr_diag[x_idx, :]
    p_diag_cut_full = plot(x_range, diag_line_data,
                           title="Diagonal C at x=3/4·L₁ (idx=$(x_idx))",
                           xlabel="x'", ylabel="C",
                           legend=false, lw=2, color=:green)

    diag_line_zeroed = zero_middle(diag_line_data, zero_indices)
    p_diag_cut_zeroed = plot(x_range, diag_line_zeroed,
                             title="Diagonal C at x=3/4·L₁ (middle zeroed)",
                             xlabel="x'", ylabel="C",
                             legend=false, lw=2, color=:green)
    scatter_zero_marks!(p_diag_cut_zeroed, zero_indices)

    positive_diag_x_positions, positive_diag_line_data = positive_half(diag_line_data, x0)
    p_diag_cut_positive = plot(positive_diag_x_positions, positive_diag_line_data,
                               title="Diagonal C at x=3/4·L₁ (Positive Half)",
                               xlabel="Distance from center", ylabel="C",
                               legend=false, lw=2, color=:purple)

    positive_diag_line_data_zeroed = zero_middle(positive_diag_line_data, zero_indices_pos)
    p_diag_cut_positive_zeroed = plot(positive_diag_x_positions, positive_diag_line_data_zeroed,
                                      title="Diagonal C at x=3/4·L₁ (Positive Half, middle zeroed)",
                                      xlabel="Distance from center", ylabel="C",
                                      legend=false, lw=2, color=:purple)
    scatter_zero_marks!(p_diag_cut_positive_zeroed, positive_diag_x_positions[zero_indices_pos])

    corr_diag_antisym = antisymmetric_with_smoothing(corr_diag, x0)
    diag_line_data_antisym = corr_diag_antisym[x_idx, :]
    p_diag_cut_antisymmetric = plot(x_range, diag_line_data_antisym,
                                    title="Diagonal C at x=3/4·L₁ (Antisymmetric)",
                                    xlabel="x'", ylabel="C",
                                    legend=false, lw=2, color=:red)

    diag_line_data_antisym_zeroed = zero_middle(diag_line_data_antisym, zero_indices)
    p_diag_cut_antisymmetric_zeroed = plot(x_range, diag_line_data_antisym_zeroed,
                                           title="Diagonal C at x=3/4·L₁ (Antisymmetric, middle zeroed)",
                                           xlabel="x'", ylabel="C",
                                           legend=false, lw=2, color=:red)
    scatter_zero_marks!(p_diag_cut_antisymmetric_zeroed, zero_indices; color=:darkred)

    # Y-axis correlations
    corr_mat_y = correlation_slice_y(state, param, x0, fix_term)
    p_corr_y_axis = heatmap(corr_mat_y,
                            title="C(x=$offset_middle,y₁; x=$offset_middle,y₂)",
                            xlabel="y₁", ylabel="y₂",
                            aspect_ratio=1, color=:plasma, colorbar=true)

    line_data_y = corr_mat_y[y_idx, :]
    p_y_cut_full = plot(y_range, line_data_y,
                        title="C at y₁=3/4·L₂ (idx=$(y_idx))",
                        xlabel="y₂", ylabel="C",
                        legend=false, lw=2, color=:cyan)

    line_data_y_zeroed = zero_middle(line_data_y, zero_indices_y)
    p_y_cut_zeroed = plot(y_range, line_data_y_zeroed,
                          title="C at y₁=3/4·L₂ (middle zeroed)",
                          xlabel="y₂", ylabel="C",
                          legend=false, lw=2, color=:cyan)
    scatter_zero_marks!(p_y_cut_zeroed, zero_indices_y)

    positive_y_positions, positive_line_data_y = positive_half(line_data_y, y0)
    p_y_cut_positive = plot(positive_y_positions, positive_line_data_y,
                            title="C at y₁=3/4·L₂ (Positive Half)",
                            xlabel="Distance from center", ylabel="C",
                            legend=false, lw=2, color=:cyan)

    positive_line_data_y_zeroed = zero_middle(positive_line_data_y, zero_indices_y_pos)
    p_y_cut_positive_zeroed = plot(positive_y_positions, positive_line_data_y_zeroed,
                                   title="C at y₁=3/4·L₂ (Positive Half, middle zeroed)",
                                   xlabel="Distance from center", ylabel="C",
                                   legend=false, lw=2, color=:cyan)
    scatter_zero_marks!(p_y_cut_positive_zeroed, positive_y_positions[zero_indices_y_pos])

    corr_mat_y_antisym = antisymmetric_with_smoothing(corr_mat_y, y0)
    line_data_y_antisym = corr_mat_y_antisym[y_idx, :]
    p_y_cut_antisymmetric = plot(y_range, line_data_y_antisym,
                                 title="C at y₁=3/4·L₂ (Antisymmetric)",
                                 xlabel="y₂", ylabel="C",
                                 legend=false, lw=2, color=:cyan)

    line_data_y_antisym_zeroed = zero_middle(line_data_y_antisym, zero_indices_y)
    p_y_cut_antisymmetric_zeroed = plot(y_range, line_data_y_antisym_zeroed,
                                        title="C at y₁=3/4·L₂ (Antisymmetric, middle zeroed)",
                                        xlabel="y₂", ylabel="C",
                                        legend=false, lw=2, color=:cyan)
    scatter_zero_marks!(p_y_cut_antisymmetric_zeroed, zero_indices_y; color=:darkcyan)

    p_empty = plot(axis=false, showaxis=false, grid=false, title="")

    display(plot(p_avg_density, p_density_x_cut, p_density_y_cut, p_current_density,
                 p_potential, p_density_x_cut_log, p_density_y_cut_log, p_empty,
                 p_corr_x_axis, p_x_cut_full, p_x_cut_positive, p_x_cut_antisymmetric,
                 p_empty, p_x_cut_zeroed, p_x_cut_positive_zeroed, p_x_cut_antisymmetric_zeroed,
                 p_corr_y_axis, p_y_cut_full, p_y_cut_positive, p_y_cut_antisymmetric,
                 p_empty, p_y_cut_zeroed, p_y_cut_positive_zeroed, p_y_cut_antisymmetric_zeroed,
                 p_corr_diag, p_diag_cut_full, p_diag_cut_positive, p_diag_cut_antisymmetric,
                 p_empty, p_diag_cut_zeroed, p_diag_cut_positive_zeroed, p_diag_cut_antisymmetric_zeroed,
                 layout=(9, 4), size=(2400, 3600),
                 plot_title="2D sweep $(sweep)"))
    return corr_mat2
end
function plot_density(density, param, state; title="Density", show_directions=false)
    dim_num = length(size(density))
    if dim_num==1
        if !show_directions
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
            # Create subplot with total density, right-moving and left-moving particles
            x_range = 1:param.dims[1]
            p1 = plot(x_range, density, 
                     title="Total Density", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter)
            
            p2 = plot(x_range, state.ρ₊,
                     title="Right-moving (ρ₊)", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter,
                     color=:red)
            
            p3 = plot(x_range, state.ρ₋,
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
        heatmap(x_range, y_range, transpose(density), 
                title=title, 
                c=cgrad(:inferno), xlims=(1, Lx), ylims=(1, Ly), 
                clims=(0,3), aspect_ratio=1, xlabel="x", ylabel="y")
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
    return p
end
function plot_data_colapse(states_params_names, power_n, indices, results_dir = "results_figures", do_fit=true; show_powerlaw=true)
    n = Float64(power_n)
    extra_power_exp = 2.0  # reference slope -2
    all_x = []
    all_y = []

    p_combined = plot(title="Combined Data Collapse f(x/y)=C(x,y)⋅y^$n", legend=:outerright, size=(1200,800))

    for (idx, (state, param, label)) in enumerate(states_params_names)
        α = param.α
        γ′ = param.γ * param.N
        dim_num = length(param.dims)

        if dim_num == 1
            L = param.dims[1]
            N = param.N
            # Setup output directories
            full_dir = "$(results_dir)/full_data"
            antisym_dir = "$(results_dir)/antisymmetric"
            sym_dir = "$(results_dir)/symmetric"
            mkpath(full_dir)
            mkpath(antisym_dir)
            mkpath(sym_dir)

            p_full_combined = plot(title="Full Data Collapse - C(x,y)⋅y^$n", legend=:outerright, size=(1000,600))
            p_antisym_combined = plot(title="Antisymmetric Data Collapse - C(x,y)⋅y^$n", legend=:outerright, size=(1000,600))
            p_sym_combined = plot(title="Symmetric Data Collapse - C(x,y)⋅y^$n", legend=:outerright, size=(1000,600))
            ref_dim_added = Dict(:full=>false, :antisym=>false, :sym=>false, :combined=>false)
            add_dimension_reference!(p_full_combined, dim_num)
            add_dimension_reference!(p_antisym_combined, dim_num)
            add_dimension_reference!(p_sym_combined, dim_num)
            if show_powerlaw
                p_full_powerlaw = plot(title="Power law check (full cut)", legend=:outerright, size=(800,600),
                                       xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_antisym_powerlaw = plot(title="Power law check (antisymmetric cut)", legend=:outerright, size=(800,600),
                                          xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_sym_powerlaw = plot(title="Power law check (symmetric cut)", legend=:outerright, size=(800,600),
                                       xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                ref_added_full = false
                ref_added_antisym = false
                ref_added_sym = false
                ref_added_full_extra = false
                ref_added_antisym_extra = false
                ref_added_sym_extra = false
            end

            for i in indices
                outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
                corr_mat = (state.ρ_matrix_avg_cuts[:full] - outer_prod_ρ) .+ (N / L^2)
                middle_spot = L ÷ 2
                point_to_look_at = Int(middle_spot + i)

                corr_mat_collapsed = corr_mat[:, point_to_look_at]
                left_value = corr_mat_collapsed[point_to_look_at - 1]
                right_value = corr_mat_collapsed[point_to_look_at + 1]
                left_side = corr_mat_collapsed[1:point_to_look_at-1]
                right_side = corr_mat_collapsed[point_to_look_at+1:end]
                full_data = vcat(left_side, [(left_value + right_value)/2], right_side)

                corr_mat_antisym = remove_symmetric_part_reflection(corr_mat, middle_spot)
                corr_mat_antisym[point_to_look_at, point_to_look_at] = (corr_mat_antisym[point_to_look_at, point_to_look_at + 1] + corr_mat_antisym[point_to_look_at, point_to_look_at - 1]) / 2
                corr_mat_antisym[point_to_look_at, L - point_to_look_at] = (corr_mat_antisym[point_to_look_at, L - (point_to_look_at + 1)] + corr_mat_antisym[point_to_look_at, L - (point_to_look_at - 1)]) / 2
                antisym_data = corr_mat_antisym[point_to_look_at, 1:end]

                corr_mat_sym = remove_antisymmetric_part_reflection(corr_mat, middle_spot)
                corr_mat_sym[point_to_look_at, point_to_look_at] = (corr_mat_sym[point_to_look_at, point_to_look_at + 1] + corr_mat_sym[point_to_look_at, point_to_look_at - 1]) / 2
                corr_mat_sym[point_to_look_at, L - point_to_look_at] = (corr_mat_sym[point_to_look_at, L - (point_to_look_at + 1)] + corr_mat_sym[point_to_look_at, L - (point_to_look_at - 1)]) / 2
                sym_data = corr_mat_sym[point_to_look_at, 1:end]

                x_positions = 1:length(full_data)
                x_scaled = (x_positions .- middle_spot) ./ (i)

                for (data, p_combined_plot, key) in zip((full_data, antisym_data, sym_data), (p_full_combined, p_antisym_combined, p_sym_combined), (:full, :antisym, :sym))
                    y_scaled = data .* i^n
                    mask = (-5 .<= x_scaled .<= 5)
                    x_filtered = x_scaled[mask]
                    y_filtered = y_scaled[mask]
                    plot!(p_combined_plot, x_filtered, y_filtered, label="y=$(i)", lw=2)
                    if !ref_dim_added[key]
                        ref_dim_added[key] = add_dimension_reference!(p_combined_plot, dim_num, x_filtered, y_filtered)
                    end
                end

                y_scaled = full_data .* i^n
                mask = (-5 .<= x_scaled .<= 5)
                x_filtered = x_scaled[mask]
                y_filtered = y_scaled[mask]
                append!(all_x, x_filtered)
                append!(all_y, y_filtered)
                plot!(p_combined, x_filtered, y_filtered, label="$(label) y=$(i)", linewidth=2)
                if !ref_dim_added[:combined]
                    ref_dim_added[:combined] = add_dimension_reference!(p_combined, dim_num, x_filtered, y_filtered)
                end
                if show_powerlaw
                    distances = abs.(x_positions .- middle_spot)
                    ref_added_full |= add_powerlaw_series!(p_full_powerlaw, distances, full_data, n; label="$(label) y=$(i)", add_reference=!ref_added_full, extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_full_extra)
                    ref_added_antisym |= add_powerlaw_series!(p_antisym_powerlaw, distances, antisym_data, n; label="$(label) y=$(i)", add_reference=!ref_added_antisym, extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_antisym_extra)
                    ref_added_sym |= add_powerlaw_series!(p_sym_powerlaw, distances, sym_data, n; label="$(label) y=$(i)", add_reference=!ref_added_sym, extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_sym_extra)
                    ref_added_full_extra = true
                    ref_added_antisym_extra = true
                    ref_added_sym_extra = true
                end
            end

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
            N = param.N
            fix_term = N / (prod(dims)^2)
            
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

            p_x_combined = plot(title="X-axis Cut Data Collapse - C(x,y)⋅y^$n", legend=:outerright, size=(1000,600))
            p_diag_combined = plot(title="Diagonal Cut Data Collapse - C(x,x)⋅y^$n", legend=:outerright, size=(1000,600))
            p_x_pos_combined = plot(title="X-axis Positive Cut Data Collapse - C(x,y)⋅y^$n", legend=:outerright, size=(1000,600))
            p_diag_pos_combined = plot(title="Diagonal Positive Cut Data Collapse - C(x,x)⋅y^$n", legend=:outerright, size=(1000,600))
            p_x_antisym_combined = plot(title="X-axis Antisymmetric Cut Data Collapse - C(x,y)⋅y^$n", legend=:outerright, size=(1000,600))
            p_diag_antisym_combined = plot(title="Diagonal Antisymmetric Cut Data Collapse - C(x,x)⋅y^$n", legend=:outerright, size=(1000,600))
            ref_dim_added = Dict(
                :x => false, :diag => false, :x_pos => false, :diag_pos => false,
                :x_antisym => false, :diag_antisym => false, :combined => false
            )
            if show_powerlaw
                p_x_powerlaw = plot(title="Power law check (x-axis cut)", legend=:outerright, size=(800,600),
                                    xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_diag_powerlaw = plot(title="Power law check (diagonal cut)", legend=:outerright, size=(800,600),
                                       xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_x_pos_powerlaw = plot(title="Power law check (x-axis positive cut)", legend=:outerright, size=(800,600),
                                        xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_diag_pos_powerlaw = plot(title="Power law check (diagonal positive cut)", legend=:outerright, size=(800,600),
                                           xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_x_antisym_powerlaw = plot(title="Power law check (x-axis antisymmetric cut)", legend=:outerright, size=(800,600),
                                            xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
                p_diag_antisym_powerlaw = plot(title="Power law check (diagonal antisymmetric cut)", legend=:outerright, size=(800,600),
                                               xscale=:log10, yscale=:log10, xlabel="|Δ|", ylabel="|C|")
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
                            plot!(p_plot, x_filtered, y_filtered, label="$(label) y=$(i)", lw=2)
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
                        plot!(p_combined, x_filtered, y_filtered, label="$(label) y=$(i)", linewidth=2)
                        if !ref_dim_added[:combined]
                            ref_dim_added[:combined] = add_dimension_reference!(p_combined, dim_num, x_filtered, y_filtered)
                        end
                    else
                        println("Warning: Skipping combined plot for i=$i due to extreme values")
                    end
                end

                if show_powerlaw
                    distances = abs.(x_positions .- middle_spot)
                    ref_added[:x] |= add_powerlaw_series!(p_x_powerlaw, distances, x_axis_data, n; label="$(label) y=$(i)", add_reference=!ref_added[:x], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:x])
                    ref_added[:diag] |= add_powerlaw_series!(p_diag_powerlaw, distances, diag_data, n; label="$(label) y=$(i)", add_reference=!ref_added[:diag], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:diag])
                    distances_pos = x_positions_pos .- middle_spot
                    ref_added[:x_pos] |= add_powerlaw_series!(p_x_pos_powerlaw, distances_pos, x_axis_positive_data, n; label="$(label) y=$(i)", add_reference=!ref_added[:x_pos], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:x_pos])
                    ref_added[:diag_pos] |= add_powerlaw_series!(p_diag_pos_powerlaw, distances_pos, diag_positive_data, n; label="$(label) y=$(i)", add_reference=!ref_added[:diag_pos], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:diag_pos])
                    ref_added[:x_antisym] |= add_powerlaw_series!(p_x_antisym_powerlaw, distances, x_axis_antisym_data, n; label="$(label) y=$(i)", add_reference=!ref_added[:x_antisym], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:x_antisym])
                    ref_added[:diag_antisym] |= add_powerlaw_series!(p_diag_antisym_powerlaw, distances, diag_antisym_data, n; label="$(label) y=$(i)", add_reference=!ref_added[:diag_antisym], extra_reference_exponent=extra_power_exp, add_extra_reference=!ref_added_extra[:diag_antisym])
                    ref_added_extra[:x] = true
                    ref_added_extra[:diag] = true
                    ref_added_extra[:x_pos] = true
                    ref_added_extra[:diag_pos] = true
                    ref_added_extra[:x_antisym] = true
                    ref_added_extra[:diag_antisym] = true
                end
            end
            
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
    
    # Create the plot with both densities
    p = plot(
        title=title,
        xlabel="Position",
        ylabel="Density",
        legend=:outerright,
        size=(1000,400)
    )
    
    # Normalize the densities
    ρ₊_avg = state.ρ₊ / sum(state.ρ₊)
    ρ₋_avg = state.ρ₋ / sum(state.ρ₋)
    
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
    
    # Calculate magnetization (ρ₊ - ρ₋)/(ρ₊ + ρ₋)
    magnetization = (state.ρ₊ - state.ρ₋) ./ (state.ρ₊ + state.ρ₋)
    
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
            if show_directions
                # Create two subplots side by side
                p1 = plot(title="Particle Densities by Direction",
                        xlabel="Position", ylabel="Density")
                
                # Plot right-moving particles
                plot!(p1, 1:param.dims[1], state.ρ₊, 
                    label="Right-moving", color=:red, 
                    marker=:circle, markersize=4)
                
                # Plot left-moving particles on the same graph
                plot!(p1, 1:param.dims[1], state.ρ₋, 
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
        if show_directions
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
