module PlotUtils
using Plots
using LsqFit
using Printf
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
function plot_sweep(sweep,state,param; label="", plot_directional=false)
    dim_num = length(param.dims)
    if dim_num==1
        normalized_dist = state.ρ_avg / sum(state.ρ_avg)
        outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
        corr_mat = state.ρ_matrix_avg_cuts[:full]-outer_prod_ρ
        p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
        p1 = plot_magnetization(state, param)
        
        # Remove potential profile plot for 1D case
        
        p4 = heatmap(corr_mat, xlabel="x", ylabel="y", 
                    title="Correlation Matrix Heatmap", color=:viridis)
        L= param.dims[1]
        middle_spot = L÷2
        
        p5 = plot(corr_mat[middle_spot,:],title="correlation matrix cut for x=$(middle_spot)")
        point_to_look_at = middle_spot+middle_spot÷4
        vline!(p4,[point_to_look_at],label="x=$(point_to_look_at)")
        left_value=corr_mat[point_to_look_at,point_to_look_at-1]
        right_value=corr_mat[point_to_look_at,point_to_look_at+1]
        left_side=corr_mat[point_to_look_at, 1:point_to_look_at-1]
        right_side=corr_mat[point_to_look_at, point_to_look_at+1:end]
        corr_mat_cut = vcat(left_side,[(left_value+right_value)/2],right_side)

        p6 = plot(corr_mat_cut,title="correlation matrix cut for x=$(point_to_look_at) ")
        #p6 = plot(vcat(left_side,[(left_value+right_value)/2],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
        vline!(p6,[point_to_look_at],label="x=$(point_to_look_at)")
        # p6 = plot(vcat(left_side,[corr_mat[point_to_look_at,point_to_look_at]],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
        # p6= plot(corr_mat[point_to_look_at,:])
        corr_mat_antisym = remove_symmetric_part_reflection(corr_mat,middle_spot)
        # left_value_sym=corr_mat_sym[point_to_look_at,point_to_look_at-1]
        # right_value_sym=corr_mat_sym[point_to_look_at,point_to_look_at+1]
        # left_side_sym=corr_mat_sym[point_to_look_at, 1:point_to_look_at-1]
        # right_side_sym =corr_mat_sym[point_to_look_at, point_to_look_at+1:end]
        # corr_mat_sym_cut = vcat(left_side_sym,[(left_value_sym+right_value_sym)/2],right_side_sym)

        corr_mat_antisym[point_to_look_at,point_to_look_at] = (corr_mat_antisym[point_to_look_at,point_to_look_at+1]+corr_mat_antisym[point_to_look_at,point_to_look_at-1])/2
        corr_mat_antisym[point_to_look_at,L-point_to_look_at] = (corr_mat_antisym[point_to_look_at,L-(point_to_look_at+1)]+corr_mat_antisym[point_to_look_at,L-(point_to_look_at-1)])/2

        p7 = plot(corr_mat_antisym[point_to_look_at,1:end],title="anti-symmetric part of corr_mat cut for x=$(point_to_look_at) ")
        p_final=plot(p0,p1,p4,p5,p6,p7, size=(2100,1000),plot_title="sweep $(sweep)",layout=grid(2,3))
        display(p_final)
        return normalized_dist, corr_mat
    elseif dim_num == 2
        # --- 2D plotting with y=0 cut ---
        dims = param.dims
        y0 = div(dims[2] + 1, 2)               # middle index for y=0

        # 1) Time‐averaged density heatmap
        normalized_dist = state.ρ_avg ./ sum(state.ρ_avg)
        p_avg_density = heatmap(normalized_dist',
                     title="⟨ρ⟩ (t=$(sweep))",
                     xlabel="x", ylabel="y",
                     aspect_ratio=1, colorbar=true)

        # Add x-axis cut of average density at y=middle
        y_middle = div(dims[2] + 1, 2)
        x_range = 1:dims[1]
        density_x_cut = normalized_dist[:, y_middle]
        p_density_x_cut = plot(x_range, density_x_cut,
                     title="⟨ρ⟩ x-cut at y=$(y_middle)",
                     xlabel="x", ylabel="Density",
                     legend=false, lw=2, color=:blue)

        # Add y-axis cut of average density at x=middle
        x_middle = div(dims[1] + 1, 2)
        y_range = 1:dims[2]
        density_y_cut = normalized_dist[x_middle, :]
        p_density_y_cut = plot(y_range, density_y_cut,
                     title="⟨ρ⟩ y-cut at x=$(x_middle)",
                     xlabel="y", ylabel="Density",
                     legend=false, lw=2, color=:green)

        # Add log plots of density cuts
        # Plot from middle toward right end only
        x_right_range = x_middle+1:dims[1]
        y_right_range = y_middle+1:dims[2]
        
        density_x_cut_right = density_x_cut[x_middle+1:end]
        density_y_cut_right = density_y_cut[y_middle+1:end]

        p_density_x_cut_log = plot(x_right_range, log10.(density_x_cut_right),
                     title="⟨ρ⟩ x-cut at y=$(y_middle) (log-log scale)",
                     xlabel="x (log)", ylabel="Density (log)",
                     legend=false, lw=2, color=:blue,
                     xscale=:log10 )

        p_density_y_cut_log = plot(y_right_range, log10.(density_y_cut_right),
                     title="⟨ρ⟩ y-cut at x=$(x_middle) (log-log scale)",
                     xlabel="y (log)", ylabel="Density (log)",
                     legend=false, lw=2, color=:green,
                     xscale=:log10 )

        # Add 2D potential profile heatmap
        p_potential = heatmap(state.potential.V',
                       title="Potential V(x,y)",
                       xlabel="x", ylabel="y",
                       aspect_ratio=1, colorbar=true,
                       color=:reds)

        # Add current density heatmap
        current_density = state.ρ ./ sum(state.ρ)
        p_current_density = heatmap(current_density',
                           title="Current ρ(x,y) (t=$(sweep))",
                           xlabel="x", ylabel="y",
                           aspect_ratio=1, colorbar=true,
                           color=:inferno)

        # 2) Extract correlation C(x1,y0; x2,y0)
        fix_term = param.N / (prod(param.dims)^2)

        slice2d_x = state.ρ_matrix_avg_cuts[:x_cut]     # dims[1]×dims[1]
        mean_vec = state.ρ_avg[:, y0]
        corr_mat2 = slice2d_x .- (mean_vec * transpose(mean_vec)) .+ fix_term

        # Remove diagonal peaks by averaging neighboring values
        for i in 1:dims[1]
            left_idx = i == 1 ? dims[1] : i - 1
            right_idx = i == dims[1] ? 1 : i + 1
            corr_mat2[i, i] = (corr_mat2[i, left_idx] + corr_mat2[i, right_idx]) / 2
        end

        # 3) Plot full 2D correlation at y=0
        p_corr_x_axis = heatmap(corr_mat2,
                     title="C(x₁,y=0; x₂,y=0)",
                     xlabel="x₁", ylabel="x₂",
                     aspect_ratio=1, colorbar=true)

        # 4) 1D cut at x₁ = 3/4·L₁
        x_idx = Int(floor(3 * dims[1] / 4))
        x_range = 1:dims[1]
        line_data = corr_mat2[x_idx, :]
        p_x_cut_full = plot(x_range, line_data,
                  title="C at x₁=3/4·L₁ (idx=$(x_idx))",
                  xlabel="x₂", ylabel="C",
                  legend=false, lw=2)

        # 5) Same 1D cut but with middle region zeroed
        line_data_zeroed = copy(line_data)
        middle_x = div(dims[1] + 1, 2)
        zero_indices = [middle_x-1, middle_x, middle_x+1]
        line_data_zeroed[zero_indices] .= 0
        
        p_x_cut_zeroed = plot(x_range, line_data_zeroed,
                  title="C at x₁=3/4·L₁ (middle zeroed)",
                  xlabel="x₂", ylabel="C",
                  legend=false, lw=2, color=:blue)
        
        # Mark the zeroed points
        scatter!(p_x_cut_zeroed, zero_indices, zeros(length(zero_indices)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # 5.1) Positive half of y=0 cut (from middle to end)
        middle_x = div(dims[1] + 1, 2)
        positive_x_range = middle_x:dims[1]
        positive_line_data = line_data[middle_x:end]
        positive_x_positions = positive_x_range .- middle_x .+ 1
        p_x_cut_positive = plot(positive_x_positions, positive_line_data,
                  title="C at x₁=3/4·L₁ (Positive Half)",
                  xlabel="Distance from center", ylabel="C",
                  legend=false, lw=2, color=:orange)

        # 5.2) Same positive half but with middle region zeroed
        positive_line_data_zeroed = copy(positive_line_data)
        zero_indices_pos = [1,2]
        positive_line_data_zeroed[zero_indices_pos] .= 0
        p_x_cut_positive_zeroed = plot(positive_x_positions, positive_line_data_zeroed,
                  title="C at x₁=3/4·L₁ (Positive Half, middle zeroed)",
                  xlabel="Distance from center", ylabel="C",
                  legend=false, lw=2, color=:blue)
        
        # Mark the zeroed points in positive half
        scatter!(p_x_cut_positive_zeroed, positive_x_positions[zero_indices_pos], 
                 zeros(sum(zero_indices_pos)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # 6) Diagonal correlation C(x,x; x',x') 
        # corr_diag = zeros(dims[1], dims[1])
        # for i in 1:dims[1], j in 1:dims[1]
        #     corr_diag[i,j] = state.ρ_matrix_avg[i, i, j, j] - state.ρ_avg[i,i] * state.ρ_avg[j,j] + fix_term
        # end
        corr_diag = state.ρ_matrix_avg_cuts[:diag] .- (state.ρ_avg * transpose(state.ρ_avg)) .+ fix_term
        
        # Apply diagonal smoothing to diagonal correlation
        for i in 1:dims[1]
            left_idx = i == 1 ? dims[1] : i - 1
            right_idx = i == dims[1] ? 1 : i + 1
            corr_diag[i, i] = (corr_diag[i, left_idx] + corr_diag[i, right_idx]) / 2
        end
        
        p_corr_diag = heatmap(corr_diag,
                     title="C(x,x; x',x') - Diagonal",
                     xlabel="x", ylabel="x'",
                     aspect_ratio=1, colorbar=true)

        # 7) 1D cut of diagonal at x = 3/4·L₁
        diag_line_data = corr_diag[x_idx, :]
        p_diag_cut_full = plot(x_range, diag_line_data,
                  title="Diagonal C at x=3/4·L₁ (idx=$(x_idx))",
                  xlabel="x'", ylabel="C",
                  legend=false, lw=2, color=:green)

        # 8) Same diagonal cut but with middle region zeroed
        diag_line_zeroed = copy(diag_line_data)
        diag_line_zeroed[zero_indices] .= 0
        
        p_diag_cut_zeroed = plot(x_range, diag_line_zeroed,
                  title="Diagonal C at x=3/4·L₁ (middle zeroed)",
                  xlabel="x'", ylabel="C",
                  legend=false, lw=2, color=:green)
        
        # Mark the zeroed points
        scatter!(p_diag_cut_zeroed, zero_indices, zeros(length(zero_indices)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # 8.1) Positive half of diagonal cut (from middle to end)
        positive_diag_x_positions = positive_x_positions
        positive_diag_line_data = diag_line_data[middle_x:end]
        p_diag_cut_positive = plot(positive_diag_x_positions, positive_diag_line_data,
                  title="Diagonal C at x=3/4·L₁ (Positive Half)",
                  xlabel="Distance from center", ylabel="C",
                  legend=false, lw=2, color=:purple)

        # 8.2) Same positive half but with middle region zeroed
        positive_diag_line_data_zeroed = copy(positive_diag_line_data)
        positive_diag_line_data_zeroed[zero_indices_pos] .= 0
        p_diag_cut_positive_zeroed = plot(positive_diag_x_positions, positive_diag_line_data_zeroed,
                  title="Diagonal C at x=3/4·L₁ (Positive Half, middle zeroed)",
                  xlabel="Distance from center", ylabel="C",
                  legend=false, lw=2, color=:purple)
        
        # Mark the zeroed points in positive half
        scatter!(p_diag_cut_positive_zeroed, positive_diag_x_positions[zero_indices_pos], 
                 zeros(sum(zero_indices_pos)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # 9) Antisymmetric parts (remove symmetric component)
        # For y=0 cut - remove symmetric part
        corr_mat2_antisym = remove_symmetric_part_reflection(corr_mat2, middle_x)
        # Apply same diagonal smoothing to antisymmetric part
        for i in 1:dims[1]
            left_idx = i == 1 ? dims[1] : i - 1
            right_idx = i == dims[1] ? 1 : i + 1
            corr_mat2_antisym[i, i] = (corr_mat2_antisym[i, left_idx] + corr_mat2_antisym[i, right_idx]) / 2
        end
        
        line_data_antisym = corr_mat2_antisym[x_idx, :]
        p_x_cut_antisymmetric = plot(x_range, line_data_antisym,
                         title="C at x₁=3/4·L₁ (Antisymmetric)",
                         xlabel="x₂", ylabel="C",
                         legend=false, lw=2, color=:red)

        # For diagonal cut - remove symmetric part  
        corr_diag_antisym = remove_symmetric_part_reflection(corr_diag, middle_x)
        # Apply same diagonal smoothing
        for i in 1:dims[1]
            left_idx = i == 1 ? dims[1] : i - 1
            right_idx = i == dims[1] ? 1 : i + 1
            corr_diag_antisym[i, i] = (corr_diag_antisym[i, left_idx] + corr_diag_antisym[i, right_idx]) / 2
        end
        
        diag_line_data_antisym = corr_diag_antisym[x_idx, :]
        p_diag_cut_antisymmetric = plot(x_range, diag_line_data_antisym,
                         title="Diagonal C at x=3/4·L₁ (Antisymmetric)",
                         xlabel="x'", ylabel="C",
                         legend=false, lw=2, color=:red)
        

        # 10) Middle-zeroed antisymmetric plots
        # X-axis antisymmetric with middle zeroed
        line_data_antisym_zeroed = copy(line_data_antisym)
        line_data_antisym_zeroed[zero_indices] .= 0
        p_x_cut_antisymmetric_zeroed = plot(x_range, line_data_antisym_zeroed,
                                title="C at x₁=3/4·L₁ (Antisymmetric, middle zeroed)",
                                xlabel="x₂", ylabel="C",
                                legend=false, lw=2, color=:red)
        
        # Mark the zeroed points in antisymmetric x-axis
        scatter!(p_x_cut_antisymmetric_zeroed, zero_indices, zeros(length(zero_indices)),
                 color=:darkred, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # Diagonal antisymmetric with middle zeroed
        diag_line_data_antisym_zeroed = copy(diag_line_data_antisym)
        diag_line_data_antisym_zeroed[zero_indices] .= 0
        p_diag_cut_antisymmetric_zeroed = plot(x_range, diag_line_data_antisym_zeroed,
                                title="Diagonal C at x=3/4·L₁ (Antisymmetric, middle zeroed)",
                                xlabel="x'", ylabel="C",
                                legend=false, lw=2, color=:red)
        
        # Mark the zeroed points in antisymmetric diagonal
        scatter!(p_diag_cut_antisymmetric_zeroed, zero_indices, zeros(length(zero_indices)),
                 color=:darkred, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # 11) Y-axis cuts (similar to x-axis cuts but along y direction)
        # Extract correlation C(x0,y1; x0,y2) where x0 = 3/4·L₁
        x0 = div(dims[1] + 1, 2)               # middle index for x=0
        # slice2d_y = state.ρ_matrix_avg[x0, :, x0, :]     # dims[2]×dims[2]
        slice2d_y = state.ρ_matrix_avg_cuts[:y_cut]     # dims[2]×dims[2]
        mean_vec_y = state.ρ_avg[x0, :]
        corr_mat_y = slice2d_y .- (mean_vec_y * transpose(mean_vec_y)) .+ fix_term

        # Remove diagonal peaks by averaging neighboring values
        for i in 1:dims[2]
            left_idx = i == 1 ? dims[2] : i - 1
            right_idx = i == dims[2] ? 1 : i + 1
            corr_mat_y[i, i] = (corr_mat_y[i, left_idx] + corr_mat_y[i, right_idx]) / 2
        end

        # Y-axis cut at y₁ = 3/4·L₂
        y_idx = Int(floor(3 * dims[2] / 4))
        y_range = 1:dims[2]
        line_data_y = corr_mat_y[y_idx, :]
        p_y_cut_full = plot(y_range, line_data_y,
                  title="C at y₁=3/4·L₂ (idx=$(y_idx))",
                  xlabel="y₂", ylabel="C",
                  legend=false, lw=2, color=:cyan)

        # Y-axis cut but with middle region zeroed
        line_data_y_zeroed = copy(line_data_y)
        middle_y = div(dims[2] + 1, 2)
        zero_indices_y = [middle_y-1, middle_y, middle_y+1]
        line_data_y_zeroed[zero_indices_y] .= 0
        
        p_y_cut_zeroed = plot(y_range, line_data_y_zeroed,
                  title="C at y₁=3/4·L₂ (middle zeroed)",
                  xlabel="y₂", ylabel="C",
                  legend=false, lw=2, color=:cyan)
        
        # Mark the zeroed points
        scatter!(p_y_cut_zeroed, zero_indices_y, zeros(length(zero_indices_y)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # Positive half of y-axis cut
        positive_y_range = middle_y:dims[2]
        positive_line_data_y = line_data_y[middle_y:end]
        positive_y_positions = positive_y_range .- middle_y .+ 1
        p_y_cut_positive = plot(positive_y_positions, positive_line_data_y,
                  title="C at y₁=3/4·L₂ (Positive Half)",
                  xlabel="Distance from center", ylabel="C",
                  legend=false, lw=2, color=:cyan)

        # Positive half with middle region zeroed
        positive_line_data_y_zeroed = copy(positive_line_data_y)
        zero_indices_y_pos = [1,2]
        positive_line_data_y_zeroed[zero_indices_y_pos] .= 0
        p_y_cut_positive_zeroed = plot(positive_y_positions, positive_line_data_y_zeroed,
                  title="C at y₁=3/4·L₂ (Positive Half, middle zeroed)",
                  xlabel="Distance from center", ylabel="C",
                  legend=false, lw=2, color=:cyan)
        
        # Mark the zeroed points in positive half
        scatter!(p_y_cut_positive_zeroed, positive_y_positions[zero_indices_y_pos], 
                 zeros(sum(zero_indices_y_pos)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # Y-axis antisymmetric part
        corr_mat_y_antisym = remove_symmetric_part_reflection(corr_mat_y, middle_y)
        # Apply same diagonal smoothing to antisymmetric part
        for i in 1:dims[2]
            left_idx = i == 1 ? dims[2] : i - 1
            right_idx = i == dims[2] ? 1 : i + 1
            corr_mat_y_antisym[i, i] = (corr_mat_y_antisym[i, left_idx] + corr_mat_y_antisym[i, right_idx]) / 2
        end
        
        line_data_y_antisym = corr_mat_y_antisym[y_idx, :]
        p_y_cut_antisymmetric = plot(y_range, line_data_y_antisym,
                         title="C at y₁=3/4·L₂ (Antisymmetric)",
                         xlabel="y₂", ylabel="C",
                         legend=false, lw=2, color=:cyan)

        # Y-axis antisymmetric with middle zeroed
        line_data_y_antisym_zeroed = copy(line_data_y_antisym)
        line_data_y_antisym_zeroed[zero_indices_y] .= 0
        p_y_cut_antisymmetric_zeroed = plot(y_range, line_data_y_antisym_zeroed,
                                title="C at y₁=3/4·L₂ (Antisymmetric, middle zeroed)",
                                xlabel="y₂", ylabel="C",
                                legend=false, lw=2, color=:cyan)
        
        # Mark the zeroed points in antisymmetric y-axis
        scatter!(p_y_cut_antisymmetric_zeroed, zero_indices_y, zeros(length(zero_indices_y)),
                 color=:darkcyan, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # Add y-axis correlation heatmap
        p_corr_y_axis = heatmap(corr_mat_y,
                     title="C(x=0,y₁; x=0,y₂)",
                     xlabel="y₁", ylabel="y₂",
                     aspect_ratio=1, colorbar=true, color=:plasma)

        # Create empty plot for organization
        p_empty = plot(axis=false, showaxis=false, grid=false, title="")

        # Layout with 36 plots total (9 rows x 4 columns)
        display(plot(p_avg_density, p_density_x_cut, p_density_y_cut, p_current_density,                         # Row 1: avg density, x-cut, y-cut, current density
                     p_potential, p_density_x_cut_log, p_density_y_cut_log, p_empty,                              # Row 2: potential, x-cut log, y-cut log, empty
                     p_corr_x_axis, p_x_cut_full, p_x_cut_positive, p_x_cut_antisymmetric,                            # Row 3: x-axis cuts: corr_x_axis_heatmap, full, positive half, antisymmetric
                     p_empty, p_x_cut_zeroed, p_x_cut_positive_zeroed, p_x_cut_antisymmetric_zeroed,            # Row 4: x-axis cuts middle zeroed: empty, full zeroed, positive half zeroed, antisymmetric zeroed
                     p_corr_y_axis, p_y_cut_full, p_y_cut_positive, p_y_cut_antisymmetric,                           # Row 5: y-axis cuts: corr_y_axis_heatmap, full, positive half, antisymmetric
                     p_empty, p_y_cut_zeroed, p_y_cut_positive_zeroed, p_y_cut_antisymmetric_zeroed,           # Row 6: y-axis cuts middle zeroed: empty, full zeroed, positive half zeroed, antisymmetric zeroed
                     p_corr_diag, p_diag_cut_full, p_diag_cut_positive, p_diag_cut_antisymmetric,                  # Row 7: diagonal cuts: corr_diag_heatmap, full, positive half, antisymmetric  
                     p_empty, p_diag_cut_zeroed, p_diag_cut_positive_zeroed, p_diag_cut_antisymmetric_zeroed,  # Row 8: diagonal cuts middle zeroed: empty, full zeroed, positive half zeroed, antisymmetric zeroed
                     layout=(9,4), size=(2400,3600),
                     plot_title="2D sweep $(sweep)"))
        return normalized_dist, corr_mat2
    else
        throw(DomainError("Only 1D or 2D plotting supported"))
    end
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
function plot_data_colapse(states_params_names, power_n, indices, results_dir = "results_figures", do_fit=true)
    n = Float64(power_n)
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

            for i in indices
                outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
                corr_mat = (state.ρ_matrix_avg - outer_prod_ρ) .+ (N / L^2)
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

                for (data, p_combined_plot) in zip((full_data, antisym_data, sym_data), (p_full_combined, p_antisym_combined, p_sym_combined))
                    y_scaled = data .* i^n
                    mask = (-5 .<= x_scaled .<= 5)
                    x_filtered = x_scaled[mask]
                    y_filtered = y_scaled[mask]
                    plot!(p_combined_plot, x_filtered, y_filtered, label="y=$(i)", lw=2)
                end

                y_scaled = full_data .* i^n
                mask = (-5 .<= x_scaled .<= 5)
                x_filtered = x_scaled[mask]
                y_filtered = y_scaled[mask]
                append!(all_x, x_filtered)
                append!(all_y, y_filtered)
                plot!(p_combined, x_filtered, y_filtered, label="$(label) y=$(i)", linewidth=2)
            end

            savefig(p_full_combined, "$(full_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_antisym_combined, "$(antisym_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_sym_combined, "$(sym_dir)/data_collapse_$(n)_indices-$(indices).png")
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

            y0 = div(dims[2] + 1, 2)  # middle y index
            
            for i in indices
                # Check for potential overflow before scaling
                scaling_factor = Float64(i)^n
                if scaling_factor > 1e10 || !isfinite(scaling_factor)
                    println("Warning: Skipping i=$i due to overflow (scaling factor: $scaling_factor)")
                    continue
                end
                
                # X-axis cut: C(x1,y0; x2,y0)
                slice2d_x = state.ρ_matrix_avg[:, y0, :, y0]
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
                for j in 1:dims[1], k in 1:dims[1]
                    corr_diag[j,k] = state.ρ_matrix_avg[j, j, k, k] - state.ρ_avg[j,j] * state.ρ_avg[k,k] + fix_term
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
                
                for (data, p_plot, cut_type, x_scale) in zip((x_axis_data, diag_data, x_axis_positive_data, diag_positive_data, x_axis_antisym_data, diag_antisym_data), 
                                                           (p_x_combined, p_diag_combined, p_x_pos_combined, p_diag_pos_combined, p_x_antisym_combined, p_diag_antisym_combined),
                                                           ("x-axis", "diagonal", "x-axis-positive", "diagonal-positive", "x-axis-antisymmetric", "diagonal-antisymmetric"),
                                                           (x_scaled, x_scaled, x_scaled_pos, x_scaled_pos, x_scaled, x_scaled))
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
                    else
                        println("Warning: Skipping combined plot for i=$i due to extreme values")
                    end
                end
            end
            
            savefig(p_x_combined, "$(x_axis_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_diag_combined, "$(diag_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_x_pos_combined, "$(x_axis_pos_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_diag_pos_combined, "$(diag_pos_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_x_antisym_combined, "$(x_axis_antisym_dir)/data_collapse_$(n)_indices-$(indices).png")
            savefig(p_diag_antisym_combined, "$(diag_antisym_dir)/data_collapse_$(n)_indices-$(indices).png")
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
                normalized_dist = state.ρ_avg / sum(state.ρ_avg)
                p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
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
            normalized_dist = state.ρ_avg / sum(state.ρ_avg)
            p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
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
    normalized_dist = state.ρ_avg / sum(state.ρ_avg)
    p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
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