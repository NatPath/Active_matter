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
        corr_mat = state.ρ_matrix_avg-outer_prod_ρ
        p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
        p1 = plot_magnetization(state, param)
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
        # p_final=plot(p0,p1,p4,p5,p6, size=(2100,1000),plot_title="sweep $(sweep)",layout=grid(2,3))
        p_final=plot(p0,p1,p4,p5,p6,p7, size=(2100,1000),plot_title="sweep $(sweep)",layout=grid(2,3))
        display(p_final)
        return normalized_dist, corr_mat
    elseif dim_num == 2
        # --- 2D plotting with y=0 cut ---
        dims = param.dims
        y0 = div(dims[2] + 1, 2)               # middle index for y=0

        # 1) Time‐averaged density heatmap
        normalized_dist = state.ρ_avg ./ sum(state.ρ_avg)
        p1 = heatmap(normalized_dist',
                     title="⟨ρ⟩ (t=$(sweep))",
                     xlabel="x", ylabel="y",
                     aspect_ratio=1, colorbar=true)

        # 2) Extract correlation C(x1,y0; x2,y0)
        fix_term = param.N / (prod(param.dims)^2)
        slice2d = state.ρ_matrix_avg[:, y0, :, y0]     # dims[1]×dims[1]
        mean_vec = state.ρ_avg[:, y0]
        corr_mat2 = slice2d .- mean_vec * transpose(mean_vec) .+ fix_term

        # Remove diagonal peaks by averaging neighboring values
        for i in 1:dims[1]
            left_idx = i == 1 ? dims[1] : i - 1
            right_idx = i == dims[1] ? 1 : i + 1
            corr_mat2[i, i] = (corr_mat2[i, left_idx] + corr_mat2[i, right_idx]) / 2
        end

        # 3) Plot full 2D correlation at y=0
        p2 = heatmap(corr_mat2,
                     title="C(x₁,y=0; x₂,y=0)",
                     xlabel="x₁", ylabel="x₂",
                     aspect_ratio=1, colorbar=true)

        # 4) 1D cut at x₁ = 3/4·L₁
        x_idx = Int(floor(3 * dims[1] / 4))
        x_range = 1:dims[1]
        line_data = corr_mat2[x_idx, :]
        p3 = plot(x_range, line_data,
                  title="C at x₁=3/4·L₁ (idx=$(x_idx))",
                  xlabel="x₂", ylabel="C",
                  legend=false, lw=2)

        # 5) Same 1D cut but with middle region zeroed
        line_data_zeroed = copy(line_data)
        middle_x = div(dims[1] + 1, 2)
        zero_indices = [middle_x-1, middle_x, middle_x+1]
        line_data_zeroed[zero_indices] .= 0
        
        p4 = plot(x_range, line_data_zeroed,
                  title="C at x₁=3/4·L₁ (middle zeroed)",
                  xlabel="x₂", ylabel="C",
                  legend=false, lw=2, color=:blue)
        
        # Mark the zeroed points
        scatter!(p4, zero_indices, zeros(length(zero_indices)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")
        # 5) Same 1D cut but with middle region zeroed
        line_data_zeroed = copy(line_data)
        middle_x = div(dims[1] + 1, 2)
        zero_indices = [middle_x-1, middle_x, middle_x+1]
        line_data_zeroed[zero_indices] .= 0
        
        p4 = plot(x_range, line_data_zeroed,
                  title="C at x₁=3/4·L₁ (middle zeroed)",
                  xlabel="x₂", ylabel="C",
                  legend=false, lw=2, color=:blue)
        
        # Mark the zeroed points
        scatter!(p4, zero_indices, zeros(length(zero_indices)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # 6) Diagonal correlation C(x,x; x',x') 
        corr_diag = zeros(dims[1], dims[1])
        for i in 1:dims[1], j in 1:dims[1]
            corr_diag[i,j] = state.ρ_matrix_avg[i, i, j, j] - state.ρ_avg[i,i] * state.ρ_avg[j,j] + fix_term
        end
        
        # Apply diagonal smoothing to diagonal correlation
        for i in 1:dims[1]
            left_idx = i == 1 ? dims[1] : i - 1
            right_idx = i == dims[1] ? 1 : i + 1
            corr_diag[i, i] = (corr_diag[i, left_idx] + corr_diag[i, right_idx]) / 2
        end
        
        p5 = heatmap(corr_diag,
                     title="C(x,x; x',x') - Diagonal",
                     xlabel="x", ylabel="x'",
                     aspect_ratio=1, colorbar=true)

        # 7) 1D cut of diagonal at x = 3/4·L₁
        diag_line_data = corr_diag[x_idx, :]
        p6 = plot(x_range, diag_line_data,
                  title="Diagonal C at x=3/4·L₁ (idx=$(x_idx))",
                  xlabel="x'", ylabel="C",
                  legend=false, lw=2, color=:green)

        # 8) Same diagonal cut but with middle region zeroed
        diag_line_zeroed = copy(diag_line_data)
        diag_line_zeroed[zero_indices] .= 0
        
        p7 = plot(x_range, diag_line_zeroed,
                  title="Diagonal C at x=3/4·L₁ (middle zeroed)",
                  xlabel="x'", ylabel="C",
                  legend=false, lw=2, color=:green)
        
        # Mark the zeroed points
        scatter!(p7, zero_indices, zeros(length(zero_indices)),
                 color=:red, markersize=6, markershape=:x,
                 label="Zeroed middle")

        # Layout: Row 1: density, Row 2: y=0 cuts, Row 3: diagonal cuts
        display(plot(p1, plot(), plot(),        # Top row: density + empty spaces
                     p2, p3, p4,               # Middle row: y=0 cuts  
                     p5, p6, p7,               # Bottom row: diagonal cuts
                     layout=(3,3), size=(1800,1200),
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
    n = power_n
    all_x = []
    all_y = []

    p_combined = plot(title="Combined Data Collapse C(x,y)*y^$n", legend=:outerright, size=(1200,800))

    for (idx, (state, param, label)) in enumerate(states_params_names)
        α = param.α
        γ′ = param.γ * param.N

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