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
# function plot_data_colapse(states_params_names,power_n,indices, results_dir = "results_figures")
#     # initial_index = param.dims[1]÷10+1
#     # index_jump = 2
#     # end_index = param.dims[1]/4
#     # initial_index = 50
#     # index_jump = 1
#     # end_index = 60
#     n = power_n
#     # Create combined plot
#     p_combined = plot(title="Combined Data Collapse C(x,y)*y^$n",
#                      legend=:outerright,
#                      size=(1200,800))
#     all_x = []
#     all_y = []
    
#     # Process each state individually first
#     for (idx, (state, param, label)) in enumerate(states_params_names)
#         p_individual = plot(title="Data Collapse - $label",
#                           legend=:outerright,
#                           size=(1000,600))
#         α = param.α
#         γ′ = param.γ * param.N
#         state_x = []
#         state_y = []
        
#         L = param.dims[1]
#         N = param.N
        
#         for i in indices
#             outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
#             corr_mat = (state.ρ_matrix_avg-outer_prod_ρ).+(N/L^2)
#             middle_spot = param.dims[1]÷2
#             point_to_look_at = Int(middle_spot+i)
            
#             corr_mat_collapsed = corr_mat[:,point_to_look_at]
#             left_value = corr_mat_collapsed[point_to_look_at-1]
#             right_value = corr_mat_collapsed[point_to_look_at+1]
#             left_side = corr_mat_collapsed[1:point_to_look_at-1]
#             right_side = corr_mat_collapsed[point_to_look_at+1:end]
#             full_data = vcat(left_side, [(left_value+right_value)/2], right_side)
            
#             corr_mat_antisym = remove_symmetric_part_reflection(corr_mat,middle_spot)

#             #for antisymmetric only part
#             corr_mat_antisym[point_to_look_at,point_to_look_at] = (corr_mat_antisym[point_to_look_at,point_to_look_at+1]+corr_mat_antisym[point_to_look_at,point_to_look_at-1])/2
#             corr_mat_antisym[point_to_look_at,L-point_to_look_at] = (corr_mat_antisym[point_to_look_at,L-(point_to_look_at+1)]+corr_mat_antisym[point_to_look_at,L-(point_to_look_at-1)])/2
#             # full_data=corr_mat_antisym[point_to_look_at,1:end]
#             #

#             x_positions = 1:length(full_data)
#             x_scaled = (x_positions .- middle_spot) ./ (i)
#             y_scaled = full_data .* i^n
#             # y_scaled = full_data .* ((α*γ′)*i^2) 
            
#             # Filter points within range
#             # mask = (-5 .<= x_scaled .<= 5)
#             # x_scaled = x_scaled[mask]
#             # y_scaled = y_scaled[mask]
            
#             # Store points for individual and combined fitting
#             append!(state_x, x_scaled)
#             append!(state_y, y_scaled)
#             append!(all_x, x_scaled)
#             append!(all_y, y_scaled)
            
#             # Plot data on both individual and combined plots
#             idx_sort = sortperm(x_scaled)
#            # ylims!(p_combined,-10^2,10^2)
#             plot!(p_individual, x_scaled[idx_sort], y_scaled[idx_sort], 
#                   label="y=$(i)", linewidth=2)
#             plot!(p_combined, x_scaled[idx_sort], y_scaled[idx_sort], 
#                   label="$(label) y=$(i)", linewidth=2)
#         end
        
#         # Fit individual state data
#         f(x, p) = (p[1] * x ./ ((1 .+ p[2]*x.^2).^2)).+p[3]
        
#         p0 = [1.0, 0.0, 0.0]
#         # fit_individual = curve_fit(f, state_x, state_y, p0)
        
#         # Add theoretical curve to individual plot
#         x_theory = range(-5, 5, length=1000)
#         # plot!(p_individual, x_theory, f(x_theory, fit_individual.param), 
#         #       label="Theoretical", color=:black, linewidth=3, linestyle=:dash)
        
#         # Save individual plot
#         savefig(p_individual, "$(results_dir)/data_collapse_$(label)_y^$(n).png")
#     end
    
#     # Fit combined data
#     f(x, p) = (p[1] * x ./ ((1 .+ p[2]*x.^2).^2)).+p[3]
#     p0 = [1.0, 0.0, 0.0]
#     # fit_combined = curve_fit(f, all_x, all_y, p0)
#     # println("Combined fit parameters: ", fit_combined.param)

#     # Add theoretical curve to combined plot
#     # x_theory = range(-5, 5, length=1000)
#     # plot!(p_combined, x_theory, f(x_theory, fit_combined.param), label="Theoretical", color=:black, linewidth=3, linestyle=:dash)
    
#     # Save combined plot
#     savefig(p_combined, "$(results_dir)/data_collapse_combined_y^$(n).png")
    
#     # Display combined plot
#     display(p_combined)
    
#     return p_combined
# end
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