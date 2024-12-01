module PlotUtils
using Plots
using LsqFit
export plot_sweep, plot_density, plot_data_colapse, plot_spatial_correlation, plot_time_correlation
function plot_sweep(sweep,state,param)
    normalized_dist = state.ρ_avg / sum(state.ρ_avg)
    p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
    # p1 = plot_density(state.ρ, param, state; title="Current density", show_directions=false)
    outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
    corr_mat = state.ρ_matrix_avg-outer_prod_ρ
    p4 = heatmap(corr_mat, xlabel="x", ylabel="y", 
                title="Correlation Matrix Heatmap", color=:viridis)
    middle_spot = param.dims[1]÷2
    p5 = plot(corr_mat[middle_spot,:],title="correlation matrix cut for x=$(middle_spot)")
    point_to_look_at = middle_spot+5
    vline!(p4,[point_to_look_at],label="x=$(point_to_look_at)")
    left_value=corr_mat[point_to_look_at,point_to_look_at-1]
    right_value=corr_mat[point_to_look_at,point_to_look_at+1]
    left_side=corr_mat[point_to_look_at, 1:point_to_look_at-1]
    right_side=corr_mat[point_to_look_at, point_to_look_at+1:end]

    p6 = plot(vcat(left_side,[(left_value+right_value)/2],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
    vline!(p6,[point_to_look_at],label="x=$(point_to_look_at)")
    # p6 = plot(vcat(left_side,[corr_mat[point_to_look_at,point_to_look_at]],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
    # p6= plot(corr_mat[point_to_look_at,:])

    p_final=plot(p0,p4,p5,p6, size=(1800,800),plot_title="sweep $(sweep)")
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
                legend=:topleft)
            
            # Secondary axis for potential
            plot!(twinx(), x_range, state.potential.V,
                  ylabel="Potential",
                  label="Potential",
                  color=:red,
                  alpha=0.3,
                  linestyle=:dash,
                  legend=:topright)
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
# function plot_data_colapse(states_params)
#     p = plot(title="Correlation Matrix Collapse: y²*corr(x,y) vs x/y", 
#              xlabel="x/y", ylabel="y²*corr(x,y)")
    
#     for (state,param) in states_params
#         for i in 8:1:param.dims[1]/5
#             outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
#             corr_mat = (state.ρ_matrix_avg-outer_prod_ρ)
#             middle_spot = param.dims[1]÷2
#             point_to_look_at = Int(middle_spot+i)
            
#             # Get correlation data
#             corr_mat_collapsed = corr_mat[:,point_to_look_at]
#             left_value = corr_mat_collapsed[point_to_look_at-1]
#             right_value = corr_mat_collapsed[point_to_look_at+1]
#             left_side = corr_mat_collapsed[1:point_to_look_at-1]
#             right_side = corr_mat_collapsed[point_to_look_at+1:end]
#             full_data = vcat(left_side, [(left_value+right_value)/2], right_side)
            
#             # Create scaled x-axis
#             x_positions = 1:length(full_data)
#             x_scaled = (x_positions .- middle_spot) ./ i
            
#             # Scale y values by i^2
#             y_scaled = full_data .* i^2
            
#             # Filter data points within range
#             mask = (-5 .<= x_scaled .<= 5)
#             x_scaled = x_scaled[mask]
#             y_scaled = y_scaled[mask]

#             # Sort by x_scaled for proper line plotting
#             idx = sortperm(x_scaled)
            
#             # Plot scaled data
#             plot!(p, x_scaled[idx], y_scaled[idx], 
#                   label="y=$i", 
#                   linewidth=2)
#         end
#     end
#     display(p)
#     return p
# end
function plot_data_colapse(states_params)
    # Initialize plot
    p = plot(title="Correlation Matrix Collapse: y²*corr(x,y) vs x/y", 
             xlabel="x/y", ylabel="y²*corr(x,y)",
             xlims=(-5,5), grid=true)
    
    # Collect all scaled data points
    all_x = Float64[]
    all_y = Float64[]
    
    for (state,param) in states_params
        for i in 16:1:param.dims[1]/5-1
            outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
            corr_mat = (state.ρ_matrix_avg-outer_prod_ρ)
            middle_spot = param.dims[1]÷2
            point_to_look_at = Int(middle_spot+i)
            
            corr_mat_collapsed = corr_mat[:,point_to_look_at]
            left_value = corr_mat_collapsed[point_to_look_at-1]
            right_value = corr_mat_collapsed[point_to_look_at+1]
            left_side = corr_mat_collapsed[1:point_to_look_at-1]
            right_side = corr_mat_collapsed[point_to_look_at+1:end]
            full_data = vcat(left_side, [(left_value+right_value)/2], right_side)
            
            x_positions = 1:length(full_data)
            x_scaled = (x_positions .- middle_spot) ./ i
            y_scaled = full_data .* i^2
            
            # Filter points within range
            mask = (-5 .<= x_scaled .<= 5)
            x_scaled = x_scaled[mask]
            y_scaled = y_scaled[mask]
            
            # Store points for fitting
            append!(all_x, x_scaled)
            append!(all_y, y_scaled)
            
            # Plot data
            idx = sortperm(x_scaled)
            plot!(p, x_scaled[idx], y_scaled[idx], 
                  label="y=$i", linewidth=2)
        end
    end
    
    # Define theoretical function
    f(x, p) = (p[1] * x ./ ((1 .+ p[2]*x.^2).^2)).+p[3]
    
    # Fit function to data
    p0 = [1.0,0.0,0.0]  # Initial parameter guess
    fit = curve_fit(f, all_x, all_y, p0)
    
    # Add theoretical curve
    x_theory = range(-5, 5, length=1000)
    plot!(p, x_theory, f(x_theory, fit.param), 
          label="Theoretical", 
          color=:black, 
          linewidth=3, 
          linestyle=:dash)
    
    display(p)
    print(fit.param)
    return p
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
        # dx_range = range(-param.dims[1] ÷ 2, param.dims[1] ÷ 2 - 1, length=param.dims[1])
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

function plot_time_correlation(time_corr, frame, fit_params=nothing)
    t = 0:(frame-1)
    p = plot(t, time_corr, 
             title="Time Correlation", 
             xlabel="Δt", ylabel="C(Δt)", 
             legend=false, lw=2, label="Data")
    
    if !isnothing(fit_params)
        plot!(p, t, fit_params[1] .* exp.(-t ./ fit_params[2]) .+ fit_params[3], 
              lw=2, ls=:dash, label="Fit")
        annotate!(p, [(frame/2, 0.8, text("τ = $(round(fit_params[2], digits=2))", 10))])
    end
    
    return p
end
end