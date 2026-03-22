using JLD2
using Plots

function plot_statistics(input_file::AbstractString,case_name)
    # Load the saved state from the input file
    statistics = load(input_file)

    # Calculate the statistics
    corr_mat = statistics["correlation_matrix"]
    normalized_dist = statistics["normalized_density"]

    # Plot the statistics
    plot(normalized_dist, xlabel="Index", ylabel="Normalized Density", label="$case_name Density")
    savefig("$(case_name)_density_plot.png")

    # Plot the correlation matrix
    heatmap(corr_mat, xlabel="Index", ylabel="Index", title="$case_name Correlation Matrix")
    # Save the plots to files
    savefig("$(case_name)_correlation_matrix_plot.png")
end

function calculate_statistics(state)
    normalized_dist = state.ρ_avg / sum(state.ρ_avg)
    outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
    corr_mat = state.ρ_matrix_avg-outer_prod_ρ
    return normalized_dist, corr_mat
end

function extract_correlation_matrix_and_density(input_file::AbstractString, output_file::AbstractString)
    # Load the saved state from the input file
    state = load(input_file)

    normalized_dist, corr_mat = calculate_statistics(state["state"])

    # Save the average correlation matrix and density to the output file
    save(output_file, "correlation_matrix", corr_mat, "normalized_density", normalized_dist)
end

# Usage example
active_input_file = "potential-smudge_fluctuation-left-right_activity-0.60_L-256_rho-1.0e+02_alpha-0.30_gammap-0.10_D-1.0_t-20000000.jld2"
active_output_file = "active_leftRight_flipping_triangle_statistics.jld2"
extract_correlation_matrix_and_density(active_input_file, active_output_file)
plot_statistics(active_output_file, "active_leftRight_flipping")

passive_input_file = "potential-smudge_Vscale-16.0_fluctuation-reflection_activity-0.00_L-256_rho-1.0e+02_alpha-0.00_gammap-1.00_D-1.0_t-154000991.jld2"
passive_output_file = "passive_plusMinus_flipping_triangle_statistics.jld2"
extract_correlation_matrix_and_density(passive_input_file, passive_output_file)
plot_statistics(passive_output_file, "passive_plusMinus_flipping")
