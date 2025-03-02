using JLD2
using Plots
using ArgParse
import Printf.@sprintf
include("potentials.jl")
include("modules_run_and_tumble.jl")  # Include the file where Param is defined
using .FP: Param  # Assuming the module is named RunAndTumble and contains Param

# Function to convert Param to a dictionary
function param_to_dict(param::Param)
    return Dict(
        :α => param.α,
        :β => param.β,
        :dims => param.dims,
        :ρ₀ => param.ρ₀,
        :N => param.N,
        :D => param.D,
        :potential_type => param.potential_type,
        :fluctuation_type => "zero-potential",
        :potential_magnitude => param.potential_magnitude
    )
end

# Function to create a Param instance from a dictionary
function dict_to_param(dict::Dict)
    return Param(
        dict[:α],
        dict[:β],
        dict[:dims],
        dict[:ρ₀],
        dict[:N],
        dict[:D],
        dict[:potential_type],
        dict[:fluctuation_type],
        dict[:potential_magnitude]
    )
end
function convert_to_param(reconstructed::JLD2.ReconstructedMutable)
    return Param(
        reconstructed.α,
        reconstructed.β,
        reconstructed.dims,
        reconstructed.ρ₀,
        reconstructed.N,
        reconstructed.D,
        reconstructed.potential_type,
        "zero-potential",  # Set the default value for fluctuation_type
        reconstructed.potential_magnitude
    )
end
# Load the JLD2 file
data_path = "/Users/nativmaor/Active_matter/run_and_tumble/saved_states/potential-2ratchet_L-64_rho-1.0e+02_alpha-0.00_betaprime-0.10_D-1.0_t-3000000.jld2"
@load data_path state param potential

# Convert param to a mutable dictionary
param = convert_to_param(param)

new_filename = "/Users/nativmaor/Active_matter/run_and_tumble/saved_states/potential-2ratchet_fluctuation-zero-potential_L-64_rho-1.0e+02_alpha-0.00_betaprime-0.10_D-1.0_t-3000000.jld2"
@save new_filename state param potential




