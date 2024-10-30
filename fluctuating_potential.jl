# #Code for fluctuating potential
# cd(@__DIR__)
# pwd()

# # package installation - you only need to run this once
# using Pkg

# Pkg.add("Plots")
# Pkg.add("Random")
# Pkg.add("FFTW")
# Pkg.add("ProgressMeter")
# Pkg.add("LsqFit")
# # end of package installation

# loading packages
using Plots
using Random
using FFTW
using ProgressMeter
using Statistics
using LsqFit 
import Printf.@sprintf

# loading module file
include("modules_fluctuating_potential.jl")
rng = MersenneTwister(123)


D = 1               # rate of hopping
dims = [64,64]      # system's dimentions
# Lx, Ly = 64, 64     # system size
ρ₀ = 0.5             # density

param = FP.setParam(D, dims, ρ₀)

pos₀ = zeros(Int64, param.N, 2)
for n in 1:param.N
    for d in eachindex(dims)
        pos₀[n,d] = rand(rng, 1:dims[d])
    end
end

state = FP.setState(0, param, pos₀)

# Increase the number of frames to see a more meaningful time correlation
n_frames = 500
# t_gap = 1/(param.N*D)
t_gap = 0.2
fps = 20
make_movie!(state, param, t_gap, n_frames, rng, "test_with_time_corr", fps)









