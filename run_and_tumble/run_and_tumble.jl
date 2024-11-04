using Plots
using Random
using FFTW
using ProgressMeter
using Statistics
using LsqFit 
import Printf.@sprintf

include("modules_run_and_tumble.jl")
rng = MersenneTwister(123)

dim_num = 1
D = 1.0                            # diffusion coefficient
α = 0.1                          # rate of tumbling 
L= 128
dims = ntuple(i->L, dim_num)     # system size
ρ₀ = 0.5                         # density
T = 1.0                           # temperature   

param = FP.setParam(α, dims, ρ₀, D)

pos₀ = zeros(Int64, param.N, dim_num)
for n in 1:param.N
    for (i,dim) in enumerate(dims)
        pos₀[n,i] = rand(rng, 1:dim)
    end
end
V = zeros(Float64, dims)
for i in 1:L
    V[i] = exp(-((i - L/2)^2) / (2 * (L)^2))  # Gaussian potential centered in the middle
end
state = FP.setState(0, rng, param, T, V)


# Increase the number of frames to see a more meaningful time correlation
make_movie!(state, param, 0.2, 500, rng, "test_with_time_corr", 20)