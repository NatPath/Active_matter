using Plots
using Random
using FFTW
using ProgressMeter
using Statistics
using LsqFit 
using LinearAlgebra
import Printf.@sprintf


include("potentials.jl")
include("modules_run_and_tumble.jl")
rng = MersenneTwister(123)

dim_num = 1
D = 1                           # diffusion coefficient
α = 0.3                          # rate of tumbling 
L= 64
dims = ntuple(i->L, dim_num)     # system size
ρ₀ =  100                    # density
T = 1.0                           # temperature   
β = 0.03/(ρ₀*L)                          # potential fluctuation rate

param = FP.setParam(α,β, dims, ρ₀, D)

pos₀ = zeros(Int64, param.N, dim_num)
for n in 1:param.N
    for (i,dim) in enumerate(dims)
        pos₀[n,i] = rand(rng, 1:dim)
    end
end
v_smudge_args = Potentials.potential_args("smudge",dims; magnitude = 0.4)
potential = Potentials.choose_potential(v_smudge_args,dims)

plot_boltzman_distribution(potential.V)
state = FP.setState(0, rng, param, T, potential)


# Increase the number of frames to see a more meaningful time correlation
# make_movie!(state, param, 1, 10000, rng, "test_with_time_corr", 10)
# show_times = [ j*10^i for i in range(3,12) for j in range(1,9)]
show_times = []
res_dist, corr_mat=run_simulation!(state, param, 1, 10^2, rng, show_times = show_times )



