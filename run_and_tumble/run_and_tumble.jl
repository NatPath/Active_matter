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
L= 64
dims = ntuple(i->L, dim_num)     # system size
ρ₀ = 0.5                       # density
T = 1.0                           # temperature   

param = FP.setParam(α, dims, ρ₀, D)

pos₀ = zeros(Int64, param.N, dim_num)
for n in 1:param.N
    for (i,dim) in enumerate(dims)
        pos₀[n,i] = rand(rng, 1:dim)
    end
end
function choose_V(v_args)
    V = zeros(Float64, dims)
    v_string = v_args["type"]
    if v_string == "well"
        for i in 1:L
            #V[i] = exp(-((i - L/2)^2) / (2 * (2)^2))  # Gaussian potential centered in the middle
            width = v_args["width"]
            height = v_args["height"]
            V[i] = i<= L/2 + width && i> L/2-width  ? height : 0
        end
    elseif v_string == "zero"
        nothing
    elseif v_string == "smudge"
        middle = Int(L//2)
        V[middle] = v_args["magnitude"]
        V[middle-1] = v_args["magnitude"]/2
    elseif v_string == "delta"
        V[middle]=v_args["magnitude"]
    else
        error("unsupported V string")
    end
    V_plot=plot(V)
    display(V_plot)
    return V
end
function plot_boltzman_distribution(V)
    display(plot(exp.(-V/T)))
end
v_well_args = Dict("type"=>"well", "width"=>L//4, "height"=>1)
v_smudge_args = Dict("type"=>"smudge", "magnitude" => 1)
v_zero_args = Dict("type"=>"zero")
V = choose_V(v_zero_args)
plot_boltzman_distribution(V)
state = FP.setState(0, rng, param, T, V)


# Increase the number of frames to see a more meaningful time correlation
make_movie!(state, param, 0.2, 500, rng, "test_with_time_corr", 20)
