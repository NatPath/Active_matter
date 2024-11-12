using Plots
using Random
using FFTW
using ProgressMeter
using Statistics
using LsqFit 
using LinearAlgebra
import Printf.@sprintf


include("modules_run_and_tumble.jl")
rng = MersenneTwister(123)

dim_num = 1
D = 1                            # diffusion coefficient
α = 0.3                          # rate of tumbling 
L= 8
dims = ntuple(i->L, dim_num)     # system size
ρ₀ =  10                    # density
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
    L = dims[1]
    middle = Int(L//2)
    x = LinRange(0,L,L)
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
        location = v_args["location"]
        V[location] = v_args["magnitude"]
        V[location-1] = v_args["magnitude"]/2
    elseif v_string == "delta"
        V[middle]=v_args["magnitude"]
    elseif v_string == "linear"
        m = v_args["slope"]
        b = v_args["shift"]
        linear_potential(m,b,x)= m*x.+b
        V = linear_potential(m,b,x)
    elseif v_string == "harmonic"
        k = v_args["k"]
        m_sign = v_args["m_sign"]
        shift = v_args["center"]
        harmonic_oscillator_potential(k,m_sign,shift,x) = k^2*(x.-shift).^2*m_sign
        V = harmonic_oscillator_potential(k,m_sign,shift,x)
    elseif v_string == "periodic"
        V0 = v_args["magnitude"]
        T = v_args["period"]
        ϕ = v_args["phase"]
        periodic_potential(V0, a, x,ϕ) = V0 * cos.(2π * (x/T) .+ ϕ )
        V = periodic_potential(V0,T,x,ϕ)
    elseif v_string == "random"
        scale = v_args["scale"]
        V = scale*rand(rng,dims...)

    else
        error("unsupported V string")
    end
    V_plot=plot(V)
    display(V_plot)
    return V
end

function plot_boltzman_distribution(V)
    exp_expression= exp.(-V/T)
    
    display(plot(exp_expression/sum(exp_expression)))
end

v_well_args = Dict("type"=>"well", "width"=>L//8, "height"=>1)
v_zero_args = Dict("type"=>"zero")
v_smudge_args = Dict("type"=>"smudge","location" => L÷2 , "magnitude" => 1)
v_delta_args = Dict("type"=>"delta", "location" => L÷2, "magnitude"=>100)
v_linear_args = Dict("type"=> "linear", "slope" => 1, "shift"=>0)
v_harmonic_args = Dict("type"=>"harmonic", "k" => 1, "m_sign"=>1, "center"=> L÷2)
v_periodic_args = Dict("type"=>"periodic", "period" => L÷4, "magnitude"=>1, "phase"=> L÷2)
v_random_args = Dict("type"=>"random", "scale"=>1 )
V = choose_V(v_harmonic_args)
plot_boltzman_distribution(V)
state = FP.setState(0, rng, param, T, V)


# Increase the number of frames to see a more meaningful time correlation
# make_movie!(state, param, 1, 100, rng, "test_with_time_corr", 10)
run_simulation!(state, param, 1, 1000000, rng )




