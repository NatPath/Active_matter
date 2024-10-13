#Code for fluctuating potential
cd(@__DIR__)
pwd()

# package installation - you only need to run this once
using Pkg

Pkg.add("Plots")
Pkg.add("Random")
Pkg.add("FFTW")
Pkg.add("ProgressMeter")
# end of package installation

# loading packages
using Plots
using Random
using FFTW

using ProgressMeter
import Printf.@sprintf

# loading module file
include("modules_fluctuating_potential.jl")
rng = MersenneTwister(123)


D = 1               # rate of hopping
Lx, Ly = 64, 64     # system size
ρ₀ = 0.5             # density

param = FP.setParam(D, Lx, Ly, ρ₀)

pos₀ = zeros(Int64, param.N, 2)
for n in 1:param.N
    pos₀[n,1] = rand(rng, 1:Lx)
    pos₀[n,2] = rand(rng, 1:Ly)
end

state = FP.setState(0, param, pos₀)

make_movie!(state, param, 0.2, 200, rng, "test", 20)









