#!/usr/bin/env julia

using Random
using Statistics
using Test

include(joinpath(@__DIR__, "src", "active_objects_sde", "modules_coupled_sde_active_objects.jl"))
using .FPCoupledSDEActiveObjects

@testset "coupled-SDE active objects" begin
    @test minimal_image(6.0, 10.0) == -4.0
    @test minimal_image(-6.0, 10.0) == 4.0
    @test oriented_separation(1.0, 8.0, 10.0) == 3.0
    @test minimum_separation(1.0, 8.0, 10.0) == 3.0

    base_param = CoupledSDEParam(
        mode=MOBILE_OBJECTS_MODE,
        L=32.0,
        rho0=0.0,
        N=3,
        D0=0.0,
        dt=0.01,
        mu_bath=1.7,
        mu_obj=0.2,
        f0=1.1,
        sigma_f=2.0,
        separation=8.0,
        n_steps=1,
        warmup_steps=0,
    )

    state = CoupledSDEState(
        0,
        0.0,
        [4.0, 8.0, 29.0],
        5.0,
        13.0,
        5.0,
        13.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    old_x = copy(state.x)
    old_XA = state.XA
    old_XB = state.XB
    work = SDEWork(base_param.N)
    SA, SB = compute_profile_sums!(work.fA, work.fB, state.x, state.XA, state.XB, base_param)
    old_fA = copy(work.fA)
    old_fB = copy(work.fB)
    dWA = 0.13
    dWB = -0.07
    dWi = zeros(Float64, base_param.N)
    obs = step_mobile_objects!(state, base_param, work, dWA, dWB, dWi)

    @test isapprox(obs.SA, SA)
    @test isapprox(obs.SB, SB)
    @test isapprox(obs.dXA, -base_param.mu_obj * SA * dWA)
    @test isapprox(obs.dXB, -base_param.mu_obj * SB * dWB)
    @test isapprox(obs.drel, obs.dXA - obs.dXB)
    @test obs.XA == old_XA
    @test obs.XB == old_XB
    for i in eachindex(old_x)
        expected = wrap_position(old_x[i] + base_param.mu_bath * (old_fA[i] * dWA + old_fB[i] * dWB), base_param.L)
        @test isapprox(state.x[i], expected)
    end

    zero_param = CoupledSDEParam(
        mode=MOBILE_OBJECTS_MODE,
        L=16.0,
        rho0=0.0,
        N=2,
        D0=0.0,
        dt=0.01,
        mu_bath=1.0,
        mu_obj=1.0,
        f0=0.0,
        sigma_f=1.0,
        separation=4.0,
    )
    zero_state = CoupledSDEState(0, 0.0, [2.0, 7.0], 3.0, 9.0, 3.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    zero_work = SDEWork(zero_param.N)
    step_mobile_objects!(zero_state, zero_param, zero_work, 0.5, -0.25, zeros(Float64, zero_param.N))
    @test isapprox(zero_state.x, [2.0, 7.0])
    @test isapprox(zero_state.XA, 3.0)
    @test isapprox(zero_state.XB, 9.0)
    @test isapprox(zero_state.last_drel, 0.0)

    fixed_param = CoupledSDEParam(
        mode=FIXED_SEPARATION_MODE,
        L=16.0,
        rho0=0.0,
        N=2,
        D0=0.0,
        dt=0.01,
        mu_bath=1.0,
        mu_obj=2.0,
        f0=1.0,
        sigma_f=1.0,
        separation=4.0,
    )
    fixed_state = CoupledSDEState(0, 0.0, [2.0, 7.0], 3.0, 9.0, 3.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    fixed_work = SDEWork(fixed_param.N)
    step_fixed_separation!(fixed_state, fixed_param, fixed_work, 0.5, -0.25, zeros(Float64, fixed_param.N))
    @test isapprox(fixed_state.XA, 3.0)
    @test isapprox(fixed_state.XB, 9.0)
    @test fixed_state.last_dXA == 0.0
    @test fixed_state.last_dXB == 0.0

    rng = MersenneTwister(1234)
    ndraw = 20_000
    dWAs = zeros(Float64, ndraw)
    dWBs = zeros(Float64, ndraw)
    dW1s = zeros(Float64, ndraw)
    for k in 1:ndraw
        dWAk, dWBk, dWik = draw_sde_noises(rng, 2, base_param.dt)
        dWAs[k] = dWAk
        dWBs[k] = dWBk
        dW1s[k] = dWik[1]
    end
    @test abs(cor(dWAs, dWBs)) < 0.03
    @test abs(cor(dWAs, dW1s)) < 0.03
    @test isapprox(var(dWAs), base_param.dt; rtol=0.05)
    @test isapprox(var(dWBs), base_param.dt; rtol=0.05)

    SA_fixed = 3.25
    SB_fixed = 1.75
    mu_obj = 0.4
    synthetic_dXA = -mu_obj .* SA_fixed .* dWAs
    synthetic_dXB = -mu_obj .* SB_fixed .* dWBs
    @test isapprox(var(synthetic_dXA), mu_obj^2 * SA_fixed^2 * base_param.dt; rtol=0.05)
    @test isapprox(var(synthetic_dXB), mu_obj^2 * SB_fixed^2 * base_param.dt; rtol=0.05)
end
