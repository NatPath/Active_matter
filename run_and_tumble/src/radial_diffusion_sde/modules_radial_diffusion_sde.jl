module RadialDiffusionSDE

using Random

export RadialDiffusionParam,
       RadialDiffusionState,
       apply_boundary,
       diffusivity,
       drift_prefactor,
       initialize_state,
       steady_state_deficit,
       steady_state_density,
       steady_state_plateau,
       step!,
       validate_param

Base.@kwdef struct RadialDiffusionParam
    dims::Int = 1
    D_inf::Float64 = 1.0
    A::Float64 = 1.0
    core_radius::Float64 = 0.5
    L::Float64 = 16.0
    dt::Float64 = 1.0e-3
    interpretation::Symbol = :ito
    boundary::Symbol = :periodic
end

mutable struct RadialDiffusionState
    positions::Matrix{Float64}
    noise::Matrix{Float64}
    step::Int
end

function validate_param(param::RadialDiffusionParam)
    param.dims == 1 || error("Only the 1D model is currently supported.")
    isfinite(param.D_inf) && param.D_inf > 0.0 || error("D_inf must be finite and positive.")
    isfinite(param.A) || error("A must be finite.")
    isfinite(param.core_radius) && param.core_radius > 0.0 ||
        error("core_radius must be finite and positive.")
    param.A > -(param.core_radius^2) ||
        error("A must be greater than -core_radius^2 so D(x) remains positive.")
    isfinite(param.L) && param.L > 0.0 || error("L must be finite and positive.")
    isfinite(param.dt) && param.dt > 0.0 || error("dt must be finite and positive.")
    param.interpretation in (:ito, :fickian) ||
        error("interpretation must be :ito or :fickian.")
    param.boundary in (:reflecting, :periodic) ||
        error("boundary must be :reflecting or :periodic.")
    return param
end

@inline function apply_boundary(x::Real, param::RadialDiffusionParam)
    half_L = 0.5 * param.L
    x_float = Float64(x)
    -half_L <= x_float < half_L && return x_float

    if param.boundary === :periodic
        return mod(x_float + half_L, param.L) - half_L
    end

    folded = mod(x_float + half_L, 2.0 * param.L)
    return folded <= param.L ? folded - half_L : 3.0 * half_L - folded
end

@inline function diffusivity(r2::Real, param::RadialDiffusionParam)
    return param.D_inf * (1.0 + param.A / (Float64(r2) + param.core_radius^2))
end

"""
Return the scalar `c(x)` for which the deterministic drift is `c(x) * x`.

For the Itô model there is no drift. For the Fickian model the drift is
`D'(x) = -2 D_inf A x / (x² + core_radius²)²`.
"""
@inline function drift_prefactor(r2::Real, param::RadialDiffusionParam)
    param.interpretation === :ito && return 0.0
    denominator = Float64(r2) + param.core_radius^2
    return -2.0 * param.D_inf * param.A / (denominator * denominator)
end

function steady_state_plateau(param::RadialDiffusionParam)
    validate_param(param)
    param.interpretation === :fickian && return 1.0 / param.L

    b = sqrt(param.core_radius^2 + param.A)
    normalization = param.L - 2.0 * param.A * atan(param.L / (2.0 * b)) / b
    return 1.0 / normalization
end

@inline function steady_state_density(x::Real, param::RadialDiffusionParam)
    plateau = steady_state_plateau(param)
    param.interpretation === :fickian && return plateau
    x2 = Float64(x)^2
    return plateau * (x2 + param.core_radius^2) /
           (x2 + param.core_radius^2 + param.A)
end

@inline function steady_state_deficit(distance::Real, param::RadialDiffusionParam)
    plateau = steady_state_plateau(param)
    param.interpretation === :fickian && return 0.0
    d2 = Float64(distance)^2
    return plateau * param.A / (d2 + param.core_radius^2 + param.A)
end

function initialize_state(
    param::RadialDiffusionParam,
    n_walkers::Integer,
    x0::AbstractVector{<:Real},
)
    validate_param(param)
    n_walkers > 0 || error("n_walkers must be positive.")
    length(x0) == param.dims ||
        error("x0 has length $(length(x0)); expected $(param.dims).")

    positions = Matrix{Float64}(undef, param.dims, Int(n_walkers))
    @inbounds for walker in axes(positions, 2), dim in axes(positions, 1)
        positions[dim, walker] = apply_boundary(x0[dim], param)
    end
    return RadialDiffusionState(positions, similar(positions), 0)
end

function step!(
    state::RadialDiffusionState,
    param::RadialDiffusionParam,
    rng::AbstractRNG,
)
    size(state.positions) == size(state.noise) || error("State arrays have incompatible sizes.")
    size(state.positions, 1) == param.dims || error("State dimension does not match param.dims.")

    randn!(rng, state.noise)
    dt = param.dt

    @inbounds @simd for walker in axes(state.positions, 2)
        x = state.positions[1, walker]
        r2 = x * x
        sigma = sqrt(2.0 * diffusivity(r2, param) * dt)
        drift = drift_prefactor(r2, param) * dt
        proposal = x + drift * x + sigma * state.noise[1, walker]
        state.positions[1, walker] = apply_boundary(proposal, param)
    end

    state.step += 1
    return state
end

end
