module SaveUtils
using Printf
using JLD2
export save_aggregation, save_state
function save_aggregation(agg_res,param,total_sweeps,save_dir)
    mkpath(save_dir)
    γ′ = param.γ * param.N
    ffr = param.forcing_fluctuation_rate * param.N
    dim = length(param.dims)
    filename = @sprintf("%s/%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gammap-%.2f_D-%.1f_f_-%.1f_ffr-%.2f_t-%d.jld2",
        save_dir,
        dim,
        param.potential_type,
        param.potential_magnitude,
        param.fluctuation_type,
        param.ϵ,
        param.dims[1],
        param.ρ₀,
        param.α,
        γ′,
        param.D,
        state.forcing.magnitude,
        ffr,
        state.t)
    potential = state.potential 
    @save filename state param potential
    return filename
end

# Original save_state function.
function save_state(state, param, save_dir)
    mkpath(save_dir)
    γ = param.γ
    ffr = param.ffr
    dim = length(param.dims)
    filename = @sprintf("%s/%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gammap-%.2f_D-%.1f_f_-%.1f_ffr-%.2f_t-%d.jld2",
        save_dir,
        dim,
        param.potential_type,
        param.potential_magnitude,
        param.fluctuation_type,
        param.ϵ,
        param.dims[1],
        param.ρ₀,
        param.α,
        γ,
        param.D,
        state.forcing.magnitude,
        ffr,
        state.t)
    # filename = @sprintf("%s/%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gammap-%.2f_D-%.1f_t-%d.jld2",
    #     save_dir,
    #     dim,
    #     param.potential_type,
    #     param.potential_magnitude,
    #     param.fluctuation_type,
    #     param.ϵ,
    #     param.dims[1],
    #     param.ρ₀,
    #     param.α,
    #     γ′,
    #     param.D,
    #     state.t)
    potential = state.potential 
    @save filename state param potential
    println("Saved a state to $filename")
    return filename
end

end