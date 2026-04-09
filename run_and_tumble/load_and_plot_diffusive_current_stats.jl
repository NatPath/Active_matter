using JLD2
using ArgParse
using Plots

include(joinpath(@__DIR__, "src", "common", "potentials.jl"))
include(joinpath(@__DIR__, "src", "diffusive", "modules_diffusive_no_activity.jl"))
include(joinpath(@__DIR__, "src", "common", "plot_utils.jl"))

using .PlotUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "saved_states"
            help = "Path(s) to saved state file(s) (.jld2)"
            required = true
            nargs = '+'
        "--out_dir"
            help = "Output directory for per-state plot folders"
            default = "results_figures/fitting"
        "--include_abs_mean_in_spatial_f_plot"
            help = "Include |<F>| in spatial F statistics panel"
            action = :store_true
        "--keep_diagonal_in_multiforce_cut"
            help = "Do not smooth the diagonal point in quarter correlation cut"
            action = :store_true
    end
    return parse_args(s)
end

function save_diffusive_1d_components(saved_state::String, state, param, potential, out_dir::String;
                                      include_abs_mean_in_spatial_f_plot::Bool=false,
                                      keep_diagonal_in_multiforce_cut::Bool=false)
    state.potential = potential

    base_name = replace(basename(saved_state), ".jld2" => "")
    state_dir = joinpath(out_dir, base_name, "current_sweep_statistics")
    mkpath(state_dir)

    components = PlotUtils.plot_sweep_1d_multiforce(
        state.t,
        state,
        param;
        remove_diagonal_for_cuts=!keep_diagonal_in_multiforce_cut,
        include_abs_mean_in_spatial_f_plot=include_abs_mean_in_spatial_f_plot,
        return_components=true,
    )

    panel_map = [
        ("00_composite", components.final_plot),
        ("01_avg_density", components.avg_density),
        ("02_inst_density", components.inst_density),
        ("03_force_averages", components.force_averages),
        ("04_spatial_f_stats", components.spatial_f_stats),
        ("05_corr_heat", components.corr_heat),
        ("06_corr_origin_cut", components.corr_origin_cut),
        ("07_corr_quarter_cut", components.corr_quarter_cut),
    ]

    for (name, plt) in panel_map
        output_file = joinpath(state_dir, string(name, ".png"))
        savefig(plt, output_file)
        println("Saved ", output_file)
    end
end

function main()
    args = parse_commandline()
    out_dir = args["out_dir"]
    include_abs_mean = get(args, "include_abs_mean_in_spatial_f_plot", false)
    keep_diag = get(args, "keep_diagonal_in_multiforce_cut", false)

    for saved_state in args["saved_states"]
        println("Processing ", saved_state)
        @load saved_state state param potential
        if length(param.dims) != 1
            println("Skipping non-1D state: ", saved_state)
            continue
        end
        save_diffusive_1d_components(
            saved_state,
            state,
            param,
            potential,
            out_dir;
            include_abs_mean_in_spatial_f_plot=include_abs_mean,
            keep_diagonal_in_multiforce_cut=keep_diag,
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
