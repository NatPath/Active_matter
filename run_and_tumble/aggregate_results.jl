using JLD2
using ArgParse
using LinearAlgebra
using Statistics
include("modules_run_and_tumble.jl")
include("save_utils.jl")
using .FP
using .SaveUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "saved_states"
            help = "Path(s) to the saved state file(s) (.jld2)"
            required = true
            nargs = '+'
        "--method"
            help = "Aggregation method: 'mean' (default) or 'svd'"
            default = "mean"
        "--cut"
            help = "Specific cut to process (e.g., 'x_cut'). Default 'auto' selects 'full' for 1D and 'x_cut' for 2D."
            default = "auto"
    end
    return parse_args(s)
end

# Helper to extract specific cut data even if only :full exists in the file
function get_cut_data(state, cut_name, param)
    cuts = state.ρ_matrix_avg_cuts
    sym_name = Symbol(cut_name)
    
    # 1. Direct match (e.g. 1D :full, or 2D :x_cut if it exists)
    if haskey(cuts, sym_name)
        return vec(cuts[sym_name])
    end
    
    # 2. Extract x_cut from Full Tensor in 2D
    if cut_name == "x_cut" && haskey(cuts, :full)
        # x-cut is typically taken at y_middle
        y_mid = div(param.dims[2], 2)
        full_tensor = cuts[:full]
        # Tensor structure is (Nx, Ny, Nx, Ny). We want [:, y_mid, :, y_mid]
        slice = full_tensor[:, y_mid, :, y_mid]
        return vec(slice)
    end
    
    return nothing
end

function main()
    args = parse_commandline()
    method = args["method"]
    files = args["saved_states"]
    n_files = length(files)
    
    println("Processing $n_files files using method: $method")
    
    # --- METHOD 1: ARITHMETIC MEAN (Default) ---
    if method == "mean"
        ρ_avg_sum = nothing
        cut_sums = Dict{Symbol,Array{Float64}}()
        total_weight = 0.0
        params_ref = nothing
        state_ref = nothing
        total_t = 0
        
        for (i, file) in enumerate(files)
            println("  [Mean] Loading $i/$n_files: $(basename(file))")
            try
                @load file state param
                if i % 20 == 0; GC.gc(); end # periodic GC
                
                params_ref = param
                total_t += state.t
                
                if state_ref === nothing
                    state_ref = state
                    ρ_avg_sum = zeros(size(state.ρ_avg))
                    for (k, arr) in state.ρ_matrix_avg_cuts
                        cut_sums[k] = zeros(size(arr))
                    end
                end
                
                weight = max(state.t, 1)
                total_weight += weight
                ρ_avg_sum .+= state.ρ_avg .* weight
                for (k, arr) in state.ρ_matrix_avg_cuts
                    if haskey(cut_sums, k)
                        cut_sums[k] .+= arr .* weight
                    end
                end
            catch e
                println("  ERROR loading $file: $e")
            end
        end
        
        if state_ref === nothing; error("No valid states found"); end
        
        avg_ρ = ρ_avg_sum ./ total_weight
        agg_cuts = Dict{Symbol,Array{Float64}}()
        for (k, sum_arr) in cut_sums
            agg_cuts[k] = sum_arr ./ total_weight
        end
        
        dummy = FP.setDummyState(state_ref, avg_ρ, agg_cuts, total_t)
        save_dir = "dummy_states_agg"
        mkpath(save_dir)
        save_state(dummy, params_ref, save_dir; tag="agg_mean")

    # --- METHOD 2: SVD (Principal Component) ---
    elseif method == "svd"
        # 1. Determine Dimensions & Target Cuts
        @load files[1] state param
        state_ref = state
        params_ref = param
        total_t = state.t
        
        # Initialize agg_cuts with EVERYTHING from the first file (Dummies)
        # This ensures diag_cut, y_cut, etc., exist even if we don't process them.
        agg_cuts = deepcopy(state.ρ_matrix_avg_cuts)
        
        # Auto-selection logic
        target_cuts = String[]
        if args["cut"] == "auto"
            dim_num = length(param.dims)
            if dim_num == 1
                push!(target_cuts, "full")
            elseif dim_num == 2
                push!(target_cuts, "x_cut") # Only process x_cut for 2D to save RAM
            end
        else
            push!(target_cuts, args["cut"])
        end
        println("  [SVD] Target cuts to process: $target_cuts")
        println("  [SVD] Other cuts will be preserved as dummies from first file.")

        # 2. Pre-allocate Memory
        rho_len = length(state.ρ_avg)
        mat_ρ = Matrix{Float64}(undef, rho_len, n_files)
        
        mat_cuts = Dict{String, Matrix{Float64}}()
        cut_shapes = Dict{String, Tuple}()
        
        for cname in target_cuts
            sample_data = get_cut_data(state, cname, param)
            if sample_data !== nothing
                len = length(sample_data)
                mat_cuts[cname] = Matrix{Float64}(undef, len, n_files)
                # Store original shape for reshaping later
                if cname == "x_cut" && haskey(state.ρ_matrix_avg_cuts, :full)
                    cut_shapes[cname] = (param.dims[1], param.dims[1])
                elseif haskey(state.ρ_matrix_avg_cuts, Symbol(cname))
                    cut_shapes[cname] = size(state.ρ_matrix_avg_cuts[Symbol(cname)])
                end
            else
                println("  WARNING: Target cut '$cname' not found in first file.")
            end
        end
        
        state = nothing; GC.gc()

        # 3. Stream Data
        valid_idxs = Int[]
        for (i, file) in enumerate(files)
            println("  [SVD] Loading $i/$n_files into memory...")
            try
                @load file state param
                if i > 1; total_t += state.t; end
                
                mat_ρ[:, i] = vec(state.ρ_avg)
                
                for (cname, mat) in mat_cuts
                    data = get_cut_data(state, cname, param)
                    if data !== nothing
                        mat[:, i] = data
                    end
                end
                push!(valid_idxs, i)
            catch e
                println("  Error: $e")
            end
            state = nothing; param = nothing; GC.gc()
        end
        
        if isempty(valid_idxs); error("No valid files"); end
        
        # Resize to valid only
        if length(valid_idxs) < n_files
            mat_ρ = mat_ρ[:, valid_idxs]
            for k in keys(mat_cuts); mat_cuts[k] = mat_cuts[k][:, valid_idxs]; end
        end

        # 4. Compute Results
        avg_ρ = reshape(mean(mat_ρ, dims=2), size(state_ref.ρ_avg))
        mat_ρ = nothing; GC.gc()
        
        # Overwrite the target cuts in agg_cuts with the SVD results
        for (cname, mat) in mat_cuts
            println("  Computing SVD for $cname...")
            F = svd(mat)
            
            mode1 = F.U[:, 1] .* (F.S[1] / sqrt(length(valid_idxs)))
            
            mean_vec = vec(mean(mat, dims=2))
            if dot(mode1, mean_vec) < 0; mode1 .*= -1; end
            
            # Update the dictionary (overwriting dummy or adding new)
            if haskey(cut_shapes, cname)
                agg_cuts[Symbol(cname)] = reshape(mode1, cut_shapes[cname])
            else
                agg_cuts[Symbol(cname)] = mode1 
            end
            
            mat = nothing; GC.gc()
        end
        
        dummy = FP.setDummyState(state_ref, avg_ρ, agg_cuts, total_t)
        save_dir = "dummy_states_agg"
        mkpath(save_dir)
        tag = "agg_svd_$(join(target_cuts, "_"))"
        save_state(dummy, params_ref, save_dir; tag=tag)
    end
end

main()
