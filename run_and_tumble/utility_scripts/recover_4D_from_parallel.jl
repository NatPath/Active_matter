using JLD2
using ArgParse
using Statistics
using Random
using Dates
include("save_utils.jl")
include("potentials.jl")
include("modules_run_and_tumble.jl")
using .FP
using .SaveUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "input"
            help = "Path to either a single parallel results .jld2 file or directory containing multiple files"
            required = true
        "--output_dir"
            help = "Output directory for the recovered dummy states"
            default = "recovered_states"
        "--batch"
            help = "Process all .jld2 files in the input directory"
            action = :store_true
        "--match_mode"
            help = "Match files from parallel_dir with broken_dir based on creation dates"
            action = :store_true
        "--allow_multiple"
            help = "Allow multiple recovered files per parameter set (adds _index suffix)"
            action = :store_true
        "--parallel_dir"
            help = "Directory containing parallel simulation results (recover_2D files)"
            default = "saved_states_parallel/recover_2D"
        "--broken_dir"
            help = "Directory containing broken aggregated results with correct parameters"
            default = "dummy_states/passive_case/to_recover"
        "--time_tolerance"
            help = "Maximum time difference in seconds for file matching"
            arg_type = Int
            default = 60
    end
    return parse_args(s)
end

function get_file_creation_time(filepath::String)
    """
    Get file creation time in seconds since epoch
    """
    return mtime(filepath)
end

function get_readable_date(filepath::String)
    """
    Get human readable creation date
    """
    return Dates.unix2datetime(mtime(filepath))
end

function match_files_by_date(parallel_dir::String, broken_dir::String, time_tolerance::Int)
    """
    Match files from two directories based on creation dates
    Returns array of tuples: (broken_file_path, parallel_file_path, time_diff)
    """
    println("Matching files between:")
    println("  Parallel dir: $parallel_dir")
    println("  Broken dir: $broken_dir")
    println("  Time tolerance: $time_tolerance seconds")
    
    # Get all .jld2 files from both directories
    parallel_files = String[]
    broken_files = String[]
    
    if !isdir(parallel_dir)
        error("Parallel directory does not exist: $parallel_dir")
    end
    
    if !isdir(broken_dir)
        error("Broken directory does not exist: $broken_dir")
    end
    
    for file in readdir(parallel_dir)
        if endswith(file, ".jld2")
            push!(parallel_files, joinpath(parallel_dir, file))
        end
    end
    
    for file in readdir(broken_dir)
        if endswith(file, ".jld2")
            push!(broken_files, joinpath(broken_dir, file))
        end
    end
    
    println("Found $(length(parallel_files)) parallel files")
    println("Found $(length(broken_files)) broken files")
    
    # Create date-file pairs
    parallel_dates = [(get_file_creation_time(f), f, get_readable_date(f)) for f in parallel_files]
    broken_dates = [(get_file_creation_time(f), f, get_readable_date(f)) for f in broken_files]
    
    # Sort by date
    sort!(parallel_dates, by=x->x[1])
    sort!(broken_dates, by=x->x[1])
    
    println("\nMatching files by creation date...")
    println("Date range analysis:")
    if !isempty(parallel_dates) && !isempty(broken_dates)
        println("  Parallel files: $(parallel_dates[1][3]) to $(parallel_dates[end][3])")
        println("  Broken files: $(broken_dates[1][3]) to $(broken_dates[end][3])")
    end
    
    matches = Tuple{String, String, Float64}[]
    unmatched_parallel = copy(parallel_files)
    used_broken_files = String[]
    
    # First pass: Find best matches for each parallel file
    for (parallel_date, parallel_file, parallel_readable) in parallel_dates
        best_match = ""
        best_match_readable = nothing
        min_diff = Inf
        
        # Find closest broken file within tolerance
        for (broken_date, broken_file, broken_readable) in broken_dates
            diff = abs(parallel_date - broken_date)
            if diff < min_diff && diff <= time_tolerance
                min_diff = diff
                best_match = broken_file
                best_match_readable = broken_readable
            end
        end
        
        if !isempty(best_match)
            push!(matches, (best_match, parallel_file, min_diff))
            println("MATCH: $(basename(best_match)) ↔ $(basename(parallel_file)) (diff: $(round(min_diff, digits=1))s)")
            println("  Broken:   $best_match_readable")
            println("  Parallel: $parallel_readable")
            
            # Remove from unmatched parallel files
            filter!(x -> x != parallel_file, unmatched_parallel)
            
            # Track which broken files are being used (allow reuse)
            if !(best_match in used_broken_files)
                push!(used_broken_files, best_match)
            end
        end
    end
    
    # Report unmatched parallel files
    unmatched_broken = String[]
    for (broken_date, broken_file, broken_readable) in broken_dates
        if !(broken_file in used_broken_files)
            push!(unmatched_broken, broken_file)
            println("UNMATCHED BROKEN: $(basename(broken_file)) - $broken_readable")
        end
    end
    
    println("\nMatching summary:")
    println("  Total matches: $(length(matches))")
    println("  Unique broken files used: $(length(used_broken_files))")
    println("  Unmatched parallel files: $(length(unmatched_parallel))")
    println("  Unmatched broken files: $(length(unmatched_broken))")
    
    # Show parameter reuse statistics
    broken_usage = Dict{String, Int}()
    for (broken_file, _, _) in matches
        broken_usage[broken_file] = get(broken_usage, broken_file, 0) + 1
    end
    
    reused_count = sum(usage > 1 for usage in values(broken_usage))
    if reused_count > 0
        println("\nParameter reuse statistics:")
        println("  Broken files used multiple times: $reused_count")
        for (broken_file, usage) in broken_usage
            if usage > 1
                println("    $(basename(broken_file)): used $usage times")
            end
        end
    end
    
    # Show some examples of unmatched parallel files
    if length(unmatched_parallel) > 0
        println("\nSample unmatched parallel files:")
        for (i, file) in enumerate(unmatched_parallel[1:min(5, length(unmatched_parallel))])
            readable_date = get_readable_date(file)
            println("  $(i). $(basename(file)) - $readable_date")
        end
        if length(unmatched_parallel) > 5
            println("  ... and $(length(unmatched_parallel) - 5) more")
        end
    end
    
    return matches, unmatched_broken
end

function recover_4D_correlation(normalized_dists, corr_mats, param)
    """
    Attempt to recover proper 4D correlation tensor from parallel results
    """
    println("Original correlation matrix dimensions: ", size(corr_mats[1]))
    println("Number of parallel results: ", length(corr_mats))
    
    # Check what we're dealing with
    corr_dims = ndims(corr_mats[1])
    dist_dims = ndims(normalized_dists[1])
    
    println("Correlation tensor has $corr_dims dimensions")
    println("Density arrays have $dist_dims dimensions") 
    
    # Aggregate the results properly based on dimensions
    if corr_dims == 2 && dist_dims == 1
        # 1D case - this should work fine
        println("Detected 1D case")
        stacked_corr = cat(corr_mats..., dims=3)
        avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)
        stacked_dists = cat(normalized_dists..., dims=2)
        avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
        
    elseif corr_dims == 4 && dist_dims == 2
        # 2D case - this is what we want
        println("Detected proper 2D case with 4D correlation tensor")
        stacked_corr = cat(corr_mats..., dims=5)
        avg_corr = dropdims(mean(stacked_corr, dims=5), dims=5)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
        
    elseif corr_dims == 3 && dist_dims == 2
        # This is likely the problematic case from cluster
        println("Detected problematic 2D case with 3D correlation tensor")
        println("Correlation tensor size: ", size(corr_mats[1]))
        
        # We can't fully recover the 4D tensor, but we can at least aggregate properly
        # and flag this as problematic
        stacked_corr = cat(corr_mats..., dims=4)
        avg_corr = dropdims(mean(stacked_corr, dims=4), dims=4)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
        
        println("WARNING: Cannot recover full 4D correlation tensor from 3D data!")
        println("The aggregated result will have reduced correlation information.")
        
    else
        error("Unexpected dimensions: corr_dims=$corr_dims, dist_dims=$dist_dims")
    end
    
    println("Final averaged correlation dimensions: ", size(avg_corr))
    println("Final averaged density dimensions: ", size(avg_dists))
    
    return avg_dists, avg_corr
end

function process_single_file(parallel_results_file, output_dir)
    """
    Process a single parallel results file
    """
    println("Processing file: $parallel_results_file")
    
    # Load the parallel results
    @load parallel_results_file normalized_dists corr_mats avg_corr avg_dists
    
    println("Analyzing loaded data...")
    println("Number of runs: ", length(normalized_dists))
    println("Density dimensions: ", size(normalized_dists[1]))
    println("Correlation dimensions: ", size(corr_mats[1]))
    
    # Load parameters from your saved dummy state file
    dummy_state_file = "dummy_states/passive_case/2D_potential-xy_slides_Vscale-16.0_fluctuation-profile_switch_activity-0.00_L-32_rho-1.0e+02_alpha-0.00_gammap-32.00_D-1.0_t-9000000.jld2"
    
    if isfile(dummy_state_file)
        println("Loading parameters from: $dummy_state_file")
        @load dummy_state_file state param potential
        t = state.t
        println("Loaded state time: $t")
        
        dims = param.dims
        println("Loaded param: L=$(param.dims), α=$(param.α), γ′=$(param.γ*param.N)")
        rng = MersenneTwister(123)
        v_args = Potentials.potential_args("xy_slides", dims; magnitude=16.0)
        potential = Potentials.choose_potential(v_args, dims; fluctuation_type="profile_switch", rng=rng)
        reference_state = FP.setState(state.t, rng, param, 1.0, potential)
        state = reference_state
    else
        println("Warning: Could not find dummy state file, using default parameters")
        L = size(normalized_dists[1], 1)
        dims = (L, L)
        
        param = FP.setParam(
            0.0, 1.0, 0.0, dims, 100.0, 1.0,
            "xy_slides", "profile_switch", 16.0
        )
        
        rng = MersenneTwister(123)
        v_args = Potentials.potential_args("xy_slides", dims; magnitude=16.0)
        potential = Potentials.choose_potential(v_args, dims; fluctuation_type="profile_switch", rng=rng)
        reference_state = FP.setState(0, rng, param, 1.0, potential)
        state = reference_state
    end
    
    # Recover the correlation tensor
    recovered_dists, recovered_corr = recover_4D_correlation(normalized_dists, corr_mats, param)
    
    # Create dummy state with recovered data
    nsteps = state.t
    dummy_state = FP.setDummyState(state, recovered_dists, recovered_corr, nsteps)
    
    # Save the recovered dummy state
    filename = save_state(dummy_state, param, output_dir)
    println("Recovered dummy state saved to: $filename")
    
    return filename
end

function process_directory(input_dir, output_dir)
    """
    Process all .jld2 files in a directory
    """
    println("Processing all .jld2 files in directory: $input_dir")
    
    # Find all .jld2 files in the directory
    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(input_dir))
    
    if isempty(jld2_files)
        println("No .jld2 files found in $input_dir")
        return
    end
    
    println("Found $(length(jld2_files)) .jld2 files to process")
    
    # Process each file
    processed_files = String[]
    failed_files = String[]
    
    for (i, filename) in enumerate(jld2_files)
        filepath = joinpath(input_dir, filename)
        println("\n=== Processing file $i/$(length(jld2_files)): $filename ===")
        
        try
            output_filename = process_single_file(filepath, output_dir)
            push!(processed_files, output_filename)
            println("✓ Successfully processed: $filename")
        catch e
            println("✗ Failed to process $filename: $e")
            push!(failed_files, filename)
        end
    end
    
    # Summary
    println("\n=== BATCH PROCESSING SUMMARY ===")
    println("Total files: $(length(jld2_files))")
    println("Successfully processed: $(length(processed_files))")
    println("Failed: $(length(failed_files))")
    
    if !isempty(failed_files)
        println("Failed files:")
        for f in failed_files
            println("  - $f")
        end
    end
end

function process_matched_pair(broken_file::String, parallel_file::String, output_dir::String, allow_multiple::Bool=false, index_suffix::String="")
    """
    Process a matched pair: use parallel results with parameters from broken file
    """
    println("\nProcessing matched pair:")
    println("  Parallel results: $(basename(parallel_file))")
    println("  Parameters from: $(basename(broken_file))")
    if !isempty(index_suffix)
        println("  Index suffix: $index_suffix")
    end
    
    # Load parallel results
    @load parallel_file normalized_dists corr_mats
    println("Loaded parallel results with $(length(normalized_dists)) runs")
    
    # Load parameters from broken file
    @load broken_file state param potential
    println("Loaded parameters: L=$(param.dims), α=$(param.α), γ′=$(param.γ*param.N)")
    
    # Recover the correlation tensor using the correct parameters
    recovered_dists, recovered_corr = recover_4D_correlation(normalized_dists, corr_mats, param)
    
    # Create dummy state with recovered data and correct parameters
    nsteps = state.t
    dummy_state = FP.setDummyState(state, recovered_dists, recovered_corr, nsteps)
    
    # Save the recovered dummy state with optional index suffix
    if allow_multiple && !isempty(index_suffix)
        filename = save_state_with_suffix(dummy_state, param, output_dir, index_suffix)
    else
        filename = save_state(dummy_state, param, output_dir)
    end
    println("Recovered dummy state saved to: $filename")
    
    return filename
end

function save_state_with_suffix(state, param, save_dir, suffix)
    """
    Save state with an additional suffix before the file extension
    """
    # Generate the normal filename first
    normal_filename = SaveUtils.generate_filename(param, state.t)
    
    # Insert suffix before .jld2 extension
    base_name = replace(normal_filename, ".jld2" => "")
    filename_with_suffix = "$(base_name)$(suffix).jld2"
    
    # Save with the modified filename
    mkpath(save_dir)
    filepath = joinpath(save_dir, filename_with_suffix)
    
    @save filepath state param
    println("State saved to: $filepath")
    return filepath
end

function process_matched_files(matches, output_dir::String, allow_multiple::Bool=false)
    """
    Process all matched file pairs, handling multiple results per parameter set if enabled
    """
    println("\n=== PROCESSING MATCHED PAIRS ===")
    println("Allow multiple results: $allow_multiple")
    
    processed_files = String[]
    failed_files = String[]
    
    if allow_multiple
        # Group matches by broken file to handle multiple parallel files per parameter set
        broken_file_groups = Dict{String, Vector{Tuple{String, String, Float64}}}()
        
        for match in matches
            broken_file, parallel_file, time_diff = match
            
            if !haskey(broken_file_groups, broken_file)
                broken_file_groups[broken_file] = Tuple{String, String, Float64}[]
            end
            push!(broken_file_groups[broken_file], match)
        end
        
        println("Grouped $(length(matches)) matches into $(length(broken_file_groups)) parameter sets")
        
        # Process each group
        for (broken_file, group_matches) in broken_file_groups
            println("\nProcessing parameter set: $(basename(broken_file)) ($(length(group_matches)) parallel files)")
            
            for (i, (broken_file, parallel_file, time_diff)) in enumerate(group_matches)
                # Add index suffix for multiple files using same parameters
                index_suffix = length(group_matches) > 1 ? "_$(i)" : ""
                
                try
                    output_filename = process_matched_pair(broken_file, parallel_file, output_dir, allow_multiple, index_suffix)
                    push!(processed_files, output_filename)
                    println("✓ Successfully processed: $(basename(broken_file))$index_suffix ← $(basename(parallel_file))")
                catch e
                    println("✗ Failed to process $(basename(broken_file))$index_suffix ← $(basename(parallel_file)): $e")
                    push!(failed_files, (broken_file, parallel_file))
                end
            end
        end
    else
        # Original single-match processing
        for (i, (broken_file, parallel_file, time_diff)) in enumerate(matches)
            println("\n--- Processing pair $i/$(length(matches)) ---")
            
            try
                output_filename = process_matched_pair(broken_file, parallel_file, output_dir, false, "")
                push!(processed_files, output_filename)
                println("✓ Successfully processed pair $i")
            catch e
                println("✗ Failed to process pair $i: $e")
                push!(failed_files, (broken_file, parallel_file))
                println(e)
            end
        end
    end
    
    # Summary
    println("\n=== MATCHING PROCESSING SUMMARY ===")
    println("Total matched pairs: $(length(matches))")
    println("Successfully processed: $(length(processed_files))")
    println("Failed: $(length(failed_files))")
    
    if !isempty(failed_files)
        println("Failed pairs:")
        for (broken, parallel) in failed_files
            println("  - $(basename(broken)) ↔ $(basename(parallel))")
        end
    end
    
    return processed_files, failed_files
end

function main()
    args = parse_commandline()
    
    input_path = args["input"]
    output_dir = args["output_dir"]
    batch_mode = args["batch"]
    match_mode = args["match_mode"]
    allow_multiple = args["allow_multiple"]
    parallel_dir = args["parallel_dir"]
    broken_dir = args["broken_dir"]
    time_tolerance = args["time_tolerance"]
    
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    if match_mode
        # Match files by creation date and process pairs
        matches, unmatched = match_files_by_date(parallel_dir, broken_dir, time_tolerance)
        
        if isempty(matches)
            println("No file matches found!")
            return
        end
        
        # Save matching report
        report_file = joinpath(output_dir, "matching_report.txt")
        open(report_file, "w") do f
            println(f, "File Matching Report - $(Dates.now())")
            println(f, "="^50)
            println(f, "Parallel directory: $parallel_dir")
            println(f, "Broken directory: $broken_dir")
            println(f, "Time tolerance: $time_tolerance seconds")
            println(f, "Allow multiple results: $allow_multiple")
            println(f, "")
            println(f, "MATCHED PAIRS:")
            for (i, (broken, parallel, diff)) in enumerate(matches)
                println(f, "$i. $(basename(broken)) ↔ $(basename(parallel)) ($(round(diff, digits=1))s)")
            end
            println(f, "")
            println(f, "UNMATCHED FILES:")
            for broken in unmatched
                println(f, "- $(basename(broken))")
            end
        end
        println("Matching report saved to: $report_file")
        
        # Process matched pairs
        processed, failed = process_matched_files(matches, output_dir, allow_multiple)
        
    elseif batch_mode || isdir(input_path)
        # Process directory
        if !isdir(input_path)
            error("Input path $input_path is not a directory")
        end
        process_directory(input_path, output_dir)
    else
        # Process single file
        if !isfile(input_path)
            error("Input file $input_path does not exist")
        end
        process_single_file(input_path, output_dir)
    end
end

main()
