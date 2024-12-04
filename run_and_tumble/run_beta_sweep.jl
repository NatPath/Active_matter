using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config"
            help = "Base configuration file path"
            required = true
        "--n-points"
            help = "Number of β′ points to simulate"
            arg_type = Int
            default = 10
        "--max-beta"
            help = "Maximum β′ value"
            arg_type = Float64
            default = 0.3
        "--min-beta"
            help = "Minimum β′ value"
            arg_type = Float64
            default = 0.0
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # Load base configuration
    config_path = args["config"]
    n_points = args["n-points"]
    max_beta = args["max-beta"]
    min_beta = args["min-beta"]
    
    # Create array of β′ values
    beta_primes = range(min_beta, max_beta, length=n_points)
    
    # Run simulations for each β′ value
    for β′ in beta_primes
        println("Running simulation for β′ = $β′")
        
        # Format β′ for filename
        beta_str = @sprintf("%.3f", β′)
        
        # Create output directory if it doesn't exist
        output_dir = "beta_sweep"
        mkpath(output_dir)
        
        # Create modified config file for this β′
        temp_config = joinpath(output_dir, "config_beta_$(beta_str).toml")
        
        # Copy base config and modify β′
        run(`cp $config_path $temp_config`)
        open(temp_config, "a") do f
            write(f, "\nβ′ = $β′\n")
        end
        
        # Run the simulation
        run(`julia run_and_tumble.jl --config $temp_config`)
    end
    
    println("Beta sweep completed!")
end

main() 