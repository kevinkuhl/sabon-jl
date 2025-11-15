module ConfigUtils

using YAML

export load_config, save_config, merge_configs, Config

"""
Base configuration with common parameters.
Example-specific configs can add their own fields.
"""
Base.@kwdef mutable struct Config
    # Data parameters
    n_train::Int = 2000
    n_val::Int = 500
    n_test::Int = 500
    n_grid::Int = 100
    sub_sampling::Int = 1
    
    # IMPORTANT: Dimensions
    input_dim::Int = 4  # Dimension of input space (e.g., 2 for 2D, 4 for 4D torus)
    output_dim::Int = 1  # Usually 1 for scalar functions
    
    # Model parameters
    n_basis::Int = 200
    hidden_size::Int = 2048
    n_hidden_layers::Int = 5
    activation::String = "relu"
    
    # Training parameters
    learning_rate::Float64 = 3e-4
    batch_size::Int = 32
    epochs::Int = 200000
    clip_norm::Float64 = 1.0
    early_stopping_patience::Int = 5000
    
    # Loss coefficients (can be toggled on/off)
    use_operator_loss::Bool = true
    use_invariance_loss::Bool = true
    use_aec_in_loss::Bool = true
    use_aec_out_loss::Bool = true
    use_sparsity_loss::Bool = true
    
    λ_operator::Float64 = 1.0
    λ_invariance::Float64 = 0.1
    λ_aec_in::Float64 = 0.1
    λ_aec_out::Float64 = 0.1
    λ_sparsity::Float64 = 0.01
    
    # Device
    device::String = "auto"
    
    # Paths (relative to example directory)
    data_dir::String = "data"
    results_dir::String = "results"
    data_filename::String = "data.jld2"  # Name of data file
    
    # Wandb
    use_wandb::Bool = true
    wandb_project::String = "sabon-experiments"
    wandb_entity::String = ""
    wandb_tags::Vector{String} = String[]
    
    # Analysis
    run_analysis_every::Int = 1000
    save_checkpoints_every::Int = 1000
    
    # Example-specific parameters (stored as a Dict for flexibility)
    example_params::Dict{String, Any} = Dict{String, Any}()
end

"""
Load config from YAML file
"""
function load_config(path::String)
    if !isfile(path)
        @warn "Config file not found: $path. Using defaults."
        return Config()
    end
    
    d = YAML.load_file(path)
    
    # Extract example_params if they exist
    example_params = Dict{String, Any}()
    if haskey(d, "example_params")
        example_params = d["example_params"]
        delete!(d, "example_params")
    end
    
    # Build config from standard fields
    config = Config()
    for (key, value) in d
        key_sym = Symbol(key)
        if hasfield(Config, key_sym)
            setfield!(config, key_sym, value)
        else
            # Unknown fields go into example_params
            example_params[key] = value
        end
    end
    
    config.example_params = example_params
    return config
end

"""
Save config to YAML file
"""
function save_config(config::Config, path::String)
    d = Dict{String, Any}()
    
    # Save all standard fields
    for field in fieldnames(Config)
        if field != :example_params
            d[String(field)] = getfield(config, field)
        end
    end
    
    # Add example_params if not empty
    if !isempty(config.example_params)
        d["example_params"] = config.example_params
    end
    
    YAML.write_file(path, d)
end

"""
Merge two configs, with override taking precedence
"""
function merge_configs(base::Config, override::Config)
    merged = deepcopy(base)
    
    for field in fieldnames(Config)
        if field != :example_params
            # Only override if the value is different from default
            setfield!(merged, field, getfield(override, field))
        end
    end
    
    # Merge example_params
    merged.example_params = merge(base.example_params, override.example_params)
    
    return merged
end

"""
Create config from base + overrides
"""
function create_config(base_path::String, overrides::Dict{String, Any})
    config = load_config(base_path)
    
    for (key, value) in overrides
        key_sym = Symbol(key)
        if hasfield(Config, key_sym)
            setfield!(config, key_sym, value)
        else
            config.example_params[key] = value
        end
    end
    
    return config
end

end  # module