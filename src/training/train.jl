module Training

using Flux, CUDA, Metal
using ProgressMeter
using Printf
using JLD2
using Statistics
using LinearAlgebra
using Wandb
using Dates

# Include utility modules
include(joinpath(@__DIR__, "..", "utils", "config.jl"))
import .ConfigUtils: load_config, save_config

include(joinpath(@__DIR__, "..", "utils", "device.jl"))
import .DeviceUtils: get_device, setup_device

include(joinpath(@__DIR__, "..", "models", "sabon.jl"))
import .SABONModel: build_sabon

include(joinpath(@__DIR__, "..", "losses", "regularization.jl"))
import .Regularization: invariance_loss, sparsity_loss

include(joinpath(@__DIR__, "..", "analysis", "metrics.jl"))
import .Metrics: compute_principal_angles_with_cat_map

export train_model

"""
    train_model(example_dir::String; config_name::String="default.yaml", run_name::String="")

Generic training function that works with any example.

# Arguments
- `example_dir`: Path to example directory (e.g., "examples/transfer_operator_anosov")
- `config_name`: Name of config file in example_dir/configs/
- `run_name`: Optional name for this run
"""
function train_model(example_dir::String;
                     config_name::String="default.yaml",
                     run_name::String="",
                     overrides::Dict=Dict())

    # Convert example_dir to absolute path
    example_dir = abspath(example_dir)

    # Setup paths
    config_path = joinpath(example_dir, "configs", config_name)
    data_dir = joinpath(example_dir, "data")
    results_dir = joinpath(example_dir, "results")
    
    # Load configuration
    config = load_config(config_path)
    
    # Apply overrides
    for (key, value) in overrides
        key_sym = Symbol(key)
        if hasfield(Config, key_sym)
            setfield!(config, key_sym, value)
        else
            config.example_params[key] = value
        end
    end
    
    # Update data_dir and results_dir to be absolute
    config.data_dir = data_dir
    config.results_dir = results_dir
    
    # Setup device
    device_type = get_device(config.device)
    to_device, ArrayType, DeviceModule = setup_device(device_type)
    @info "Using device: $device_type"
    
    # Create results directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    run_name_final = run_name != "" ? run_name : "run_$(timestamp)"
    run_dir = joinpath(results_dir, run_name_final)
    mkpath(run_dir)
    
    # Save config
    save_config(config, joinpath(run_dir, "config.yaml"))
    
    # Initialize Wandb
    lg = nothing
    if config.use_wandb
        wandb_config = Dict(
            "example" => basename(example_dir),
            "input_dim" => config.input_dim,
            "n_train" => config.n_train,
            "n_val" => config.n_val,
            "n_basis" => config.n_basis,
            "hidden_size" => config.hidden_size,
            "learning_rate" => config.learning_rate,
            "batch_size" => config.batch_size,
            "λ_operator" => config.λ_operator,
            "λ_invariance" => config.λ_invariance,
            "λ_aec_in" => config.λ_aec_in,
            "λ_aec_out" => config.λ_aec_out,
            "λ_sparsity" => config.λ_sparsity,
            "device" => string(device_type)
        )
        
        # Add example-specific params
        for (key, value) in config.example_params
            wandb_config["example_$(key)"] = value
        end
        
        lg = WandbLogger(
            project = config.wandb_project,
            entity = config.wandb_entity,
            name = run_name_final,
            config = wandb_config,
            tags = vcat(config.wandb_tags, [basename(example_dir)])
        )
    end
    
    # Load data
    @info "Loading data from $(config.data_dir)..."
    data_file = joinpath(config.data_dir, config.data_filename)
    
    if !isfile(data_file)
        error("Data file not found: $data_file. Please generate data first.")
    end
    
    loaded = JLD2.load(data_file)
    
    # Generic data loading (assumes standard naming)
    in_funcs = loaded["xdata"][:, 1:config.n_train]
    out_funcs = loaded["ydata"][:, 1:config.n_train]
    val_in_funcs = loaded["xdata"][:, config.n_train+1:config.n_train+config.n_val]
    val_out_funcs = loaded["ydata"][:, config.n_train+1:config.n_train+config.n_val]
    test_in_funcs = loaded["xdata"][:, config.n_train+config.n_val+1:end]
    test_out_funcs = loaded["ydata"][:, config.n_train+config.n_val+1:end]
    gridmat = loaded["gridmat"]
    grid = haskey(loaded, "grid") ? loaded["grid"] : nothing
    
    # Move to device
    in_funcs_dev = to_device(in_funcs)
    out_funcs_dev = to_device(out_funcs)
    gridmat_dev = to_device(gridmat)
    val_in_funcs_dev = to_device(val_in_funcs)
    val_out_funcs_dev = to_device(val_out_funcs)
    
    # Pre-compute A_hat for invariance loss
    @info "Computing operator surrogate..."
    A_hat = (out_funcs * pinv(in_funcs)) |> to_device
    
    # Build model
    @info "Building model with input_dim=$(config.input_dim)..."
    activation = config.activation == "relu" ? relu : 
                 config.activation == "tanh" ? tanh : 
                 config.activation == "gelu" ? gelu : relu
    
    model = build_sabon(
        input_dim = config.input_dim,
        n_basis = config.n_basis,
        hidden_size = config.hidden_size,
        n_layers = config.n_hidden_layers,
        activation = activation
    ) |> to_device
    
    # Setup optimizer
    opt = Flux.setup(
        Flux.Optimisers.OptimiserChain(
            ClipNorm(config.clip_norm),
            Flux.Optimisers.Adam(config.learning_rate)
        ),
        model
    )
    
    # Loss function
    function total_loss(model, in_f, out_f, gmat, compute_all=true)
        # Get basis
        basis = model.basis(gmat)  # n_basis × n_grid_points
        
        # Forward pass
        pred = model(in_f, gmat)
        
        # Operator loss (always compute)
        loss_op = config.use_operator_loss ? Flux.Losses.mse(pred, out_f) : 0.0f0

        if !compute_all
            return loss_op, loss_op, 0.0f0, 0.0f0, 0.0f0, 0.0f0, basis
        end

        # Invariance loss
        loss_inv = 0.0f0
        if config.use_invariance_loss
            loss_inv = invariance_loss(basis, A_hat, config.n_grid)
        end

        # AEC losses
        loss_aec_in = 0.0f0
        loss_aec_out = 0.0f0
        if config.use_aec_in_loss || config.use_aec_out_loss
            if config.use_aec_in_loss
                β_in = model.linear_map(basis * in_f / Float32(size(gmat, 2)))
                recon_in = basis' * β_in
                loss_aec_in = Flux.Losses.mse(recon_in, in_f)
            end

            if config.use_aec_out_loss
                β_out = model.linear_map(basis * out_f / Float32(size(gmat, 2)))
                recon_out = basis' * β_out
                loss_aec_out = Flux.Losses.mse(recon_out, out_f)
            end
        end
        
        # Sparsity loss
        loss_sparse = 0.0f0
        if config.use_sparsity_loss
            loss_sparse = sparsity_loss(basis)
        end
        
        # Total weighted loss
        total = (Float32(config.λ_operator) * loss_op +
                 Float32(config.λ_invariance) * loss_inv +
                 Float32(config.λ_aec_in) * loss_aec_in +
                 Float32(config.λ_aec_out) * loss_aec_out +
                 Float32(config.λ_sparsity) * loss_sparse)
        
        return total, loss_op, loss_inv, loss_aec_in, loss_aec_out, loss_sparse, basis
    end
    
    # Load analysis functions if available
    analysis_available = false
    if isfile(joinpath(example_dir, "scripts", "analyze.jl"))
        include(joinpath(example_dir, "scripts", "analyze.jl"))
        analysis_available = true
    end
    
    # Training loop
    best_val_loss = Inf
    patience_counter = 0
    
    @info "Starting training for $(config.epochs) epochs..."
    @showprogress for epoch in 1:config.epochs
        # Training step
        loss, grads = Flux.withgradient(model) do m
            total, _, _, _, _, _, _ = total_loss(m, in_funcs_dev, out_funcs_dev, gridmat_dev, true)
            total
        end
        
        Flux.update!(opt, model, grads[1])
        
        # Compute detailed metrics every N epochs
        if epoch % 100 == 0 || epoch == 1
            # Training metrics
            _, train_op, train_inv, train_aec_in, train_aec_out, train_sparse, basis = 
                total_loss(model, in_funcs_dev, out_funcs_dev, gridmat_dev, true)
            
            # Validation metrics
            val_total, val_op, val_inv, val_aec_in, val_aec_out, val_sparse, _ = 
                total_loss(model, val_in_funcs_dev, val_out_funcs_dev, gridmat_dev, true)
            
            # Log to Wandb
            log_dict = Dict(
                "epoch" => epoch,
                "train/total_loss" => loss,
                "train/operator_loss" => train_op,
                "val/total_loss" => val_total,
                "val/operator_loss" => val_op,
            )
            
            # Add optional losses
            if config.use_invariance_loss
                log_dict["train/invariance_loss"] = train_inv
                log_dict["val/invariance_loss"] = val_inv
            end
            if config.use_aec_in_loss
                log_dict["train/aec_in_loss"] = train_aec_in
                log_dict["val/aec_in_loss"] = val_aec_in
            end
            if config.use_aec_out_loss
                log_dict["train/aec_out_loss"] = train_aec_out
                log_dict["val/aec_out_loss"] = val_aec_out
            end
            if config.use_sparsity_loss
                log_dict["train/sparsity_loss"] = train_sparse
                log_dict["val/sparsity_loss"] = val_sparse
            end
            
            # Example-specific metrics (if analysis available)
            if analysis_available && grid !== nothing
                try
                    basis_cpu = cpu(basis)
                    pa = compute_principal_angles_with_cat_map(basis_cpu, grid, cpu(gridmat))
                    
                    log_dict["metrics/principal_angle_min"] = minimum(pa)
                    log_dict["metrics/principal_angle_max"] = maximum(pa)
                    log_dict["metrics/principal_angle_mean"] = mean(pa)
                    
                    sin_pa = sin.(pa .* π / 180)
                    log_dict["metrics/sin_principal_angle_max"] = maximum(sin_pa)
                catch e
                    @warn "Could not compute principal angles: $e"
                end
            end
            
            if config.use_wandb && lg !== nothing
                Wandb.log(lg, log_dict)
            end
            
            # Print progress
            @printf("Epoch %d: Train Op=%.6f Val Op=%.6f\n", epoch, train_op, val_op)
            
            # Save best model
            if val_op < best_val_loss
                best_val_loss = val_op
                patience_counter = 0
                
                model_cpu = cpu(model)
                JLD2.jldsave(
                    joinpath(run_dir, "best_model.jld2");
                    model_state = Flux.state(model_cpu),
                    epoch = epoch,
                    val_loss = val_op
                )
                @info "New best model saved (val_op = $val_op)"
            else
                patience_counter += 100
            end
            
            # Early stopping
            if patience_counter >= config.early_stopping_patience
                @info "Early stopping triggered at epoch $epoch"
                break
            end
        end
        
        # Run full analysis periodically
        if analysis_available && epoch % config.run_analysis_every == 0
            @info "Running full analysis..."
            try
                run_analysis(model, gridmat, grid, in_funcs, out_funcs, 
                            val_in_funcs, val_out_funcs, test_in_funcs, test_out_funcs,
                            config, joinpath(run_dir, "analysis_epoch_$epoch"))
            catch e
                @warn "Analysis failed: $e"
            end
        end
        
        # Save checkpoint
        if epoch % config.save_checkpoints_every == 0
            model_cpu = cpu(model)
            JLD2.jldsave(
                joinpath(run_dir, "checkpoint_epoch_$epoch.jld2");
                model_state = Flux.state(model_cpu),
                opt_state = opt,
                epoch = epoch
            )
        end
    end
    
    # Final analysis
    if analysis_available
        @info "Running final analysis..."
        try
            run_analysis(model, gridmat, grid, in_funcs, out_funcs, 
                        val_in_funcs, val_out_funcs, test_in_funcs, test_out_funcs,
                        config, joinpath(run_dir, "final_analysis"))
        catch e
            @warn "Final analysis failed: $e"
        end
    end
    
    if config.use_wandb && lg !== nothing
        close(lg)
    end
    
    @info "Training complete! Results saved to: $run_dir"
    return best_val_loss
end

end  # module