#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using Hyperopt

include("src/training/train.jl")
using .Training

"""
Run hyperparameter optimization for any example.

Usage:
    julia run_hyperopt.jl examples/transfer_operator_anosov
"""

function run_hyperopt(example_dir::String; n_trials::Int=50)
    
    @hyperopt for i=n_trials,
        sampler = RandomSampler(),
        
        # Hyperparameters to tune
        learning_rate = exp10.(LinRange(-4, -2.5, n_trials)),
        hidden_size = [1024, 2048, 4096],
        n_basis = [100, 200, 300],
        λ_operator = [0.5, 1.0, 2.0],
        λ_invariance = LinRange(0.01, 0.5, n_trials),
        λ_aec_in = LinRange(0.01, 0.5, n_trials),
        λ_aec_out = LinRange(0.01, 0.5, n_trials),
        λ_sparsity = exp10.(LinRange(-3, -1, n_trials))
        
        # Create override dict
        overrides = Dict(
            "learning_rate" => learning_rate,
            "hidden_size" => hidden_size,
            "n_basis" => n_basis,
            "λ_operator" => λ_operator,
            "λ_invariance" => λ_invariance,
            "λ_aec_in" => λ_aec_in,
            "λ_aec_out" => λ_aec_out,
            "λ_sparsity" => λ_sparsity,
            "epochs" => 10000,  # Shorter for hyperopt
        )
        
        # Train with overrides
        val_loss = train_model(
            example_dir; 
            run_name="hyperopt_$i",
            overrides=overrides
        )
        
        # Return validation loss to minimize
        val_loss
    end
end

function main(args)
    if length(args) < 1
        println("Usage: julia run_hyperopt.jl <example_dir> [n_trials]")
        return
    end
    
    example_dir = args[1]
    n_trials = length(args) >= 2 ? parse(Int, args[2]) : 50
    
    @info "Running hyperparameter optimization for: $example_dir"
    @info "Number of trials: $n_trials"
    
    best = run_hyperopt(example_dir; n_trials=n_trials)
    
    @info "Best hyperparameters:"
    println(best)
end

main(ARGS)