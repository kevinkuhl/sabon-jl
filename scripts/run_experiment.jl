#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

include("../src/training/train.jl")
using .Training

"""
Usage:
    julia run_experiment.jl examples/transfer_operator_anosov
    julia run_experiment.jl examples/transfer_operator_anosov default.yaml my_run
    julia run_experiment.jl examples/simple_koopman_2d custom_config.yaml
"""

function main(args)
    if length(args) < 1
        println("Usage: julia run_experiment.jl <example_dir> [config_name] [run_name]")
        println("Example: julia run_experiment.jl examples/transfer_operator_anosov")
        return
    end
    
    example_dir = args[1]
    config_name = length(args) >= 2 ? args[2] : "default.yaml"
    run_name = length(args) >= 3 ? args[3] : ""
    
    @info "Running experiment: $example_dir with config: $config_name"
    
    train_model(example_dir; config_name=config_name, run_name=run_name)
end

main(ARGS)