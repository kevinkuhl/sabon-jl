#!/usr/bin/env julia

"""
Generate training data for the perturbed Arnold's cat map (Transfer Operator).

This generates functions as random linear combinations of Fourier basis functions,
then applies the transfer operator (composition with the cat map).

Usage:
    julia generate_data.jl
    julia generate_data.jl --n_functions 3000 --n_grid 100 --max_order 4
"""

using LinearAlgebra
using JLD2
using Printf
using Random
using Plots
using Statistics

# Parse command line arguments
function parse_args(args)
    n_functions = 3000  # Total: 2000 train + 500 val + 500 test
    n_grid = 100
    max_order = 4
    delta = 0.01  # Perturbation parameter
    output_dir = "../data"
    seed = 42
    
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--n_functions" && i < length(args)
            i += 1
            n_functions = parse(Int, args[i])
        elseif arg == "--n_grid" && i < length(args)
            i += 1
            n_grid = parse(Int, args[i])
        elseif arg == "--max_order" && i < length(args)
            i += 1
            max_order = parse(Int, args[i])
        elseif arg == "--delta" && i < length(args)
            i += 1
            delta = parse(Float64, args[i])
        elseif arg == "--output_dir" && i < length(args)
            i += 1
            output_dir = args[i]
        elseif arg == "--seed" && i < length(args)
            i += 1
            seed = parse(Int, args[i])
        end
        i += 1
    end
    
    return n_functions, n_grid, max_order, delta, output_dir, seed
end

"""
Create 1D Fourier basis functions.
Returns a list of (type, order, scale) tuples.
"""
function make_1d_basis(max_order::Int)
    r2 = sqrt(2.0)
    basis = [("c", 0, 1.0)]  # Constant term
    
    for k in 1:max_order
        push!(basis, ("c", k, r2))  # cos(k*θ)
        push!(basis, ("s", k, r2))  # sin(k*θ)
    end
    
    return basis
end

"""
Create 2D Fourier basis as tensor product of 1D bases.
"""
function make_2d_basis(max_order::Int)
    basis_1d = make_1d_basis(max_order)
    basis_2d = []
    
    for (t1, k1, s1) in basis_1d
        for (t2, k2, s2) in basis_1d
            push!(basis_2d, (t1, k1, t2, k2, s1 * s2))
        end
    end
    
    return basis_2d
end

"""
Evaluate 2D Fourier basis at points on the 2-torus.
Points are given as (x1, y1, x2, y2) where (x1, y1) and (x2, y2) are on unit circles.
"""
function eval_basis_2d(basis_2d, x1, y1, x2, y2)
    n_basis = length(basis_2d)
    n_points = length(x1)
    values = zeros(n_basis, n_points)
    
    # Convert to angles
    tp = 2π
    t1 = (atan.(y1, x1) .+ 2π) .% (2π) ./ tp
    t2 = (atan.(y2, x2) .+ 2π) .% (2π) ./ tp
    
    for (i, (type1, k1, type2, k2, scale)) in enumerate(basis_2d)
        # Evaluate basis function
        f1 = type1 == "c" ? cos.(tp * k1 * t1) : sin.(tp * k1 * t1)
        f2 = type2 == "c" ? cos.(tp * k2 * t2) : sin.(tp * k2 * t2)
        values[i, :] = scale * f1 .* f2
    end
    
    return values
end

"""
Perturbed Arnold's cat map (Transfer operator version).
Maps (θ, ψ) → (θ', ψ') on the 2-torus.
"""
function cat_map(x, delta=0.01)
    # x is [x1, y1, x2, y2] where (x1,y1) and (x2,y2) are on unit circles
    # Convert to angles
    theta = atan(x[2], x[1])
    psi = atan(x[4], x[3])
    
    # Apply perturbed cat map
    theta_new = 2*theta + psi + 2*delta*cos(2π*theta)
    psi_new = theta + psi + delta*sin(4π*psi + 1)
    
    # Map back to [0, 2π]
    theta_new = mod(theta_new, 2π)
    psi_new = mod(psi_new, 2π)
    
    return [cos(theta_new), sin(theta_new), cos(psi_new), sin(psi_new)]
end

"""
Compute inverse cat map on grid for transfer operator.
Uses Newton's method to find preimages.
"""
function inverse_cat_map_grid(n_grid::Int, delta::Float64)
    # Create grid points
    pts = (collect(0:n_grid-1) .+ 0.5) ./ n_grid
    theta_grid = 2π * pts
    psi_grid = 2π * pts
    
    # Create all grid points as 4D vectors
    grid = []
    for psi in psi_grid
        for theta in theta_grid
            push!(grid, [cos(theta), sin(theta), cos(psi), sin(psi)])
        end
    end
    
    # For each grid point, find its preimage using Newton's method
    preimages = similar(grid)
    determinants = zeros(length(grid))
    
    for (i, y) in enumerate(grid)
        # Target angles
        theta_target = atan(y[2], y[1])
        psi_target = atan(y[4], y[3])
        
        # Initial guess (linear inverse)
        x = [2*theta_target - psi_target, -theta_target + 2*psi_target]
        x = mod.(x, 2π)
        
        # Newton's method
        for iter in 1:30
            theta, psi = x
            
            # Residual
            theta_new = 2*theta + psi + 2*delta*cos(2π*theta)
            psi_new = theta + psi + delta*sin(4π*psi + 1)
            
            r = [(theta_new - theta_target + π) % (2π) - π,
                 (psi_new - psi_target + π) % (2π) - π]
            
            if norm(r) < 1e-12
                break
            end
            
            # Jacobian
            J = [2 - 4π*delta*sin(2π*theta)  1;
                 1  1 + 4π*delta*cos(4π*psi + 1)]
            
            # Update
            x = (x - J \ r) .% (2π)
        end
        
        theta, psi = x
        preimages[i] = [cos(theta), sin(theta), cos(psi), sin(psi)]
        
        # Compute Jacobian determinant
        J_det = (2 - 4π*delta*sin(2π*theta)) * (1 + 4π*delta*cos(4π*psi + 1)) - 1
        determinants[i] = abs(J_det)
    end
    
    return preimages, determinants
end

"""
Generate training data for the transfer operator.
"""
function generate_data(n_functions::Int, n_grid::Int, max_order::Int, delta::Float64, seed::Int)
    Random.seed!(seed)
    
    @info "Generating $n_functions functions on a $(n_grid)×$(n_grid) grid"
    @info "Max Fourier order: $max_order, Perturbation: δ=$delta"
    
    # Create Fourier basis
    basis_2d = make_2d_basis(max_order)
    n_basis = length(basis_2d)
    @info "Number of Fourier basis functions: $n_basis"
    
    # Create random linear combinations
    @info "Creating random function coefficients..."
    coefficients = 2 * rand(n_functions, n_basis) .- 1  # Random in [-1, 1]
    
    # Create grid on 4D torus (product of two circles)
    @info "Creating grid..."
    pts = (collect(0:n_grid-1) .+ 0.5) ./ n_grid
    theta_vals = 2π * pts
    psi_vals = 2π * pts
    
    grid = []
    x1_vals = Float64[]
    y1_vals = Float64[]
    x2_vals = Float64[]
    y2_vals = Float64[]
    
    for psi in psi_vals
        for theta in theta_vals
            push!(grid, [[cos(theta), sin(theta)], [cos(psi), sin(psi)]])
            push!(x1_vals, cos(theta))
            push!(y1_vals, sin(theta))
            push!(x2_vals, cos(psi))
            push!(y2_vals, sin(psi))
        end
    end
    
    # Grid in matrix form (4 × n_grid^2)
    gridmat = hcat([[x1_vals[i], y1_vals[i], x2_vals[i], y2_vals[i]] for i in 1:n_grid^2]...)
    
    # Evaluate basis at grid points
    @info "Evaluating basis functions at grid points..."
    basis_values = eval_basis_2d(basis_2d, x1_vals, y1_vals, x2_vals, y2_vals)
    
    # Generate input functions (xdata)
    @info "Generating input functions..."
    xdata = coefficients * basis_values  # n_functions × n_grid^2
    
    # Compute inverse cat map for transfer operator
    @info "Computing inverse cat map (this may take a moment)..."
    preimages, determinants = inverse_cat_map_grid(n_grid, delta)
    
    # Extract preimage coordinates
    pre_x1 = [p[1] for p in preimages]
    pre_y1 = [p[2] for p in preimages]
    pre_x2 = [p[3] for p in preimages]
    pre_y2 = [p[4] for p in preimages]
    
    # Evaluate basis at preimages and divide by determinant
    @info "Evaluating basis at preimages..."
    basis_at_preimages = eval_basis_2d(basis_2d, pre_x1, pre_y1, pre_x2, pre_y2)
    
    # Apply transfer operator: divide by Jacobian determinant
    for i in 1:n_grid^2
        basis_at_preimages[:, i] ./= determinants[i]
    end
    
    # Generate output functions (ydata) - transfer operator applied
    @info "Generating output functions (transfer operator applied)..."
    ydata = coefficients * basis_at_preimages  # n_functions × n_grid^2
    
    # Reshape for consistency with your format
    # Each row should be rotated 180 degrees
    xdata_reshaped = zeros(n_grid^2, n_functions)
    ydata_reshaped = zeros(n_grid^2, n_functions)
    
    for i in 1:n_functions
        x_mat = reshape(xdata[i, :], n_grid, n_grid)
        y_mat = reshape(ydata[i, :], n_grid, n_grid)
        
        # Rotate 180 degrees
        x_mat = rot180(x_mat)
        y_mat = rot180(y_mat)
        
        xdata_reshaped[:, i] = vec(x_mat)
        ydata_reshaped[:, i] = vec(y_mat)
    end
    
    return xdata_reshaped, ydata_reshaped, grid, gridmat, basis_2d
end

"""
Generate diagnostic plots.
"""
function generate_diagnostic_plots(xdata, ydata, basis_2d, n_grid, output_dir)
    @info "Generating diagnostic plots..."
    
    # Plot a few basis functions
    p_basis = []
    basis_indices = [1, 25, length(basis_2d)]
    
    for idx in basis_indices
        t1, k1, t2, k2, scale = basis_2d[idx]
        title_str = "Basis $idx: $(t1)$(k1) × $(t2)$(k2)"
        
        # Reconstruct 2D grid values
        pts = (collect(0:n_grid-1) .+ 0.5) ./ n_grid
        theta_vals = 2π * pts
        psi_vals = 2π * pts
        
        vals = zeros(n_grid, n_grid)
        for (i, psi) in enumerate(psi_vals)
            for (j, theta) in enumerate(theta_vals)
                f1 = t1 == "c" ? cos(2π*k1*theta/(2π)) : sin(2π*k1*theta/(2π))
                f2 = t2 == "c" ? cos(2π*k2*psi/(2π)) : sin(2π*k2*psi/(2π))
                vals[j, i] = scale * f1 * f2
            end
        end
        
        push!(p_basis, heatmap(vals, title=title_str, c=:RdBu, clim=(-maximum(abs.(vals)), maximum(abs.(vals)))))
    end
    
    # Plot a few input/output pairs
    p_data = []
    data_indices = [1, 50, 100]
    
    for idx in data_indices
        x_vals = reshape(xdata[:, idx], n_grid, n_grid)
        y_vals = reshape(ydata[:, idx], n_grid, n_grid)
        
        push!(p_data, heatmap(x_vals, title="Input $idx", c=:viridis))
        push!(p_data, heatmap(y_vals, title="Output $idx (T applied)", c=:viridis))
    end
    
    # Combine plots
    p_combined = plot(p_basis..., p_data..., layout=(3, 3), size=(1200, 1200))
    
    savefig(p_combined, joinpath(output_dir, "data_diagnostic.png"))
    @info "Diagnostic plot saved to $(joinpath(output_dir, "data_diagnostic.png"))"
end

function main(args)
    n_functions, n_grid, max_order, delta, output_dir, seed = parse_args(args)
    
    # Create output directory
    mkpath(output_dir)
    
    # Generate data
    xdata, ydata, grid, gridmat, basis_2d = generate_data(n_functions, n_grid, max_order, delta, seed)
    
    # Print statistics
    @info "\nData Statistics:"
    @info "  Input range: [$(minimum(xdata)), $(maximum(xdata))]"
    @info "  Output range: [$(minimum(ydata)), $(maximum(ydata))]"
    @info "  Input mean/std: $(mean(xdata)) / $(std(xdata))"
    @info "  Output mean/std: $(mean(ydata)) / $(std(ydata))"
    
    # Save data
    output_file = joinpath(output_dir, "transfer_operator_data.jld2")
    @info "\nSaving data to $output_file..."
    
    JLD2.jldsave(output_file;
        xdata = xdata,
        ydata = ydata,
        grid = grid,
        gridmat = gridmat,
        n_functions = n_functions,
        n_grid = n_grid,
        max_order = max_order,
        delta = delta,
        n_basis_functions = length(basis_2d)
    )
    
    # Generate diagnostic plots
    generate_diagnostic_plots(xdata, ydata, basis_2d, n_grid, output_dir)
    
    @info "\n✓ Data generation complete!"
    @info "  Total functions: $n_functions"
    @info "  Grid size: $(n_grid)×$(n_grid)"
    @info "  Data shape: $(size(xdata))"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end