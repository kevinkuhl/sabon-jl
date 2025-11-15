module Metrics

using LinearAlgebra
using Statistics

export compute_principal_angles, compute_spectrum

"""
    compute_principal_angles(T1, T2)

Compute principal angles between two subspaces represented by column matrices T1 and T2.
Returns angles in degrees.
"""
function compute_principal_angles(T1::AbstractMatrix, T2::AbstractMatrix)
    # Orthonormalize both subspaces
    Q1 = Matrix(qr(T1).Q)
    Q2 = Matrix(qr(T2).Q)
    
    # Compute SVD of Q1' * Q2
    S = svd(Q1' * Q2)
    
    # Clamp singular values to [0, 1] to avoid numerical issues
    cos_angles = clamp.(S.S, 0.0, 1.0)
    
    # Convert to angles in degrees
    angles = acos.(cos_angles) .* (180 / π)
    
    return angles
end

"""
    compute_cat_map_on_grid(grid, delta=0.01)

Apply the (perturbed) cat map to grid points.
Returns the permutation and transformed grid.
"""
function compute_cat_map_on_grid(grid, delta=0.01)
    cat_map_point(x, y) = begin
        θ = angle(complex(x...))
        ψ = angle(complex(y...))
        θnew = mod(2θ + ψ + 2delta * cos(2π * θ), 2π)
        ψnew = mod(θ + ψ + delta * sin(4π * ψ + 1), 2π)
        return [[cos(θnew), sin(θnew)], [cos(ψnew), sin(ψnew)]]
    end
    
    # Apply to all grid points
    cat_on_grid = [cat_map_point(x[1], x[2]) for x in grid]
    
    # Create matrix representation
    cmat = hcat([vcat(cat_on_grid[i]...) for i in eachindex(cat_on_grid)]...)
    
    return cmat
end

"""
    compute_principal_angles_with_cat_map(basis, grid, gridmat, delta=0.01)

Compute principal angles between span(basis) and its image under the cat map.
"""
function compute_principal_angles_with_cat_map(basis, grid, gridmat, delta=0.01)
    # T1: basis evaluated at grid points (transpose for column format)
    T1 = basis'
    
    # Get cat map transformation
    cmat = compute_cat_map_on_grid(grid, delta)
    
    # Find permutation: for each grid point, find where cat map sends it
    difference = norm(gridmat[:,1] - gridmat[:,2])
    p = [findfirst(x -> norm(x - col) < difference, eachcol(cmat)) for col in eachcol(gridmat)]
    
    # T2: basis evaluated at transformed points
    T2 = T1[p, :]
    
    # Compute principal angles
    return compute_principal_angles(T1, T2)
end

"""
    compute_spectrum(model, gridmat)

Compute eigenvalues of the learned Koopman/Transfer operator.
"""
function compute_spectrum(linear_map, basis_values)
    # Compute Gram matrix
    G = basis_values * basis_values' / size(basis_values, 2)
    
    # Get operator matrix
    A = linear_map(G)
    
    # Compute eigenvalues
    eig_result = eigen(A)
    
    return eig_result.values, eig_result.vectors
end

end  # module