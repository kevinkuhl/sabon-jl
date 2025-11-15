module Regularization

using LinearAlgebra
using Flux
using Statistics

export invariance_loss, aec_loss, sparsity_loss

"""
    invariance_loss(basis, A_hat, grid_size)

Compute invariance loss: ||Φ(A_hat @ Φ^T) - A_hat @ Φ^T||_F / ||A_hat @ Φ^T||_F
This enforces that the learned basis spans an invariant subspace.
"""
function invariance_loss(basis, A_hat, grid_size::Int)
    # basis is n_basis × (grid_size^2)
    # A_hat is (grid_size^2) × (grid_size^2)
    
    Lt = A_hat * basis'  # (grid_size^2) × n_basis
    
    # Coefficients: project Lt onto basis
    trap_weight = 1.0f0 / Float32(grid_size^2)
    coeffs = (basis * trap_weight) * Lt  # n_basis × n_basis
    
    # Reconstruction
    recon = coeffs' * basis  # (grid_size^2) × n_basis -> n_basis × (grid_size^2)
    
    # Relative error
    error_norm = norm(recon' - Lt)
    Lt_norm = norm(Lt)

    return error_norm / (Lt_norm + 1f-12)
end

"""
    aec_loss(reconstructed, original)

Autoencoder reconstruction loss (MSE).
"""
function aec_loss(reconstructed, original)
    return Flux.Losses.mse(reconstructed, original)
end

"""
    sparsity_loss(basis)

L1 sparsity regularization on basis functions.
"""
function sparsity_loss(basis)
    return mean(sum(abs, basis, dims=2))
end

end  # module