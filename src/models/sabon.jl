module SABONModel

using Flux

export SABON, build_sabon

struct SABON{T1<:Chain, T2<:Chain}
    linear_map::T1
    basis::T2
end

"""
Forward pass for SABON model.
Works with any input dimension.
"""
function (m::SABON)(f, grid_matrix)
    t = m.basis(grid_matrix)
    β = m.linear_map(t * f / Float32(size(grid_matrix, 2)))
    return t' * β
end

Flux.@layer SABON

"""
    build_sabon(; kwargs...)

Build a SABON model with specified architecture.

# Arguments
- `input_dim::Int`: Dimension of input coordinates
- `n_basis::Int`: Number of basis functions
- `hidden_size::Int`: Size of hidden layers
- `n_layers::Int`: Number of hidden layers
- `activation`: Activation function
- `use_bias::Bool`: Whether to use bias in linear map
"""
function build_sabon(;
    input_dim::Int,
    n_basis::Int,
    hidden_size::Int,
    n_layers::Int,
    activation=relu,
    use_bias::Bool=false
)
    # Linear map (always n_basis -> n_basis)
    linear_map = Chain(Dense(n_basis, n_basis, identity; bias=use_bias))
    
    # Basis network (input_dim -> n_basis)
    layers = []
    
    # First layer
    push!(layers, Dense(input_dim, hidden_size, activation))
    
    # Hidden layers
    for _ in 1:(n_layers-1)
        push!(layers, Dense(hidden_size, hidden_size, activation))
    end
    
    # Output layer
    push!(layers, Dense(hidden_size, n_basis, identity))
    
    basis = Chain(layers...)
    
    return SABON(linear_map, basis)
end

end  # module