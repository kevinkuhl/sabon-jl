module DeviceUtils

using CUDA
using Metal  # For MPS on Mac

export get_device, setup_device

"""
    get_device(device_str::String)

Get appropriate device based on string specification.
Options: "cuda", "mps", "cpu", "auto"
"""
function get_device(device_str::String)
    device_str = lowercase(device_str)
    
    if device_str == "auto"
        if CUDA.functional()
            return :cuda
        elseif Metal.functional()
            return :mps
        else
            return :cpu
        end
    elseif device_str == "cuda"
        if !CUDA.functional()
            @warn "CUDA not available, falling back to CPU"
            return :cpu
        end
        return :cuda
    elseif device_str == "mps"
        if !Metal.functional()
            @warn "MPS not available, falling back to CPU"
            return :cpu
        end
        return :mps
    else
        return :cpu
    end
end

"""
    setup_device(device_symbol::Symbol)

Returns gpu/cpu function based on device
"""
function setup_device(device_symbol::Symbol)
    if device_symbol == :cuda
        return CUDA.cu, CUDA.CuArray, CUDA
    elseif device_symbol == :mps
        return Metal.mtl, Metal.MtlArray, Metal
    else
        return identity, Array, nothing
    end
end

end  # module