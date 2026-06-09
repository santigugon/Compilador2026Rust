import torch
import triton
import triton.language as tl

def fused_tile_exp(input, dims, *, out=None):
    # Prepend 1s to dims if needed to match input dimensions
    if len(dims) < input.ndim:
        dims = (1,) * (input.ndim - len(dims)) + dims
    
    # Calculate output shape
    output_shape = tuple(input.shape[i] * dims[i] for i in range(len(dims)))
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output tensor shape must match the expected tiled shape"
        assert out.dtype == input.dtype, "Output tensor dtype must match input dtype"
        assert out.device == input.device, "Output tensor device must match input device"
    
    # Handle scalar input case
    if input.numel() == 1:
        out.fill_(torch.exp(input.item()))
        return out
    
    # Flatten input and output for easier processing
    input_flat = input.flatten()
    out_flat = out.flatten()
    
    # Calculate total elements
    total_elements = out_flat.numel()
    
    # Define block size
    BLOCK = 256
    grid = (triton.cdiv(total_elements, BLOCK),)
    
    # Define kernel
    @triton.jit
    def _fused_tile_exp_kernel(input_ptr, out_ptr, total_elements: tl.constexpr, 
                               input_elements: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements
        
        # Calculate which input element each output element corresponds to
        input_indices = offsets % input_elements
        
        # Load input elements
        x = tl.load(input_ptr + input_indices, mask=mask, other=0.0)
        
        # Apply exponential function
        y = tl.exp(x)
        
        # Store result
        tl.store(out_ptr + offsets, y, mask=mask)
    
    # Launch kernel
    _fused_tile_exp_kernel[grid](input_flat, out_flat, total_elements, input_flat.numel(), BLOCK=BLOCK)
    
    return out