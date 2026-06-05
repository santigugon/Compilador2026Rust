import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr, other_ptr, output_ptr,
    alpha, approximate,
    input_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    
    # Subtract other scaled by alpha from input
    x = input - alpha * other
    
    # Apply GELU
    if approximate == "tanh":
        # Approximate GELU using tanh
        gelu_x = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    else:
        # Exact GELU
        gelu_x = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    
    tl.store(output_ptr + offsets, gelu_x, mask=mask)

def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and other have the same dtype
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other = other.to(input.dtype)
    
    # Flatten tensors to 1D for kernel processing
    input_flat = input.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    # Ensure tensors are contiguous
    input_flat = input_flat.contiguous()
    other_flat = other_flat.contiguous()
    out_flat = out_flat.contiguous()
    
    # Determine the size
    size = input_flat.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    # Determine approximate mode
    approx_mode = "tanh" if approximate == "tanh" else "none"
    
    gelu_kernel[grid](
        input_flat,
        other_flat,
        out_flat,
        alpha,
        approx_mode,
        size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.reshape(input.shape)
