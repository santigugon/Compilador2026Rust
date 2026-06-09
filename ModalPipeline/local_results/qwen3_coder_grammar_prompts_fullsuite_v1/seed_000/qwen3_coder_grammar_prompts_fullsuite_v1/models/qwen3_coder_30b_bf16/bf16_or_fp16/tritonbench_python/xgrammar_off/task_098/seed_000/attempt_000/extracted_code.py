import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr, other_ptr, output_ptr,
    alpha, approximate,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_block = tl.load(input_ptr + offset + tl.arange(0, BLOCK_SIZE))
    other_block = tl.load(other_ptr + offset + tl.arange(0, BLOCK_SIZE))
    
    # Subtract other scaled by alpha from input
    x = input_block - alpha * other_block
    
    # Apply GELU
    if approximate == "none":
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = x / sqrt_2
        erf_val = tl.math.erf(erf_arg)
        gelu_val = x * 0.5 * (1.0 + erf_val)
    else:
        # Approximate GELU using tanh
        # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = 0.7978845608028654
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_val = tl.math.tanh(tanh_arg)
        gelu_val = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(output_ptr + offset + tl.arange(0, BLOCK_SIZE), gelu_val)

def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and other have the same dtype
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other = other.to(input.dtype)
    
    # Ensure other has the same shape as input or is broadcastable
    if other.numel() == 1:
        other = other.expand_as(input)
    elif other.shape != input.shape:
        raise ValueError("other tensor must be broadcastable to input tensor shape")
    
    # Calculate total elements
    N = input.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gelu_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        alpha=alpha,
        approximate=approximate,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
