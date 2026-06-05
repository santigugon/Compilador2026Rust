import torch
import triton
import triton.language as tl

@triton.jit
def batch_norm_hardsigmoid_kernel(
    x_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    weight_ptr, 
    bias_ptr,
    output_ptr,
    N,
    C,
    training,
    momentum,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Get channel index for batch norm
    channel_idx = (offsets // (N // C)) % C
    
    # Load running stats
    mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < C)
    var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < C)
    
    # Batch normalization
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Apply learnable weight and bias
    if weight_ptr is not None and bias_ptr is not None:
        weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < C)
        bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < C)
        x_norm = x_norm * weight + bias
    
    # Hardsigmoid activation
    hardsigmoid = tl.where(x_norm >= 0, tl.where(x_norm <= 6, x_norm / 6 + 0.5, 1.0), 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, hardsigmoid, mask=mask)

def fused_hardsigmoid_batch_norm(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False
) -> torch.Tensor:
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    N = x.numel()
    C = running_mean.shape[0]  # Channel dimension
    
    # Prepare output tensor
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    num_warps = 4
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Handle optional weight and bias
    weight_ptr = weight.data_ptr() if weight is not None else None
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    batch_norm_hardsigmoid_kernel[grid](
        x_ptr=x.data_ptr(),
        running_mean_ptr=running_mean.data_ptr(),
        running_var_ptr=running_var.data_ptr(),
        weight_ptr=weight_ptr,
        bias_ptr=bias_ptr,
        output_ptr=output.data_ptr(),
        N=N,
        C=C,
        training=training,
        momentum=momentum,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output
