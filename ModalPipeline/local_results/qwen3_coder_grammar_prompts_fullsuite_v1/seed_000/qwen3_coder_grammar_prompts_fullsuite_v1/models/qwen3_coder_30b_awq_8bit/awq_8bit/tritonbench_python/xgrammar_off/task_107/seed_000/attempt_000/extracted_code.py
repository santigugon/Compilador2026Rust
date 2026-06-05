import torch
import triton
import triton.language as tl

@triton.jit
def gelu_min_kernel(
    input_ptr, 
    output_ptr, 
    min_indices_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # GELU computation
    if APPROXIMATE == 0:  # exact
        sqrt_2_over_pi = 0.7978845608028654
        cdf = 0.5 * (1.0 + tl.tanh(sqrt_2_over_pi * (input_vals + 0.044715 * input_vals * input_vals * input_vals)))
    else:  # tanh approximation
        sqrt_2 = 1.4142135623730951
        cdf = 0.5 * (1.0 + tl.tanh(sqrt_2 * input_vals * 0.5))
    
    gelu_vals = input_vals * cdf
    
    # Find minimum
    min_val = tl.min(gelu_vals)
    min_idx = tl.argmin(gelu_vals)
    
    tl.store(output_ptr + pid, min_val, mask=mask[0])
    tl.store(min_indices_ptr + pid, min_idx, mask=mask[0])

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    if dim is None:
        # Compute minimum over all elements
        input_flat = input.flatten()
        N = input_flat.numel()
        BLOCK_SIZE = 1024
        num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty(num_blocks, dtype=torch.float32, device=input.device)
        min_indices = torch.empty(num_blocks, dtype=torch.long, device=input.device)
        
        gelu_min_kernel[(num_blocks,)](
            input_flat.data_ptr(),
            output.data_ptr(),
            min_indices.data_ptr(),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            APPROXIMATE=0 if approximate == 'none' else 1
        )
        
        min_val = output.min()
        min_idx = min_indices[output.argmin()].item()
        
        if out is not None:
            out.copy_(min_val)
            return out, torch.tensor(min_idx)
        else:
            return min_val, torch.tensor(min_idx)
    else:
        # Compute minimum along specified dimension
        input_gelu = input.clone()
        if approximate == 'none':
            sqrt_2_over_pi = 0.7978845608028654
            input_gelu = 0.5 * input * (1.0 + torch.tanh(sqrt_2_over_pi * (input + 0.044715 * input * input * input)))
        else:
            sqrt_2 = 1.4142135623730951
            input_gelu = 0.5 * input * (1.0 + torch.tanh(sqrt_2 * input * 0.5))
        
        min_val, min_idx = torch.min(input_gelu, dim=dim, keepdim=keepdim)
        
        if out is not None:
            out.copy_(min_val)
            return out, min_idx
        else:
            return min_val, min_idx
