import torch
import triton
import triton.language as tl

@triton.jit
def fused_hardshrink_dropout_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    p, 
    lambd, 
    training, 
    inplace,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptrs = input_ptr + offsets
    output_ptrs = output_ptr + offsets
    
    input_vals = tl.load(input_ptrs, mask=mask)
    
    if training:
        # Generate random numbers for dropout
        rand_vals = tl.random.random([BLOCK_SIZE], seed=pid)
        dropout_mask = rand_vals > p
        input_vals = tl.where(dropout_mask, input_vals, tl.zeros_like(input_vals))
    
    # Apply hard shrinkage
    # Hard shrinkage: f(x) = x if |x| > lambd, else 0
    shrinkage_mask = tl.abs(input_vals) > lambd
    output_vals = tl.where(shrinkage_mask, input_vals, tl.zeros_like(input_vals))
    
    if inplace:
        tl.store(output_ptrs, output_vals, mask=mask)
    else:
        tl.store(output_ptrs, output_vals, mask=mask)

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor:
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    if input.numel() == 0:
        return output
    
    # Ensure input is contiguous
    input = input.contiguous()
    output = output.contiguous()
    
    # Launch kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_hardshrink_dropout_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        p=p,
        lambd=lambd,
        training=training,
        inplace=inplace,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
