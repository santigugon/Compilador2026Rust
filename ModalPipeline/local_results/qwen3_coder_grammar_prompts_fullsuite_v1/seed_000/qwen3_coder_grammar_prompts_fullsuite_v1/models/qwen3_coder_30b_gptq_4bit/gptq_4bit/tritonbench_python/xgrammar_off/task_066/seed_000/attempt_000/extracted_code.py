import torch
import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, output_ptr,
    n_elements, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tl.load(mask_ptr + offsets, mask=offsets < n_elements, other=0)
    input = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    other = tl.load(other_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Select elements based on mask
    selected = tl.where(mask, input, 0.0)
    
    # Add other tensor scaled by alpha
    result = selected + alpha * other
    
    # Apply GELU activation
    result = result * 0.5 * (1.0 + tl.erf(result / tl.sqrt(2.0)))
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    assert input.shape == mask.shape, "input and mask must have the same shape"
    assert input.shape == other.shape, "input and other must have the same shape"
    
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_masked_select_add_gelu_kernel[grid](
        input_ptr=input.data_ptr(),
        mask_ptr=mask.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
