import torch
import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(
    input_ptr, mask_ptr, other_ptr, output_ptr,
    input_size,
    alpha,
    other_is_scalar,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.load(mask_ptr + offsets, mask=offsets < input_size)
    
    input_vals = tl.load(input_ptr + offsets, mask=offsets < input_size)
    
    if other_is_scalar:
        other_vals = alpha * other_ptr[0]
    else:
        other_vals = tl.load(other_ptr + offsets, mask=offsets < input_size)
        other_vals = other_vals * alpha
    
    selected_vals = tl.where(mask, input_vals + other_vals, 0.0)
    
    # GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x = selected_vals
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    gelu_val = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(output_ptr + offsets, gelu_val, mask=offsets < input_size)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    if input.dtype != torch.float32:
        input = input.float()
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if other.dtype != torch.float32:
        other = other.float()
    
    input_size = input.numel()
    BLOCK_SIZE = 1024
    num_blocks = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    other_is_scalar = other.dim() == 0 or (other.numel() == 1)
    
    fused_masked_select_add_gelu_kernel[
        num_blocks
    ](
        input_ptr=input.data_ptr(),
        mask_ptr=mask.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        input_size=input_size,
        alpha=alpha,
        other_is_scalar=other_is_scalar,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F


def test_fused_masked_select_add_gelu():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask1 = torch.tensor([True, False, True, False], device='cuda')
    other1 = 1.0
    results["test_case_1"] = fused_masked_select_add_gelu(input1, mask1, other1)
    
    # Test case 2: Test with alpha parameter
    input2 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask2 = torch.tensor([True, True, False, False], device='cuda')
    other2 = 2.0
    results["test_case_2"] = fused_masked_select_add_gelu(input2, mask2, other2, alpha=0.5)
    
    # Test case 3: Test with approximate='tanh'
    input3 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask3 = torch.tensor([False, True, True, False], device='cuda')
    other3 = 1.0
    results["test_case_3"] = fused_masked_select_add_gelu(input3, mask3, other3, approximate='tanh')
    
    # Test case 4: Test with out parameter
    input4 = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    mask4 = torch.tensor([True, False, True, True], device='cuda')
    other4 = 1.0
    out4 = torch.empty(3, device='cuda')
    results["test_case_4"] = fused_masked_select_add_gelu(input4, mask4, other4, out=out4)
    
    return results

test_results = test_fused_masked_select_add_gelu()
