import torch
import triton
import triton.language as tl

@triton.jit
def _selu_kernel(x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    exp_x = tl.exp(x)
    selu_val = scale * (tl.maximum(0, x) + tl.minimum(0, alpha * (exp_x - 1.0)))
    tl.store(out_ptr + offsets, selu_val, mask=mask)

def selu(input, inplace=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _selu_kernel[grid](input, out, n, alpha, scale, BLOCK=block)
    
    return out

##################################################################################################################################################



def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    return results

test_results = test_selu()
