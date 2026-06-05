import torch
import triton
import triton.language as tl

@triton.jit
def _elu_linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n: tl.constexpr, k: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Linear transformation: y = x @ w.T + b
    # For simplicity, we assume input is 1D and weight is 1D for this kernel
    # In practice, this would need to handle proper matrix multiplication
    # Here we do a simplified version for demonstration
    w = tl.load(w_ptr, mask=tl.arange(0, k) < k, other=0.0)
    if b_ptr is not None:
        b = tl.load(b_ptr, mask=tl.arange(0, k) < k, other=0.0)
    else:
        b = tl.zeros((k,), dtype=tl.float32)
    
    # Compute linear transformation
    linear_result = tl.sum(x[:, None] * w[None, :], axis=0) + b
    
    # Apply ELU
    elu_result = tl.where(linear_result > 0, linear_result, alpha * (tl.exp(linear_result) - 1.0))
    
    tl.store(out_ptr + offsets, elu_result, mask=mask)

@triton.jit
def _elu_kernel(x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x > 0, x, alpha * (tl.exp(x) - 1.0))
    tl.store(out_ptr + offsets, y, mask=mask)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Handle the case where input is 1D and weight is 1D for simplicity
    # In practice, this would need to handle proper matrix multiplication
    
    # Compute linear transformation
    if bias is not None:
        linear_output = torch.nn.functional.linear(input, weight, bias)
    else:
        linear_output = torch.nn.functional.linear(input, weight)
    
    # Apply ELU
    if inplace:
        # Apply ELU in-place
        out = linear_output
        n = linear_output.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _elu_kernel[grid](linear_output, out, n, alpha, BLOCK=block)
    else:
        # Apply ELU out-of-place
        out = torch.empty_like(linear_output)
        n = linear_output.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _elu_kernel[grid](linear_output, out, n, alpha, BLOCK=block)
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
#     output = F.linear(input, weight, bias)
#     return F.elu(output, alpha=alpha, inplace=inplace)

def test_elu_linear():
    results = {}

    # Test case 1: Basic test with bias, alpha=1.0, inplace=False
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = elu_linear(input1, weight1, bias1)

    # Test case 2: Without bias, alpha=1.0, inplace=False
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    results["test_case_2"] = elu_linear(input2, weight2)

    # Test case 3: With bias, alpha=0.5, inplace=False
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight3 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = elu_linear(input3, weight3, bias3, alpha=0.5)

    # Test case 4: With bias, alpha=1.0, inplace=True
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device='cuda')
    bias4 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_4"] = elu_linear(input4, weight4, bias4, inplace=True)

    return results

test_results = test_elu_linear()
