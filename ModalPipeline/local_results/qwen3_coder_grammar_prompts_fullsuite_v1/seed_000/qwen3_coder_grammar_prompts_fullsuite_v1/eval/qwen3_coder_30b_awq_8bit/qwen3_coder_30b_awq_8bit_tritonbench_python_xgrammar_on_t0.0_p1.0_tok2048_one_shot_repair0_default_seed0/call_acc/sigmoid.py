import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)

def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    sigmoid_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
##################################################################################################################################################



import torch
import torch.special

def test_sigmoid():
    results = {}

    # Test case 1: Simple tensor on GPU
    input_tensor_1 = torch.tensor([0.0, 1.0, -1.0], device='cuda')
    results["test_case_1"] = sigmoid(input_tensor_1)

    # Test case 2: Larger tensor with positive and negative values on GPU
    input_tensor_2 = torch.tensor([0.5, -0.5, 2.0, -2.0], device='cuda')
    results["test_case_2"] = sigmoid(input_tensor_2)

    # Test case 3: 2D tensor on GPU
    input_tensor_3 = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], device='cuda')
    results["test_case_3"] = sigmoid(input_tensor_3)

    # Test case 4: Tensor with all zeros on GPU
    input_tensor_4 = torch.zeros(3, 3, device='cuda')
    results["test_case_4"] = sigmoid(input_tensor_4)

    return results

test_results = test_sigmoid()
