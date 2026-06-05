import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y + offset, y, mask=mask)

def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    N = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    sigmoid_kernel[grid](input, out, N, BLOCK_SIZE=BLOCK_SIZE)
    
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
