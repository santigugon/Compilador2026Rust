import torch
import triton
import triton.language as tl

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input1.dtype, device=input1.device)
    
    # Ensure other is broadcastable with input1
    if other.dim() == 0:
        other = other.expand_as(input1)
    
    # Element-wise multiplication and addition
    x = input1 * input2 + other
    
    # Apply log-softmax
    if dim == -1:
        dim = x.dim() - 1
    
    # Use PyTorch's log_softmax for numerical stability
    x = torch.log_softmax(x, dim=dim)
    
    # Apply dropout
    if training:
        mask = torch.rand_like(x) > p
        x = x * mask / (1.0 - p)
    
    # Batch matrix multiplication
    out_tensor = torch.bmm(x, mat2)
    
    # Return result
    return out_tensor
##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_mul_add_logsoftmax_dropout_bmm():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2)

    # Test case 2: Different dropout probability
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.3)

    # Test case 3: In-place operation
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, inplace=True)

    # Test case 4: Different dimension for log-softmax
    input1 = torch.rand(2, 3, 4, device='cuda')
    input2 = torch.rand(2, 3, 4, device='cuda')
    other = torch.rand(2, 3, 4, device='cuda')
    mat2 = torch.rand(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, dim=1)

    return results

test_results = test_fused_mul_add_logsoftmax_dropout_bmm()
