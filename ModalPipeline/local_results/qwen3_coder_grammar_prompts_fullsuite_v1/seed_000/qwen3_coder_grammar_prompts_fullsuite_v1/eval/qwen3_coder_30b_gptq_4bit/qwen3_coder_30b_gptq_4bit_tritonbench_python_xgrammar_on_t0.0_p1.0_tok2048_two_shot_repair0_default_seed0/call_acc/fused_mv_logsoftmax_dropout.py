import torch
import triton
import triton.language as tl

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if not inplace:
            out = out.clone()

    # Handle scalar vector case
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec, dtype=input.dtype, device=input.device)

    # Ensure vec is a column vector
    if vec.dim() == 1:
        vec = vec.unsqueeze(1)

    # Get dimensions
    m, n = input.shape
    k = vec.shape[0]
    
    # Check if matrix-vector multiplication is valid
    if n != k:
        raise ValueError("Matrix and vector dimensions do not match for multiplication")

    # Perform matrix-vector multiplication
    mv_out = torch.mm(input, vec)
    
    # Apply log-softmax along specified dimension
    if dim == 0:
        # Apply log-softmax to each row
        log_softmax_out = torch.log_softmax(mv_out, dim=0)
    else:
        # Apply log-softmax to each column
        log_softmax_out = torch.log_softmax(mv_out, dim=1)
    
    # Apply dropout
    if training:
        # Create dropout mask
        dropout_mask = torch.rand_like(log_softmax_out) > p
        # Apply dropout
        out = log_softmax_out * dropout_mask / (1.0 - p)
    else:
        out = log_softmax_out
    
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_mv_logsoftmax_dropout():
    results = {}

    # Test case 1: Basic functionality
    input1 = torch.randn(3, 4, device='cuda')
    vec1 = torch.randn(4, device='cuda')
    results["test_case_1"] = fused_mv_logsoftmax_dropout(input1, vec1)

    # Test case 2: Dropout with p=0.2
    input2 = torch.randn(3, 4, device='cuda')
    vec2 = torch.randn(4, device='cuda')
    results["test_case_2"] = fused_mv_logsoftmax_dropout(input2, vec2, p=0.2)

    # Test case 3: Dropout in evaluation mode (training=False)
    input3 = torch.randn(3, 4, device='cuda')
    vec3 = torch.randn(4, device='cuda')
    results["test_case_3"] = fused_mv_logsoftmax_dropout(input3, vec3, training=False)

    # Test case 4: Inplace operation
    input4 = torch.randn(3, 4, device='cuda')
    vec4 = torch.randn(4, device='cuda')
    results["test_case_4"] = fused_mv_logsoftmax_dropout(input4, vec4, inplace=True)

    return results

test_results = test_fused_mv_logsoftmax_dropout()
