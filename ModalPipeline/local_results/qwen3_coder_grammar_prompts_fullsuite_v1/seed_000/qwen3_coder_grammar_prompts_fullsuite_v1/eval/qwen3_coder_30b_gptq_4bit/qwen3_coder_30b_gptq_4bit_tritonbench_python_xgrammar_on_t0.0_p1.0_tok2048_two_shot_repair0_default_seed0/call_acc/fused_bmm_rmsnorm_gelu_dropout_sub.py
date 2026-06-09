import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Ensure other tensor is broadcastable to (B, N, P)
    if other.shape != (B, N, P):
        other = other.expand(B, N, P)
    
    # Initialize output tensor
    if out is None:
        out = torch.empty_like(input1)
    else:
        assert out.shape == (B, N, P), "Output tensor shape must match (B, N, P)"
    
    # Perform batch matrix multiplication
    output = torch.bmm(input1, input2)
    
    # Apply RMS normalization
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Compute RMS normalization
    mean_square = output.pow(2).mean(dim=-1, keepdim=True)
    rms = (mean_square + eps).sqrt()
    output = output / rms
    
    # Apply GELU activation
    if approximate == 'none':
        output = output * 0.5 * (1 + torch.erf(output / (2 ** 0.5)))
    elif approximate == 'tanh':
        output = output * 0.5 * (1 + torch.tanh(0.7978845608 * (output + 0.044715 * (output ** 3))))
    else:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Apply dropout
    if training and dropout_p > 0:
        dropout_mask = (torch.rand_like(output) > dropout_p)
        output = output * dropout_mask / (1.0 - dropout_p)
    
    # Subtract other tensor
    output = output - other
    
    return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_bmm_rmsnorm_gelu_dropout_sub():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    other = torch.randn(2, 3, 5, device='cuda')
    normalized_shape = 5
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape)

    # Test case 2: Test with different dropout probability
    dropout_p = 0.3
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=dropout_p)

    # Test case 3: Test with training set to False
    training = False
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, training=training)

    # Test case 4: Test with approximate GELU
    approximate = 'tanh'
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, approximate=approximate)

    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout_sub()
