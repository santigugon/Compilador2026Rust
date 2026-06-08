import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(x_ptr, weight_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    # Compute RMS
    mean_square = tl.sum(x * x, axis=0) / n
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    # Normalize and scale
    y = x * inv_rms * weight
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if approximate == 'tanh':
        # GELU with tanh approximation
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    else:
        # Standard GELU
        y = 0.5 * x * (1.0 + tl.erf(0.70710678118654757 * x))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if training:
        # Generate random mask
        # Note: In practice, we'd use a proper random number generator
        # For simplicity, we'll use a deterministic approach
        # This is a placeholder - real implementation would use proper RNG
        rand_val = tl.rand()  # This is not valid in Triton, we need to use a different approach
        # For now, we'll simulate dropout with a simple approach
        # In a real implementation, we'd need to pass in a proper random mask
        # This is a simplified version for demonstration
        y = x * (rand_val > p) / (1.0 - p)
    else:
        y = x
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Validate inputs
    if input1.dim() != 3 or input2.dim() != 3:
        raise ValueError("Both input1 and input2 must be 3D tensors")
    
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    
    if B != B2 or M != M2:
        raise ValueError("Batch size and middle dimension must match between input1 and input2")
    
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    if len(normalized_shape) != 1 or normalized_shape[0] != P:
        raise ValueError("normalized_shape must be [P] where P is the last dimension of input2")
    
    # Compute batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)
    
    # Initialize output tensor
    if out is None:
        out = torch.empty_like(bmm_out)
    else:
        if out.shape != bmm_out.shape:
            raise ValueError("out tensor must have the same shape as the bmm output")
    
    # RMS normalization
    # We'll compute RMS normalization on the last dimension (P)
    n = P
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create weight tensor for RMS normalization
    weight = torch.ones(P, dtype=input1.dtype, device=input1.device)
    
    # Apply RMS normalization
    rms_out = torch.empty_like(bmm_out)
    _rms_norm_kernel[grid](bmm_out, weight, rms_out, n, eps, BLOCK=block)
    
    # Apply GELU activation
    gelu_out = torch.empty_like(rms_out)
    approximate_val = 1 if approximate == 'tanh' else 0
    _gelu_kernel[grid](rms_out, gelu_out, n, approximate_val, BLOCK=block)
    
    # Apply dropout
    dropout_out = torch.empty_like(gelu_out)
    if training:
        # For simplicity, we'll use a basic dropout approach
        # In a real implementation, we'd need proper random number generation
        dropout_mask = torch.rand_like(gelu_out) > dropout_p
        dropout_out = gelu_out * dropout_mask / (1.0 - dropout_p)
    else:
        dropout_out = gelu_out
    
    # Copy result to output tensor
    out.copy_(dropout_out)
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-05, training=True, approximate='none', *, out=None):
#     z1 = torch.bmm(input1, input2)
#     rms_norm = F.rms_norm(z1, normalized_shape=(normalized_shape,), eps=eps)
#     gelu_out = F.gelu(rms_norm, approximate=approximate)
#     output = F.dropout(gelu_out, p=dropout_p, training=training)
#     if out is not None:
#         out.copy_(output)
#         return out
#     return output

def test_fused_bmm_rmsnorm_gelu_dropout():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5)
    
    # Test case 2: Different dropout probability
    results["test_case_2"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, dropout_p=0.2)
    
    # Test case 3: Non-training mode
    results["test_case_3"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, training=False)
    
    # Test case 4: Different approximation method for GELU
    results["test_case_4"] = fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape=5, approximate='tanh')
    
    return results

test_results = test_fused_bmm_rmsnorm_gelu_dropout()
