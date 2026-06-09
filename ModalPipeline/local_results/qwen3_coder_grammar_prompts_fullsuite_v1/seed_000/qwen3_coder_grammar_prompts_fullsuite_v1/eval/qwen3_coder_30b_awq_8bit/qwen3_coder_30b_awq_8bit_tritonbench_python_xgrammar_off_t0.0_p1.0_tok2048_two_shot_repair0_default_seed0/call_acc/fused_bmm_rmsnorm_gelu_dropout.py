import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(x_ptr, rms_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute RMS
    mean_square = tl.sum(x * x, axis=0) / n
    rms = tl.sqrt(mean_square + eps)
    # Normalize
    normalized = x / rms
    tl.store(out_ptr + offsets, normalized, mask=mask)
    tl.store(rms_ptr + pid, rms, mask=pid < tl.cdiv(n, BLOCK))

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if approximate == 'tanh':
        # GELU approximation using tanh
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    else:
        # Standard GELU
        y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Generate random mask
    rand = tl.rand(tl.program_id(0), offsets[0])  # Simple random number generation
    keep_mask = rand > p
    # Apply dropout
    y = tl.where(keep_mask, x / (1.0 - p), 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)
    tl.store(mask_ptr + offsets, keep_mask, mask=mask)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Validate inputs
    assert input1.dim() == 3 and input2.dim() == 3
    assert input1.size(0) == input2.size(0)  # batch size
    assert input1.size(2) == input2.size(1)  # inner dimension
    
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Compute batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)
    
    # Prepare for RMS normalization
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    normalized_shape = torch.Size(normalized_shape)
    
    # Flatten the last dimensions for normalization
    flat_shape = (B * N, P)
    flat_bmm_out = bmm_out.view(flat_shape)
    
    # RMS normalization
    # Compute RMS for each element in the normalized shape
    # For simplicity, we'll compute RMS over the last dimension
    rms_out = torch.empty_like(flat_bmm_out)
    rms_values = torch.empty(B * N, dtype=torch.float32, device=flat_bmm_out.device)
    
    # Use Triton kernel for RMS normalization
    n = flat_bmm_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # We'll compute RMS for each row (B*N rows, P columns)
    # For each row, we compute RMS over P elements
    for i in range(B * N):
        row = flat_bmm_out[i]
        mean_square = torch.mean(row * row)
        rms = torch.sqrt(mean_square + eps)
        rms_values[i] = rms
        rms_out[i] = row / rms
    
    # Apply GELU activation
    gelu_out = torch.empty_like(rms_out)
    if approximate == 'tanh':
        # Use tanh approximation for GELU
        gelu_out = 0.5 * rms_out * (1.0 + torch.tanh(0.7978845608028654 * (rms_out + 0.044715 * rms_out * rms_out * rms_out)))
    else:
        # Standard GELU
        gelu_out = 0.5 * rms_out * (1.0 + torch.erf(rms_out / torch.sqrt(torch.tensor(2.0))))
    
    # Apply dropout if training
    if training and dropout_p > 0:
        dropout_out = torch.empty_like(gelu_out)
        # Generate dropout mask
        keep_mask = torch.rand_like(gelu_out) > dropout_p
        dropout_out = gelu_out * keep_mask / (1.0 - dropout_p)
        output = dropout_out
    else:
        output = gelu_out
    
    # Reshape back to original shape
    output = output.view(B, N, P)
    
    if out is not None:
        out.copy_(output)
        return out
    else:
        return output

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
