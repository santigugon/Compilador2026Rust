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
    square = x * x
    mean = tl.sum(square) / n
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    tl.store(rms_ptr + pid, inv_rms)
    # Normalize and store
    normalized = x * inv_rms
    tl.store(out_ptr + offsets, normalized, mask=mask)

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if approximate == 'none':
        # Standard GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        x_over_sqrt2 = x / sqrt_2
        erf_x = 2.0 * tl.sigmoid(1.4142135623730951 * x) - 1.0
        y = 0.5 * x * (1.0 + erf_x)
    else:  # approximate == 'tanh'
        # Approximate GELU using tanh
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if training:
        # Generate random mask
        seed = tl.load(mask_ptr + 0)
        rand_val = tl.rand(seed, pid * BLOCK + tl.arange(0, BLOCK))  # Use program_id as seed offset
        keep_mask = rand_val > p
        y = tl.where(keep_mask, x / (1.0 - p), 0.0)
    else:
        y = x
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Validate inputs
    assert input1.dim() == 3 and input2.dim() == 3
    assert input1.size(0) == input2.size(0)  # batch size
    assert input1.size(2) == input2.size(1)  # inner dimension
    assert isinstance(normalized_shape, (int, list, torch.Size))
    
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Compute batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)  # Shape: (B, N, P)
    
    # Prepare for RMS normalization
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    normalized_shape = list(normalized_shape)
    assert len(normalized_shape) == 1 and normalized_shape[0] == P
    
    # Flatten for normalization
    flat_shape = (B * N, P)
    bmm_flat = bmm_out.view(flat_shape)
    
    # RMS normalization
    out_shape = bmm_out.shape
    rms_out = torch.empty_like(bmm_flat)
    rms_values = torch.empty(B * N, dtype=torch.float32, device=bmm_flat.device)
    
    # Compute RMS normalization
    n = P
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Compute RMS values
    for i in range(B * N):
        start = i * n
        end = start + n
        x = bmm_flat[i]
        square = x * x
        mean = torch.sum(square) / n
        inv_rms = 1.0 / torch.sqrt(mean + eps)
        rms_values[i] = inv_rms
        rms_out[i] = x * inv_rms
    
    # Apply GELU
    gelu_out = torch.empty_like(rms_out)
    n = P
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Apply GELU activation
    approximate_val = 'none' if approximate == 'none' else 'tanh'
    for i in range(B * N):
        x = rms_out[i]
        if approximate_val == 'none':
            # Standard GELU
            sqrt_2 = 1.4142135623730951
            x_over_sqrt2 = x / sqrt_2
            erf_x = 2.0 * torch.sigmoid(1.4142135623730951 * x) - 1.0
            y = 0.5 * x * (1.0 + erf_x)
        else:
            # Approximate GELU using tanh
            y = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        gelu_out[i] = y
    
    # Apply dropout
    dropout_out = torch.empty_like(gelu_out)
    if training and dropout_p > 0:
        # Generate random mask
        torch.manual_seed(42)  # For reproducibility
        rand_mask = torch.rand_like(gelu_out) > dropout_p
        dropout_out = gelu_out * rand_mask / (1.0 - dropout_p)
    else:
        dropout_out = gelu_out
    
    # Reshape back to original shape
    result = dropout_out.view(out_shape)
    
    if out is not None:
        out.copy_(result)
        return out
    return result
