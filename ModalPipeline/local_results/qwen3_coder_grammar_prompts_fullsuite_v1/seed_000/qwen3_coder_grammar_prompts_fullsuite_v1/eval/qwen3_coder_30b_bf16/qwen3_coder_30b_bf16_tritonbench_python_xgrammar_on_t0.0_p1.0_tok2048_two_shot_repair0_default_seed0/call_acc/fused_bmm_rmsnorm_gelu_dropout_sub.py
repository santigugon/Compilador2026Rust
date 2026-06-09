import torch
import triton
import triton.language as tl

def _rmsnorm_kernel(x_ptr, rms_ptr, normalized_shape: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (x_ptr.shape[1] // BLOCK)
    seq_idx = pid % (x_ptr.shape[1] // BLOCK)
    
    # Load data
    offsets = batch_idx * x_ptr.stride(0) + seq_idx * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < x_ptr.shape[0], other=0.0)
    
    # Compute mean square
    mean_square = tl.sum(x * x, axis=0) / normalized_shape
    # Compute RMS
    rms = tl.sqrt(mean_square + eps)
    # Store RMS
    rms_offsets = batch_idx * rms_ptr.stride(0) + seq_idx * BLOCK
    tl.store(rms_ptr + rms_offsets, rms)
    
    # Normalize
    x_norm = x / rms
    # Store normalized values
    tl.store(x_ptr + offsets, x_norm, mask=offsets < x_ptr.shape[0])

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation
    y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, y, mask=mask)

def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, dropout_p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if training:
        # Generate random mask
        rand = tl.random.rand(0, 0)  # Placeholder for actual random generation
        # For simplicity, we'll use a fixed pattern
        # In practice, you'd use tl.random.rand() properly
        keep_prob = 1.0 - dropout_p
        # Simple mask generation
        mask_val = tl.where(rand < keep_prob, 1.0, 0.0)
        # Apply dropout
        y = x * mask_val / keep_prob
    else:
        y = x
    
    tl.store(out_ptr + offsets, y, mask=mask)
    # Store mask for backward pass if needed
    tl.store(mask_ptr + offsets, mask_val, mask=mask)

@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x - y, mask=mask)

@triton.jit
def _bmm_kernel(x_ptr, y_ptr, out_ptr, batch_size: tl.constexpr, seq_len: tl.constexpr, hidden_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len // BLOCK)
    seq_idx = pid % (seq_len // BLOCK)
    
    # Compute batch matrix multiplication
    for k in range(hidden_size):
        x_val = tl.load(x_ptr + batch_idx * x_ptr.stride(0) + seq_idx * BLOCK + k)
        y_val = tl.load(y_ptr + batch_idx * y_ptr.stride(0) + k * y_ptr.stride(1) + seq_idx * BLOCK)
        tl.store(out_ptr + batch_idx * out_ptr.stride(0) + seq_idx * BLOCK + k, x_val * y_val)

@triton.jit
def _bmm_kernel_v2(x_ptr, y_ptr, out_ptr, batch_size: tl.constexpr, seq_len: tl.constexpr, hidden_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len // BLOCK)
    seq_idx = pid % (seq_len // BLOCK)
    
    # Load x and y
    x_offsets = batch_idx * x_ptr.stride(0) + seq_idx * BLOCK + tl.arange(0, BLOCK)
    y_offsets = batch_idx * y_ptr.stride(0) + tl.arange(0, BLOCK) * y_ptr.stride(1) + seq_idx * BLOCK
    
    # Compute dot product
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(hidden_size):
        x_val = tl.load(x_ptr + batch_idx * x_ptr.stride(0) + seq_idx * BLOCK + k, mask=k < hidden_size)
        y_val = tl.load(y_ptr + batch_idx * y_ptr.stride(0) + k * y_ptr.stride(1) + seq_idx * BLOCK, mask=k < hidden_size)
        acc += x_val * y_val
    
    # Store result
    out_offsets = batch_idx * out_ptr.stride(0) + seq_idx * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + out_offsets, acc, mask=out_offsets < out_ptr.shape[0])

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Validate inputs
    assert input1.dim() == 3 and input2.dim() == 3
    assert input1.shape[0] == input2.shape[0]  # batch size
    assert input1.shape[2] == input2.shape[1]  # inner dimension
    
    batch_size, seq_len, hidden_size = input1.shape[0], input1.shape[1], input1.shape[2]
    
    # Perform batch matrix multiplication
    bmm_out = torch.empty((batch_size, seq_len, input2.shape[2]), dtype=input1.dtype, device=input1.device)
    
    # Use a simple implementation for now
    bmm_out = torch.bmm(input1, input2)
    
    # RMS normalization
    rms = torch.empty((batch_size, seq_len), dtype=torch.float32, device=input1.device)
    
    # Apply RMS normalization
    for i in range(batch_size):
        for j in range(seq_len):
            x = bmm_out[i, j]
            mean_square = torch.mean(x * x)
            rms[i, j] = torch.sqrt(mean_square + eps)
            bmm_out[i, j] = x / rms[i, j]
    
    # GELU activation
    gelu_out = torch.empty_like(bmm_out)
    n = bmm_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](bmm_out, gelu_out, n, BLOCK=block)
    
    # Dropout
    dropout_out = torch.empty_like(gelu_out)
    mask = torch.empty_like(gelu_out)
    if training:
        # Simple dropout implementation
        keep_prob = 1.0 - dropout_p
        rand = torch.rand_like(gelu_out)
        mask = (rand < keep_prob).to(torch.float32)
        dropout_out = gelu_out * mask / keep_prob
    else:
        dropout_out = gelu_out
        mask = torch.ones_like(gelu_out)
    
    # Subtraction
    if out is None:
        out = torch.empty_like(dropout_out)
    
    # Handle broadcasting
    if other.shape != dropout_out.shape:
        other = other.expand(dropout_out.shape)
    
    # Use Triton kernel for subtraction
    n = dropout_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sub_kernel[grid](dropout_out, other, out, n, BLOCK=block)
    
    return out
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
