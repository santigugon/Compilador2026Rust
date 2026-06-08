import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, out_ptr, 
    dropout_mask_ptr,
    batch_size, n, m, p,
    p_dropout: tl.constexpr,
    training: tl.constexpr,
    approximate: tl.constexpr,
    input1_stride_0, input1_stride_1, input1_stride_2,
    input2_stride_0, input2_stride_1, input2_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    dropout_mask_stride_0, dropout_mask_stride_1, dropout_mask_stride_2,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Compute batch offset
    batch_offset = pid_batch * n * m
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, m, BLOCK_K):
        # Load input1 slice
        input1_block = tl.load(
            input1_ptr + batch_offset + 
            (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * input1_stride_1 + 
            (k + tl.arange(0, BLOCK_K)[None, :]) * input1_stride_2,
            mask=(pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < n) &
                  (k + tl.arange(0, BLOCK_K)[None, :] < m),
            other=0.0
        )
        
        # Load input2 slice
        input2_block = tl.load(
            input2_ptr + 
            (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) * input2_stride_1 + 
            (k + tl.arange(0, BLOCK_K)[:, None]) * input2_stride_2,
            mask=(pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < p) &
                  (k + tl.arange(0, BLOCK_K)[:, None] < m),
            other=0.0
        )
        
        # Matrix multiplication
        acc += tl.dot(input1_block, input2_block)
    
    # Apply dropout if training
    if training:
        # Load dropout mask
        dropout_mask = tl.load(
            dropout_mask_ptr + 
            batch_offset + 
            (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * dropout_mask_stride_1 + 
            (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) * dropout_mask_stride_2,
            mask=(pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < n) &
                  (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < p),
            other=0.0
        )
        acc = acc * (1.0 / (1.0 - p_dropout)) * dropout_mask
    else:
        # No dropout in inference mode
        pass
    
    # Apply GELU activation
    if approximate == 'tanh':
        # GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        x = acc
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    else:
        # Standard GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        x = acc
        sqrt_2 = 1.4142135623730951
        erf_arg = x / sqrt_2
        # Approximate erf using polynomial
        # erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
        t = 1.0 / (1.0 + 0.3275911 * tl.abs(erf_arg))
        erf_approx = 1.0 - (
            0.254829592 * t - 
            0.284496736 * t * t + 
            1.421413741 * t * t * t - 
            1.453152027 * t * t * t * t + 
            1.061405429 * t * t * t * t * t
        ) * tl.exp(-erf_arg * erf_arg)
        gelu = 0.5 * x * (1.0 + tl.where(erf_arg >= 0, erf_approx, -erf_approx))
    
    # Store result
    tl.store(
        out_ptr + 
        batch_offset + 
        (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * out_stride_1 + 
        (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) * out_stride_2,
        gelu,
        mask=(pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < n) &
              (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :] < p)
    )

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    # Validate inputs
    assert input1.dim() == 3, "input1 must be a 3D tensor"
    assert input2.dim() == 3, "input2 must be a 3D tensor"
    assert input1.size(0) == input2.size(0), "Batch sizes must match"
    assert input1.size(2) == input2.size(1), "Matrix dimensions must be compatible for multiplication"
    
    batch_size, n, m = input1.shape
    _, _, p = input2.shape
    
    # Create output tensor
    if out is not None:
        assert out.shape == (batch_size, n, p), "Output tensor shape must match expected output shape"
        output = out
    else:
        output = torch.empty((batch_size, n, p), dtype=input1.dtype, device=input1.device)
    
    # Create dropout mask if needed
    dropout_mask = None
    if training:
        dropout_mask = torch.empty((batch_size, n, p), dtype=torch.float32, device=input1.device)
        dropout_mask = torch.bernoulli(1.0 - p, dropout_mask)
        dropout_mask = dropout_mask.to(input1.dtype)
    
    # Launch kernel
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    grid = (
        batch_size,
        triton.cdiv(n, BLOCK_M),
        triton.cdiv(p, BLOCK_N)
    )
    
    # Prepare strides
    input1_stride_0, input1_stride_1, input1_stride_2 = input1.stride()
    input2_stride_0, input2_stride_1, input2_stride_2 = input2.stride()
    out_stride_0, out_stride_1, out_stride_2 = output.stride()
    
    if dropout_mask is not None:
        dropout_mask_stride_0, dropout_mask_stride_1, dropout_mask_stride_2 = dropout_mask.stride()
    else:
        dropout_mask_stride_0 = dropout_mask_stride_1 = dropout_mask_stride_2 = 0
    
    # Launch kernel
    _bmm_dropout_gelu_kernel[grid](
        input1, input2, output,
        dropout_mask,
        batch_size, n, m, p,
        p_dropout=p,
        training=training,
        approximate=approximate,
        input1_stride_0=input1_stride_0,
        input1_stride_1=input1_stride_1,
        input1_stride_2=input1_stride_2,
        input2_stride_0=input2_stride_0,
        input2_stride_1=input2_stride_1,
        input2_stride_2=input2_stride_2,
        out_stride_0=out_stride_0,
        out_stride_1=out_stride_1,
        out_stride_2=out_stride_2,
        dropout_mask_stride_0=dropout_mask_stride_0,
        dropout_mask_stride_1=dropout_mask_stride_1,
        dropout_mask_stride_2=dropout_mask_stride_2,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return output
