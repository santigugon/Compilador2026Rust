import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, output_ptr, 
    dropout_mask_ptr,
    B, N, M, P,
    p,
    training,
    inplace,
    approximate,
    stride_input1_b, stride_input1_n, stride_input1_m,
    stride_input2_b, stride_input2_m, stride_input2_p,
    stride_output_b, stride_output_n, stride_output_p,
    stride_mask_b, stride_mask_n, stride_mask_p,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get block indices
    block_b = tl.program_id(0)
    block_n = tl.program_id(1)
    block_p = tl.program_id(2)
    
    # Compute offsets
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_p = block_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, M, BLOCK_SIZE_K):
        # Load input1 and input2
        input1 = tl.load(
            input1_ptr + 
            block_b * stride_input1_b + 
            offs_n[:, None] * stride_input1_n + 
            offs_k[None, :] * stride_input1_m,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < M)
        )
        input2 = tl.load(
            input2_ptr + 
            block_b * stride_input2_b + 
            offs_k[:, None] * stride_input2_m + 
            offs_p[None, :] * stride_input2_p,
            mask=(offs_k[:, None] < M) & (offs_p[None, :] < P)
        )
        
        # Matrix multiplication
        acc += tl.dot(input1, input2)
    
    # Apply dropout if training
    if training:
        # Generate random mask
        mask = tl.rand(
            block_b * stride_mask_b + 
            offs_n[:, None] * stride_mask_n + 
            offs_p[None, :] * stride_mask_p,
            seed=0
        ) > p
        acc = tl.where(mask, acc / (1.0 - p), 0.0)
    
    # Apply GELU activation
    if approximate == 'tanh':
        # Approximate GELU using tanh
        acc = 0.5 * acc * (1.0 + tl.tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)))
    else:
        # Exact GELU
        acc = 0.5 * acc * (1.0 + tl.erf(acc / tl.sqrt(2.0)))
    
    # Store result
    tl.store(
        output_ptr + 
        block_b * stride_output_b + 
        offs_n[:, None] * stride_output_n + 
        offs_p[None, :] * stride_output_p,
        acc,
        mask=(offs_n[:, None] < N) & (offs_p[None, :] < P)
    )

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    # Validate inputs
    assert input1.dim() == 3 and input2.dim() == 3
    assert input1.size(0) == input2.size(0)  # Batch size must match
    assert input1.size(2) == input2.size(1)  # Inner dimension must match
    
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Create output tensor
    if out is None:
        output = torch.empty(B, N, P, dtype=input1.dtype, device=input1.device)
    else:
        output = out
    
    # Create dropout mask if needed
    if training:
        dropout_mask = torch.empty(B, N, P, dtype=torch.float32, device=input1.device)
    else:
        dropout_mask = None
    
    # Launch kernel
    grid = (B, triton.cdiv(N, 128), triton.cdiv(P, 128))
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    
    _bmm_dropout_gelu_kernel[grid](
        input1, input2, output, dropout_mask,
        B, N, M, P,
        p,
        training,
        inplace,
        approximate,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        dropout_mask.stride(0) if dropout_mask is not None else 0,
        dropout_mask.stride(1) if dropout_mask is not None else 0,
        dropout_mask.stride(2) if dropout_mask is not None else 0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output
