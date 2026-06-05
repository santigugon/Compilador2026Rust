import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, output_ptr, 
    dropout_mask_ptr, 
    B, N, M, P, 
    dropout_p, 
    training,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    NUM_WARPS
):
    # Compute batch matrix multiplication
    pid = tl.program_id(axis=0)
    pid_batch = pid // (N * P)
    pid_m = (pid % (N * P)) // P
    pid_n = (pid % (N * P)) % P
    
    # Load input tensors
    input1 = tl.load(input1_ptr + pid_batch * N * M + pid_m * M + tl.arange(0, BLOCK_SIZE_K))
    input2 = tl.load(input2_ptr + pid_batch * M * P + tl.arange(0, BLOCK_SIZE_K) * P + pid_n)
    
    # Compute dot product
    accumulator = tl.sum(input1 * input2)
    
    # Apply dropout if training
    if training:
        # Generate random mask
        rand_val = tl.random.rand(1)
        dropout_mask = rand_val > dropout_p
        accumulator = accumulator * dropout_mask / (1.0 - dropout_p)
    
    # Apply GELU activation
    if approximate == 'tanh':
        # Approximate GELU using tanh
        gelu_result = 0.5 * accumulator * (1 + tl.tanh(1.702 * accumulator))
    else:
        # Exact GELU
        gelu_result = 0.5 * accumulator * (1 + tl.erf(accumulator / tl.sqrt(2.0)))
    
    # Store result
    tl.store(output_ptr + pid, gelu_result)

def fused_bmm_dropout_gelu(
    input1: torch.Tensor,
    input2: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    approximate: str = 'none',
    *,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Validate inputs
    assert input1.dim() == 3, "input1 must be 3D tensor"
    assert input2.dim() == 3, "input2 must be 3D tensor"
    assert input1.shape[0] == input2.shape[0], "Batch sizes must match"
    assert input1.shape[2] == input2.shape[1], "Matrix dimensions must be compatible"
    
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(B, N, P, dtype=input1.dtype, device=input1.device)
    
    # Set kernel parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    NUM_WARPS = 4
    
    # Launch kernel
    grid = (B * N * P,)
    _fused_bmm_dropout_gelu_kernel[grid](
        input1_ptr=input1.data_ptr(),
        input2_ptr=input2.data_ptr(),
        output_ptr=out.data_ptr(),
        dropout_mask_ptr=None,
        B=B, N=N, M=M, P=P,
        dropout_p=p,
        training=training,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_WARPS=NUM_WARPS
    )
    
    return out
