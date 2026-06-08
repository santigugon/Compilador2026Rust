import torch
import triton
import triton.language as tl

@triton.jit
def _fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, out_ptr, 
    dropout_mask_ptr,
    batch_size: tl.constexpr,
    seq_len1: tl.constexpr,
    seq_len2: tl.constexpr,
    dropout_prob: tl.constexpr,
    training: tl.constexpr,
    approximate: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq1_idx = tl.program_id(1)
    seq2_idx = tl.program_id(2)
    
    # Compute the base pointers for the current batch and sequences
    input1_base = input1_ptr + batch_idx * seq_len1 * seq_len2
    input2_base = input2_ptr + batch_idx * seq_len2 * seq_len2
    out_base = out_ptr + batch_idx * seq_len1 * seq_len2
    
    # Compute the dot product for the current sequence pair
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over the M dimension
    for k in range(0, seq_len2, BLOCK_SIZE):
        # Load input1 and input2
        input1_offsets = seq1_idx * seq_len2 + tl.arange(0, BLOCK_SIZE) + k
        input2_offsets = tl.arange(0, BLOCK_SIZE) * seq_len2 + seq2_idx + k
        
        # Ensure we don't go out of bounds
        mask1 = (k + tl.arange(0, BLOCK_SIZE)) < seq_len2
        mask2 = (k + tl.arange(0, BLOCK_SIZE)) < seq_len2
        
        input1_vals = tl.load(input1_ptr + input1_offsets, mask=mask1, other=0.0)
        input2_vals = tl.load(input2_ptr + input2_offsets, mask=mask2, other=0.0)
        
        # Compute the dot product
        acc += tl.sum(input1_vals * input2_vals)
    
    # Store the result
    out_offsets = seq1_idx * seq_len2 + seq2_idx
    tl.store(out_ptr + out_offsets, acc, mask=True)

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    # Validate inputs
    if input1.dim() != 3 or input2.dim() != 3:
        raise ValueError("Both input1 and input2 must be 3D tensors")
    
    B, N, M = input1.shape
    _, M2, P = input2.shape
    
    if M != M2:
        raise ValueError("The last dimension of input1 must match the second last dimension of input2")
    
    if out is None:
        out = torch.empty(B, N, P, dtype=input1.dtype, device=input1.device)
    else:
        if out.shape != (B, N, P):
            raise ValueError("Output tensor shape must match the expected output shape")
    
    # Perform batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)
    
    # Apply dropout if training is True
    if training and p > 0:
        # Create dropout mask
        dropout_mask = torch.rand(B, N, P, device=input1.device) > p
        bmm_out = bmm_out * dropout_mask / (1.0 - p)
    
    # Apply GELU activation
    if approximate == 'tanh':
        # Use tanh approximation for GELU
        out = 0.5 * bmm_out * (1 + torch.tanh(0.7978845608 * (bmm_out + 0.044715 * bmm_out * bmm_out * bmm_out)))
    else:
        # Use standard GELU
        out = 0.5 * bmm_out * (1 + torch.erf(bmm_out / (2 ** 0.5)))
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
#     Z = torch.bmm(input1, input2)
#     D = torch.nn.functional.dropout(Z, p=p, training=training, inplace=inplace)
#     O = torch.nn.functional.gelu(D, approximate=approximate)
#     if out is not None:
#         out.copy_(O)
#         return out
#     return O

def test_fused_bmm_dropout_gelu():
    results = {}
    
    # Test case 1: Default parameters
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_1"] = fused_bmm_dropout_gelu(input1, input2)
    
    # Test case 2: Dropout with p=0.3 and training=False
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_2"] = fused_bmm_dropout_gelu(input1, input2, p=0.3, training=False)
    
    # Test case 3: In-place dropout
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_3"] = fused_bmm_dropout_gelu(input1, input2, inplace=True)
    
    # Test case 4: GELU with tanh approximation
    input1 = torch.randn(2, 3, 4, device='cuda')
    input2 = torch.randn(2, 4, 5, device='cuda')
    results["test_case_4"] = fused_bmm_dropout_gelu(input1, input2, approximate='tanh')
    
    return results

test_results = test_fused_bmm_dropout_gelu()
