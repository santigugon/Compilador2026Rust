import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Ensure inputs are contiguous
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    
    # Get dimensions
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Check if normalized_shape is a list or tuple
    if isinstance(normalized_shape, (list, tuple)):
        normalized_shape = torch.Size(normalized_shape)
    
    # Flatten for easier processing
    input1_flat = input1.view(B * N, M)
    input2_flat = input2.view(B * N, M)
    
    # Perform batch matrix multiplication
    # Result shape: (B*N, P)
    bmm_out = torch.bmm(input1.view(B, N, M), input2.view(B, M, P))
    
    # Reshape for RMS normalization
    bmm_out = bmm_out.view(B * N, P)
    
    # RMS normalization
    # Compute mean of squares
    mean_square = torch.mean(bmm_out * bmm_out, dim=1, keepdim=True)
    # Add epsilon and take square root
    rms = torch.sqrt(mean_square + eps)
    # Normalize
    normalized = bmm_out / rms
    
    # Apply GELU activation
    if approximate == 'none':
        gelu_out = torch.nn.functional.gelu(normalized)
    elif approximate == 'tanh':
        gelu_out = 0.5 * normalized * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (normalized + 0.044715 * normalized * normalized * normalized)))
    else:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Apply dropout
    if training and dropout_p > 0:
        # Create dropout mask
        dropout_mask = torch.rand_like(gelu_out) > dropout_p
        # Apply mask
        dropout_out = gelu_out * dropout_mask / (1.0 - dropout_p)
    else:
        dropout_out = gelu_out
    
    # Reshape back to original shape
    output = dropout_out.view(B, N, P)
    
    # Return output tensor
    if out is not None:
        out.copy_(output)
        return out
    else:
        return output