import torch
import triton
import triton.language as tl

def _cosine_embedding_loss_with_normalization_kernel(input1_ptr, input2_ptr, target_ptr, output_ptr, n_features, n_samples, margin, reduction, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load data
    input1_block = tl.load(input1_ptr + offset, mask=offset < n_samples * n_features)
    input2_block = tl.load(input2_ptr + offset, mask=offset < n_samples * n_features)
    target_block = tl.load(target_ptr + offset // n_features, mask=offset // n_features < n_samples)
    
    # Compute cosine similarity
    dot_product = tl.sum(input1_block * input2_block)
    norm1 = tl.sqrt(tl.sum(input1_block * input1_block) + 1e-8)
    norm2 = tl.sqrt(tl.sum(input2_block * input2_block) + 1e-8)
    cosine_sim = dot_product / (norm1 * norm2)
    
    # Compute loss
    loss = tl.where(target_block == 1, 1 - cosine_sim, tl.maximum(cosine_sim - margin, 0))
    
    # Store result
    tl.store(output_ptr + offset // n_features, loss, mask=offset // n_features < n_samples)

@triton.jit
def _cosine_embedding_loss_with_normalization_kernel_v2(input1_ptr, input2_ptr, target_ptr, output_ptr, n_features, n_samples, margin, reduction, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load data
    input1_block = tl.load(input1_ptr + offset, mask=offset < n_samples * n_features)
    input2_block = tl.load(input2_ptr + offset, mask=offset < n_samples * n_features)
    target_block = tl.load(target_ptr + offset // n_features, mask=offset // n_features < n_samples)
    
    # Compute cosine similarity
    dot_product = tl.sum(input1_block * input2_block)
    norm1 = tl.sqrt(tl.sum(input1_block * input1_block) + 1e-8)
    norm2 = tl.sqrt(tl.sum(input2_block * input2_block) + 1e-8)
    cosine_sim = dot_product / (norm1 * norm2)
    
    # Compute loss
    loss = tl.where(target_block == 1, 1 - cosine_sim, tl.maximum(cosine_sim - margin, 0))
    
    # Store result
    tl.store(output_ptr + offset // n_features, loss, mask=offset // n_features < n_samples)

@triton.jit
def _reduce_mean_kernel(input_ptr, output_ptr, n_samples, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load data
    input_block = tl.load(input_ptr + offset, mask=offset < n_samples)
    
    # Compute mean
    mean_val = tl.sum(input_block) / n_samples
    
    # Store result
    tl.store(output_ptr, mean_val)

@triton.jit
def _reduce_sum_kernel(input_ptr, output_ptr, n_samples, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    
    # Load data
    input_block = tl.load(input_ptr + offset, mask=offset < n_samples)
    
    # Compute sum
    sum_val = tl.sum(input_block)
    
    # Store result
    tl.store(output_ptr, sum_val)

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    # Validate inputs
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert target.shape[0] == input1.shape[0], "target must have the same number of samples as input tensors"
    assert reduction in ['none', 'mean', 'sum'], "reduction must be 'none', 'mean', or 'sum'"
    
    # Flatten input tensors
    n_samples, n_features = input1.shape
    input1_flat = input1.view(-1)
    input2_flat = input2.view(-1)
    target_flat = target.view(-1)
    
    # Allocate output tensor
    if reduction == 'none':
        output = torch.empty(n_samples, dtype=torch.float32, device=input1.device)
    else:
        output = torch.empty(1, dtype=torch.float32, device=input1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (n_samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if reduction == 'none':
        _cosine_embedding_loss_with_normalization_kernel_v2[
            grid_size
        ](
            input1_flat, input2_flat, target_flat, output, n_features, n_samples, margin, reduction, BLOCK_SIZE
        )
    else:
        # Compute intermediate losses
        intermediate_output = torch.empty(n_samples, dtype=torch.float32, device=input1.device)
        _cosine_embedding_loss_with_normalization_kernel_v2[
            grid_size
        ](
            input1_flat, input2_flat, target_flat, intermediate_output, n_features, n_samples, margin, 'none', BLOCK_SIZE
        )
        
        # Apply reduction
        if reduction == 'mean':
            _reduce_mean_kernel[1](intermediate_output, output, n_samples, BLOCK_SIZE)
        else:  # sum
            _reduce_sum_kernel[1](intermediate_output, output, n_samples, BLOCK_SIZE)
    
    return output if reduction != 'none' else output.view(n_samples, 1)