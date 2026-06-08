import torch
import triton
import triton.language as tl

@triton.jit
def cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    input1_size, input2_size, target_size,
    margin, reduction,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = tl.minimum(input1_size, input2_size)
    
    # Initialize output
    loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process elements in blocks
    for i in range(0, num_elements, BLOCK_SIZE):
        # Load data
        input1_offsets = i + tl.arange(0, BLOCK_SIZE)
        input2_offsets = i + tl.arange(0, BLOCK_SIZE)
        target_offsets = i + tl.arange(0, BLOCK_SIZE)
        
        input1_mask = input1_offsets < input1_size
        input2_mask = input2_offsets < input2_size
        target_mask = target_offsets < target_size
        
        input1_data = tl.load(input1_ptr + input1_offsets, mask=input1_mask, other=0.0)
        input2_data = tl.load(input2_ptr + input2_offsets, mask=input2_mask, other=0.0)
        target_data = tl.load(target_ptr + target_offsets, mask=target_mask, other=0.0)
        
        # Compute dot product
        dot_product = tl.sum(input1_data * input2_data)
        
        # Compute L2 norms
        norm1 = tl.sqrt(tl.sum(input1_data * input1_data) + 1e-8)
        norm2 = tl.sqrt(tl.sum(input2_data * input2_data) + 1e-8)
        
        # Compute cosine similarity
        cos_sim = dot_product / (norm1 * norm2)
        
        # Compute loss
        loss_val = tl.where(
            target_data == 1,
            1 - cos_sim,
            tl.maximum(cos_sim + margin, 0)
        )
        
        # Accumulate loss
        loss = loss + loss_val
    
    # Store result
    tl.store(output_ptr + pid, tl.sum(loss))

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor, 
    input2: torch.Tensor, 
    target: torch.Tensor, 
    margin: float = 0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    # Ensure inputs are on the same device and have the same dtype
    device = input1.device
    if input2.device != device or target.device != device:
        raise ValueError("All input tensors must be on the same device")
    
    if input1.dtype != input2.dtype or input1.dtype != target.dtype:
        raise ValueError("All input tensors must have the same dtype")
    
    # Flatten tensors for processing
    input1_flat = input1.view(-1)
    input2_flat = input2.view(-1)
    target_flat = target.view(-1)
    
    # Create output tensor
    output = torch.empty(1, dtype=torch.float32, device=device)
    
    # Determine block size and grid size
    BLOCK_SIZE = 256
    num_elements = max(input1_flat.size(0), input2_flat.size(0))
    grid_size = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    cosine_embedding_loss_kernel[grid_size](
        input1_flat, input2_flat, target_flat, output,
        input1_flat.size(0), input2_flat.size(0), target_flat.size(0),
        margin, reduction, BLOCK_SIZE
    )
    
    # Apply reduction
    if reduction == 'mean':
        return output / target_flat.size(0)
    elif reduction == 'sum':
        return output
    else:  # reduction == 'none'
        return output
    
    return output
