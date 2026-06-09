import torch
import triton
import triton.language as tl

@triton.jit
def _log_softmax_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    n_cols, 
    dim_size,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    if row >= n_cols:
        return
    
    # Load input data for this row
    offsets = row * DIM_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute log softmax
    # First, find max for numerical stability
    max_val = tl.max(input_vals, axis=0)
    # Subtract max to prevent overflow
    shifted = input_vals - max_val
    # Compute sum of exponentials
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    # Compute log softmax
    log_sum_exp = tl.log(sum_exp)
    log_softmax_vals = shifted - log_sum_exp
    
    # Store result
    tl.store(output_ptr + offsets, log_softmax_vals, mask=mask)

@triton.jit
def _cross_entropy_kernel(
    log_softmax_ptr,
    target_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    n_cols,
    dim_size,
    ignore_index,
    label_smoothing,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    if row >= n_cols:
        return
    
    # Load log softmax values for this row
    offsets = row * DIM_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    log_softmax_vals = tl.load(log_softmax_ptr + offsets, mask=mask, other=0.0)
    
    # Load target for this row
    target_offset = row
    target_val = tl.load(target_ptr + target_offset, mask=True, other=0)
    
    # Check if target should be ignored
    if target_val == ignore_index:
        tl.store(output_ptr + target_offset, 0.0, mask=True)
        return
    
    # Compute cross entropy loss
    # For label smoothing, we need to adjust the target
    if label_smoothing > 0.0:
        # Apply label smoothing
        num_classes = DIM_SIZE
        smooth_val = label_smoothing / (num_classes - 1)
        # Create smoothed target distribution
        ce_loss = 0.0
        for i in range(DIM_SIZE):
            if i == target_val:
                ce_loss += (1.0 - label_smoothing) * log_softmax_vals[i]
            else:
                ce_loss += smooth_val * log_softmax_vals[i]
    else:
        # Standard cross entropy
        ce_loss = -log_softmax_vals[target_val]
    
    # Apply weight if provided
    if weight_ptr is not None:
        weight_val = tl.load(weight_ptr + target_val, mask=True, other=1.0)
        ce_loss *= weight_val
    
    # Store result
    tl.store(output_ptr + target_offset, ce_loss, mask=True)

def fused_cross_entropy_log_softmax(
    input: torch.Tensor, 
    target: torch.Tensor, 
    dim: int = 1, 
    weight: torch.Tensor = None, 
    ignore_index: int = -100, 
    reduction: str = 'mean', 
    label_smoothing: float = 0.0
) -> torch.Tensor:
    # Validate inputs
    if input.dim() != target.dim() + 1:
        raise ValueError("input should have one more dimension than target")
    
    # Get dimensions
    input_shape = input.shape
    target_shape = target.shape
    n_cols = 1
    dim_size = input_shape[dim]
    
    # Compute total elements
    n_elements = input.numel()
    
    # Compute number of rows (elements along all dimensions except the specified one)
    for i, s in enumerate(input_shape):
        if i != dim:
            n_cols *= s
    
    # Create output tensor
    if reduction == 'none':
        output = torch.empty(target_shape, dtype=torch.float32, device=input.device)
    else:
        output = torch.empty((), dtype=torch.float32, device=input.device)
    
    # Handle special case where input is scalar or empty
    if n_elements == 0:
        if reduction == 'none':
            return output
        else:
            return torch.tensor(0.0, dtype=torch.float32, device=input.device)
    
    # Prepare for kernel launch
    BLOCK_SIZE = 256
    DIM_SIZE = dim_size
    
    # First compute log softmax
    log_softmax_output = torch.empty_like(input, dtype=torch.float32)
    
    # Launch log softmax kernel
    if n_cols > 0:
        grid = (n_cols,)
        _log_softmax_kernel[grid](
            input, 
            log_softmax_output, 
            n_elements, 
            n_cols, 
            dim_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DIM_SIZE=DIM_SIZE
        )
    
    # Compute cross entropy loss
    if reduction == 'none':
        # For each element, compute cross entropy
        if weight is not None:
            weight_ptr = weight
        else:
            weight_ptr = None
            
        _cross_entropy_kernel[grid](
            log_softmax_output,
            target,
            weight_ptr,
            output,
            n_elements,
            n_cols,
            dim_size,
            ignore_index,
            label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            DIM_SIZE=DIM_SIZE
        )
    else:
        # Compute total loss and apply reduction
        temp_output = torch.empty(n_cols, dtype=torch.float32, device=input.device)
        
        if weight is not None:
            weight_ptr = weight
        else:
            weight_ptr = None
            
        _cross_entropy_kernel[grid](
            log_softmax_output,
            target,
            weight_ptr,
            temp_output,
            n_elements,
            n_cols,
            dim_size,
            ignore_index,
            label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            DIM_SIZE=DIM_SIZE
        )
        
        # Apply reduction
        if reduction == 'sum':
            output = torch.sum(temp_output)
        elif reduction == 'mean':
            # For mean, we need to handle the ignore_index properly
            # Count valid elements (not ignored)
            valid_count = (target != ignore_index).sum().item()
            if valid_count > 0:
                output = torch.sum(temp_output) / valid_count
            else:
                output = torch.tensor(0.0, dtype=torch.float32, device=input.device)
    
    return output
