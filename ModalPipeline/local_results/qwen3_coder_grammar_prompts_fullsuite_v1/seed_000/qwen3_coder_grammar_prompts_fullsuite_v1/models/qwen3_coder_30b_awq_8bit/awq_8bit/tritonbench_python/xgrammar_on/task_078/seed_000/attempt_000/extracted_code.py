import torch
import triton
import triton.language as tl

def _log_softmax_kernel(input_ptr, output_ptr, weights_ptr, target_ptr, loss_ptr, n_elements, n_classes, dim_size, reduction, ignore_index, label_smoothing, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each program handles one row
    row = pid
    if row >= dim_size:
        return
    
    # Load target for this row
    target_val = tl.load(target_ptr + row, mask=(row < dim_size), other=ignore_index)
    
    # Compute log softmax
    # Load input for this row
    input_row = input_ptr + row * n_classes
    output_row = output_ptr + row * n_classes
    
    # Find max for numerical stability
    max_val = tl.load(input_row + 0)
    for i in range(1, n_classes):
        max_val = tl.maximum(max_val, tl.load(input_row + i))
    
    # Compute sum of exp
    sum_exp = 0.0
    for i in range(n_classes):
        exp_val = tl.exp(tl.load(input_row + i) - max_val)
        tl.store(output_row + i, exp_val)
        sum_exp += exp_val
    
    # Normalize and compute log
    for i in range(n_classes):
        log_val = tl.log(tl.load(output_row + i) / sum_exp)
        tl.store(output_row + i, log_val)
    
    # Apply label smoothing if needed
    if label_smoothing > 0.0:
        smooth_val = label_smoothing / n_classes
        for i in range(n_classes):
            smoothed_val = (1.0 - label_smoothing) * tl.load(output_row + i) + smooth_val
            tl.store(output_row + i, smoothed_val)
    
    # Compute loss
    if target_val != ignore_index:
        if weights_ptr is not None:
            weight_val = tl.load(weights_ptr + target_val)
        else:
            weight_val = 1.0
        
        loss_val = -tl.load(output_row + target_val) * weight_val
        tl.store(loss_ptr + row, loss_val)
    else:
        tl.store(loss_ptr + row, 0.0)

@triton.jit
def _reduce_kernel(loss_ptr, output_ptr, n_elements, reduction, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    loss_vals = tl.load(loss_ptr + offsets, mask=mask, other=0.0)
    
    if reduction == 'mean':
        sum_loss = tl.sum(loss_vals, axis=0)
        count = tl.sum(tl.where(mask, 1.0, 0.0))
        result = sum_loss / count
    elif reduction == 'sum':
        result = tl.sum(loss_vals, axis=0)
    else:  # 'none'
        result = loss_vals
    
    tl.store(output_ptr + pid * BLOCK + tl.arange(0, BLOCK), result, mask=mask)

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Validate inputs
    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    
    if label_smoothing < 0 or label_smoothing > 1:
        raise ValueError("label_smoothing must be between 0 and 1")
    
    # Handle scalar inputs
    if input.dim() == 0:
        input = input.unsqueeze(0)
    if target.dim() == 0:
        target = target.unsqueeze(0)
    
    # Get dimensions
    n_classes = input.shape[dim]
    dim_size = input.shape[0] if dim == 1 else input.shape[1]
    n_elements = input.numel() // n_classes
    
    # Create output tensors
    output = torch.empty_like(input)
    loss = torch.empty(input.shape[0] if dim == 1 else input.shape[1], dtype=torch.float32, device=input.device)
    
    # Handle weight tensor
    weights_ptr = None
    if weight is not None:
        if weight.shape[0] != n_classes:
            raise ValueError("weight tensor must have the same number of elements as classes")
        weights_ptr = weight
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n_elements, block),)
    
    _log_softmax_kernel[grid](
        input, output, weights_ptr, target, loss, n_elements, n_classes, dim_size, reduction, ignore_index, label_smoothing, BLOCK=block
    )
    
    # Handle reduction
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    else:  # sum
        return torch.sum(loss)