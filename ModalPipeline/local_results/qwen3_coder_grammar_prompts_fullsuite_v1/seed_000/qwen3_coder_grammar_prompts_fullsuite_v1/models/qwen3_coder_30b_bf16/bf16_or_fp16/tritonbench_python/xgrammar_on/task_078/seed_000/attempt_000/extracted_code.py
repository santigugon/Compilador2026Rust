import torch
import triton
import triton.language as tl

def _log_softmax_kernel(input_ptr, output_ptr, n_elements, n_cols, dim, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    if row >= n_elements // n_cols:
        return
    
    # Calculate the starting offset for this row
    row_start = row * n_cols
    
    # Load data for this row
    offsets = row_start + tl.arange(0, BLOCK)
    mask = offsets < row_start + n_cols
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Subtract max to prevent overflow
    x = x - max_val
    
    # Compute sum of exponentials
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute log softmax
    log_sum_exp = tl.log(sum_exp)
    log_softmax = x - log_sum_exp
    
    # Store result
    tl.store(output_ptr + offsets, log_softmax, mask=mask)

def _cross_entropy_kernel(input_ptr, target_ptr, weight_ptr, output_ptr, n_elements, n_cols, ignore_index, reduction, label_smoothing, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    if row >= n_elements // n_cols:
        return
    
    row_start = row * n_cols
    offsets = row_start + tl.arange(0, BLOCK)
    mask = offsets < row_start + n_cols
    
    # Load target
    target = tl.load(target_ptr + row, mask=True, other=0)
    
    # Check if target should be ignored
    if target == ignore_index:
        tl.store(output_ptr + row, 0.0, mask=True)
        return
    
    # Load input for this row
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Load weight if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + target, mask=True, other=1.0)
    else:
        weight = 1.0
    
    # Apply label smoothing
    if label_smoothing > 0:
        # Apply label smoothing
        smooth_loss = 0.0
        for i in range(n_cols):
            if i == target:
                smooth_loss += (1.0 - label_smoothing) * x[i]
            else:
                smooth_loss += label_smoothing / (n_cols - 1) * x[i]
        loss = -smooth_loss
    else:
        # Standard cross entropy
        loss = -x[target]
    
    # Apply weight
    loss = loss * weight
    
    # Store result
    tl.store(output_ptr + row, loss, mask=True)

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Validate inputs
    if input.dim() != target.dim() + 1:
        raise ValueError("input should have one more dimension than target")
    
    # Handle scalar target
    if target.dim() == 0:
        target = target.unsqueeze(0)
    
    # Get dimensions
    n_elements = input.numel()
    n_cols = input.size(dim)
    
    # Create output tensor
    if reduction == 'none':
        out = torch.empty(target.shape, dtype=torch.float, device=input.device)
    else:
        out = torch.empty((), dtype=torch.float, device=input.device)
    
    # Handle case where input is scalar
    if input.numel() == 1:
        # For scalar input, we can't use the normal kernel
        if target.item() == ignore_index:
            return torch.tensor(0.0, dtype=torch.float, device=input.device)
        
        # Compute log softmax
        log_softmax_val = torch.log_softmax(input, dim=dim)
        
        # Compute cross entropy
        if label_smoothing > 0:
            # Apply label smoothing
            target_prob = torch.zeros_like(input)
            target_prob.scatter_(dim, target, 1.0)
            target_prob = target_prob * (1.0 - label_smoothing) + label_smoothing / (n_cols - 1)
            loss = -(target_prob * log_softmax_val).sum(dim)
        else:
            loss = -log_softmax_val.gather(dim, target)
        
        if weight is not None:
            loss = loss * weight[target]
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    # For multi-element tensors
    # First compute log softmax
    log_softmax_out = torch.empty_like(input)
    
    block = 256
    grid = (triton.cdiv(n_elements, block),)
    
    # Compute log softmax
    _log_softmax_kernel[grid](input, log_softmax_out, n_elements, n_cols, dim, BLOCK=block)
    
    # Compute cross entropy
    if reduction == 'none':
        _cross_entropy_kernel[grid](log_softmax_out, target, weight, out, n_elements, n_cols, ignore_index, reduction, label_smoothing, BLOCK=block)
    else:
        # For reduction, we need to compute the final result
        temp_out = torch.empty(target.shape, dtype=torch.float, device=input.device)
        _cross_entropy_kernel[grid](log_softmax_out, target, weight, temp_out, n_elements, n_cols, ignore_index, 'none', label_smoothing, BLOCK=block)
        
        if reduction == 'mean':
            out = temp_out.mean()
        elif reduction == 'sum':
            out = temp_out.sum()
        else:
            out = temp_out
    
    return out