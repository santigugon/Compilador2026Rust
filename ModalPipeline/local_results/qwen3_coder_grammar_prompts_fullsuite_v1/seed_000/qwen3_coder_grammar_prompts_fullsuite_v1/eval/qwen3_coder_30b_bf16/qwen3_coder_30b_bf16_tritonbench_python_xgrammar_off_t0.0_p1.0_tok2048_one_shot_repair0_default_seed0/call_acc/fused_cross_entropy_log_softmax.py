import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    input_ptr, target_ptr, weight_ptr, output_ptr,
    N, C, dim, ignore_index, reduction, label_smoothing,
    BLOCK_SIZE: tl.constexpr
):
    # Get the thread index
    tid = tl.program_id(0)
    
    # Each block handles one sample
    if tid >= N:
        return
    
    # Compute the offset for this sample
    input_offset = tid * C
    target_offset = tid
    output_offset = tid
    
    # Initialize loss and max_val
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    sum_exp = tl.full([1], 0.0, dtype=tl.float32)
    
    # Compute max for numerical stability
    for i in range(0, C, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < C
        if mask:
            val = tl.load(input_ptr + input_offset + idx, mask=mask, other=-float('inf'))
            max_val = tl.maximum(max_val, val)
    
    # Compute sum of exponentials
    for i in range(0, C, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < C
        if mask:
            val = tl.load(input_ptr + input_offset + idx, mask=mask, other=-float('inf'))
            exp_val = tl.exp(val - max_val)
            sum_exp += exp_val
    
    # Compute log softmax
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Compute cross entropy loss
    target_val = tl.load(target_ptr + target_offset)
    
    # Check if target is ignored
    if target_val == ignore_index:
        loss = tl.full([1], 0.0, dtype=tl.float32)
    else:
        # Apply label smoothing
        if label_smoothing > 0.0:
            # Smoothed target probabilities
            smooth_val = label_smoothing / (C - 1)
            target_prob = tl.full([1], smooth_val, dtype=tl.float32)
            # Set the correct class probability
            correct_prob = 1.0 - label_smoothing + smooth_val
            # Compute loss with smoothing
            log_prob = tl.load(input_ptr + input_offset + target_val) - log_sum_exp
            loss = -log_prob * (1.0 - label_smoothing) - smooth_val * (C - 1) * log_prob
        else:
            # Standard case
            log_prob = tl.load(input_ptr + input_offset + target_val) - log_sum_exp
            loss = -log_prob
            
        # Apply weight if provided
        if weight_ptr is not None:
            weight_val = tl.load(weight_ptr + target_val)
            loss *= weight_val
    
    # Store result
    tl.store(output_ptr + output_offset, loss)

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Validate inputs
    assert input.dim() >= 2, "input must be at least 2D"
    assert target.dim() == input.dim() - 1, "target must be 1D less than input"
    assert dim < input.dim(), "dim must be less than input dimensions"
    
    # Ensure input is contiguous
    input = input.contiguous()
    target = target.contiguous()
    
    # Get dimensions
    N = input.shape[0]  # batch size
    C = input.shape[dim]  # number of classes
    
    # Handle reduction
    if reduction == 'none':
        output = torch.empty(N, dtype=torch.float32, device=input.device)
    else:
        output = torch.empty(1, dtype=torch.float32, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (N, 1, 1)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    target_ptr = target.data_ptr()
    output_ptr = output.data_ptr()
    weight_ptr = weight.data_ptr() if weight is not None else None
    
    # Launch kernel
    fused_cross_entropy_log_softmax_kernel[grid](
        input_ptr, target_ptr, weight_ptr, output_ptr,
        N, C, dim, ignore_index, reduction, label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply reduction
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_cross_entropy_log_softmax():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_1"] = fused_cross_entropy_log_softmax(input, target)
    
    # Test case 2: Test with label smoothing
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_2"] = fused_cross_entropy_log_softmax(input, target, label_smoothing=0.1)
    
    # Test case 3: Test with weight
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    weight = torch.tensor([1.0, 0.5, 2.0], device='cuda')
    results["test_case_3"] = fused_cross_entropy_log_softmax(input, target, weight=weight)
    
    # Test case 4: Test with sum reduction
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    target = torch.tensor([2, 1], device='cuda')
    results["test_case_4"] = fused_cross_entropy_log_softmax(input, target, reduction='sum')
    
    return results

test_results = test_fused_cross_entropy_log_softmax()
