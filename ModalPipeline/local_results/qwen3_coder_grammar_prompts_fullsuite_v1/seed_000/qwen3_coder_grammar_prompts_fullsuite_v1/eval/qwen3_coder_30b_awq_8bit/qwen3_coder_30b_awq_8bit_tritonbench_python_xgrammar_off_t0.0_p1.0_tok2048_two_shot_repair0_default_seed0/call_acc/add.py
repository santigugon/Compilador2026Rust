import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + alpha * y, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        return input + alpha * other
    
    # Ensure tensors have the same dtype for computation
    if input.dtype != other.dtype:
        # Promote to a common dtype
        common_dtype = torch.result_type(input, other)
        input = input.to(common_dtype)
        other = other.to(common_dtype)
    
    # Handle broadcasting
    # We'll use PyTorch's broadcasting rules by computing the output shape
    # and then using a simple element-wise kernel
    out_shape = torch.broadcast_shapes(input.shape, other.shape)
    
    # Create output tensor
    if out is None:
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != out_shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected broadcast shape {out_shape}")
    
    # Flatten tensors for kernel execution
    input_flat = input.reshape(-1)
    other_flat = other.reshape(-1)
    out_flat = out.reshape(-1)
    
    n = out_flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _add_kernel[grid](input_flat, other_flat, out_flat, n, alpha, BLOCK=block)
    return out

##################################################################################################################################################



import torch

def test_add():
    results = {}

    # Test case 1: Adding two tensors with default alpha
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other1 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_1"] = add(input1, other1)

    # Test case 2: Adding a tensor and a scalar with default alpha
    input2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other2 = 2.0
    results["test_case_2"] = add(input2, other2)

    # Test case 3: Adding two tensors with a specified alpha
    input3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other3 = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    results["test_case_3"] = add(input3, other3, alpha=0.5)

    # Test case 4: Adding a tensor and a scalar with a specified alpha
    input4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other4 = 2.0
    results["test_case_4"] = add(input4, other4, alpha=0.5)

    return results

test_results = test_add()
