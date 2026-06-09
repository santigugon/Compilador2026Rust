import torch
import triton
import triton.language as tl

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    if dim is None:
        # Compute mean over all elements
        input = input.float() if input.dtype != torch.float32 else input
        out_tensor = torch.empty((), dtype=torch.float32, device=input.device)
        
        # Compute exp and mean in one kernel
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        @triton.jit
        def _exp_mean_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.exp(x)
            tl.store(out_ptr + offsets, y, mask=mask)
        
        # Allocate intermediate tensor for exp results
        exp_tensor = torch.empty_like(input)
        _exp_mean_kernel[grid](input, exp_tensor, n, BLOCK=block)
        
        # Compute mean
        mean_val = exp_tensor.mean()
        out_tensor.fill_(mean_val)
        
        if out is not None:
            out.copy_(out_tensor)
            return out
        else:
            return out_tensor
    else:
        # Compute mean along specified dimension
        input = input.float() if input.dtype != torch.float32 else input
        if dtype is not None:
            output_dtype = dtype
        else:
            output_dtype = input.dtype
        
        # Get output shape
        output_shape = list(input.shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        
        out_tensor = torch.empty(output_shape, dtype=output_dtype, device=input.device)
        
        # Compute exp and mean along dimension
        exp_tensor = torch.exp(input)
        mean_tensor = exp_tensor.mean(dim=dim, keepdim=keepdim)
        
        if out is not None:
            out.copy_(mean_tensor)
            return out
        else:
            return mean_tensor
##################################################################################################################################################



import torch

def test_exp_mean():
    results = {}

    # Test case 1: Basic test with a 1D tensor on GPU
    input_tensor_1d = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = exp_mean(input_tensor_1d)

    # Test case 2: 2D tensor with dim specified
    input_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = exp_mean(input_tensor_2d, dim=0)

    # Test case 3: 2D tensor with keepdim=True
    results["test_case_3"] = exp_mean(input_tensor_2d, dim=1, keepdim=True)

    # Test case 4: 3D tensor with no dim specified (mean over all elements)
    input_tensor_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_4"] = exp_mean(input_tensor_3d)

    return results

test_results = test_exp_mean()
