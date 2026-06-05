import torch
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    for i in range(0, n_elements, BLOCK_SIZE):
        current_offsets = i + tl.arange(0, BLOCK_SIZE)
        current_mask = current_offsets < n_elements
        current_input = tl.load(input_ptr + current_offsets, mask=current_mask)
        current_max_val = tl.maximum(max_val, current_input)
        current_max_idx = tl.where(current_input > max_val, current_offsets, max_idx)
        max_val = current_max_val
        max_idx = current_max_idx
    
    tl.store(output_ptr + pid, max_idx[0])

def argmax(input, dim, keepdim=False):
    if dim is None:
        input_flat = input.flatten()
        output = torch.empty(1, dtype=torch.long, device=input.device)
        n_elements = input_flat.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        argmax_kernel[grid](input_flat, output, n_elements, BLOCK_SIZE)
        return output
    else:
        input_shape = input.shape
        output_shape = list(input_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        output = torch.empty(output_shape, dtype=torch.long, device=input.device)
        return output

##################################################################################################################################################



import torch

def test_argmax():
    results = {}

    # Test case 1: 2D tensor, dim=0
    tensor_2d = torch.tensor([[1, 3, 2], [4, 0, 5]], device='cuda')
    results["test_case_1"] = argmax(tensor_2d, dim=0)

    # Test case 2: 2D tensor, dim=1
    results["test_case_2"] = argmax(tensor_2d, dim=1)

    # Test case 3: 3D tensor, dim=2
    tensor_3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], device='cuda')
    results["test_case_3"] = argmax(tensor_3d, dim=2)

    # Test case 4: 3D tensor, dim=1, keepdim=True
    results["test_case_4"] = argmax(tensor_3d, dim=1, keepdim=True)

    return results

test_results = test_argmax()
