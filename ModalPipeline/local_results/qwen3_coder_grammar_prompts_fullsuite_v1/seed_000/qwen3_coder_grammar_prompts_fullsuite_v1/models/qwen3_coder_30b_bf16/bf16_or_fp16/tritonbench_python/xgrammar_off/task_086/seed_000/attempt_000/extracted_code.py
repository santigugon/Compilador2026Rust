import torch
import triton
import triton.language as tl

@triton.jit
def log_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    log_input = tl.log(input)
    output = tl.tanh(log_input)
    tl.store(output_ptr + offsets, output, mask=mask)

def log_tanh(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    assert input.dtype == torch.float32, "Input tensor must be float32"
    assert input.device == out.device, "Input and output tensors must be on the same device"
    assert input.shape == out.shape, "Input and output tensors must have the same shape"
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    log_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
