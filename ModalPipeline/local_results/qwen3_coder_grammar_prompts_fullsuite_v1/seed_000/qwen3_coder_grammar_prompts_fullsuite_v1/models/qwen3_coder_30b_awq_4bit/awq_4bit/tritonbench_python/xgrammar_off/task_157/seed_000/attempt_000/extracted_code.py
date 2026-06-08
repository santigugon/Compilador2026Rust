import torch
import triton
import triton.language as tl

@triton.jit
def signbit_bitwise_and_kernel(
    input_ptr, other_ptr, signbit_ptr, bitwise_and_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    
    signbit_result = tl.where(input_vals < 0.0, True, False)
    bitwise_and_result = input_vals & other_vals
    
    tl.store(signbit_ptr + offsets, signbit_result, mask=mask)
    tl.store(bitwise_and_ptr + offsets, bitwise_and_result, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float32 or input.dtype == torch.float64
    assert other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]
    assert input.shape == other.shape
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    output_shape = input.shape
    
    signbit_result = torch.empty_like(input, dtype=torch.bool)
    bitwise_and_result = torch.empty_like(other, dtype=other.dtype)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    signbit_bitwise_and_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        signbit_ptr=signbit_result,
        bitwise_and_ptr=bitwise_and_result,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return signbit_result, bitwise_and_result
