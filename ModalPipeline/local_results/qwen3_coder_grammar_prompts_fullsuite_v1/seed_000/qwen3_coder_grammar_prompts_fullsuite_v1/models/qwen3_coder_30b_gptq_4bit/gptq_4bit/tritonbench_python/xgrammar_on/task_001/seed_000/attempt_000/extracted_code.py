import torch
import triton
import triton.language as tl

def _get_rounding_fn(rounding_mode):
    if rounding_mode is None:
        return lambda x: x
    elif rounding_mode == 'trunc':
        return lambda x: tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    elif rounding_mode == 'floor':
        return lambda x: tl.floor(x)
    else:
        raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")

@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)
    result = x / y
    rounding_fn = _get_rounding_fn(rounding_mode)
    result = rounding_fn(result)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _div_scalar_kernel(x_ptr, y_val, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x / y_val
    rounding_fn = _get_rounding_fn(rounding_mode)
    result = rounding_fn(result)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _div_complex_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=1.0)
    # For complex division, we compute (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    # Simplified for real numbers
    real_x = tl.real(x)
    imag_x = tl.imag(x)
    real_y = tl.real(y)
    imag_y = tl.imag(y)
    denominator = real_y * real_y + imag_y * imag_y
    real_result = (real_x * real_y + imag_x * imag_y) / denominator
    imag_result = (imag_x * real_y - real_x * imag_y) / denominator
    result = tl.complex(real_result, imag_result)
    rounding_fn = _get_rounding_fn(rounding_mode)
    result = rounding_fn(result)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _div_complex_scalar_kernel(x_ptr, y_val, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x / y_val
    rounding_fn = _get_rounding_fn(rounding_mode)
    result = rounding_fn(result)
    tl.store(out_ptr + offsets, result, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        if out is not None:
            out = torch.empty_like(out)
        else:
            out = torch.empty_like(input)
        
        # Determine the dtype of the result
        if input.is_complex() or torch.is_complex(other):
            result_dtype = torch.complex128 if input.is_complex() else torch.complex128
        else:
            result_dtype = torch.promote_types(input.dtype, torch.tensor(other).dtype)
        
        # Cast input to appropriate dtype
        input = input.to(result_dtype)
        other = torch.tensor(other, dtype=result_dtype)
        
        # Handle complex case
        if input.is_complex() or other.is_complex():
            n = input.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            _div_complex_scalar_kernel[grid](input, other, out, n, rounding_mode, BLOCK=block)
        else:
            n = input.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            _div_scalar_kernel[grid](input, other, out, n, rounding_mode, BLOCK=block)
        return out
    
    # Handle tensor other
    if out is not None:
        out = torch.empty_like(out)
    else:
        out = torch.empty_like(input)
    
    # Determine the dtype of the result
    if input.is_complex() or other.is_complex():
        result_dtype = torch.complex128 if input.is_complex() else torch.complex128
    else:
        result_dtype = torch.promote_types(input.dtype, other.dtype)
    
    # Cast inputs to appropriate dtype
    input = input.to(result_dtype)
    other = other.to(result_dtype)
    
    # Handle complex case
    if input.is_complex() or other.is_complex():
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _div_complex_kernel[grid](input, other, out, n, rounding_mode, BLOCK=block)
    else:
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _div_kernel[grid](input, other, out, n, rounding_mode, BLOCK=block)
    return out