import torch
import triton
import triton.language as tl

def mul_relu(input, other, inplace=False, out=None):
    if inplace:
        if out is not None:
            raise ValueError("Cannot specify both 'inplace' and 'out'")
        if not torch.is_tensor(other):
            input *= other
            input = torch.relu(input)
            return input
        else:
            input *= other
            input = torch.relu(input)
            return input
    else:
        if out is not None:
            output = out
        else:
            output = torch.empty_like(input)
        
        if not torch.is_tensor(other):
            output = input * other
            output = torch.relu(output)
            return output
        else:
            n = input.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            
            @triton.jit
            def _mul_relu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offsets = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offsets < n
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
                result = x * y
                relu_result = tl.where(result > 0, result, 0.0)
                tl.store(out_ptr + offsets, relu_result, mask=mask)
            
            _mul_relu_kernel[grid](input, other, output, n, BLOCK=block)
            return output