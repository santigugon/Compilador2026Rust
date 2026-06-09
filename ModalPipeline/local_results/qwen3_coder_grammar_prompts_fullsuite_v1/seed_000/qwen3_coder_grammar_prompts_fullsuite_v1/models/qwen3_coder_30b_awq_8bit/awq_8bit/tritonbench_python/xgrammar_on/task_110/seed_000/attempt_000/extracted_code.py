import torch
import triton
import triton.language as tl

@triton.jit
def _exp_mean_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x: tl.constexpr, stride_out: tl.constexpr, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute output index
    if keepdim:
        out_idx = pid
    else:
        out_idx = pid // dim_size
        
    # Load input elements
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply exponential
    exp_x = tl.exp(x)
    
    # Compute sum
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute mean
    mean_exp = sum_exp / dim_size
    
    # Store result
    tl.store(out_ptr + out_idx, mean_exp, mask=keepdim or (out_idx < 1))

@triton.jit
def _exp_mean_kernel_1d(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    mean_exp = sum_exp / n
    tl.store(out_ptr, mean_exp, mask=True)

@triton.jit
def _exp_mean_kernel_2d(x_ptr, out_ptr, rows: tl.constexpr, cols: tl.constexpr, stride_x_row: tl.constexpr, stride_x_col, stride_out_row: tl.constexpr, stride_out_col: tl.constexpr, dim: tl.constexpr, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    if dim == 0:  # Reduce along rows
        # Each thread block handles one column
        col = pid
        if col >= cols:
            return
        
        # Compute sum for this column
        sum_exp = 0.0
        for row in range(rows):
            x_val = tl.load(x_ptr + row * stride_x_row + col * stride_x_col)
            exp_val = tl.exp(x_val)
            sum_exp += exp_val
        
        mean_exp = sum_exp / rows
        
        if keepdim:
            tl.store(out_ptr + col * stride_out_col, mean_exp)
        else:
            tl.store(out_ptr + col * stride_out_col, mean_exp)
    
    else:  # Reduce along columns
        # Each thread block handles one row
        row = pid
        if row >= rows:
            return
        
        # Compute sum for this row
        sum_exp = 0.0
        for col in range(cols):
            x_val = tl.load(x_ptr + row * stride_x_row + col * stride_x_col)
            exp_val = tl.exp(x_val)
            sum_exp += exp_val
        
        mean_exp = sum_exp / cols
        
        if keepdim:
            tl.store(out_ptr + row * stride_out_row, mean_exp)
        else:
            tl.store(out_ptr + row * stride_out_row, mean_exp)

def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    if dim is None:
        # Compute mean over all elements
        input_flat = input.flatten()
        n = input_flat.numel()
        out_tensor = torch.empty((), dtype=dtype or input.dtype, device=input.device)
        
        block = 256
        grid = (triton.cdiv(n, block),)
        _exp_mean_kernel_1d[grid](input_flat, out_tensor, n, BLOCK=block)
        return out_tensor
    
    else:
        # Handle specific dimension
        input_shape = input.shape
        if dim < 0:
            dim = len(input_shape) + dim
        
        if dim >= len(input_shape):
            raise ValueError(f"dim {dim} is out of range for tensor with shape {input_shape}")
        
        # Compute output shape
        output_shape = list(input_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        
        # Create output tensor
        out_tensor = torch.empty(output_shape, dtype=dtype or input.dtype, device=input.device)
        
        if len(input_shape) == 1:
            # 1D case
            n = input.numel()
            block = 256
            grid = (triton.cdiv(n, block),)
            _exp_mean_kernel_1d[grid](input, out_tensor, n, BLOCK=block)
        
        elif len(input_shape) == 2:
            # 2D case
            rows, cols = input_shape
            if dim == 0:
                # Reduce along rows
                block = 256
                grid = (cols,)
                _exp_mean_kernel_2d[grid](input, out_tensor, rows, cols, input.stride(0), input.stride(1), 
                                        out_tensor.stride(0) if len(output_shape) > 0 else 1, 
                                        out_tensor.stride(0) if len(output_shape) > 0 else 1,
                                        dim, keepdim, BLOCK=block)
            else:
                # Reduce along columns
                block = 256
                grid = (rows,)
                _exp_mean_kernel_2d[grid](input, out_tensor, rows, cols, input.stride(0), input.stride(1), 
                                        out_tensor.stride(0) if len(output_shape) > 0 else 1, 
                                        out_tensor.stride(0) if len(output_shape) > 0 else 1,
                                        dim, keepdim, BLOCK=block)
        
        else:
            # For higher dimensions, use PyTorch implementation
            if out is not None:
                out.copy_(torch.exp(input).mean(dim=dim, keepdim=keepdim))
                return out
            else:
                return torch.exp(input).mean(dim=dim, keepdim=keepdim)
        
        return out_tensor