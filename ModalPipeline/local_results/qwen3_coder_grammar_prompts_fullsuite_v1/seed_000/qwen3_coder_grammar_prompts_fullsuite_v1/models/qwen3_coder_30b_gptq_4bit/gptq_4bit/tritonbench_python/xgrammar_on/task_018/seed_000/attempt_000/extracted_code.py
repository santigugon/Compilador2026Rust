import torch
import triton
import triton.language as tl

def argmax(input, dim, keepdim=False):
    if dim is None:
        # Flatten the input tensor
        input_flat = input.flatten()
        # Create output tensor
        out = torch.empty((), dtype=torch.long)
        # Use PyTorch's argmax for the flattened tensor
        out = torch.argmax(input_flat)
        return out
    else:
        # Get the shape and strides of the input tensor
        shape = input.shape
        strides = input.stride()
        # Create output tensor
        out_shape = list(shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            out_shape.pop(dim)
        out = torch.empty(out_shape, dtype=torch.long)
        # Determine the size of the dimension to reduce
        reduce_size = shape[dim]
        # Determine the size of the output tensor
        out_size = 1
        for i in range(len(shape)):
            if i != dim:
                out_size *= shape[i]
        # Create a kernel to compute argmax
        @triton.jit
        def _argmax_kernel(input_ptr, out_ptr, reduce_size: tl.constexpr, out_size: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < out_size
            # Load input data
            input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            # Compute argmax
            max_val = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
            max_idx = tl.full((BLOCK,), 0, dtype=tl.int32)
            for i in range(reduce_size):
                # Load data from the reduced dimension
                data = tl.load(input_ptr + offsets * reduce_size + i, mask=mask, other=0.0)
                # Update max_val and max_idx
                mask_new = data > max_val
                max_val = tl.where(mask_new, data, max_val)
                max_idx = tl.where(mask_new, i, max_idx)
            # Store result
            tl.store(out_ptr + offsets, max_idx, mask=mask)
        # Launch kernel
        block = 256
        grid = (triton.cdiv(out_size, block),)
        _argmax_kernel[grid](input, out, reduce_size, out_size, BLOCK=block)
        return out