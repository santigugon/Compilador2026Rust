import torch
import triton
import triton.language as tl

def argmax(input, dim, keepdim=False):
    if dim is None:
        # Flatten the input and find argmax
        input_flat = input.flatten()
        out = torch.empty((), dtype=torch.long, device=input.device)
        
        @triton.jit
        def _argmax_kernel_flat(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
            
            # Find max value and its index
            max_val = tl.max(x)
            max_idx = tl.arg_max(x)
            
            # Store result
            tl.store(out_ptr, max_idx)
        
        n = input_flat.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _argmax_kernel_flat[grid](input_flat, out, n, BLOCK=block)
        return out
    else:
        # Handle specific dimension
        input = input.contiguous()
        input_shape = input.shape
        input_ndim = input.dim()
        
        # Normalize negative dimension
        if dim < 0:
            dim = input_ndim + dim
        
        # Validate dimension
        if dim < 0 or dim >= input_ndim:
            raise IndexError(f'dim {dim} out of range')
        
        # Calculate output shape
        output_shape = list(input_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        
        out = torch.empty(output_shape, dtype=torch.long, device=input.device)
        
        # Calculate strides
        input_strides = input.stride()
        output_strides = out.stride() if out.numel() > 0 else []
        
        @triton.jit
        def _argmax_kernel_dim(x_ptr, out_ptr, 
                              input_shape_ptr, input_strides_ptr,
                              output_shape_ptr, output_strides_ptr,
                              input_ndim: tl.constexpr,
                              dim: tl.constexpr,
                              keepdim: tl.constexpr,
                              BLOCK: tl.constexpr):
            # Calculate global thread index
            pid = tl.program_id(0)
            
            # Calculate output indices
            output_idx = pid
            output_indices = []
            temp = output_idx
            for i in range(input_ndim - 1, -1, -1):
                if i != dim:
                    output_indices.append(temp % input_shape_ptr[i])
                    temp //= input_shape_ptr[i]
                else:
                    output_indices.append(0)  # Will be set later
            
            # Calculate input offset
            input_offset = 0
            for i in range(input_ndim):
                if i != dim:
                    input_offset += output_indices[input_ndim - 1 - i] * input_strides_ptr[i]
                else:
                    # For the reduced dimension, we'll iterate through it
                    pass
            
            # Find argmax along the specified dimension
            max_val = -float('inf')
            max_idx = 0
            
            # Get the size of the dimension we're reducing
            dim_size = input_shape_ptr[dim]
            
            # Iterate through the dimension
            for i in range(dim_size):
                current_offset = input_offset + i * input_strides_ptr[dim]
                val = tl.load(x_ptr + current_offset)
                if val > max_val:
                    max_val = val
                    max_idx = i
            
            # Store result
            tl.store(out_ptr + output_idx, max_idx)
        
        # For simplicity, use a more straightforward approach
        # Calculate output size
        output_size = 1
        for i in range(input_ndim):
            if i != dim:
                output_size *= input_shape[i]
        
        if output_size == 0:
            return out
        
        # Create a kernel that works with the specific dimension
        @triton.jit
        def _argmax_kernel(x_ptr, out_ptr, 
                          input_shape_ptr, input_strides_ptr,
                          output_size: tl.constexpr,
                          dim_size: tl.constexpr,
                          dim: tl.constexpr,
                          keepdim: tl.constexpr,
                          BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            
            # Calculate which output element we're working on
            if pid >= output_size:
                return
            
            # Calculate the offset in the input tensor
            # We need to compute the indices in the output tensor
            # and then map them to the input tensor
            
            # For each output element, we need to find the argmax along the specified dimension
            # This is a complex indexing problem, so we'll use a simpler approach
            
            # Calculate the stride for the dimension we're reducing
            dim_stride = input_strides_ptr[dim]
            
            # Calculate the base offset for this output element
            base_offset = 0
            temp_pid = pid
            for i in range(input_ndim - 1, -1, -1):
                if i != dim:
                    # This is a dimension we're not reducing
                    dim_idx = temp_pid % input_shape_ptr[i]
                    base_offset += dim_idx * input_strides_ptr[i]
                    temp_pid //= input_shape_ptr[i]
                else:
                    # This is the dimension we're reducing
                    pass
            
            # Now find argmax along the specified dimension
            max_val = -float('inf')
            max_idx = 0
            
            for i in range(dim_size):
                current_offset = base_offset + i * dim_stride
                val = tl.load(x_ptr + current_offset)
                if val > max_val:
                    max_val = val
                    max_idx = i
            
            tl.store(out_ptr + pid, max_idx)
        
        # Calculate the size of the dimension we're reducing
        dim_size = input_shape[dim]
        
        # Calculate output size
        output_size = 1
        for i in range(input_ndim):
            if i != dim:
                output_size *= input_shape[i]
        
        if output_size == 0:
            return out
        
        # Launch kernel
        block = 256
        grid = (triton.cdiv(output_size, block),)
        
        # Create shape and stride arrays for kernel
        input_shape_ptr = tl.static_input_ptr(input_shape)
        input_strides_ptr = tl.static_input_ptr(input_strides)
        
        _argmax_kernel[grid](input, out, 
                           input_shape_ptr, input_strides_ptr,
                           output_size, dim_size, dim, keepdim, BLOCK=block)
        return out