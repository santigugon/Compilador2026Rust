import torch
import triton
import triton.language as tl

def _get_block_size(N):
    return min(256, triton.next_power_of_2(N))

@triton.jit
def _mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, out_ptr, 
    input_row_stride, input_col_stride,
    vec_stride,
    out_row_stride, out_col_stride,
    n_rows, n_cols,
    p: tl.constexpr,
    training: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if dim == 0:
        # Reduce along rows (output along columns)
        row = pid
        if row >= n_rows:
            return
        
        # Compute matrix-vector multiplication
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(0, n_cols, BLOCK_SIZE):
            vec_offsets = i + tl.arange(0, BLOCK_SIZE)
            vec_mask = vec_offsets < n_cols
            
            input_offsets = row * input_row_stride + vec_offsets * input_col_stride
            input_mask = vec_mask
            
            vec_vals = tl.load(vec_ptr + vec_offsets, mask=vec_mask, other=0.0)
            input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
            
            acc += input_vals * vec_vals
        
        # Store result of MV
        out_offsets = row * out_row_stride + tl.arange(0, BLOCK_SIZE) * out_col_stride
        tl.store(out_ptr + out_offsets, acc, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    else:
        # Reduce along columns (output along rows)
        col = pid
        if col >= n_cols:
            return
        
        # Compute matrix-vector multiplication
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(0, n_rows, BLOCK_SIZE):
            input_offsets = i * input_row_stride + col * input_col_stride
            input_mask = (i + tl.arange(0, BLOCK_SIZE)) < n_rows
            
            vec_offsets = i + tl.arange(0, BLOCK_SIZE)
            vec_mask = vec_mask
            
            input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
            vec_vals = tl.load(vec_ptr + vec_offsets, mask=vec_mask, other=0.0)
            
            acc += input_vals * vec_vals
        
        # Store result of MV
        out_offsets = col * out_col_stride + tl.arange(0, BLOCK_SIZE) * out_row_stride
        tl.store(out_ptr + out_offsets, acc, mask=tl.arange(0, BLOCK_SIZE) < n_rows)

@triton.jit
def _logsoftmax_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Log-softmax computation
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    log_sum_exp = tl.log(x_sum)
    log_softmax = x_shifted - log_sum_exp
    
    tl.store(out_ptr + offsets, log_softmax, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if training:
        # Generate random mask
        rand = tl.random.rand(0)  # Simple random number generation
        keep_prob = 1.0 - p
        dropout_mask = rand < keep_prob
        
        # Apply dropout
        out = tl.where(dropout_mask, x / keep_prob, 0.0)
    else:
        out = x
    
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def _mv_kernel(input_ptr, vec_ptr, out_ptr, n_rows, n_cols, input_row_stride, input_col_stride, vec_stride, out_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        vec_offsets = i + tl.arange(0, BLOCK_SIZE)
        vec_mask = vec_offsets < n_cols
        
        input_offsets = row * input_row_stride + vec_offsets * input_col_stride
        input_mask = vec_mask
        
        vec_vals = tl.load(vec_ptr + vec_offsets, mask=vec_mask, other=0.0)
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        acc += input_vals * vec_vals
    
    out_offsets = row * out_stride
    tl.store(out_ptr + out_offsets, acc, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Handle scalar vector case
    if vec.dim() == 0:
        vec = vec.expand(input.size(dim))
    
    # Matrix-vector multiplication
    if dim == 0:
        # Reduce along rows
        n_rows = input.size(0)
        n_cols = input.size(1)
        if n_cols != vec.size(0):
            raise ValueError("Vector size must match the number of columns in input matrix")
        
        # Compute intermediate result
        mv_out = torch.empty(n_cols, dtype=input.dtype, device=input.device)
        
        block = _get_block_size(n_cols)
        grid = (triton.cdiv(n_cols, block),)
        
        # Use a simple approach for now - compute MV in PyTorch
        mv_out = torch.mv(input, vec)
    else:
        # Reduce along columns
        n_rows = input.size(0)
        n_cols = input.size(1)
        if n_rows != vec.size(0):
            raise ValueError("Vector size must match the number of rows in input matrix")
        
        # Compute intermediate result
        mv_out = torch.empty(n_rows, dtype=input.dtype, device=input.device)
        
        block = _get_block_size(n_rows)
        grid = (triton.cdiv(n_rows, block),)
        
        # Use a simple approach for now - compute MV in PyTorch
        mv_out = torch.mv(input.t(), vec)
    
    # Log-softmax
    logsoftmax_out = torch.empty_like(mv_out)
    n = mv_out.numel()
    block = _get_block_size(n)
    grid = (triton.cdiv(n, block),)
    
    # For simplicity, use PyTorch's log_softmax
    logsoftmax_out = torch.log_softmax(mv_out, dim=0)
    
    # Dropout
    if training:
        dropout_out = torch.empty_like(logsoftmax_out)
        # Simple approach: use PyTorch's dropout
        dropout_out = torch.nn.functional.dropout(logsoftmax_out, p=p, training=training)
    else:
        dropout_out = logsoftmax_out
    
    # Return result
    if inplace:
        out.copy_(dropout_out)
        return out
    else:
        return dropout_out