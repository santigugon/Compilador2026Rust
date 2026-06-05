import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, output_ptr, 
    dropout_mask_ptr, 
    input_row_stride, vec_stride, 
    output_row_stride, 
    n_cols, 
    p, 
    training: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    input_row = tl.load(input_ptr + row_idx * input_row_stride, mask=row_idx < tl.cdiv(n_cols, BLOCK_SIZE))
    vec = tl.load(vec_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Matrix-vector multiplication
    dot_product = tl.sum(input_row * vec, axis=0)
    
    # Log-softmax computation
    max_val = tl.max(input_row, axis=0)
    exp_vals = tl.exp(input_row - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    log_softmax = input_row - max_val - tl.log(sum_exp)
    
    # Apply dropout
    if training:
        dropout_mask = tl.random.rand(1, BLOCK_SIZE) > p
        output = log_softmax * dropout_mask / (1.0 - p)
    else:
        output = log_softmax
    
    # Store result
    tl.store(output_ptr + row_idx * output_row_stride, output, mask=row_idx < tl.cdiv(n_cols, BLOCK_SIZE))

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    if out is None:
        out = torch.empty(input.size(0), dtype=torch.float32, device=input.device)
    
    if inplace:
        raise ValueError("Inplace operation not supported in this implementation")
    
    # Ensure input and vec are contiguous
    input = input.contiguous()
    vec = vec.contiguous()
    
    # Get dimensions
    n_rows, n_cols = input.shape
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1, 1)
    
    # Create dropout mask if training
    if training:
        dropout_mask = torch.rand(n_rows, n_cols, device=input.device) > p
    else:
        dropout_mask = None
    
    # Launch kernel
    fused_mv_logsoftmax_dropout_kernel[grid](
        input_ptr=input.data_ptr(),
        vec_ptr=vec.data_ptr(),
        output_ptr=out.data_ptr(),
        dropout_mask_ptr=dropout_mask.data_ptr() if dropout_mask is not None else 0,
        input_row_stride=n_cols,
        vec_stride=1,
        output_row_stride=n_cols,
        n_cols=n_cols,
        p=p,
        training=training,
        dim=dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
