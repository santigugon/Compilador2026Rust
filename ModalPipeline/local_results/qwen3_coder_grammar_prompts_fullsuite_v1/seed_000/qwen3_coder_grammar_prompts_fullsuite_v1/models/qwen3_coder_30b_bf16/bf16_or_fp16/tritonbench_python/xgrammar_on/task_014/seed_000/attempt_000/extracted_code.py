import torch
import triton
import triton.language as tl

def _get_block_size(N):
    return min(256, triton.next_power_of_2(N))

@triton.jit
def _mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, out_ptr,
    n_rows: tl.constexpr, n_cols: tl.constexpr,
    p: tl.constexpr,
    training: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if dim == 0:
        # Reduce along rows (output has shape [n_cols])
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Matrix-vector multiplication
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(n_rows):
            vec_val = tl.load(vec_ptr + i, mask=i < n_rows, other=0.0)
            input_vals = tl.load(input_ptr + i * n_cols + offsets, mask=mask, other=0.0)
            acc += vec_val * input_vals
        
        # Log-softmax
        max_val = tl.max(acc, axis=0)
        exp_sum = tl.sum(tl.exp(acc - max_val), axis=0)
        log_softmax = acc - max_val - tl.log(exp_sum)
        
        # Dropout
        if training:
            keep_prob = 1.0 - p
            rand_vals = tl.random.rand(0, BLOCK_SIZE)
            mask = rand_vals < keep_prob
            log_softmax = log_softmax * (1.0 / keep_prob) * mask
        
        tl.store(out_ptr + offsets, log_softmax, mask=mask)
    else:
        # Reduce along columns (output has shape [n_rows])
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_rows
        
        # Matrix-vector multiplication
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(n_cols):
            vec_val = tl.load(vec_ptr + i, mask=i < n_cols, other=0.0)
            input_vals = tl.load(input_ptr + offsets * n_cols + i, mask=mask, other=0.0)
            acc += vec_val * input_vals
        
        # Log-softmax
        max_val = tl.max(acc, axis=0)
        exp_sum = tl.sum(tl.exp(acc - max_val), axis=0)
        log_softmax = acc - max_val - tl.log(exp_sum)
        
        # Dropout
        if training:
            keep_prob = 1.0 - p
            rand_vals = tl.random.rand(0, BLOCK_SIZE)
            mask = rand_vals < keep_prob
            log_softmax = log_softmax * (1.0 / keep_prob) * mask
        
        tl.store(out_ptr + offsets, log_softmax, mask=mask)

@triton.jit
def _mv_logsoftmax_dropout_kernel_v2(
    input_ptr, vec_ptr, out_ptr,
    n_rows: tl.constexpr, n_cols: tl.constexpr,
    p: tl.constexpr,
    training: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    if dim == 0:
        # Reduce along rows (output has shape [n_cols])
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Matrix-vector multiplication
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(n_rows):
            vec_val = tl.load(vec_ptr + i, mask=i < n_rows, other=0.0)
            input_vals = tl.load(input_ptr + i * n_cols + offsets, mask=mask, other=0.0)
            acc += vec_val * input_vals
        
        # Log-softmax
        max_val = tl.max(acc, axis=0)
        exp_sum = tl.sum(tl.exp(acc - max_val), axis=0)
        log_softmax = acc - max_val - tl.log(exp_sum)
        
        # Dropout
        if training:
            keep_prob = 1.0 - p
            rand_vals = tl.random.rand(0, BLOCK_SIZE)
            mask = rand_vals < keep_prob
            log_softmax = log_softmax * (1.0 / keep_prob) * mask
        
        tl.store(out_ptr + offsets, log_softmax, mask=mask)
    else:
        # Reduce along columns (output has shape [n_rows])
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_rows
        
        # Matrix-vector multiplication
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(n_cols):
            vec_val = tl.load(vec_ptr + i, mask=i < n_cols, other=0.0)
            input_vals = tl.load(input_ptr + offsets * n_cols + i, mask=mask, other=0.0)
            acc += vec_val * input_vals
        
        # Log-softmax
        max_val = tl.max(acc, axis=0)
        exp_sum = tl.sum(tl.exp(acc - max_val), axis=0)
        log_softmax = acc - max_val - tl.log(exp_sum)
        
        # Dropout
        if training:
            keep_prob = 1.0 - p
            rand_vals = tl.random.rand(0, BLOCK_SIZE)
            mask = rand_vals < keep_prob
            log_softmax = log_softmax * (1.0 / keep_prob) * mask
        
        tl.store(out_ptr + offsets, log_softmax, mask=mask)

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    if out is None:
        if dim == 0:
            out = torch.empty(input.shape[1], dtype=input.dtype, device=input.device)
        else:
            out = torch.empty(input.shape[0], dtype=input.dtype, device=input.device)
    else:
        if dim == 0:
            assert out.shape == (input.shape[1],), f"Output shape mismatch: expected {(input.shape[1],)}, got {out.shape}"
        else:
            assert out.shape == (input.shape[0],), f"Output shape mismatch: expected {(input.shape[0],)}, got {out.shape}"
    
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec, dtype=input.dtype, device=input.device)
    
    n_rows, n_cols = input.shape
    
    if dim == 0:
        # Output shape is [n_cols]
        block_size = _get_block_size(n_cols)
        grid_size = triton.cdiv(n_cols, block_size)
        _mv_logsoftmax_dropout_kernel[grid_size](
            input, vec, out,
            n_rows, n_cols,
            p,
            training,
            dim,
            BLOCK_SIZE=block_size
        )
    else:
        # Output shape is [n_rows]
        block_size = _get_block_size(n_rows)
        grid_size = triton.cdiv(n_rows, block_size)
        _mv_logsoftmax_dropout_kernel[grid_size](
            input, vec, out,
            n_rows, n_cols,
            p,
            training,
            dim,
            BLOCK_SIZE=block_size
        )
    
    return out