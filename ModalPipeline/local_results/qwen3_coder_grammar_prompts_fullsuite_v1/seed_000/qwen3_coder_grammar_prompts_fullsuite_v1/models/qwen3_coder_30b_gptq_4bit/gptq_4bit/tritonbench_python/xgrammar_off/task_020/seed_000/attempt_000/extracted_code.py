import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(
    x1_ptr, x2_ptr, output_ptr,
    x1_row_stride, x2_row_stride, output_row_stride,
    n_features, n_x1, n_x2,
    p_distance, eps_distance,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_x1:
        return
    
    output_row = output_ptr + row * output_row_stride
    x1_row = x1_ptr + row * x1_row_stride
    
    for col in range(0, n_x2, BLOCK_SIZE):
        col_end = min(col + BLOCK_SIZE, n_x2)
        for c in range(col, col_end):
            sum = 0.0
            for i in range(0, n_features, BLOCK_SIZE):
                i_end = min(i + BLOCK_SIZE, n_features)
                for j in range(i, i_end):
                    x1_val = tl.load(x1_row + j, mask=j < n_features)
                    x2_val = tl.load(x2_ptr + c * x2_row_stride + j, mask=j < n_features)
                    diff = x1_val - x2_val
                    sum += diff * diff
                if p_distance != 2.0:
                    sum = sum ** (p_distance / 2.0)
            dist = tl.sqrt(sum + eps_distance)
            tl.store(output_row + c, dist)

@triton.jit
def normalize_kernel(
    input_ptr, output_ptr, norm_ptr,
    input_row_stride, output_row_stride, norm_row_stride,
    n_features, n_rows, dim_norm, p_norm, eps_norm,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    
    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride
    norm_row = norm_ptr + row * norm_row_stride
    
    # Compute norm along specified dimension
    norm = 0.0
    for i in range(0, n_features, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, n_features)
        for j in range(i, i_end):
            val = tl.load(input_row + j, mask=j < n_features)
            norm += val * val
        if p_norm != 2.0:
            norm = norm ** (p_norm / 2.0)
    norm = tl.sqrt(norm + eps_norm)
    
    # Store norm
    tl.store(norm_row, norm)
    
    # Normalize
    for i in range(0, n_features, BLOCK_SIZE):
        i_end = min(i + BLOCK_SIZE, n_features)
        for j in range(i, i_end):
            val = tl.load(input_row + j, mask=j < n_features)
            normalized_val = val / norm
            tl.store(output_row + j, normalized_val)

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    assert dim_norm >= 0 and dim_norm < len(x1.shape), "Invalid dim_norm"
    
    n_x1, n_features = x1.shape[0], x1.shape[1]
    n_x2 = x2.shape[0]
    
    # Compute pairwise distances
    output = torch.empty(n_x1, n_x2, dtype=torch.float32, device=x1.device)
    
    # Launch pairwise distance kernel
    BLOCK_SIZE = 128
    grid = (n_x1,)
    pairwise_distance_kernel[grid](
        x1, x2, output,
        x1.stride(0), x2.stride(0), output.stride(0),
        n_features, n_x1, n_x2,
        p_distance, eps_distance,
        BLOCK_SIZE
    )
    
    # Normalize along specified dimension
    if dim_norm == 1:
        # Normalize along the second dimension (columns)
        output_normalized = torch.empty_like(output)
        norm = torch.empty(n_x1, dtype=torch.float32, device=x1.device)
        
        grid = (n_x1,)
        normalize_kernel[grid](
            output, output_normalized, norm,
            output.stride(0), output_normalized.stride(0), norm.stride(0),
            n_x2, n_x1, dim_norm, p_norm, eps_norm,
            BLOCK_SIZE
        )
        
        if keepdim:
            output_normalized = output_normalized.unsqueeze(dim_norm)
            norm = norm.unsqueeze(dim_norm)
        
        return output_normalized
    else:
        # For other dimensions, we would need to handle differently
        # This implementation assumes dim_norm=1 for simplicity
        raise NotImplementedError("Only dim_norm=1 is supported in this implementation")
