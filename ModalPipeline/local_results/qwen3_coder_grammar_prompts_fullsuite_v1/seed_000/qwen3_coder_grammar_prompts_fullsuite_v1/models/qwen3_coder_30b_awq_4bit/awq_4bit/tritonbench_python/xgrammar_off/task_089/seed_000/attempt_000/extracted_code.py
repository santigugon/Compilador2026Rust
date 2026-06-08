import torch
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_row_stride, x2_row_stride, out_row_stride,
    n_features, n_samples,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_samples:
        return
    
    x1_row = x1_ptr + row * x1_row_stride
    x2_row = x2_ptr + row * x2_row_stride
    out_row = out_ptr + row * out_row_stride
    
    # Compute cosine similarity for this row
    dot_product = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x1_norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x2_norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(0, n_features, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        
        x1_vals = tl.load(x1_row + offsets, mask=mask, other=0.0)
        x2_vals = tl.load(x2_row + offsets, mask=mask, other=0.0)
        
        dot_product += x1_vals * x2_vals
        x1_norm += x1_vals * x1_vals
        x2_norm += x2_vals * x2_vals
    
    dot_product = tl.sum(dot_product)
    x1_norm = tl.sum(x1_norm)
    x2_norm = tl.sum(x2_norm)
    
    x1_norm = tl.sqrt(x1_norm)
    x2_norm = tl.sqrt(x2_norm)
    
    norm_product = x1_norm * x2_norm + eps
    cosine_sim = dot_product / norm_product
    
    tl.store(out_row, cosine_sim)

@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_row_stride, input_col_stride,
    output_row_stride, output_col_stride,
    height, width, kernel_size, stride, padding,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    out_row = tl.program_id(0)
    out_col = tl.program_id(1)
    
    if out_row >= height or out_col >= width:
        return
    
    # Calculate input indices
    start_h = out_row * stride - padding
    start_w = out_col * stride - padding
    
    # Initialize sum
    sum_val = tl.zeros([1], dtype=tl.float32)
    count = 0
    
    # Iterate over kernel
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h = start_h + kh
            w = start_w + kw
            
            # Check bounds
            if h >= 0 and h < height and w >= 0 and w < width:
                input_idx = h * input_row_stride + w * input_col_stride
                sum_val += tl.load(input_ptr + input_idx)
                count += 1
    
    # Average
    if count > 0:
        avg = sum_val / count
        output_idx = out_row * output_row_stride + out_col * output_col_stride
        tl.store(output_ptr + output_idx, avg)

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    
    # Ensure inputs are on the same device and are contiguous
    device = x1.device
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Compute cosine similarity
    n_samples, n_features = x1.shape
    cosine_sim = torch.empty(n_samples, 1, device=device, dtype=torch.float32)
    
    # Launch cosine similarity kernel
    BLOCK_SIZE = 256
    grid = (n_samples, 1)
    cosine_similarity_kernel[grid](
        x1, x2, cosine_sim,
        x1.stride(0), x2.stride(0), cosine_sim.stride(0),
        n_features, n_samples,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)
    
    # Apply 2D average pooling
    height, width = cosine_sim.shape[2], cosine_sim.shape[3]
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    output = torch.empty(out_height, out_width, device=device, dtype=torch.float32)
    
    # Launch pooling kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = (out_height, out_width)
    avg_pool2d_kernel[grid](
        cosine_sim, output,
        cosine_sim.stride(0), cosine_sim.stride(1),
        output.stride(0), output.stride(1),
        height, width, kernel_size, stride, padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return output
