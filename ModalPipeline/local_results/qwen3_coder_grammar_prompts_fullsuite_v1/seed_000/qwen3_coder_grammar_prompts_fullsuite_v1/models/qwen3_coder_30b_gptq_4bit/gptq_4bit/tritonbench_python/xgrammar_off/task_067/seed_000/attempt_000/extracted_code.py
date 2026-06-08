import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_stride_0, x1_stride_1, x1_stride_2, x1_stride_3,
    x2_stride_0, x2_stride_1, x2_stride_2, x2_stride_3,
    out_stride_0, out_stride_1,
    output_size_0, output_size_1,
    p_val: tl.constexpr,
    eps_val: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (output_size_0 * output_size_1)
    h_id = (pid % (output_size_0 * output_size_1)) // output_size_1
    w_id = (pid % (output_size_0 * output_size_1)) % output_size_1
    
    if batch_id >= 1:
        return
    
    # Compute the distance for this element
    x1_val = tl.load(x1_ptr + batch_id * x1_stride_0 + h_id * x1_stride_2 + w_id * x1_stride_3)
    x2_val = tl.load(x2_ptr + batch_id * x2_stride_0 + h_id * x2_stride_2 + w_id * x2_stride_3)
    
    diff = x1_val - x2_val
    diff_pow = tl.power(tl.abs(diff), p_val)
    
    # Store the result
    tl.store(out_ptr + h_id * out_stride_1 + w_id, diff_pow)

def fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size, p=2.0, eps=1e-6, keepdim=False):
    # Handle output_size as int or tuple
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Apply adaptive average pooling
    x1_pooled = torch.nn.functional.adaptive_avg_pool2d(x1, output_size)
    x2_pooled = torch.nn.functional.adaptive_avg_pool2d(x2, output_size)
    
    # Ensure tensors are contiguous for easier handling
    x1_pooled = x1_pooled.contiguous()
    x2_pooled = x2_pooled.contiguous()
    
    # Get output shape
    batch_size, channels, h_out, w_out = x1_pooled.shape
    
    # Create output tensor
    if keepdim:
        out = torch.empty((batch_size, 1, h_out, w_out), dtype=torch.float32, device=x1.device)
    else:
        out = torch.empty((batch_size, h_out, w_out), dtype=torch.float32, device=x1.device)
    
    # Compute pairwise distance
    n = batch_size * h_out * w_out
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Prepare strides
    x1_stride_0 = x1_pooled.stride(0)
    x1_stride_1 = x1_pooled.stride(1)
    x1_stride_2 = x1_pooled.stride(2)
    x1_stride_3 = x1_pooled.stride(3)
    
    x2_stride_0 = x2_pooled.stride(0)
    x2_stride_1 = x2_pooled.stride(1)
    x2_stride_2 = x2_pooled.stride(2)
    x2_stride_3 = x2_pooled.stride(3)
    
    out_stride_0 = out.stride(0)
    out_stride_1 = out.stride(1) if keepdim else out.stride(0)
    
    # Launch kernel
    _pairwise_distance_kernel[grid](
        x1_pooled, x2_pooled, out,
        x1_stride_0, x1_stride_1, x1_stride_2, x1_stride_3,
        x2_stride_0, x2_stride_1, x2_stride_2, x2_stride_3,
        out_stride_0, out_stride_1,
        output_size[0], output_size[1],
        p, eps, BLOCK=block
    )
    
    # Sum over the channel dimension if needed
    if keepdim:
        out = out.sum(dim=1, keepdim=True)
    else:
        out = out.sum(dim=1)
    
    # Apply p-norm
    out = torch.pow(out + eps, 1.0 / p)
    
    return out
