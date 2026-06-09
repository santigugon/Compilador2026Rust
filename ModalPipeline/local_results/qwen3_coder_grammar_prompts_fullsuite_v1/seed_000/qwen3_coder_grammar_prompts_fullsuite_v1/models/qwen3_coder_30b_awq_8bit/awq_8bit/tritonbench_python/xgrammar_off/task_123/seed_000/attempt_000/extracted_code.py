import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    conv_stride_h, conv_stride_w,
    conv_padding_h, conv_padding_w,
    conv_dilation_h, conv_dilation_w,
    conv_groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output dimensions
    batch_size, in_channels, ih, iw = input_shape
    out_channels, _, kh, kw = weight_shape
    
    # Each thread block handles one output element
    output_h = (ih + 2 * conv_padding_h - (kh - 1) * conv_dilation_h - 1) // conv_stride_h + 1
    output_w = (iw + 2 * conv_padding_w - (kw - 1) * conv_dilation_w - 1) // conv_stride_w + 1
    
    # Calculate which output element this block handles
    batch_idx = pid // (output_h * output_w)
    h_idx = (pid % (output_h * output_w)) // output_w
    w_idx = (pid % (output_h * output_w)) % output_w
    
    if batch_idx >= batch_size:
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over groups
    for g in range(conv_groups):
        # Calculate group-specific indices
        in_c_per_group = in_channels // conv_groups
        out_c_per_group = out_channels // conv_groups
        
        # Calculate output channel range for this group
        out_c_start = g * out_c_per_group
        out_c_end = (g + 1) * out_c_per_group
        
        # Loop over kernel elements
        for k in range(kh * kw):
            kh_idx = k // kw
            kw_idx = k % kw
            
            # Calculate input indices
            ih_start = h_idx * conv_stride_h - conv_padding_h
            iw_start = w_idx * conv_stride_w - conv_padding_w
            
            ih_k = ih_start + kh_idx * conv_dilation_h
            iw_k = iw_start + kw_idx * conv_dilation_w
            
            # Check bounds
            if ih_k >= 0 and ih_k < ih and iw_k >= 0 and iw_k < iw:
                # Load input and weight
                input_offset = batch_idx * (in_channels * ih * iw) + g * in_c_per_group * ih * iw + ih_k * iw + iw_k
                weight_offset = out_c_start * (in_c_per_group * kh * kw) + k
                
                # Load input and weight
                input_val = tl.load(input_ptr + input_offset, mask=True)
                weight_val = tl.load(weight_ptr + weight_offset, mask=True)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        for c in range(out_c_start, out_c_end):
            bias_val = tl.load(bias_ptr + c, mask=True)
            acc += bias_val
    
    # Store result
    output_offset = batch_idx * (out_channels * output_h * output_w) + out_c_start * (output_h * output_w) + h_idx * output_w + w_idx
    tl.store(output_ptr + output_offset, acc)

def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
    pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    # Handle scalar inputs
    if not isinstance(conv_stride, (tuple, list)):
        conv_stride = (conv_stride, conv_stride)
    if not isinstance(conv_padding, (tuple, list)):
        conv_padding = (conv_padding, conv_padding)
    if not isinstance(conv_dilation, (tuple, list)):
        conv_dilation = (conv_dilation, conv_dilation)
    if not isinstance(pool_kernel_size, (tuple, list)):
        pool_kernel_size = (pool_kernel_size, pool_kernel_size)
    if pool_stride is None:
        pool_stride = pool_kernel_size
    if not isinstance(pool_stride, (tuple, list)):
        pool_stride = (pool_stride, pool_stride)
    if not isinstance(pool_padding, (tuple, list)):
        pool_padding = (pool_padding, pool_padding)
    if not isinstance(pool_dilation, (tuple, list)):
        pool_dilation = (pool_dilation, pool_dilation)
    
    # Convolution parameters
    conv_stride_h, conv_stride_w = conv_stride
    conv_padding_h, conv_padding_w = conv_padding
    conv_dilation_h, conv_dilation_w = conv_dilation
    
    # Pooling parameters
    pool_kernel_h, pool_kernel_w = pool_kernel_size
    pool_stride_h, pool_stride_w = pool_stride
    pool_padding_h, pool_padding_w = pool_padding
    pool_dilation_h, pool_dilation_w = pool_dilation
    
    # Perform convolution
    conv_out = torch.nn.functional.conv2d(
        input, weight, bias, conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    # Perform max pooling
    pool_out = torch.nn.functional.max_pool2d(
        conv_out, pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode
    )
    
    # Apply ReLU
    if inplace:
        pool_out = torch.nn.functional.relu_(pool_out)
    else:
        pool_out = torch.nn.functional.relu(pool_out)
    
    return pool_out
