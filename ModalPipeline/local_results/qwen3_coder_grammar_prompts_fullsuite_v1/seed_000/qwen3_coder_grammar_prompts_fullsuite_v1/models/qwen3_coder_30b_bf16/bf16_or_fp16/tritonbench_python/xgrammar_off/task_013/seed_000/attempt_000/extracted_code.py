import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (C_out * H * W)
    out_c = (pid % (C_out * H * W)) // (H * W)
    out_h = (pid % (C_out * H * W)) // W % H
    out_w = pid % W
    
    if batch_id >= N or out_c >= C_out or out_h >= H or out_w >= W:
        return
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for c in range(C_in):
        for kh in range(kH):
            for kw in range(kW):
                ih = out_h * stride_h - padding_h + kh * dilation_h
                iw = out_w * stride_w - padding_w + kw * dilation_w
                
                if 0 <= ih < input_stride_2 and 0 <= iw < input_stride_3:
                    input_val = tl.load(input_ptr + 
                                       batch_id * input_stride_0 + 
                                       c * input_stride_1 + 
                                       ih * input_stride_2 + 
                                       iw * input_stride_3)
                    weight_val = tl.load(weight_ptr + 
                                        out_c * weight_stride_0 + 
                                        c * weight_stride_1 + 
                                        kh * weight_stride_2 + 
                                        kw * weight_stride_3)
                    acc += input_val * weight_val
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c)
        acc += bias_val
    
    tl.store(output_ptr + 
             batch_id * output_stride_0 + 
             out_c * output_stride_1 + 
             out_h * output_stride_2 + 
             out_w * output_stride_3, 
             acc)

@triton.jit
def _batch_norm_kernel(
    input_ptr, output_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    N, C, H, W, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (C * H * W)
    channel_id = (pid % (C * H * W)) // (H * W)
    h = (pid % (C * H * W)) // W % H
    w = pid % W
    
    if batch_id >= N or channel_id >= C or h >= H or w >= W:
        return
    
    mean_val = tl.load(mean_ptr + channel_id)
    var_val = tl.load(var_ptr + channel_id)
    weight_val = tl.load(weight_ptr + channel_id)
    bias_val = tl.load(bias_ptr + channel_id)
    
    input_val = tl.load(input_ptr + 
                       batch_id * input_stride_0 + 
                       channel_id * input_stride_1 + 
                       h * input_stride_2 + 
                       w * input_stride_3)
    
    normalized = (input_val - mean_val) / tl.sqrt(var_val + eps)
    output_val = weight_val * normalized + bias_val
    
    tl.store(output_ptr + 
             batch_id * output_stride_0 + 
             channel_id * output_stride_1 + 
             h * output_stride_2 + 
             w * output_stride_3, 
             output_val)

@triton.jit
def _relu_kernel(
    input_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    N, C, H, W, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (C * H * W)
    channel_id = (pid % (C * H * W)) // (H * W)
    h = (pid % (C * H * W)) // W % H
    w = pid % W
    
    if batch_id >= N or channel_id >= C or h >= H or w >= W:
        return
    
    input_val = tl.load(input_ptr + 
                       batch_id * input_stride_0 + 
                       channel_id * input_stride_1 + 
                       h * input_stride_2 + 
                       w * input_stride_3)
    
    output_val = tl.maximum(input_val, 0.0)
    
    tl.store(output_ptr + 
             batch_id * output_stride_0 + 
             channel_id * output_stride_1 + 
             h * output_stride_2 + 
             w * output_stride_3, 
             output_val)

@triton.jit
def _dropout_kernel(
    input_ptr, output_ptr, mask_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    N, C, H, W, p: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (C * H * W)
    channel_id = (pid % (C * H * W)) // (H * W)
    h = (pid % (C * H * W)) // W % H
    w = pid % W
    
    if batch_id >= N or channel_id >= C or h >= H or w >= W:
        return
    
    input_val = tl.load(input_ptr + 
                       batch_id * input_stride_0 + 
                       channel_id * input_stride_1 + 
                       h * input_stride_2 + 
                       w * input_stride_3)
    
    mask_val = tl.rand() > p
    output_val = input_val * mask_val / (1.0 - p)
    
    tl.store(output_ptr + 
             batch_id * output_stride_0 + 
             channel_id * output_stride_1 + 
             h * output_stride_2 + 
             w * output_stride_3, 
             output_val)
    
    tl.store(mask_ptr + 
             batch_id * input_stride_0 + 
             channel_id * input_stride_1 + 
             h * input_stride_2 + 
             w * input_stride_3, 
             mask_val)

def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor, 
    weight: torch.Tensor, 
    bias=None, 
    stride=1, 
    padding=0, 
    dilation=1, 
    groups=1, 
    p=0.5, 
    training=True, 
    inplace=False
) -> torch.Tensor:
    # Get dimensions
    N, C_in, H, W = input.shape
    C_out, _, kH, kW = weight.shape
    
    # Handle stride, padding, dilation, groups
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Compute output dimensions
    H_out = (H + 2 * padding_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    W_out = (W + 2 * padding_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Allocate output tensor
    output = torch.empty(N, C_out, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Conv2d
    conv_output = torch.empty(N, C_out, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Conv2d kernel launch
    BLOCK_SIZE = 256
    grid_size = (N * C_out * H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _conv2d_kernel[grid_size](
        input, weight, bias, conv_output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        conv_output.stride(0), conv_output.stride(1), conv_output.stride(2), conv_output.stride(3),
        N, C_in, H_out, W_out, C_out, kH, kW, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Batch normalization
    # Compute mean and variance
