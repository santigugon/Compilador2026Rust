import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

@triton.jit
def conv2d_gelu_kernel(
    input_ptr,  # pointer to input tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor
    output_ptr,  # pointer to output tensor
    input_shape,  # (batch, in_channels, height, width)
    weight_shape,  # (out_channels, in_channels, kH, kW)
    bias_shape,  # (out_channels)
    stride_h,  # stride height
    stride_w,  # stride width
    padding_h,  # padding height
    padding_w,  # padding width
    dilation_h,  # dilation height
    dilation_w,  # dilation width
    groups,  # number of groups
    out_channels,  # number of output channels
    in_channels,  # number of input channels
    batch_size,  # batch size
    iH,  # input height
    iW,  # input width
    oH,  # output height
    oW,  # output width
    kH,  # kernel height
    kW,  # kernel width
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(oW, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(oH, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute the output position
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Loop over the kernel
    for k in range(0, kH * kW * in_channels // groups, BLOCK_SIZE_K):
        # Load input and weight
        input_offset = (m_start * stride_h - padding_h) * iW + (n_start * stride_w - padding_w)
        weight_offset = k
        
        # Load input
        input_tile = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * iW + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < oH) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < oW))
        
        # Load weight
        weight_tile = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE_K)[None, :], mask=tl.arange(0, BLOCK_SIZE_K)[None, :] < kH * kW * in_channels // groups)
        
        # Compute convolution
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias
    if bias_ptr is not None:
        bias_tile = tl.load(bias_ptr + tl.arange(0, out_channels))
        acc += bias_tile[:, None, None]
    
    # Apply GELU
    acc = gelu(acc)
    
    # Store output
    output_offset = m_start * oW + n_start
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * oW + tl.arange(0, BLOCK_SIZE_N)[None, :], acc)


def gelu_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Handle stride
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride
    
    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            padding_h, padding_w = 0, 0
        elif padding == 'same':
            # For simplicity, we'll assume padding is calculated
            padding_h, padding_w = 0, 0
        else:
            raise ValueError("Invalid padding type")
    elif isinstance(padding, int):
        padding_h, padding_w = padding, padding
    else:
        padding_h, padding_w = padding
    
    # Handle dilation
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Get tensor shapes
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    oH = (iH + 2 * padding_h - (kH - 1) * dilation_h - 1) // stride_h + 1
    oW = (iW + 2 * padding_w - (kW - 1) * dilation_w - 1) // stride_w + 1
    
    # Create output tensor
    if out is None:
        out = torch.empty((batch_size, out_channels, oH, oW), dtype=input.dtype, device=input.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Launch kernel
    grid = (triton.cdiv(oH, BLOCK_SIZE_M) * triton.cdiv(oW, BLOCK_SIZE_N),)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = out.data_ptr()
    
    # Launch kernel
    conv2d_gelu_kernel[grid](
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        input.shape,
        weight.shape,
        bias.shape if bias is not None else (0,),
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        out_channels,
        in_channels,
        batch_size,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out