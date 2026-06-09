import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple

@triton.jit
def _gelu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Calculate output dimensions
    output_h = oH
    output_w = oW

    # Load input and weight
    input_block = tl.load(input_ptr + pid_h * BLOCK_SIZE_H + pid_w * BLOCK_SIZE_W + pid_c * BLOCK_SIZE_C, mask=tl.arange(0, BLOCK_SIZE_H) < output_h & tl.arange(0, BLOCK_SIZE_W) < output_w & tl.arange(0, BLOCK_SIZE_C) < in_channels)
    weight_block = tl.load(weight_ptr + pid_c * kH * kW, mask=tl.arange(0, kH) < kH & tl.arange(0, kW) < kW)

    # Perform convolution
    output = 0.0
    for kh in range(kH):
        for kw in range(kW):
            input_val = input_block[0]
            weight_val = weight_block[kh * kW + kw]
            output += input_val * weight_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_c, mask=pid_c < out_channels)
        output += bias_val

    # Apply GELU activation
    output = 0.5 * output * (1.0 + tl.erf(output / tl.sqrt(2.0)))

    # Store result
    tl.store(output_ptr + pid_h * BLOCK_SIZE_H + pid_w * BLOCK_SIZE_W + pid_c * BLOCK_SIZE_C, output)

def gelu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = 'none',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h = pad_w = 0
        elif padding == 'same':
            pad_h = pad_w = 0
        else:
            raise ValueError("Invalid padding string")
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    # Handle dilation
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    # Get input dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1

    # Create output tensor
    if out is None:
        out = torch.empty(minibatch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (minibatch, out_channels, oH, oW)

    # Handle groups
    if groups > 1:
        # For simplicity, we assume groups=1 for this implementation
        raise NotImplementedError("Groups > 1 not supported in this implementation")

    # Launch kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    grid = (
        triton.cdiv(oH, BLOCK_SIZE_H),
        triton.cdiv(oW, BLOCK_SIZE_W),
        triton.cdiv(out_channels, BLOCK_SIZE_C)
    )

    # Create pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = out.data_ptr()

    # Call kernel
    _gelu_conv2d_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        iH, iW, oH, oW, in_channels, out_channels, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    return out