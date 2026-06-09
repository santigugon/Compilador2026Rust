import torch
import triton
import triton.language as tl

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(other):
        alpha = alpha
        other_scalar = other
        other = torch.tensor(other_scalar, dtype=input.dtype, device=input.device)
    else:
        alpha = alpha
        other_scalar = None

    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            padding = (0, 0)
        elif padding == 'same':
            # For 'same' padding, we compute padding to make output size equal to input size
            # This is a simplified approach; in practice, more complex logic may be needed
            padding = (0, 0)
        else:
            raise ValueError("Padding must be 'valid', 'same', or a tuple of integers")
    elif isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2:
        pass
    else:
        raise ValueError("Padding must be an integer, 'valid', 'same', or a tuple of two integers")

    # Handle stride
    if isinstance(stride, int):
        stride = (stride, stride)
    elif isinstance(stride, tuple) and len(stride) == 2:
        pass
    else:
        raise ValueError("Stride must be an integer or a tuple of two integers")

    # Handle dilation
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif isinstance(dilation, tuple) and len(dilation) == 2:
        pass
    else:
        raise ValueError("Dilation must be an integer or a tuple of two integers")

    # Get dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Compute output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation

    oH = (iH + 2 * padH - (dilationH * (kH - 1) + 1)) // strideH + 1
    oW = (iW + 2 * padW - (dilationW * (kW - 1) + 1)) // strideW + 1

    # Initialize output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty((batch_size, out_channels, oH, oW), dtype=input.dtype, device=input.device)

    # Apply convolution
    _conv2d_kernel(input, weight, output, bias, stride, padding, dilation, groups)

    # Add other tensor or scalar
    if other is not None:
        if other_scalar is not None:
            output += alpha * other_scalar
        else:
            output += alpha * other

    return output

@triton.jit
def _conv2d_kernel(input_ptr, weight_ptr, output_ptr, bias_ptr, stride, padding, dilation, groups):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)

    # Get input dimensions
    batch_size, in_channels, iH, iW = tl.load(input_ptr + 0).shape
    out_channels, _, kH, kW = tl.load(weight_ptr + 0).shape

    # Get stride, padding, dilation
    strideH, strideW = stride
    padH, padW = padding
    dilationH, dilationW = dilation

    # Get output dimensions
    oH = (iH + 2 * padH - (dilationH * (kH - 1) + 1)) // strideH + 1
    oW = (iW + 2 * padW - (dilationW * (kW - 1) + 1)) // strideW + 1

    # Get input and weight pointers
    input_batch_ptr = input_ptr + batch_idx * in_channels * iH * iW
    weight_channel_ptr = weight_ptr + out_channel_idx * (in_channels // groups) * kH * kW

    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)

    # Loop over input channels and kernel elements
    for c in range(in_channels // groups):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input indices
                ih = out_h_idx * strideH - padH + kh * dilationH
                iw = out_w_idx * strideW - padW + kw * dilationW

                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Load input value
                    input_val = tl.load(input_batch_ptr + c * iH * iW + ih * iW + iw)
                    # Load weight value
                    weight_val = tl.load(weight_channel_ptr + c * kH * kW + kh * kW + kw)
                    # Accumulate
                    acc += input_val * weight_val

    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val

    # Store result
    output_ptr += batch_idx * out_channels * oH * oW + out_channel_idx * oH * oW + out_h_idx * oW + out_w_idx
    tl.store(output_ptr, acc)
