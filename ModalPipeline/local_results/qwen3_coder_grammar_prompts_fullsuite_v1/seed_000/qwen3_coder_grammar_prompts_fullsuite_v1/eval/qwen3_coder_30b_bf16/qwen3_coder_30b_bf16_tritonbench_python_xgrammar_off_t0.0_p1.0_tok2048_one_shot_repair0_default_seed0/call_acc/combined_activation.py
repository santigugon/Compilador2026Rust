import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, output_ptr,
    input_row_stride, input_col_stride,
    weight1_row_stride, weight1_col_stride,
    weight2_row_stride, weight2_col_stride,
    bias_row_stride, bias_col_stride,
    output_row_stride, output_col_stride,
    N, D_in, D_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE)
    num_pid_d = tl.cdiv(D_out, BLOCK_SIZE)
    pid_n = pid // num_pid_d
    pid_d = pid % num_pid_d
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(N, D_in),
        strides=(input_row_stride, input_col_stride),
        offsets=(pid_n * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, D_in),
        order=(1, 0)
    )
    weight1_block_ptr = tl.make_block_ptr(
        base=weight1_ptr,
        shape=(D_in, D_out),
        strides=(weight1_row_stride, weight1_col_stride),
        offsets=(0, pid_d * BLOCK_SIZE),
        block_shape=(D_in, BLOCK_SIZE),
        order=(0, 1)
    )
    input_data = tl.load(input_block_ptr)
    weight1_data = tl.load(weight1_block_ptr)
    acc = tl.dot(input_data, weight1_data)
    acc = tl.sigmoid(acc) * tl.tanh(acc)
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(D_out,),
        strides=(bias_row_stride,),
        offsets=(pid_d * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    bias_data = tl.load(bias_block_ptr)
    acc += bias_data
    weight2_block_ptr = tl.make_block_ptr(
        base=weight2_ptr,
        shape=(D_out,),
        strides=(weight2_row_stride,),
        offsets=(pid_d * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    weight2_data = tl.load(weight2_block_ptr)
    acc *= weight2_data
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, D_out),
        strides=(output_row_stride, output_col_stride),
        offsets=(pid_n * BLOCK_SIZE, pid_d * BLOCK_SIZE),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(1, 0)
    )
    tl.store(output_block_ptr, acc)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    if out is None:
        out = torch.empty(input.shape[:-1] + (weight1.shape[1],), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == input.shape[:-1] + (weight1.shape[1],)
        assert out.dtype == input.dtype
        assert out.device == input.device
    N, D_in = input.shape[-2], input.shape[-1]
    D_out = weight1.shape[1]
    assert weight1.shape == (D_in, D_out)
    assert weight2.shape == (D_out,)
    assert bias.shape == (D_out,)
    assert input.shape[-1] == weight1.shape[0]
    BLOCK_SIZE = 32
    num_pid_n = triton.cdiv(N, BLOCK_SIZE)
    num_pid_d = triton.cdiv(D_out, BLOCK_SIZE)
    grid = (num_pid_n * num_pid_d,)
    input_ptr = input.data_ptr()
    weight1_ptr = weight1.data_ptr()
    weight2_ptr = weight2.data_ptr()
    bias_ptr = bias.data_ptr()
    output_ptr = out.data_ptr()
    input_row_stride = input.stride(-2) if input.ndim > 1 else 0
    input_col_stride = input.stride(-1) if input.ndim > 1 else 0
    weight1_row_stride = weight1.stride(0)
    weight1_col_stride = weight1.stride(1)
    weight2_row_stride = weight2.stride(0)
    bias_row_stride = bias.stride(0)
    output_row_stride = out.stride(-2) if out.ndim > 1 else 0
    output_col_stride = out.stride(-1) if out.ndim > 1 else 0
    combined_activation_kernel[grid](
        input_ptr, weight1_ptr, weight2_ptr, bias_ptr, output_ptr,
        input_row_stride, input_col_stride,
        weight1_row_stride, weight1_col_stride,
        weight2_row_stride, weight2_col_stride,
        bias_row_stride, bias_col_stride,
        output_row_stride, output_col_stride,
        N, D_in, D_out,
        BLOCK_SIZE
    )
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def combined_activation(input, weight1, weight2, bias, *, out=None):
#     """
#     Perform the combined activation function which includes matrix multiplication,
#     sigmoid, tanh, element-wise multiplication, and addition.

#     Args:
#         input (Tensor): Input tensor of shape (*, N, D_in), where * denotes any batch dimensions.
#         weight1 (Tensor): Weight matrix of shape (D_in, D_out).
#         weight2 (Tensor): Weight tensor for element-wise multiplication, must be broadcastable 
#                           to the shape of the intermediate activation.
#         bias (Tensor): Bias tensor, must be broadcastable to the shape of the output.
#         out (Tensor, optional): Output tensor to store the result, ignored if None.

#     Returns:
#         Tensor: Output tensor of shape (*, N, D_out).
#     """
#     z = torch.mm(input, weight1)
#     s = torch.sigmoid(z)
#     t = torch.tanh(s)
#     m = t * weight2
#     y = m + bias
#     if out is not None:
#         out.copy_(y)
#         return out
#     return y

def test_combined_activation():
    results = {}

    # Test case 1
    input1 = torch.randn(2, 3, device='cuda')
    weight1_1 = torch.randn(3, 4, device='cuda')
    weight2_1 = torch.randn(2, 4, device='cuda')
    bias1 = torch.randn(2, 4, device='cuda')
    results["test_case_1"] = combined_activation(input1, weight1_1, weight2_1, bias1)

    # Test case 2
    input2 = torch.randn(3, 3, device='cuda')
    weight1_2 = torch.randn(3, 5, device='cuda')
    weight2_2 = torch.randn(3, 5, device='cuda')
    bias2 = torch.randn(3, 5, device='cuda')
    results["test_case_2"] = combined_activation(input2, weight1_2, weight2_2, bias2)

    # Test case 3
    input3 = torch.randn(4, 3, device='cuda')
    weight1_3 = torch.randn(3, 6, device='cuda')
    weight2_3 = torch.randn(4, 6, device='cuda')
    bias3 = torch.randn(4, 6, device='cuda')
    results["test_case_3"] = combined_activation(input3, weight1_3, weight2_3, bias3)

    # Test case 4
    input4 = torch.randn(5, 3, device='cuda')
    weight1_4 = torch.randn(3, 7, device='cuda')
    weight2_4 = torch.randn(5, 7, device='cuda')
    bias4 = torch.randn(5, 7, device='cuda')
    results["test_case_4"] = combined_activation(input4, weight1_4, weight2_4, bias4)

    return results

test_results = test_combined_activation()
