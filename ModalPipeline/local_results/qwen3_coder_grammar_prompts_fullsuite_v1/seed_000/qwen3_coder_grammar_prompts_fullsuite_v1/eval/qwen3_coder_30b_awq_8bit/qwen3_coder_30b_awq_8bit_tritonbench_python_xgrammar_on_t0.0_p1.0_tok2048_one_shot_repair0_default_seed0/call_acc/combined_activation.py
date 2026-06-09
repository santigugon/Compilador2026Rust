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
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + offs_m[:, None] * input_row_stride + offs_k[None, :] * input_col_stride
    weight1_ptrs = weight1_ptr + offs_k[:, None] * weight1_row_stride + offs_n[None, :] * weight1_col_stride
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(input_ptrs)
        b = tl.load(weight1_ptrs)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K * input_col_stride
        weight1_ptrs += BLOCK_SIZE_K * weight1_row_stride
    
    output_ptrs = output_ptr + offs_m[:, None] * output_row_stride + offs_n[None, :] * output_col_stride
    
    if ACTIVATION_TYPE == 0:
        # Sigmoid
        result = tl.sigmoid(accumulator)
    elif ACTIVATION_TYPE == 1:
        # Tanh
        result = tl.tanh(accumulator)
    else:
        # Identity
        result = accumulator
    
    # Apply weight2 and bias
    weight2_ptrs = weight2_ptr + offs_n[None, :] * weight2_col_stride
    bias_ptrs = bias_ptr + offs_n[None, :] * bias_col_stride
    
    weight2_vals = tl.load(weight2_ptrs)
    bias_vals = tl.load(bias_ptrs)
    
    result = result * weight2_vals + bias_vals
    
    tl.store(output_ptrs, result)

@triton.jit
def combined_activation_kernel_batched(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, output_ptr,
    input_row_stride, input_col_stride,
    weight1_row_stride, weight1_col_stride,
    weight2_row_stride, weight2_col_stride,
    bias_row_stride, bias_col_stride,
    output_row_stride, output_col_stride,
    batch_size, M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N))
    tile_id = pid % (tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N))
    
    pid_m = tile_id // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_SIZE_N)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + batch_id * input_row_stride + offs_m[:, None] * input_row_stride + offs_k[None, :] * input_col_stride
    weight1_ptrs = weight1_ptr + offs_k[:, None] * weight1_row_stride + offs_n[None, :] * weight1_col_stride
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(input_ptrs)
        b = tl.load(weight1_ptrs)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K * input_col_stride
        weight1_ptrs += BLOCK_SIZE_K * weight1_row_stride
    
    output_ptrs = output_ptr + batch_id * output_row_stride + offs_m[:, None] * output_row_stride + offs_n[None, :] * output_col_stride
    
    if ACTIVATION_TYPE == 0:
        # Sigmoid
        result = tl.sigmoid(accumulator)
    elif ACTIVATION_TYPE == 1:
        # Tanh
        result = tl.tanh(accumulator)
    else:
        # Identity
        result = accumulator
    
    # Apply weight2 and bias
    weight2_ptrs = weight2_ptr + offs_n[None, :] * weight2_col_stride
    bias_ptrs = bias_ptr + offs_n[None, :] * bias_col_stride
    
    weight2_vals = tl.load(weight2_ptrs)
    bias_vals = tl.load(bias_ptrs)
    
    result = result * weight2_vals + bias_vals
    
    tl.store(output_ptrs, result)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    if out is None:
        out = torch.empty(input.shape[:-1] + (weight1.shape[1],), dtype=input.dtype, device=input.device)
    
    # Ensure input and weight1 are compatible
    assert input.shape[-1] == weight1.shape[0], "Input and weight1 dimensions do not match"
    
    # Ensure weight2 and bias are broadcastable
    assert weight2.shape[-1] == weight1.shape[1], "Weight2 and weight1 dimensions do not match"
    assert bias.shape[-1] == weight1.shape[1], "Bias and weight1 dimensions do not match"
    
    # Flatten batch dimensions
    input_flat = input.view(-1, input.shape[-1])
    out_flat = out.view(-1, out.shape[-1])
    
    M, K = input_flat.shape
    N = weight1.shape[1]
    
    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    combined_activation_kernel[grid](
        input_flat, weight1, weight2, bias, out_flat,
        input_flat.stride(0), input_flat.stride(1),
        weight1.stride(0), weight1.stride(1),
        weight2.stride(0), weight2.stride(1),
        bias.stride(0), bias.stride(1),
        out_flat.stride(0), out_flat.stride(1),
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        0  # ACTIVATION_TYPE: 0 for sigmoid
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
