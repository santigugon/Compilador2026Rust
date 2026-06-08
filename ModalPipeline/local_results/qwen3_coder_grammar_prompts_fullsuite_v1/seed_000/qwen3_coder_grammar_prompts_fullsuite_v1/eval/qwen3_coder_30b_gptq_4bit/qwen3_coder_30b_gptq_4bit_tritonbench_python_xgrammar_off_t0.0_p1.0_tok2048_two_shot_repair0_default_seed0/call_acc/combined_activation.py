import torch
import triton
import triton.language as tl

@triton.jit
def _combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
    n_batch: tl.constexpr, n_in: tl.constexpr, n_out: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // n_out
    out_id = pid % n_out
    
    # Compute matrix multiplication for this batch and output
    acc = 0.0
    for i in range(0, n_in, BLOCK):
        # Load input and weight1 blocks
        input_offsets = batch_id * n_in + i + tl.arange(0, BLOCK)
        weight1_offsets = i + out_id * n_in + tl.arange(0, BLOCK)
        
        mask = (input_offsets < (batch_id + 1) * n_in) & (weight1_offsets < n_in * n_out)
        
        input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        weight1_vals = tl.load(weight1_ptr + weight1_offsets, mask=mask, other=0.0)
        
        acc += tl.sum(input_vals * weight1_vals)
    
    # Compute activation functions
    # Apply sigmoid and tanh
    activation = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0  # tanh
    activation = 1.0 / (1.0 + tl.exp(-activation))  # sigmoid
    
    # Element-wise multiplication with weight2
    # weight2 is broadcastable to output shape, so we compute it per batch
    weight2_val = tl.load(weight2_ptr + out_id, mask=out_id < n_out, other=0.0)
    activation = activation * weight2_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + out_id, mask=out_id < n_out, other=0.0)
    activation = activation + bias_val
    
    # Store result
    out_offsets = batch_id * n_out + out_id
    tl.store(out_ptr + out_offsets, activation, mask=out_id < n_out)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate dimensions
    batch_shape = input.shape[:-2]
    n_in = input.shape[-1]
    n_out = weight1.shape[-1]
    
    # Check compatibility
    assert weight1.shape[-2] == n_in, "Weight1 dimension mismatch"
    assert weight2.shape[-1] == n_out, "Weight2 dimension mismatch"
    assert bias.shape[-1] == n_out, "Bias dimension mismatch"
    
    # Compute output shape
    output_shape = batch_shape + (n_out,)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output tensor shape mismatch"
    
    # Handle batch dimensions
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Launch kernel
    block = 32
    grid = (batch_size * n_out,)
    
    # Flatten input and weight1 for kernel processing
    input_flat = input.view(-1, n_in)
    weight1_flat = weight1.view(n_in, n_out)
    weight2_flat = weight2.view(-1, n_out)
    bias_flat = bias.view(-1, n_out)
    
    # Create output tensor with flattened shape
    out_flat = out.view(-1, n_out)
    
    # Launch kernel
    _combined_activation_kernel[grid](
        input_flat, weight1_flat, weight2_flat, bias_flat, out_flat,
        batch_size, n_in, n_out, BLOCK=block
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
