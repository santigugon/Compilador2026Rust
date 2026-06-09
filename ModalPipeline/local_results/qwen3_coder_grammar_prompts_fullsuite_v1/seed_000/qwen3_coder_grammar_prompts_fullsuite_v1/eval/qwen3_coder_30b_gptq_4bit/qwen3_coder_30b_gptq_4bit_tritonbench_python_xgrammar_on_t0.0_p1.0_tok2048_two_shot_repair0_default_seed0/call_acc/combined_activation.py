import torch
import triton
import triton.language as tl

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Check dimensions for compatibility
    assert input.shape[-1] == weight1.shape[-2], "Input and weight1 dimensions incompatible for matrix multiplication"
    
    # Compute intermediate activation: input @ weight1
    intermediate = torch.matmul(input, weight1)
    
    # Apply sigmoid and tanh
    sigmoid_out = torch.sigmoid(intermediate)
    tanh_out = torch.tanh(intermediate)
    
    # Element-wise multiplication with weight2
    if weight2 is not None:
        # Handle broadcasting
        if weight2.shape == (1,):  # scalar
            weight2 = weight2.expand_as(sigmoid_out)
        elif weight2.shape[-1] == 1:  # broadcastable
            weight2 = weight2.expand_as(sigmoid_out)
        elif weight2.shape[-1] == sigmoid_out.shape[-1]:  # same size
            pass
        else:
            raise ValueError("weight2 is not broadcastable to intermediate activation shape")
        
        # Apply element-wise multiplication
        combined = sigmoid_out * weight2
    else:
        combined = sigmoid_out
    
    # Add bias
    if bias is not None:
        # Handle broadcasting
        if bias.shape == (1,):  # scalar
            bias = bias.expand_as(combined)
        elif bias.shape[-1] == 1:  # broadcastable
            bias = bias.expand_as(combined)
        elif bias.shape[-1] == combined.shape[-1]:  # same size
            pass
        else:
            raise ValueError("bias is not broadcastable to combined activation shape")
        
        combined = combined + bias
    
    # Return result
    if out is not None:
        out.copy_(combined)
        return out
    else:
        return combined
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
