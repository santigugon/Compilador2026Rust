import torch
import triton
import triton.language as tl

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Handle in-place operation
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Linear transformation
    if bias is not None:
        linear_out = torch.nn.functional.linear(input, weight, bias)
    else:
        linear_out = torch.nn.functional.linear(input, weight)
    
    # Apply sigmoid
    sigmoid_out = torch.sigmoid(linear_out)
    
    # Apply dropout if in training mode
    if training and p > 0:
        # Create dropout mask
        dropout_mask = (torch.rand_like(sigmoid_out) > p).to(torch.float32)
        # Apply dropout
        output = sigmoid_out * dropout_mask / (1.0 - p)
    else:
        output = sigmoid_out
    
    return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
#     """
#     Applies a linear transformation followed by a sigmoid activation and dropout.

#     Args:
#         input (torch.Tensor): Input tensor of shape (*, in_features).
#         weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
#         bias (torch.Tensor, optional): Bias tensor of shape (out_features). Default: None.
#         p (float, optional): Probability of an element to be zeroed in dropout. Default: 0.5.
#         training (bool, optional): If True, applies dropout during training. Default: True.
#         inplace (bool, optional): If True, performs the operation in-place. Default: False.

#     Returns:
#         torch.Tensor: The resulting tensor after applying the linear transformation, sigmoid activation, and dropout.
#     """
#     output = F.linear(input, weight, bias)
#     output = torch.sigmoid(output)
#     if training:
#         output = F.dropout(output, p=p, training=training, inplace=inplace)
#     return output

def test_dropout_sigmoid_linear():
    results = {}
    
    # Test case 1: Basic test with bias, training=True, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_1"] = dropout_sigmoid_linear(input, weight, bias)
    
    # Test case 2: No bias, training=True, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    results["test_case_2"] = dropout_sigmoid_linear(input, weight)
    
    # Test case 3: With bias, training=False, inplace=False
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_3"] = dropout_sigmoid_linear(input, weight, bias, training=False)
    
    # Test case 4: With bias, training=True, inplace=True
    input = torch.randn(2, 3, device='cuda')
    weight = torch.randn(4, 3, device='cuda')
    bias = torch.randn(4, device='cuda')
    results["test_case_4"] = dropout_sigmoid_linear(input, weight, bias, inplace=True)
    
    return results

test_results = test_dropout_sigmoid_linear()
