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