import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr,
    alpha: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr,
    vec_stride_0: tl.constexpr,
    other_stride_0: tl.constexpr,
    out_stride_0: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each block handles one row of the output
    row = pid
    if row >= n:
        return
    
    # Initialize accumulator for matrix-vector multiplication
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    
    # Loop over the m dimension in chunks
    for k in range(0, m, BLOCK):
        # Load vector chunk
        vec_offsets = k + tl.arange(0, BLOCK)
        vec_mask = vec_offsets < m
        vec_vals = tl.load(vec_ptr + vec_offsets * vec_stride_0, mask=vec_mask, other=0.0)
        
        # Load input chunk
        input_offsets = row * input_stride_0 + vec_offsets * input_stride_1
        input_mask = vec_offsets < m
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Accumulate
        acc += input_vals * vec_vals
    
    # Reduce across the m dimension
    result = tl.sum(acc, axis=0)
    
    # Apply sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-result))
    
    # Load other value
    other_val = tl.load(other_ptr)
    
    # Apply subtraction with alpha
    final_result = sigmoid_result - alpha * other_val
    
    # Store result
    tl.store(out_ptr + row * out_stride_0, final_result)


def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Validate input shapes
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec dimensions must match"
    
    n, m = input.shape
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    
    # Create output tensor
    if out is None:
        out = torch.empty(n, dtype=torch.float32, device=input.device)
    else:
        assert out.shape == (n,), "out tensor must have shape (n,)"
        assert out.dtype == torch.float32, "out tensor must be float32"
        assert out.device == input.device, "out tensor must be on the same device as input"
    
    # Launch kernel
    block = 256
    grid = (n,)
    
    _fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m,
        alpha,
        input.stride(0), input.stride(1),
        vec.stride(0),
        other.stride(0) if other.dim() > 0 else 0,
        out.stride(0),
        BLOCK=block
    )
    
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
#     """
#     Performs a fused operation combining matrix-vector multiplication, sigmoid activation, and subtraction.

#     Args:
#         input (Tensor): Input matrix A of shape (n, m).
#         vec (Tensor): Input vector v of shape (m).
#         other (Tensor or Number): Tensor or scalar b to subtract from the sigmoid output, scaled by alpha.
#         alpha (Number, optional): Scalar multiplier for other. Default: 1.
#         out (Tensor, optional): Output tensor. Ignored if None. Default: None.

#     Returns:
#         Tensor: The result of the fused operation.
#     """
#     z = torch.mv(input, vec)
#     s = torch.sigmoid(z)
#     y = torch.sub(s, other, alpha=alpha)
#     if out is not None:
#         out.copy_(y)
#         return out
#     return y

def test_fused_mv_sigmoid_sub():
    results = {}
    
    # Test case 1: Basic functionality
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec1 = torch.tensor([1.0, 1.0], device='cuda')
    other1 = torch.tensor([0.5, 0.5], device='cuda')
    results["test_case_1"] = fused_mv_sigmoid_sub(input1, vec1, other1)
    
    # Test case 2: Scalar other
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec2 = torch.tensor([1.0, 1.0], device='cuda')
    other2 = 0.5
    results["test_case_2"] = fused_mv_sigmoid_sub(input2, vec2, other2)
    
    # Test case 3: Different alpha
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec3 = torch.tensor([1.0, 1.0], device='cuda')
    other3 = torch.tensor([0.5, 0.5], device='cuda')
    results["test_case_3"] = fused_mv_sigmoid_sub(input3, vec3, other3, alpha=2)
    
    # Test case 4: Output tensor provided
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    vec4 = torch.tensor([1.0, 1.0], device='cuda')
    other4 = torch.tensor([0.5, 0.5], device='cuda')
    out4 = torch.empty(2, device='cuda')
    results["test_case_4"] = fused_mv_sigmoid_sub(input4, vec4, other4, out=out4)
    
    return results

test_results = test_fused_mv_sigmoid_sub()
