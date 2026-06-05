import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, output_ptr,
    n, m,
    alpha,
    input_stride_0, input_stride_1,
    vec_stride_0,
    other_stride_0,
    output_stride_0,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    if row >= n:
        return
    
    # Load vector
    vec = tl.load(vec_ptr + tl.arange(0, BLOCK_SIZE) * vec_stride_0)
    
    # Compute dot product
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, m, BLOCK_SIZE):
        mask = (tl.arange(0, BLOCK_SIZE) + i) < m
        input_vals = tl.load(input_ptr + row * input_stride_0 + (i + tl.arange(0, BLOCK_SIZE)) * input_stride_1, mask=mask)
        acc += input_vals * vec[tl.arange(0, BLOCK_SIZE)]
    
    # Compute sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    
    # Load other value
    other_val = tl.load(other_ptr)
    other_scaled = other_val * alpha
    
    # Subtract and store
    result = sigmoid_val - other_scaled
    tl.store(output_ptr + row * output_stride_0 + tl.arange(0, BLOCK_SIZE), result, mask=tl.arange(0, BLOCK_SIZE) < m)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    assert input.dim() == 2
    assert vec.dim() == 1
    assert input.size(1) == vec.size(0)
    
    n, m = input.shape
    
    if out is None:
        out = torch.empty(n, m, dtype=torch.float32, device=input.device)
    
    # Ensure tensors are contiguous
    input = input.contiguous()
    vec = vec.contiguous()
    out = out.contiguous()
    
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    else:
        other = other.contiguous()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n,)
    
    fused_mv_sigmoid_sub_kernel[grid](
        input_ptr=input.data_ptr(),
        vec_ptr=vec.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n=n,
        m=m,
        alpha=alpha,
        input_stride_0=input.stride(0),
        input_stride_1=input.stride(1),
        vec_stride_0=vec.stride(0),
        other_stride_0=other.stride(0),
        output_stride_0=out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
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
