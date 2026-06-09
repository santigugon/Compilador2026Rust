import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    A_ptr, vec_ptr, other_ptr, out_ptr,
    n, m,
    alpha,
    stride_a_row, stride_a_col,
    stride_vec,
    stride_other,
    stride_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(n, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < n
    mask_n = offs_n < m
    
    a_ptrs = A_ptr + offs_m[:, None] * stride_a_row + offs_n[None, :] * stride_a_col
    vec_ptrs = vec_ptr + offs_n * stride_vec
    
    a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)
    vec = tl.load(vec_ptrs, mask=mask_n, other=0.0)
    
    # Matrix-vector multiplication
    dot = tl.sum(a * vec[None, :], axis=1)
    
    # Sigmoid activation
    sigmoid_dot = 1.0 / (1.0 + tl.exp(-dot))
    
    # Subtract alpha * other
    other_ptrs = other_ptr + offs_m * stride_other
    other = tl.load(other_ptrs, mask=mask_m, other=0.0)
    result = sigmoid_dot - alpha * other
    
    # Store result
    out_ptrs = out_ptr + offs_m * stride_out
    tl.store(out_ptrs, result, mask=mask_m)


def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec dimensions must match"
    
    n, m = input.shape
    
    if out is None:
        out = torch.empty(n, dtype=torch.float32, device=input.device)
    
    # Ensure other is a tensor
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    
    if other.dim() == 0:
        other = other.expand(n)
    elif other.dim() == 1:
        assert other.size(0) == n, "other must have the same size as the first dimension of input"
    else:
        raise ValueError("other must be a scalar or 1D tensor")
    
    # Launch kernel
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    num_warps = 4
    
    grid = (triton.cdiv(n, BLOCK_SIZE_M),)
    
    _fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m,
        alpha,
        input.stride(0), input.stride(1),
        vec.stride(0),
        other.stride(0),
        out.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps
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
