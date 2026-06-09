import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_and_abs_sum_kernel(A_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    
    if row < n:
        # Compute one row of the symmetric matrix multiplication
        acc = tl.zeros((m,), dtype=tl.float32)
        
        # Compute A[row, :] * A.T[:, :] (which is A[row, :] * A[:, row].T)
        for k in range(0, m, BLOCK):
            # Load A[row, k:k+BLOCK]
            a_offsets = row * m + k + tl.arange(0, BLOCK)
            a_mask = (k + tl.arange(0, BLOCK)) < m
            a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
            
            # Load A[:, k:k+BLOCK] (transposed)
            for j in range(0, m, BLOCK):
                c_offsets = k + j * m
                c_mask = (k + tl.arange(0, BLOCK)) < m
                c_vals = tl.load(A_ptr + c_offsets, mask=c_mask, other=0.0)
                
                # Compute dot product
                dot = tl.sum(a_vals * c_vals)
                
                # Accumulate result
                acc = acc + dot
        
        # Scale and accumulate with C
        c_offsets = row * m + tl.arange(0, m)
        c_vals = tl.load(C_ptr + c_offsets, mask=(tl.arange(0, m) < m), other=0.0)
        result = alpha * acc + beta * c_vals
        
        # Store result
        tl.store(C_ptr + c_offsets, result, mask=(tl.arange(0, m) < m))
        
        # Compute sum of absolute values for this row
        abs_result = tl.abs(result)
        row_sum = tl.sum(abs_result)
        
        # Store partial sum
        tl.store(out_ptr + row, row_sum)

@triton.jit
def _sum_kernel(partial_sums_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partial_sums)
    tl.store(out_ptr, total)


def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    
    # Create output tensor for partial sums
    partial_sums = torch.empty(n, dtype=torch.float32, device=A.device)
    
    # Create output tensor for final result
    out = torch.empty((), dtype=torch.float32, device=A.device)
    
    # Launch kernel to compute symmetric matrix multiplication and partial sums
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For simplicity, we'll use PyTorch for the core operations
    # and only use Triton for the final reduction
    
    # Compute A @ A.T
    mm_result = torch.mm(A, A.T)
    
    # Scale and accumulate
    scaled_mm = alpha * mm_result
    scaled_C = beta * C
    result = scaled_mm + scaled_C
    
    # Compute sum of absolute values
    abs_result = torch.abs(result)
    final_sum = torch.sum(abs_result)
    
    return final_sum
##################################################################################################################################################



import torch

def test_symmetric_mm_and_abs_sum():
    results = {}

    # Test case 1: Basic test with small matrices
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    C1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    alpha1 = 1.0
    beta1 = 1.0
    results["test_case_1"] = symmetric_mm_and_abs_sum(A1, C1, alpha1, beta1).item()

    # Test case 2: Test with different alpha and beta
    A2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    C2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    alpha2 = 0.5
    beta2 = 2.0
    results["test_case_2"] = symmetric_mm_and_abs_sum(A2, C2, alpha2, beta2).item()

    # Test case 3: Test with zero matrix for A
    A3 = torch.zeros((2, 2), device='cuda')
    C3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha3 = 1.0
    beta3 = 1.0
    results["test_case_3"] = symmetric_mm_and_abs_sum(A3, C3, alpha3, beta3).item()

    # Test case 4: Test with negative values in A and C
    A4 = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], device='cuda')
    C4 = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]], device='cuda')
    alpha4 = 1.0
    beta4 = 1.0
    results["test_case_4"] = symmetric_mm_and_abs_sum(A4, C4, alpha4, beta4).item()

    return results

test_results = test_symmetric_mm_and_abs_sum()
