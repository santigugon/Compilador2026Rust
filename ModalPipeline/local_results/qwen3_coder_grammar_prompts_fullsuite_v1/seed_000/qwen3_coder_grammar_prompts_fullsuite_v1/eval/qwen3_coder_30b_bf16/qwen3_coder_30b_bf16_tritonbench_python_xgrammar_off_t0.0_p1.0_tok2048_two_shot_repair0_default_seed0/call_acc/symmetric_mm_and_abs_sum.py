import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_and_abs_sum_kernel(A_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute the symmetric matrix multiplication and accumulation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Load A and C tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for boundaries
    mask_m = offs_m < n
    mask_n = offs_n < m
    
    # Load A tile
    a_tile = tl.load(A_ptr + offs_m[:, None] * m + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Compute A @ A.T for this tile
    # We'll compute the dot product of rows of A with columns of A
    # For symmetric case, we compute A @ A.T where result[i,j] = sum(A[i,:] * A[j,:])
    # But we only need to compute the upper triangular part and then mirror it
    
    # For simplicity, we'll compute the full matrix multiplication
    # and then accumulate with C
    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute A @ A.T
    for k in range(0, m, BLOCK_N):
        # Load A row slice
        a_row = tl.load(A_ptr + offs_m[:, None] * m + tl.arange(0, BLOCK_N) + k, mask=mask_m[:, None] & (tl.arange(0, BLOCK_N) + k < m), other=0.0)
        # Load A column slice (transposed)
        a_col = tl.load(A_ptr + (tl.arange(0, BLOCK_N) + k)[:, None] * m + offs_n[None, :], mask=(tl.arange(0, BLOCK_N) + k < m)[:, None] & mask_n[None, :], other=0.0)
        # Compute dot product
        result += tl.dot(a_row, a_col)
    
    # Scale by alpha
    result *= alpha
    
    # Load C tile
    c_tile = tl.load(C_ptr + offs_m[:, None] * m + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Accumulate with C scaled by beta
    result += c_tile * beta
    
    # Store the result back to C
    tl.store(C_ptr + offs_m[:, None] * m + offs_n[None, :], result, mask=mask_m[:, None] & mask_n[None, :])
    
    # Compute sum of absolute values
    abs_result = tl.abs(result)
    # Reduce across all elements in this tile
    tile_sum = tl.sum(abs_result)
    
    # Store partial sum
    tl.atomic_add(out_ptr, tile_sum)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    A = A.contiguous()
    C = C.contiguous()
    
    # Validate shapes
    assert A.dim() == 2, "A must be a 2D tensor"
    assert C.dim() == 2, "C must be a 2D tensor"
    assert A.shape == C.shape, "A and C must have the same shape"
    
    n, m = A.shape
    
    # Create output tensor for the sum
    out = torch.zeros(1, dtype=torch.float32, device=A.device)
    
    # Define block size
    BLOCK_M = 16
    BLOCK_N = 16
    
    # Grid size
    grid_m = triton.cdiv(n, BLOCK_M)
    grid_n = triton.cdiv(m, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    _symmetric_mm_and_abs_sum_kernel[grid](
        A, C, out, n, m, alpha, beta, BLOCK_M, BLOCK_N
    )
    
    return out

# Alternative implementation that's more straightforward
@triton.jit
def _symmetric_mm_kernel(A_ptr, C_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_row = row < n
    
    # Compute A @ A.T and accumulate with C
    for i in range(0, n, BLOCK_SIZE):
        # Compute partial dot products
        col = i + tl.arange(0, BLOCK_SIZE)
        mask_col = col < n
        
        # Load A row and column
        a_row = tl.load(A_ptr + row[:, None] * m + col[None, :], mask=mask_row[:, None] & mask_col[None, :], other=0.0)
        a_col = tl.load(A_ptr + col[:, None] * m + row[None, :], mask=mask_col[:, None] & mask_row[None, :], other=0.0)
        
        # Compute dot product
        result = tl.dot(a_row, a_col)
        
        # Scale and accumulate
        result *= alpha
        
        # Load C and accumulate
        c_vals = tl.load(C_ptr + row[:, None] * m + col[None, :], mask=mask_row[:, None] & mask_col[None, :], other=0.0)
        result += c_vals * beta
        
        # Store result
        tl.store(C_ptr + row[:, None] * m + col[None, :], result, mask=mask_row[:, None] & mask_col[None, :])

@triton.jit
def _abs_sum_kernel(C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_row = row < n
    
    # Compute sum of absolute values
    for i in range(0, m, BLOCK_SIZE):
        col = i + tl.arange(0, BLOCK_SIZE)
        mask_col = col < m
        
        # Load C values
        c_vals = tl.load(C_ptr + row[:, None] * m + col[None, :], mask=mask_row[:, None] & mask_col[None, :], other=0.0)
        
        # Compute absolute values and sum
        abs_vals = tl.abs(c_vals)
        partial_sum = tl.sum(abs_vals)
        
        # Atomic add to global sum
        tl.atomic_add(out_ptr, partial_sum)

def symmetric_mm_and_abs_sum_v2(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    A = A.contiguous()
    C = C.contiguous()
    
    # Validate shapes
    assert A.dim() == 2, "A must be a 2D tensor"
    assert C.dim() == 2, "C must be a 2D tensor"
    assert A.shape == C.shape, "A and C must have the same shape"
    
    n, m = A.shape
    
    # Create output tensor for the sum
    out = torch.zeros(1, dtype=torch.float32, device=A.device)
    
    # Define block size
    BLOCK_SIZE = 32
    
    # First compute the symmetric matrix multiplication and accumulation
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    _symmetric_mm_kernel[grid_size](A, C, n, m, alpha, beta, BLOCK_SIZE)
    
    # Then compute sum of absolute values
    _abs_sum_kernel[grid_size](C, out, n, m, BLOCK_SIZE)
    
    return out

# Final implementation using a cleaner approach
@triton.jit
def _symmetric_mm_and_abs_sum_kernel_v3(A_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_row = row < n
    
    # Compute A @ A.T and accumulate with C
    for i in range(0, n, BLOCK_SIZE):
        # Compute partial dot products
        col = i + tl.arange(0, BLOCK_SIZE)
        mask_col = col < n
        
        # Load A row and column
        a_row = tl.load(A_ptr + row[:, None] * m + col[None, :], mask=mask_row[:, None] & mask_col[None, :], other=0.0)
        a_col = tl.load(A_ptr + col[:, None] * m + row[None, :], mask=mask_col[:, None] & mask_row[None, :], other=0.0)
        
        # Compute dot product
        result = tl.dot(a_row, a_col)
        
        # Scale and accumulate
        result *= alpha
        
        #

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
