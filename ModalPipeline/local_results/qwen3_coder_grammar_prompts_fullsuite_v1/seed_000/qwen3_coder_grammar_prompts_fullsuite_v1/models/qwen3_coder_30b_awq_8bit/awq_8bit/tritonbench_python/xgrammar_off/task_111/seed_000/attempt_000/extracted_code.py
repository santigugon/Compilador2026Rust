import torch
import triton
import triton.language as tl

@triton.jit
def _svd_kernel(
    A_ptr, U_ptr, S_ptr, V_ptr,
    m, n, k,
    stride_am, stride_an,
    stride_um, stride_un,
    stride_sm,
    stride_vm, stride_vn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_REDUCE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Shared memory for tiles
    tile_A = tl.shared_ptr(A_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_K), (stride_am, stride_an))
    tile_U = tl.shared_ptr(U_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_K), (stride_um, stride_un))
    tile_V = tl.shared_ptr(V_ptr, (BLOCK_SIZE_K, BLOCK_SIZE_N), (stride_vm, stride_vn))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over tiles
    for k_iter in range(0, tl.cdiv(n, BLOCK_SIZE_K)):
        # Load tile
        a = tl.load(tile_A + (pid_m * BLOCK_SIZE_M, k_iter * BLOCK_SIZE_K))
        b = tl.load(tile_V + (k_iter * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N))
        
        # Compute partial dot product
        acc += tl.dot(a, b)
    
    # Store result
    if IS_REDUCE:
        tl.store(S_ptr + (pid_m * BLOCK_SIZE_M + pid_n), acc[0, 0])
    else:
        tl.store(U_ptr + (pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), acc)

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None) -> torch.Tensor:
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Unsupported dtype")
    
    if k <= 0:
        raise ValueError("k must be positive")
    
    # Get dimensions
    *batch_dims, m, n = A.shape
    if k > min(m, n):
        raise ValueError("k must be <= min(m, n)")
    
    # Determine output shape
    if full_matrices:
        u_shape = (*batch_dims, m, m)
        v_shape = (*batch_dims, n, n)
    else:
        u_shape = (*batch_dims, m, k)
        v_shape = (*batch_dims, k, n)
    
    # Allocate output tensors
    if out is not None:
        U = out[0] if isinstance(out, tuple) else out
        S = out[1] if isinstance(out, tuple) else None
        V = out[2] if isinstance(out, tuple) else None
    else:
        U = torch.empty(u_shape, dtype=A.dtype, device=A.device)
        S = torch.empty((*batch_dims, min(m, n)), dtype=torch.float32, device=A.device)
        V = torch.empty(v_shape, dtype=A.dtype, device=A.device)
    
    # Launch kernel
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    _svd_kernel[grid](
        A, U, S, V,
        m, n, k,
        A.stride(-2), A.stride(-1),
        U.stride(-2), U.stride(-1),
        S.stride(-1),
        V.stride(-2), V.stride(-1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
        IS_REDUCE=False
    )
    
    # Return approximation
    if out is not None:
        return (U, S, V) if isinstance(out, tuple) else U
    else:
        return torch.matmul(U, torch.diag_embed(S)) @ V
