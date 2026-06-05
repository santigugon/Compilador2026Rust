import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_dot_kernel(
    A_ptr, x_ptr, y_ptr,
    alpha, beta,
    n, m,
    A_stride_0, A_stride_1,
    x_stride,
    y_stride,
    BLOCK_SIZE_M, BLOCK_SIZE_N
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the matrix-vector product part
    if pid_m < n:
        y_offset = pid_m * y_stride
        for i in range(0, m, BLOCK_SIZE_N):
            if i + pid_n < m:
                a = tl.load(A_ptr + pid_m * A_stride_0 + (i + pid_n) * A_stride_1)
                x_val = tl.load(x_ptr + (i + pid_n) * x_stride)
                tl.atomic_add(y_ptr + y_offset, alpha * a * x_val)
    
    # Update y with beta * y
    if pid_m < n:
        y_offset = pid_m * y_stride
        y_val = tl.load(y_ptr + y_offset)
        tl.store(y_ptr + y_offset, beta * y_val)

@triton.jit
def dot_product_kernel(
    y_ptr, x_ptr,
    n, m,
    y_stride, x_stride,
    BLOCK_SIZE_M
):
    pid = tl.program_id(0)
    if pid < n:
        y_offset = pid * y_stride
        x_offset = pid * x_stride
        y_val = tl.load(y_ptr + y_offset)
        x_val = tl.load(x_ptr + x_offset)
        tl.atomic_add(y_ptr + y_offset, y_val * x_val)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.shape[0] == y.shape[0], "Matrix rows must match y vector length"
    assert A.shape[1] == x.shape[0], "Matrix columns must match x vector length"
    
    n, m = A.shape[0], A.shape[1]
    
    # Launch kernel for matrix-vector product and update y
    grid = (n, 1)
    matrix_vector_dot_kernel[grid](
        A, x, y,
        alpha, beta,
        n, m,
        A.stride(0), A.stride(1),
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32
    )
    
    # Compute dot product of updated y and x
    y_dot_x = torch.dot(y, x)
    return y_dot_x
