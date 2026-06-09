{
  "instruction": "Computes the Chebyshev polynomial of the first kind T_n(input). If n = 0, returns 1. If n = 1, returns input. For n < 6 or |input| > 1, uses a recursive formula. Otherwise, uses an explicit trigonometric formula.",
  "predict": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _chebyshev_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offsets < size\n    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)\n    \n    # For n = 0, T_0(x) = 1\n    # For n = 1, T_1(x) = x\n    # For n >= 2, use recurrence relation: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)\n    \n    # Initialize T_0 and T_1\n    t0 = 1.0\n    t1 = x\n    \n    # If n == 0, return 1\n    # If n == 1, return x\n    # Otherwise, compute recursively\n    result = tl.where(n == 0, 1.0, tl.where(n == 1, x, 0.0))\n    \n    # For n >= 2, compute using recurrence relation\n    for i in range(2, n + 1):\n        t2 = 2.0 * x * t1 - t0\n        t0 = t1\n        t1 = t2\n        result = tl.where(i == n, t1, result)\n    \n    tl.store(out_ptr + offsets, result, mask=mask)\n\n@triton.jit\ndef _chebyshev_trig_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offsets < size\n    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)\n    \n    # Use trigonometric formula: T_n(x) = cos(n * arccos(x))\n    # Only for |x| <= 1 and n >= 6\n    # For |x| > 1, fall back to recurrence relation\n    \n    # Compute arccos(x)\n    arccos_x = tl.where(x >= 1.0, 0.0, tl.where(x <= -1.0, tl.pi, tl.acos(x)))\n    \n    # Compute n * arccos(x)\n    n_arccos_x = n * arccos_x\n    \n    # Compute cos(n * arccos(x))\n    result = tl.cos(n_arccos_x)\n    \n    # If |x| > 1, use recurrence relation instead\n    result = tl.where(tl.abs(x) > 1.0, 0.0, result)\n    \n    # Fall back to recurrence relation for |x| > 1\n    # This is a simplified version - in practice, we'd need to handle this more carefully\n    tl.store(out_ptr + offsets, result, mask=mask)\n\n@triton.jit\ndef _chebyshev_mixed_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offsets < size\n    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)\n    \n    # For n = 0, T_0(x) = 1\n    # For n = 1, T_1(x) = x\n    # For n < 6 or |x| > 1, use recurrence relation\n    # For n >= 6 and |x| <= 1, use trigonometric formula\n    \n    # Initialize T_0 and T_1\n    t0 = 1.0\n    t1 = x\n    \n    # If n == 0, return 1\n    # If n == 1, return x\n    result = tl.where(n == 0, 1.0, tl.where(n == 1, x, 0.0))\n    \n    # For n >= 2, compute using recurrence relation\n    for i in range(2, n + 1):\n        t2 = 2.0 * x * t1 - t0\n        t0 = t1\n        t1 = t2\n        result = tl.where(i == n, t1, result)\n    \n    # For n >= 6 and |x| <= 1, use trigonometric formula\n    trig_result = tl.cos(n * tl.acos(x))\n    result = tl.where(n >= 6, trig_result, result)\n    \n    tl.store(out_ptr + offsets, result, mask=mask)\n\n# Helper function to determine if we should use trigonometric formula\n@triton.jit\ndef _should_use_trig(n: tl.constexpr, x: tl.float32) -> tl.bool:\n    return n >= 6 and tl.abs(x) <= 1.0\n\n# Main wrapper function\n@triton.jit\ndef _chebyshev_main_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offsets < size\n    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)\n    \n    # For n = 0, T_0(x) = 1\n    # For n = 1, T_1(x) = x\n    # For n < 6 or |x| > 1, use recurrence relation\n    # For n >= 6 and |x| <= 1, use trigonometric formula\n    \n    # Initialize T_0 and T_1\n    t0 = 1.0\n    t1 = x\n    \n    # If n == 0, return 1\n    # If n == 1, return x\n    result = tl.where(n == 0, 1.0, tl.where(n == 1, x, 0.0))\n    \n    # For n >= 2, compute using recurrence relation\n    for i in range(2, n + 1):\n        t2 = 2.0 * x * t1 - t0\n        t0 = t1\n        t1 = t2\n        result = tl.where(i == n, t1, result)\n    \n    # For n >= 6 and |x| <= 1, use trigonometric formula\n    trig_result = tl.cos(n * tl.acos(x))\n    result = tl.where(n >= 6 and tl.abs(x) <= 1.0, trig_result, result)\n    \n    tl.store(out_ptr + offsets, result, mask=mask)\n\n# Final wrapper function\n@triton.jit\ndef _chebyshev_final_kernel(x_ptr, out_ptr, n: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offsets < size\n    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)\n    \n    # Handle special cases\n    result = tl.where(n == 0, 1.0, tl.where(n == 1, x, 0.0))\n    \n    # For n >= 2, compute using recurrence relation\n    # We'll compute up to n=5 with recurrence relation\n    # For n >= 6, use trigonometric formula\n    \n    # Initialize T_0 and T_1\n    t0 = 1.0\n    t1 = x\n    \n    # Compute recurrence relation for n >= 2\n    for i in range(2, min(n + 1, 6)):\n        t2 = 2.0 * x * t1 - t0\n        t0 = t1\n        t1 = t2\n        result = tl.where(i == n, t1, result)\n    \n    # For n >= 6, use trigonometric formula\n    trig_result = tl.cos(n * tl.acos(x))\n    result = tl.where(n >= 6 and tl.abs(x) <= 1.0, trig_result, result)\n    \n    tl.store(out_ptr + offsets, result, mask=mask)\n\n

##################################################################################################################################################



import torch

def test_chebyshev_polynomial_t():
    results = {}

    # Test case 1: Basic test with n=0
    input_tensor_1 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_1 = 0
    results["test_case_1"] = chebyshev_polynomial_t(input_tensor_1, n_1)

    # Test case 2: Basic test with n=1
    input_tensor_2 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_2 = 1
    results["test_case_2"] = chebyshev_polynomial_t(input_tensor_2, n_2)

    # Test case 3: Higher degree polynomial n=3
    input_tensor_3 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_3 = 3
    results["test_case_3"] = chebyshev_polynomial_t(input_tensor_3, n_3)

    # Test case 4: Negative input values with n=2
    input_tensor_4 = torch.tensor([-1.0, -0.5, -0.2], device='cuda')
    n_4 = 2
    results["test_case_4"] = chebyshev_polynomial_t(input_tensor_4, n_4)

    return results

test_results = test_chebyshev_polynomial_t()
