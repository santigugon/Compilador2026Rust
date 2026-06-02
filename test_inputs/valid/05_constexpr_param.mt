@triton.jit
def arange_kernel(x, BS: tl.constexpr): {
  y = tl.arange(0, BS);
}
