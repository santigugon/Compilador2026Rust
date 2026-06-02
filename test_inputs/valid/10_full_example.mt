@triton.jit
def k(x, out, BS: tl.constexpr): {
  pid = tl.program_id(0);
  offs = pid * BS + tl.arange(0, BS);
  tl.store(out + offs, tl.load(x + offs));
}
