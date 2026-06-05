@triton.jit
def bad_call(x): {
  y = tl.math.load(x);
}
