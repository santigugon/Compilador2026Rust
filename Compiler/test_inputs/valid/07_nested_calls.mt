@triton.jit
def nested(x, out): {
  tl.store(out, tl.load(x));
}
