from quadracheer.util import memoized

@memoized
def fibonacci(n):
   "Return the nth fibonacci number."
   fibonacci.save.append(n)
   if n in (0, 1):
      return n
   return fibonacci(n-1) + fibonacci(n-2)
fibonacci.save = []

def test_memoized_fib():
    val = fibonacci(12)
    assert(val == 144)
    # Thirteen values including 0.
    assert(len(fibonacci.save) == 13)
