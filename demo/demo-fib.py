from memo import memo
import jax
import jax.numpy as np
import functools

'''
This file is useful for exploring/tinkering with the memo compiler, especially
understanding the subtleties of statically-known parameters and recursion.
'''

Unit = [0]

@functools.cache
@memo(debug_print_compiled=True, debug_trace=True)
def fib[a: Unit](n):
    return 1 if n < 2 else fib[a](n - 1) + fib[a](n - 2)

print([fib(n) for n in range(10 + 1)])  # this works
#  print(jax.vmap(fib)(np.arange(10 + 1)))  # this rightfully doesn't work
