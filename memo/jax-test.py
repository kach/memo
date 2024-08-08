import functools
import jax

# jax.config.update('jax_log_compiles', True)
# jax.config.update('jax_explain_cache_misses', True)
# jax.config.update('jax_raise_persistent_cache_errors', True)
# jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)

@functools.partial(jax.jit, static_argnums=(0,))
def fib(t):
    cache = {}
    def fib_(t):
        if t <= 2:
            return 1
        f1 = fib_(t - 1) if t - 1 not in cache else cache[t - 1]
        f2 = fib_(t - 2) if t - 2 not in cache else cache[t - 2]
        out = f1 + f2
        cache[t] = out
        return out
    return fib_(t)

print(fib(30))
# print(fib.lower(23).as_text())  # grows linearly with t
# print(jax.make_jaxpr(fib, static_argnums=(0,))(10))  # grows exponentially with t