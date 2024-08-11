import jax
import jax.numpy as jnp

def marg(t, dims):
    if dims == ():
        return t
    return t.sum(axis=tuple(-1 - d for d in dims), keepdims=True)

def pad(t, total):
    count = total - len(t.shape)
    for _ in range(count):
        t = jnp.expand_dims(t, 0)
    return t

def ffi(f, *args):
    if len(args) == 0:
        return f()
    args = jax.numpy.broadcast_arrays(*args)
    target_shape = args[0].shape
    args = [arg.reshape(-1) for arg in args]
    if isinstance(f, jax.lib.xla_extension.PjitFunction):
        return jax.vmap(f)(*args).reshape(target_shape)
    else:
        raise NotImplementedError