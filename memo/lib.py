import jax
import jax.numpy as jnp
import time
from functools import cache

from .core import MemoError, AuxInfo, memo_result

def marg(t, dims):
    if dims == ():
        return t
    return t.sum(axis=tuple(-1 - d for d in dims), keepdims=True)

def maxx(t, dims):
    if dims == ():
        return t
    return jnp.max(t, axis=tuple(-1 - d for d in dims), keepdims=True)

def pad(t, total):
    count = total - len(t.shape)
    for _ in range(count):
        t = jnp.expand_dims(t, 0)
    return t

def check_scalar_param(x, name):
    if not jnp.isscalar(x):
        raise MemoError(
            f"Parameter {name} was not a numeric scalar, but rather an array. By default, all memo parameters must be scalars. Annotate this parameter as `{name}: ...` if you really did intend to pass in a non-scalar.",
            hint=None,
            user=True,
            ctxt=None,
            loc=None
        )

def check_exotic_param(x, name):
    if not isinstance(x, jnp.ndarray):
        raise MemoError(
            f"Parameter {name} was not a JAX array, despite being annotated as `{name}: ...`.",
            hint=None,
            user=True,
            ctxt=None,
            loc=None
        )

def ffi(f, statics, *args):
    if jax.eval_shape(
        f,
        *[(z if static else jax.ShapeDtypeStruct((), jnp.int32)) for z, static in zip(args, statics)]
    ).shape != ():
        raise MemoError(
            f"The function {f.__name__}(...) is not scalar-in-scalar-out. memo can only handle external (@jax.jit) functions that take scalars as input and return a single scalar as output.",
            hint=None,
            user=True,
            ctxt=None,
            loc=None
        )
    # if not isinstance(f, jax.lib.xla_extension.PjitFunction):
    #     raise MemoError(
    #         f"Tried to call non-JAX function `{f.__name__}`. Use @jax.jit to mark as JAX.",
    #         hint=None,
    #         user=True,
    #         ctxt=None,
    #         loc=None
    #     )
    nonstatic_args = [arg for arg, static in zip(args, statics) if not static]
    if len(nonstatic_args) == 0:
        return f(*args)
    target_shape = jax.numpy.broadcast_shapes(*[arg.shape for arg in nonstatic_args])
    args = [arg if static else jax.numpy.broadcast_to(arg, target_shape).reshape(-1) for arg, static in zip(args, statics)]
    return jax.vmap(f, in_axes=[None if static else 0 for arg, static in zip(args, statics)])(*args).reshape(target_shape)

def check_which_retval(num_retvals, which_retval):
    if num_retvals == 1:
        assert which_retval is None, "You are calling a memo model with only one return value, so you do not have to specify specify which return value you want."
    else:
        assert which_retval is not None, "You are calling a memo model that has multiple return values, but you have not specified which return value you want. Try writing model[0][...](...) to get the first return value."
        assert 0 <= which_retval
        assert which_retval < num_retvals, "Your memo model does not have enough return values."

def check_domains(tgt, src):
    if len(tgt) > len(src):
        raise Exception("Not enough arguments to memo call!")
    if len(src) > len(tgt):
        raise Exception("Too many arguments to memo call!")
    for i, (t, s) in enumerate(zip(tgt, src)):
        if t != s:
            raise Exception(f"Domain mismatch in memo call argument {i + 1}: {t} != {s}.")

def pprint_table(f, z):
    z = z.at[jnp.isclose(z, 1., atol=1e-5)].set(1)
    z = z.at[jnp.isclose(z, 0., atol=1e-5)].set(0)

    def pprint(val):
        if isinstance(val, jnp.ndarray):
            return str(val.item())
        from enum import Enum
        if isinstance(val, Enum):
            return f'{val.name}'
        return str(val)

    rows = []
    if f._num_retvals == 1:
        rows.append(tuple([f'{ax}: {dom}' for ax, dom in zip(f._axes, f._doms)]) + (f"{f.__name__}",))
    else:
        rows.append(tuple([f'{ax}: {dom}' for ax, dom in zip(f._axes, f._doms)]) + tuple(f"{f.__name__}[{i}]" for i in range(f._num_retvals)))
    import itertools
    for row in itertools.product(*[enumerate(v) for v in f._vals]):
        idx = tuple([r[0] for r in row])
        lead = tuple([pprint(r[1]) for r in row])
        if f._num_retvals == 1:
            rows.append(lead + (pprint(z[idx]),))
        else:
            rows.append(lead + tuple([pprint(z[(k,) + idx]) for k in range(f._num_retvals)]))

    widths = []
    for col in range(len(rows[0])):
        widths.append(max([len(row[col]) for row in rows]))

    def hr():
        for w, c in zip(widths, rows[0]):
            print('+', end='-')
            print('-' * w, end='-')
        print('-+')

    hr()
    for ri, row in enumerate(rows):
        for w, c in zip(widths, row):
            print('|', end=' ')
            print(c + ' ' * (w - len(c)), end=' ')
        print(' |')
        if ri == 0:
            hr()
    hr()


def make_pandas_data(f, z):
    import itertools
    def pprint(val):
        if isinstance(val, jnp.ndarray):
            return val.item()
        from enum import Enum
        if isinstance(val, Enum):
            return val.name
        return val

    data = dict()
    for ax, dom in zip(f._axes, f._doms):
        data[f"{ax}"] = list()
    data[f"{f.__name__[5:]}"] = list()


    for row in itertools.product(*[enumerate(v) for v in f._vals]):
        idx = tuple([r[0] for r in row])
        lead = tuple([pprint(r[1]) for r in row])
        row_data = lead + (pprint(z[idx]),)

        for (dom, val) in zip(data.keys(), row_data):
            data[dom].append(val)

    import pandas as pd
    return pd.DataFrame(data)


def make_xarray_data(f, z):
    def parse(val):
        if isinstance(val, jnp.ndarray):
            return val.item()
        from enum import Enum
        if isinstance(val, Enum):
            return val.name
        return val

    coords = {}
    for (ax, dom, vals) in zip(f._axes, f._doms, f._vals):
        coords[f"{ax}"] = [parse(v) for v in vals]

    import xarray as xr
    return xr.DataArray(name=f"{f.__name__[5:]}", data=z, coords=coords)

def collapse_diagonal(A, i, j):
    # Check that the specified axes have the same size
    if A.shape[i] != A.shape[j]:
        raise ValueError(f"Axes {i} and {j} must have the same size (got {A.shape[i]} and {A.shape[j]})")
    # Move the axes of interest to the end for easier manipulation
    A_swapped = jnp.moveaxis(A, [i, j], [-2, -1])
    # Extract the diagonal
    diag = jnp.diagonal(A_swapped, axis1=-2, axis2=-1)
    # Add a new axis at the end (which will become axis j)
    diag_expanded = jnp.expand_dims(diag, axis=-1)
    # Now move the axes to the correct order
    result = jnp.moveaxis(diag_expanded, [-2, -1], [i, j])
    return result


def array_index(arr, *idxs):
    return arr[idxs]

from enum import IntEnum
class Bool(IntEnum):
    NO = 0
    YES = 1