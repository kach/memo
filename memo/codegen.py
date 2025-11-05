from __future__ import annotations

from .core import *
from .parse import *
from .version import __version__

import textwrap
import os, sys, platform, inspect
from io import StringIO
from typing import Any, Optional, Literal, Protocol, overload, cast, TYPE_CHECKING
from collections.abc import Callable
import warnings
import linecache

if TYPE_CHECKING:
    import jax

from . import lib
lib_dir = ', '.join([key for key in dir(lib) if not key.startswith('_')])

class MemoCompiled(Protocol):
    @overload
    def __call__(
        self,
        *args: jax.typing.ArrayLike,
        return_aux: Literal[False] = ...,
        return_pandas: bool = ...,
        return_xarray: bool = ...,
        return_cost: bool = ...,
        print_table: bool = ...,
        **kwargs: jax.typing.ArrayLike
    ) -> jax.Array:
        ...

    @overload
    def __call__(
        self,
        *args: jax.typing.ArrayLike,
        return_aux: Literal[True] = ...,
        return_pandas: bool = ...,
        return_xarray: bool = ...,
        return_cost: bool = ...,
        print_table: bool = ...,
        **kwargs: jax.typing.ArrayLike
    ) -> memo_result:
        ...

    def __call__(
        self,
        *args: jax.typing.ArrayLike,
        return_aux: bool = False,
        return_pandas: bool = False,
        return_xarray: bool = False,
        return_cost: bool = False,
        print_table: bool = False,
        **kwargs: jax.typing.ArrayLike
    ) -> jax.Array | memo_result:
        ...

def make_static_parameter_list(pctxt: ParsingContext) -> str:
    out = ''
    for sp, sd in zip(pctxt.static_parameters, pctxt.static_defaults):
        if sd is None:
            out += f'{sp}'
        else:
            out += f'{sp}={sd}'
        out += ', '
    return out

def codegen(
    pctxt: ParsingContext,
    stmts: list[Stmt],
    retvals: list[Expr],
    debug_print_compiled: bool=False,
    debug_trace: bool=False,
    save_comic: Optional[str]=None,
    install_module: Optional[Callable[[str], Any]] = None,
    cache: bool = False
) -> MemoCompiled:
    f_name = pctxt.loc_name
    ctxt = Context(frame=Frame(name=ROOT_FRAME_NAME), pctxt=pctxt)
    ctxt.hoisted_syms.extend(pctxt.static_parameters)
    with ctxt.hoist():
        for param in pctxt.static_parameters:
            if param not in pctxt.exotic_parameters:
                ctxt.emit(f'check_scalar_param({param}, "{param}")')
            else:
                ctxt.emit(f'check_exotic_param({param}, "{param}")')
        ctxt.emit(f"cost_ = 0")
        if debug_trace:
            ctxt.emit(f"""_time_ = time.time()""")
            ctxt.emit(f"""print(f' --> {pctxt.loc_name}({{ {", ".join(pctxt.static_parameters) if len(pctxt.static_parameters) > 0 else '""'} }})')""")
    for i, stmt_ in enumerate(stmts):
        ctxt.continuation = stmts[i + 1:]
        eval_stmt(stmt_, ctxt)

    vals = []
    for retval in retvals:
        val = eval_expr(retval, ctxt)
        # for ax_name, ax_dom in pctxt.axes:
        #     if (Name('self'), ax_name) not in val.deps:
        #         warnings.warn(f"memo {pctxt.loc_name}'s return value does not depend on axis {ax_name} (of type {ax_dom}). Are you sure this is what you want? Please note that memo will avoid redundant work by returning an array where the dimension along that axis is of length 1.")

        if not val.known:
            raise MemoError(
                "Returning a value that the observer has uncertainty over",
                hint="Did you mean to use E[...] after your return statement?",
                ctxt=ctxt,
                user=True,
                loc=retval.loc
            )
        squeeze_axes = [
            -1 - i
            for i in range(ctxt.next_idx)
            if i not in [z[0] for z in ctxt.forall_idxs]
        ]
        ctxt.emit(f"# prepare output")
        ctxt.emit(f"{val.tag} = jnp.array({val.tag})  # ensure output is an array")
        ctxt.emit(f"{val.tag} = pad({val.tag}, {ctxt.next_idx})")
        ctxt.emit(f"{val.tag} = {val.tag}.squeeze(axis={tuple(squeeze_axes)}).transpose()")
        vals.append(val.tag)

    with ctxt.hoist():
        ctxt.emit(f"""\
_out_ = _jit_{f_name}({", ".join(ctxt.hoisted_syms)})
""")

        ctxt.emit(f"""\
if return_cost:
    #  https://jax.readthedocs.io/en/latest/aot.html
    _lowered_ = _jit_{f_name}.lower({", ".join(ctxt.hoisted_syms)})
    _cost_ = _lowered_.cost_analysis()
    _cost_ = dict(
        flops=_cost_.get('flops', 0),
        transcendentals=_cost_.get('transcendentals', 0),
        bytes=_cost_.get('bytes accessed', 0)
    )
    aux.cost += _cost_['flops'] + _cost_['transcendentals']
""")
        if debug_trace:
            ctxt.emit(f"""print(f'<--  {pctxt.loc_name}({{ {", ".join(pctxt.static_parameters) if len(pctxt.static_parameters) > 0 else '""'} }}) has shape {{ _out_.shape }}')""")
            ctxt.emit(f"""\
if return_cost:
    print(f'     cost = {{aux.cost}} operations')
""")
            ctxt.emit(f"""print(f'     time = {{time.time() - _time_:.6f}} sec')""")
        ctxt.emit(f"if print_table: pprint_table(_out_{f_name}, _out_)")
        ctxt.emit(f"if return_pandas: aux.pandas = make_pandas_data(_out_{f_name}, _out_)")
        ctxt.emit(f"if return_xarray: aux.xarray = make_xarray_data(_out_{f_name}, _out_)")
        ctxt.emit(f"""return memo_result(data=_out_, aux=aux) if return_aux else _out_""")
    if len(retvals) == 1:
        ctxt.emit(f"return {vals[0]}")
    else:
        ctxt.emit(f"return jnp.stack(jnp.broadcast_arrays({', '.join(vals)}), axis=0)")

    out = f"""\
def _make_{f_name}():
    from memo.lib import {lib_dir}

    @jax.jit
    def _jit_{f_name}({", ".join(ctxt.hoisted_syms)}):
{textwrap.indent(ctxt.regular_buf.getvalue(), "    " * 2)}

{"    @cache" if cache else ""}
    def _out_{f_name}(
        {make_static_parameter_list(pctxt)}*,
        return_aux=False,
        return_pandas=False,
        return_xarray=False,
        return_cost=False,
        print_table=False
    ):
        aux = AuxInfo()
        if return_pandas or return_xarray:
            return_aux = True
        if return_cost:
            return_aux = True
            aux.cost = 0.
{textwrap.indent(ctxt.hoisted_buf.getvalue(), "    " * 2)}

    _out_{f_name}._shape = tuple([{", ".join(f"len({p[1]})" for p in pctxt.axes)}])
    _out_{f_name}._axes = tuple([{", ".join(f"{repr(p[0])}" for p in pctxt.axes)}])
    _out_{f_name}._doms = tuple([{", ".join(f"{repr(p[1])}" for p in pctxt.axes)}])
    _out_{f_name}._vals = tuple([{", ".join(f"{p[1]}" for p in pctxt.axes)}])
    _out_{f_name}._num_retvals = {len(retvals)}
    return _out_{f_name}

{f_name} = _make_{f_name}()
{f_name}.__name__ = '{f_name}'
{f_name}.__qualname__ = '{pctxt.qualname}'
{f_name}.__doc__ = {repr(pctxt.doc)}
"""

    if debug_print_compiled:
        for i, line in enumerate(out.splitlines()):
            print(f"{line}  # {i + 1: 5d}")

    if save_comic is not None:
        from .comic import comic
        comic(ctxt.frame, fname=save_comic)

    globals_of_caller = inspect.stack()[3].frame.f_globals
    locals_of_caller  = inspect.stack()[3].frame.f_locals
    if globals_of_caller != locals_of_caller and install_module is None:
        warnings.warn(f"memo works best in the global (module) scope. Defining memos within function definitions is currently not officially supported, though if you know what you are doing then go ahead and do it!")
    if install_module is not None:
        ret = install_module(out)[f"{f_name}"]
        return cast(MemoCompiled, lambda _: print("Call me from inside the module!"))

    scope_retvals: dict[Any, Any] = {}

    exec(out, globals_of_caller, scope_retvals)
    return cast(MemoCompiled, scope_retvals[f"{f_name}"])

def memo_(f: Callable[..., Any], **kwargs: Any) -> MemoCompiled:
    try:
        pctxt, stmts, retvals = parse_memo(f)
        return codegen(pctxt, stmts, retvals, **kwargs)
    except MemoError as e:
        if e.loc:
            e.add_note(f"  file: \"{os.path.basename(e.loc.file)}\", line {e.loc.line}, in @memo {e.loc.name}")
            e.add_note(f"    {linecache.getline(e.loc.file, e.loc.line)[:-1]}")
            e.add_note(f"    {' ' * e.loc.offset}^")
        if e.hint is not None:
            e.add_note('')
            for line in textwrap.wrap(
                e.hint, initial_indent="  hint: ", subsequent_indent="        "
            ):
                e.add_note(line)
        if e.ctxt:  # TODO
            e.add_note('')
            frame_name = f"{e.ctxt.frame.name}"
            z = e.ctxt.frame
            while z.parent is not None and z.parent.name != ROOT_FRAME_NAME:
                z = z.parent
                frame_name += f", as modeled by {z.name}"
            ctxt_note = f'''\
This error was encountered in the frame of {frame_name}.

In that frame, {e.ctxt.frame.name} is currently modeling the following {len(e.ctxt.frame.choices)} choices: {", ".join([v if k == Name("self") else f"{k}.{v}" for k, v in e.ctxt.frame.choices.keys()])}.
'''
            for line in textwrap.wrap(
                ctxt_note, initial_indent="  ctxt: ", subsequent_indent="        "
            ):
                e.add_note(line)
        if not e.user:
            e.add_note("")
            e.add_note(
                "[We think this may be a bug in memo: if you don't understand what is going on, please get in touch with us!]"
            )
        e.add_note("")

        # Describe environment...
        import jax
        e.add_note(f"  info: You are using memo {__version__}, JAX {jax.__version__}, Python {platform.python_version()} on {platform.system()}.")

        raise e.with_traceback(None) from None

import sys, traceback
old_excepthook = sys.excepthook
def new_excepthook(typ, val, tb):  # type: ignore
    if typ is MemoError:
        return traceback.print_exception(val, limit=0)
    old_excepthook(typ, val, tb)
sys.excepthook = new_excepthook

warnings.showwarning = lambda msg, *args: print('Warning:', msg, file=sys.stderr)

try:
    ipython = get_ipython()  # type: ignore
    old_showtraceback = ipython.showtraceback
    def new_showtraceback(*args, **kwargs):  # type: ignore
        info = sys.exc_info()
        if info[0] is MemoError:
            return traceback.print_exception(info[1], limit=0)
        old_showtraceback(sys.exc_info(), **kwargs)
    ipython.showtraceback = new_showtraceback
except NameError:
    pass

@overload
def memo(f: None=None, **kwargs: Any) -> Callable[[Callable[..., Any]], MemoCompiled]:
    ...

@overload
def memo(f: Callable[..., Any], **kwargs: Any) -> MemoCompiled:
    ...

def memo(f: None | Callable[..., Any] = None, **kwargs: Any) -> Callable[[Callable[..., Any]], MemoCompiled] | MemoCompiled:
    if f is None:
        return lambda f: memo_(f, **kwargs)
    return memo_(f, **kwargs)

def memo_test(mod, expect='pass', item=None, *args, **kwargs):  # type: ignore
    def helper(f):  # type: ignore
        name = f.__name__
        outcome = None
        err: BaseException
        try:
            memo(f, install_module=mod.install, **kwargs)
            f = mod.__getattribute__(name)
            out = f(*args)
        except MemoError as e:
            outcome = 'ce'
            err = e
        except Exception as e:
            outcome = 're'
            err = e
        else:
            outcome = 'pass'

        if outcome == 'pass' and item is not None and abs(out.item() - item) > 1e-6:
            print(f'[fail {name}, {out} != {item}]')
        elif expect == outcome:
            print(f'[ pass {name} ]')
            return f
        else:
            print(f'[!fail {name}, {outcome} != {expect} ]')
            raise err
    return helper
