from .core import *
from .parse import *
from .version import __version__

import textwrap
import os, sys, platform, inspect
from io import StringIO
import jax
from typing import Any

def codegen(
    pctxt: ParsingContext,
    stmts: list[Stmt],
    retval: Expr,
    debug_print_compiled: bool=False,
    debug_trace: bool=False,
    save_comic: str|None=None
) -> Any:
    f_name = pctxt.loc_name
    ctxt = Context(frame=Frame(name=ROOT_FRAME_NAME))
    ctxt.hoisted_syms.extend(pctxt.static_parameters)
    with ctxt.hoist():
        if debug_trace:
            ctxt.emit(f"""print(f' -> {pctxt.loc_name}({{ {", ".join(pctxt.static_parameters)} }})')""")
    for stmt_ in stmts:
        eval_stmt(stmt_, ctxt)

    val = eval_expr(retval, ctxt)
    if not val.known:
        raise MemoError(
            "Returning a value that the observer has uncertainty over",
            hint="TODO",
            ctxt=ctxt,
            user=True,
            loc=retval.loc
        )
    squeeze_axes = [
        -1 - i
        for i in range(ctxt.next_idx)
        if i not in [z[0] for z in ctxt.forall_idxs]
    ]
    ctxt.emit(f"{val.tag} = jnp.array({val.tag})")
    ctxt.emit(
        f"{val.tag} = pad({val.tag}, {ctxt.next_idx}).squeeze(axis={tuple(squeeze_axes)}).transpose()"
    )

    with ctxt.hoist():
        ctxt.emit(f"""_jit_ = _jit_{f_name}({", ".join(ctxt.hoisted_syms)})""")
        if debug_trace:
            ctxt.emit(f"""print(f'<-  {pctxt.loc_name}({{ {", ".join(pctxt.static_parameters)} }})')""")
        ctxt.emit(f"""return _jit_""")
    ctxt.emit(f"return {val.tag}")

    out = f"""\
def _make_{f_name}():
    from memo.lib import marg, pad, ffi, jax, jnp
    import functools

    @jax.jit
    def _jit_{f_name}({", ".join(ctxt.hoisted_syms)}):
{textwrap.indent(ctxt.regular_buf.getvalue(), "    " * 2)}

    def _out_{f_name}({", ".join(pctxt.static_parameters)}):
{textwrap.indent(ctxt.hoisted_buf.getvalue(), "    " * 2)}

    _out_{f_name}._shape = tuple([{", ".join(f"len({p[1]})" for p in pctxt.axes)}])
    return _out_{f_name}

{f_name} = _make_{f_name}()
"""

    if debug_print_compiled:
        for i, line in enumerate(out.splitlines()):
            print(f"{i + 1: 5d}  {line}")

    if save_comic is not None:
        from .comic import comic
        comic(ctxt.frame, val, fname=save_comic)

    globals_of_caller = inspect.stack()[3].frame.f_globals
    retvals: dict[Any, Any] = {}
    exec(out, globals_of_caller, retvals)
    return retvals[f"{f_name}"]


def memo_(f, **kwargs):  # type: ignore
    pctxt, stmts, retval = parse_memo(f)
    return codegen(pctxt, stmts, retval, **kwargs)


def memo(f=None, **kwargs):  # type: ignore
    try:
        if f is None:
            return lambda f: memo_(f, **kwargs)  # type: ignore
        return memo_(f, **kwargs)  # type: ignore
    except MemoError as e:
        if e.loc:
            e.add_note('')
            e.add_note(
                f"    at: @memo {e.loc.name} in {os.path.basename(e.loc.file)}, line {e.loc.line}, column {e.loc.offset + 1}"
            )
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
        e.add_note(f"  P.S.: You are currently using...")
        e.add_note(f"        + memo version {__version__} and JAX version {jax.__version__}")
        e.add_note(f"        + on Python version {platform.python_version()} on the {platform.system()} platform")
        raise