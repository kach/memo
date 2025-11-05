from __future__ import annotations

from typing import NewType, Any, Tuple, Literal, Iterator, NamedTuple, TYPE_CHECKING

from contextlib import contextmanager
import itertools
from enum import Enum
import dataclasses
from dataclasses import dataclass, field
import copy
from functools import singledispatch

import textwrap
from io import StringIO

import warnings
import linecache

if TYPE_CHECKING:
    import jax
    import pandas as pd
    import xarray as xr

try:
    from icecream import ic  # type: ignore
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

@dataclass
class AuxInfo:
    cost: float | None = None
    pandas: pd.DataFrame | None = None
    xarray: xr.DataArray | None = None

class memo_result(NamedTuple):
    data: jax.Array
    aux: AuxInfo


@dataclass(frozen=True)
class SourceLocation:
    file: str
    line: int
    offset: int
    name: str


class MemoError(Exception):
    def __init__(
        self: MemoError,
        message: str,
        hint: str | None,
        user: bool,
        ctxt: Context | None,
        loc: SourceLocation | None,
    ) -> None:
        self.message = message
        self.hint = hint
        self.user = user
        self.ctxt = ctxt
        self.loc = loc


Name = NewType("Name", str)
Id = NewType("Id", str)
Dom = NewType("Dom", str)


@dataclass(frozen=True)
class Value:
    tag: str
    known: bool
    deps: set[tuple[Name, Id]]


@dataclass
class Choice:
    tag: str
    idx: int
    known: bool
    domain: Dom


@dataclass
class Frame:
    name: Name
    choices: dict[tuple[Name, Id], Choice] = field(default_factory=dict)
    children: dict[Name, Frame] = field(default_factory=dict)
    conditions: dict[tuple[Name, Id], tuple[Name, Id]] = field(default_factory=dict)
    # key is a choice in this frame, val is a choice in the parent frame
    # used to create "aliases" in child's choices, e.g. in observe/knows
    goals: dict[Id, Expr] = field(default_factory=dict)
    ll: str | None = None
    parent: Frame | None = None

    def ensure_child(self, who: Name) -> None:
        if who not in self.children:
            self.children[who] = Frame(name=who, parent=self)


ROOT_FRAME_NAME = Name("observer")

@dataclass(frozen=True, kw_only=True)
class SyntaxNode:
    loc: SourceLocation | None

@dataclass(frozen=True, kw_only=True)
class ExprSyntaxNode(SyntaxNode):
    static: bool

@dataclass(frozen=True)
class ELit(ExprSyntaxNode):
    value: float | str


Op = Enum(
    "Op",
    [
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "POW",
        "MOD",
        "EQ",
        "NEQ",
        "LT",
        "LTE",
        "GT",
        "GTE",
        "AND",
        "OR",
        "XOR",
        "EXP",
        "ABS",
        "LOG",
        "UADD",
        "NEG",
        "INV",
        "ITE",
    ],
)


@dataclass(frozen=True)
class EOp(ExprSyntaxNode):
    op: Op
    args: list[Expr]


@dataclass(frozen=True)
class EFFI(ExprSyntaxNode):
    name: str
    args: list[Expr]


@dataclass(frozen=True)
class EMemo(ExprSyntaxNode):
    name: str
    which_retval: int | None
    args: list[Expr]
    ids: list[Tuple[Id, Name, Id]]


@dataclass(frozen=True)
class EChoice(ExprSyntaxNode):
    id: Id


@dataclass(frozen=True)
class EExpect(ExprSyntaxNode):
    expr: Expr
    reduction: Literal["expectation", "variance"]
    warn: bool = True


@dataclass(frozen=True)
class EEntropy(ExprSyntaxNode):
    rvs: list[tuple[Name, Id]]

@dataclass(frozen=True)
class EKL(ExprSyntaxNode):
    p_who: Name
    p_id: Id
    q_who: Name
    q_id: Id

@dataclass(frozen=True)
class EWith(ExprSyntaxNode):
    who: Name
    expr: Expr


@dataclass(frozen=True)
class EImagine(ExprSyntaxNode):
    do: list[Stmt]
    then: Expr


@dataclass(frozen=True)
class ECost(ExprSyntaxNode):
    name: str
    args: list[Expr]

@dataclass(frozen=True)
class EInline(ExprSyntaxNode):
    val: str

@dataclass(frozen=True)
class EPosterior(ExprSyntaxNode):
    query: list[tuple[Name, Id]]
    var: list[tuple[Name, Id]]

@dataclass(frozen=True)
class EPredict(ExprSyntaxNode):
    expr: Expr

@dataclass(frozen=True)
class EUtil(ExprSyntaxNode):
    goal: Id

Expr = (
    ELit
    | EOp
    | EFFI
    | EMemo
    | EChoice
    | EExpect
    | EEntropy
    | EKL
    | EWith
    | EImagine
    | ECost
    | EInline
    | EPosterior
    | EPredict
    | EUtil
)


@dataclass(frozen=True)
class SPass(SyntaxNode):
    pass


@dataclass(frozen=True)
class SChoose(SyntaxNode):
    who: Name
    choices: list[tuple[Id, Dom]]
    wpp: Expr
    reduction: Literal["normalize", "maximize"]


@dataclass(frozen=True)
class SObserve(SyntaxNode):
    who: Name
    id: Id


@dataclass(frozen=True)
class SWith(SyntaxNode):
    who: Name
    stmt: Stmt


@dataclass(frozen=True)
class SShow(SyntaxNode):
    who: Name
    target_who: Name
    target_id: Id
    source_who: Name
    source_id: Id

@dataclass(frozen=True)
class SObserves(SyntaxNode):
    who: Name
    what: Expr
    how: Literal["boolean", "probability"]

@dataclass(frozen=True)
class SForAll(SyntaxNode):
    id: Id
    domain: Dom


@dataclass(frozen=True)
class SKnows(SyntaxNode):
    who: Name
    source_who: Name
    source_id: Id

@dataclass(frozen=True)
class SSnapshot(SyntaxNode):
    who: Name
    alias: Name

@dataclass(frozen=True)
class STrace(SyntaxNode):
    who: Name

@dataclass(frozen=True)
class SWants(SyntaxNode):
    who: Name
    what: Id
    how: Expr

@dataclass(frozen=True)
class SGuess(SyntaxNode):
    who: Name
    id: Id
    target_who: Name
    target_id: Id

Stmt = (
    SPass
    | SChoose
    | SObserve
    | SObserves  # TODO we need to change this name...
    | SWith
    | SShow
    | SForAll
    | SKnows
    | SSnapshot
    | STrace
    | SWants
    | SGuess
)


class Buffer:
    def __init__(self: Buffer) -> None:
        self.io: StringIO = StringIO()
        self.tab_level: int = 0

    def indent(self: Buffer) -> None:
        self.tab_level += 1

    def dedent(self: Buffer) -> None:
        self.tab_level -= 1

    def emit(self: Buffer, line: str) -> None:
        print("    " * self.tab_level + line, file=self.io)

    def getvalue(self: Buffer) -> str:
        return self.io.getvalue()


@dataclass
class ParsingContext:
    cast: None | list[str]
    static_parameters: list[str]
    exotic_parameters: set[str]
    static_defaults: list[None | str]
    axes: list[tuple[str, str]]
    loc_name: str
    loc_file: str
    qualname: str
    doc: str | None = None


@dataclass
class Context:
    frame: Frame
    pctxt: ParsingContext
    continuation: list[Stmt] = field(default_factory=list)

    hoisted_buf: Buffer = field(default_factory=Buffer)
    regular_buf: Buffer = field(default_factory=Buffer)
    hoisted: bool = False
    hoisted_syms: list[str] = field(default_factory=list)

    next_idx: int = 0
    _sym: int = -1
    path_condition: list[str] = field(default_factory=list)

    def current_buf(self: Context) -> Buffer:
        return self.hoisted_buf if self.hoisted else self.regular_buf

    def emit(self: Context, line: str) -> None:
        self.current_buf().emit(line)

    def indent(self: Context) -> None:
        self.current_buf().indent()

    def dedent(self: Context) -> None:
        self.current_buf().dedent()

    def sym(self: Context, hint: str = "") -> str:
        self._sym += 1
        sym = f"{hint}_{self._sym}"
        if self.hoisted:
            self.hoisted_syms.append(sym)
        return sym

    @contextmanager
    def hoist(self: Context, state: bool = True) -> Iterator[None]:
        old_state = self.hoisted
        self.hoisted = state
        yield
        self.hoisted = old_state

    @contextmanager
    def path_depends(self: Context, pc: str) -> Iterator[None]:
        self.path_condition.append(pc)
        yield
        self.path_condition.pop()

    forall_idxs: list[tuple[int, Id, Dom]] = field(default_factory=list)


def assemble_tags(tags: list[str], **kwargs: Any) -> str:
    kwarg_thunk = ', '.join(
        f'{k}={v}' for k, v in kwargs.items()
    )
    posarg_thunk = ', '.join(tags)

    if len(tags) == 0:
        return kwarg_thunk
    if len(kwargs) == 0:
        return posarg_thunk
    return f"{posarg_thunk}, {kwarg_thunk}"

@singledispatch
def eval_expr(e: Expr, ctxt: Context) -> Value:
    raise NotImplementedError

@eval_expr.register
def _(e: ELit, ctxt: Context) -> Value:
    val = e.value
    with ctxt.hoist():
        out = ctxt.sym("lit")
        ctxt.emit(f"{out} = {val}")
    return Value(tag=out, known=True, deps=set())

@eval_expr.register
def _(e: EChoice, ctxt: Context) -> Value:
    id = e.id
    if (Name("self"), id) not in ctxt.frame.choices:
        raise MemoError(
            f"Unknown choice {ctxt.frame.name}.{id}",
            hint=f"Did you perhaps misspell {id}?" + (
                f" Or, did you forget to include {id} as an axis in the definition of this memo?"
                if ctxt.frame.parent is None else
                f" {ctxt.frame.name} is not yet aware of any choice called {id}. Or, did you forget to call {ctxt.frame.name}.chooses({id} ...) or {ctxt.frame.name}.knows({id}) earlier in the memo?"
            ),
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )
    ch = ctxt.frame.choices[(Name("self"), id)]
    return Value(
        tag=ch.tag, known=ch.known, deps=set([(Name("self"), id)])
    )

@eval_expr.register
def _(e: EFFI, ctxt: Context) -> Value:
    name, args = e.name, e.args
    args_out = []
    for arg in args:
        args_out.append(eval_expr(arg, ctxt))
    known = all(arg.known for arg in args_out)
    deps = set().union(*(arg.deps for arg in args_out))
    with ctxt.hoist(e.static):
        out = ctxt.sym(f"ffi_{name}")
        arg_statics = ", ".join([repr(arg.static) for arg in args])
        arg_tags = ", ".join(arg.tag for arg in args_out)
        ctxt.emit(f'{out} = ffi({name}, [{arg_statics}], {arg_tags})')
        if e.static:
            ctxt.emit(f'{out} = jnp.array({out}).item()')
    return Value(tag=out, known=known, deps=deps)

@eval_expr.register
def _(e: ECost, ctxt: Context) -> Value:
    name, args = e.name, e.args
    args_out = []
    with ctxt.hoist():
        for arg in args:
            args_out.append(eval_expr(arg, ctxt))
            if not arg.static:
                raise MemoError(
                    "parameter not statically known",
                    hint="""When calling a memo, you can only pass in parameters that are fixed ("static") values that memo can compute without reasoning about agents. Such values cannot depend on any agents' choices -- only on literal numeric values and other parameters. This constraint is what enables memo to help you fit/optimize parameters fast by gradient descent.""",
                    user=True,
                    ctxt=ctxt,
                    loc=arg.loc,
                )
        res = ctxt.sym(f"result_cost_{name}")
        ctxt.emit(f'if {" and ".join(ctxt.path_condition) if len(ctxt.path_condition) > 0 else "True"}:')
        ctxt.indent()
        ctxt.emit(f'_, {res} = {name}({assemble_tags([arg.tag for arg in args_out], return_cost=True)})')
        ctxt.emit(f'{res} = {res}.cost')
        ctxt.dedent()
        ctxt.emit('else:')
        ctxt.indent()
        ctxt.emit(f'{res} = 0')
        ctxt.dedent()
    return Value(tag=res, known=True, deps=set())

@eval_expr.register
def _(e: EMemo, ctxt: Context) -> Value:
    name, args, ids, which_retval = e.name, e.args, e.ids, e.which_retval
    args_out = []
    with ctxt.hoist():
        for arg in args:
            args_out.append(eval_expr(arg, ctxt))
            if not arg.static:
                raise MemoError(
                    "parameter not statically known",
                    hint="""When calling a memo, you can only pass in parameters that are fixed ("static") values that memo can compute without reasoning about agents. Such values cannot depend on any agents' choices -- only on literal numeric values and other parameters. This constraint is what enables memo to help you fit/optimize parameters fast by gradient descent.""",
                    user=True,
                    ctxt=ctxt,
                    loc=arg.loc,
                )

    for _, source_name, source_id in ids:
        if (source_name, source_id) not in ctxt.frame.choices:
            raise MemoError(
                "Unknown choice referenced in a memo call",
                hint=f"{ctxt.frame.name} does not yet model {source_name}'s choice of {source_id}.",
                user=True,
                ctxt=ctxt,
                loc=e.loc
            )

    with ctxt.hoist():
        res = ctxt.sym(f"result_array_{name}")
        doms = [ctxt.frame.choices[source_name, source_id].domain for _, source_name, source_id in ids]
        ctxt.emit(f"""check_which_retval({name}._num_retvals, {which_retval})""")
        ctxt.emit(f"""check_domains({name}._doms, {repr(tuple(str(d) for d in doms))})""")
        ctxt.emit(f'if {" and ".join(ctxt.path_condition) if len(ctxt.path_condition) > 0 else "True"}:')
        ctxt.indent()
        ctxt.emit(f'{res}, res_aux = {name}({assemble_tags([arg.tag for arg in args_out], return_aux=True, return_cost='return_cost')})')
        if which_retval is not None:
            ctxt.emit(f'{res} = {res}[{which_retval}]')
        ctxt.emit(f"if return_cost: aux.cost += res_aux.cost")
        ctxt.dedent()
        ctxt.emit('else:')
        ctxt.indent()
        ctxt.emit(f'{res} = jnp.zeros({name}._shape)')
        ctxt.dedent()
        ctxt.emit(f"{res} = {res}.transpose()")

    out_idxs = []
    for target_id, source_name, source_id in reversed(ids):
        out_idxs.append(ctxt.frame.choices[(source_name, source_id)].idx)
        # TODO: assert domains match here, too

    ctxt.emit(
        f'{res} = jnp.expand_dims({res}, ({",".join(str(-i - 1) for i in range(max(out_idxs) + 1 - len(out_idxs)))}))'
    )
    permuted_dims: list[None | int] = [None] * (max(out_idxs) + 1)

    for permuted_idx, source_idx in enumerate(out_idxs):
        permuted_dims[-1 - source_idx] = permuted_idx

    filler_count = len(out_idxs)
    for idx in range(len(permuted_dims)):
        if permuted_dims[idx] is None:
            permuted_dims[idx] = filler_count
            filler_count += 1

    ctxt.emit(f"{res} = jnp.permute_dims({res}, {tuple(permuted_dims)})")

    return Value(
        tag=res,
        known=all(ctxt.frame.choices[sn, si].known for _, sn, si in ids),
        deps=set((sn, si) for _, sn, si in ids)
    )

@eval_expr.register
def _(e: EOp, ctxt: Context) -> Value:
    op, args = e.op, e.args
    if op in [
        Op.ADD,
        Op.SUB,
        Op.MUL,
        Op.DIV,
        Op.POW,
        Op.MOD,
        Op.EQ,
        Op.NEQ,
        Op.LT,
        Op.LTE,
        Op.GT,
        Op.GTE,
        Op.AND,
        Op.OR,
        Op.XOR
    ]:
        assert len(args) == 2
        l = eval_expr(args[0], ctxt)
        r = eval_expr(args[1], ctxt)
        with ctxt.hoist(e.static):
            out = ctxt.sym(f"op_{op.name.lower()}")
            match op:
                case Op.ADD:
                    ctxt.emit(f"{out} = {l.tag} + {r.tag}")
                case Op.SUB:
                    ctxt.emit(f"{out} = {l.tag} - {r.tag}")
                case Op.MUL:
                    ctxt.emit(f"{out} = {l.tag} * {r.tag}")
                case Op.DIV:
                    ctxt.emit(f"{out} = {l.tag} / {r.tag}")
                case Op.POW:
                    ctxt.emit(f"{out} = {l.tag} ** {r.tag}")
                case Op.MOD:
                    ctxt.emit(f"{out} = {l.tag} % {r.tag}")
                case Op.EQ:
                    ctxt.emit(f"{out} = {l.tag} == {r.tag}")
                case Op.NEQ:
                    ctxt.emit(f"{out} = {l.tag} != {r.tag}")
                case Op.LT:
                    ctxt.emit(f"{out} = {l.tag} < {r.tag}")
                case Op.LTE:
                    ctxt.emit(f"{out} = {l.tag} <= {r.tag}")
                case Op.GT:
                    ctxt.emit(f"{out} = {l.tag} > {r.tag}")
                case Op.GTE:
                    ctxt.emit(f"{out} = {l.tag} >= {r.tag}")
                case Op.AND:
                    ctxt.emit(f"{out} = {l.tag} & {r.tag}")
                case Op.OR:
                    ctxt.emit(f"{out} = {l.tag} | {r.tag}")
                case Op.XOR:
                    ctxt.emit(f"{out} = {l.tag} ^ {r.tag}")
        return Value(
            tag=out,
            known=l.known and r.known,
            deps=l.deps | r.deps
        )
    elif op in [Op.EXP, Op.ABS, Op.LOG, Op.UADD, Op.NEG, Op.INV]:
        assert len(args) == 1
        l = eval_expr(args[0], ctxt)
        with ctxt.hoist(e.static):
            out = ctxt.sym(f"op_{op.name.lower()}")
            match op:
                case Op.EXP:
                    ctxt.emit(f"{out} = jnp.exp({l.tag})")
                case Op.ABS:
                    ctxt.emit(f"{out} = jnp.abs({l.tag})")
                case Op.LOG:
                    ctxt.emit(f"{out} = jnp.log({l.tag})")
                case Op.UADD:
                    ctxt.emit(f"{out} = +({l.tag})")
                case Op.NEG:
                    ctxt.emit(f"{out} = -({l.tag})")
                case Op.INV:
                    ctxt.emit(f"{out} = True ^ {l.tag}")
        return Value(tag=out, known=l.known, deps=l.deps)
    elif op == Op.ITE:
        assert len(args) == 3
        with ctxt.hoist(e.static):
            out = ctxt.sym(f"op_{op.name.lower()}")
            c = eval_expr(args[0], ctxt)
            if args[0].static:
                with ctxt.path_depends(c.tag):
                    t = eval_expr(args[1], ctxt)
                inv_out = eval_expr(EOp(Op.INV, [ELit(c.tag, loc=None, static=True)], loc=None, static=True), ctxt)
                with ctxt.path_depends(inv_out.tag):
                    f = eval_expr(args[2], ctxt)
                ctxt.emit(f"{out} = jnp.where({c.tag}, {t.tag}, {f.tag})")
                return Value(
                    tag=out,
                    known=c.known and t.known and f.known,
                    deps=c.deps | t.deps | f.deps
                )
            else:
                t = eval_expr(args[1], ctxt)
                f = eval_expr(args[2], ctxt)
                ctxt.emit(f"{out} = jnp.where({c.tag}, {t.tag}, {f.tag})")
                return Value(
                    tag=out,
                    known=c.known and t.known and f.known,
                    deps=c.deps | t.deps | f.deps
                )
    else:
        raise NotImplementedError

@eval_expr.register
def _(e: EPosterior, ctxt: Context) -> Value:
    knw, var = e.query, e.var
    var_idxs = {ctxt.frame.choices[var_].idx for var_ in var}
    idxs_to_marginalize = tuple(set(
        c.idx for _, c in ctxt.frame.choices.items() if not (c.known or c.idx in var_idxs)
    ))
    out = ctxt.sym("posterior")
    ctxt.emit(f"{out} = marg({ctxt.frame.ll}, {idxs_to_marginalize})")
    ctxt.emit(f"{out} = pad({out}, {ctxt.next_idx})")
    for knw_, var_ in zip(knw, var):
        # in theory, knw_c is a unitary dimension (known choice ll)
        # in theory, var_c is a non-unitary dimension (unknown choice ll)
        # if we transpose their indices, then all is well
        # however, sometimes knw_c is non-unitary, so we have to instead extract the diag
        knw_c = ctxt.frame.choices[knw_]
        var_c = ctxt.frame.choices[var_]
        ctxt.emit(f"if {out}.shape[-1 - {knw_c.idx}] == 1:")
        ctxt.indent()
        ctxt.emit(f"{out} = jnp.swapaxes({out}, -1 - {var_c.idx}, -1 - {knw_c.idx})")
        ctxt.dedent()
        ctxt.emit("else:")
        ctxt.indent()
        ctxt.emit(f"{out} = collapse_diagonal({out}, -1 - {knw_c.idx}, -1 - {var_c.idx})")
        ctxt.dedent()
    return Value(tag=out, known=True, deps={c for c, cc in ctxt.frame.choices.items() if cc.known})

@eval_expr.register
def _(e: EExpect, ctxt: Context) -> Value:
    expr, reduction = e.expr, e.reduction
    knw, var = [], []
    def epost_eligible(lc: Id, rw: Name, rc: Id) -> bool:
        if (Name("self"), lc) not in ctxt.frame.choices:
            return False
        if (rw, rc) not in ctxt.frame.choices:
            return False

        lcc = ctxt.frame.choices[Name("self"), lc]
        rcc = ctxt.frame.choices[rw, rc]
        return lcc.known and (not rcc.known) and lcc.domain == rcc.domain

    def check_eposterior(expr_: Expr) -> bool:
        match expr_:
            case EOp(Op.EQ, (
                [EChoice(id=lc), EWith(rw, EChoice(rc))] |
                [EWith(rw, EChoice(rc)), EChoice(id=lc)]
            )) if epost_eligible(lc, rw, rc):
                knw.append((Name("self"), lc))
                var.append((rw, rc))
                return True
            case EOp(
                Op.AND, [fst, snd]
            ) if check_eposterior(fst) and check_eposterior(snd):
                return True
        return False
    if check_eposterior(expr):
        axes_distinct = len(
            {ctxt.frame.choices[c].idx for c in knw + var}
        ) == len(knw) + len(var)
        if axes_distinct:
            return eval_expr(
                EPosterior(knw, var, loc=e.loc, static=e.static), ctxt
            )

    val_ = eval_expr(expr, ctxt)
    if all(ctxt.frame.choices[c].known for c in sorted(val_.deps)):
        if e.warn:
            warnings.warn(f"""\
Redundant expectation, not marginalizing...
| {linecache.getline(e.loc.file, e.loc.line)[:-1] if e.loc is not None else ''}
| {' ' * e.loc.offset if e.loc is not None else ''}^
    """)
        if reduction == "expectation":
            return val_
    idxs_to_marginalize = tuple(set(
        # TODO: ideally, dedup by looking at frame.conditions
        c.idx for _, c in ctxt.frame.choices.items() if not c.known
    ))
    ctxt.emit(f"# {ctxt.frame.name} expectation")

    out = ctxt.sym("exp")
    if reduction == "expectation":
        ctxt.emit(
            f"{out} = marg({ctxt.frame.ll} * {val_.tag}, {idxs_to_marginalize})"
        )
    elif reduction == "variance":
        ctxt.emit(
            f"{out} = marg({ctxt.frame.ll} * {val_.tag} ** 2, {idxs_to_marginalize}) - marg({ctxt.frame.ll} * {val_.tag}, {idxs_to_marginalize}) ** 2"
        )
    deps = ({  # TODO: this lets in too many deps!!
        c for c, _ in ctxt.frame.choices.items()
        if ctxt.frame.choices[c].known
    })
    return Value(
        tag=out,
        known=True,
        # deps={(name, id) for (name, id) in val_.deps if ctxt.frame.choices[(name, id)].known}
        deps=deps
    )

@eval_expr.register
def _(e: EEntropy, ctxt: Context) -> Value:
    rvs = e.rvs
    for rv in rvs:
        if ctxt.frame.choices[rv].known:
            raise MemoError(
                "Taking entropy of known variable",
                hint=f"{rv[0]}.{rv[1]} is already known to {ctxt.frame.name}, so its entropy is zero.",
                user=True,
                ctxt=ctxt,
                loc=e.loc
            )
    idxs_a = tuple(
        set(c.idx for n, c in ctxt.frame.choices.items() if (not c.known) and (n not in rvs))
    )
    deps = {n for n, c in ctxt.frame.choices.items() if c.known}
    ctxt.emit(f"# {ctxt.frame.name} entropy")

    out = ctxt.sym("entropy")
    marginal = ctxt.sym("marginal")
    ctxt.emit(f"{marginal} = marg({ctxt.frame.ll}, {idxs_a})")

    idxs_b = tuple(
        set(c.idx for n, c in ctxt.frame.choices.items() if (not c.known) and (n in rvs))
    )
    ctxt.emit(
        f"{out} = -marg({marginal} * jnp.nan_to_num(jnp.log({marginal})), {idxs_b})"
    )
    return Value(
        tag=out,
        known=True,
        deps=deps
    )

@eval_expr.register
def _(e: EKL, ctxt: Context) -> Value:
    p_who, p_id, q_who, q_id = e.p_who, e.p_id, e.q_who, e.q_id
    if (p_who, p_id) not in ctxt.frame.choices:
        raise MemoError(
            f"Unknown choice {p_who}.{p_id}",
            hint=f"Did you misspell something?",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )
    if (q_who, q_id) not in ctxt.frame.choices:
        raise MemoError(
            f"Unknown choice {q_who}.{q_id}",
            hint=f"Did you misspell something?",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )
    p_c = ctxt.frame.choices[p_who, p_id]
    q_c = ctxt.frame.choices[q_who, q_id]
    if p_c.known:
        raise MemoError(
            f"Cannot take KL-divergence over known variable {p_who}.{p_id}",
            hint=f"It only makes sense to take KL-divergence over uncertain variables.",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )
    if q_c.known:
        raise MemoError(
            f"Cannot take KL-divergence over known variable {q_who}.{q_id}",
            hint=f"It only makes sense to take KL-divergence over uncertain variables.",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )
    if p_c.domain != q_c.domain:
        raise MemoError(
            f"KL-divergence mismatched support",
            hint=f"Domains of {p_who}.{p_id} ({p_c.domain}) and {q_who}.{q_id} ({q_c.domain}) do not match.",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )

    idxs_p = tuple(set(c.idx for n, c in ctxt.frame.choices.items() if (not c.known) and (c != p_c)))
    idxs_q = tuple(set(c.idx for n, c in ctxt.frame.choices.items() if (not c.known) and (c != q_c)))

    ctxt.emit(f"# {ctxt.frame.name} KL-divergence")

    out = ctxt.sym("out")
    p_p = ctxt.sym("p_p")
    ctxt.emit(f"{p_p} = marg({ctxt.frame.ll}, {idxs_p})")
    q_p = ctxt.sym("q_p")
    ctxt.emit(f"{q_p} = marg({ctxt.frame.ll}, {idxs_q})")

    ctxt.emit(
        f"{out} = marg({p_p} * jnp.nan_to_num(jnp.log({p_p}) - jnp.log(jnp.swapaxes({q_p}, {-1 - p_c.idx}, {-1 - q_c.idx}))), [{p_c.idx}])"
    )
    return Value(
        tag=out,
        known=True,
        deps={n for n, c in ctxt.frame.choices.items() if c.known}
    )

@eval_expr.register
def _(e: EWith, ctxt: Context) -> Value:
    who, expr = e.who, e.expr
    if who == Name("self"):
        return eval_expr(expr, ctxt)
    ctxt.frame.ensure_child(who)

    old_frame = ctxt.frame
    ctxt.frame = ctxt.frame.children[who]
    val_ = eval_expr(expr, ctxt)
    ctxt.frame = old_frame
    if not val_.known:
        raise MemoError(
            "Asking an agent for an unknown value",
            hint=f"{who} has uncertainty about the value of the expression that {ctxt.frame.name} is imagining {who} computing. Did you perhaps mean to take {who}'s *expected* value of that expression, using E[...]?",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )

    deps = set()
    # print()
    # ic(ctxt.frame.name, who, e)
    # ic(ctxt.frame.children[who].conditions)
    # ic(val_.deps)
    for who_, id in val_.deps:
        if (who_, id) in ctxt.frame.children[who].conditions:
            z_who, z_id = ctxt.frame.children[who].conditions[(who_, id)]
            # ic(who, who_, id, z_who, z_id)
            deps.add((z_who, z_id))
        elif who_ == Name("self"):
            deps.add((who, id))
        else:
            ic(ctxt.frame.name, who, e, who_, id)
            assert False  # should never happen
    # ic(deps)
    try:
        # "all" short-circuits!!!
        known = all(ctxt.frame.choices[(who_, id_)].known for (who_, id_) in reversed(sorted(deps)))
    except Exception:
        raise
    return Value(tag=val_.tag, known=known, deps=deps)

@eval_expr.register
def _(e: EImagine, ctxt: Context) -> Value:
    do, then = e.do, e.then
    ctxt.emit(f"# {ctxt.frame.name} imagines")

    who = ctxt.frame.name

    # if ctxt.frame.parent is not None:
        # ctxt.frame = ctxt.frame.parent
        # eval_stmt(SSnapshot(who, Name(f'future_{who}'), loc=e.loc), ctxt)
        # ctxt.frame = ctxt.frame.children[who]

    old_frame = ctxt.frame
    current_frame_copy = copy.deepcopy(ctxt.frame)
    current_frame_copy.name = Name(f'imagined_{who}')
    fresh_lls(ctxt, current_frame_copy)

    ctxt.frame = current_frame_copy
    cont_saved = ctxt.continuation
    for i, stmt in enumerate(do):
        ctxt.continuation = do[i + 1:]
        eval_stmt(stmt, ctxt)
    val_ = eval_expr(then, ctxt)
    ctxt.continuation = cont_saved
    if not val_.known:
        raise MemoError(
            'trying to imagine an unknown value',
            hint=f"In this hypothetical imagined world, {who} won't be able to compute the return value you requested. Perhaps you meant to wrap the return value in E[...]?",
            user=True,
            ctxt=ctxt,
            loc=e.then.loc
        )

    new_deps = set()
    for dep in val_.deps:
        if dep in old_frame.choices:
            new_deps.add(dep)
            continue
        if dep in current_frame_copy.conditions:
            # deal with env: knows(a) scenario
            if current_frame_copy.conditions[dep][0] == current_frame_copy.name:
                new_deps.add((Name('self'), current_frame_copy.conditions[dep][1]))
                continue
        assert False  ## something bad has happened
    val_ = dataclasses.replace(val_, deps=new_deps)

    ctxt.frame = old_frame
    return val_

@eval_expr.register
def _(e: EInline, ctxt: Context) -> Value:
    val = e.val
    with ctxt.hoist():
        tag = ctxt.sym("inline")
        ctxt.emit(f'{tag} = {val}')
    return Value(
        tag=tag,
        known=True,
        deps=set()
    )

@eval_expr.register
def _(e: EPredict, ctxt: Context) -> Value:
    assert ctxt.frame.parent is not None
    expr = e.expr

    prefix: list[Name] = []
    f = ctxt.frame
    while f.parent is not None and not f.name.startswith('imagined_'):
        prefix.insert(0, f.name)
        f = f.parent

    name = Name(ctxt.sym(f'future_{ctxt.frame.name}'))
    frame_saved = ctxt.frame
    ctxt.frame = ctxt.frame.parent
    eval_stmt(SSnapshot(frame_saved.name, name, loc=e.loc), ctxt)
    ctxt.frame = frame_saved

    to_imagine: list[Stmt] = []
    for stmt in ctxt.continuation:
        for who in prefix[:-1]:
            if isinstance(stmt, SWith) and stmt.who == who:
                stmt = stmt.stmt
            else:
                break
        else:  # if no break
            if isinstance(stmt, SForAll):
                raise NotImplementedError
            elif isinstance(stmt, SPass):
                to_imagine.append(stmt)
                continue
            elif stmt.who == prefix[-1]:
                if isinstance(stmt, SKnows):
                    raise MemoError(
                        message="A 'knows' statement cannot appear after a 'wants' statement or a 'Predict' expression.",
                        hint=f"This is because {ctxt.frame.name} cannot evaluate utilities without knowing all of the relevant variables. Can you move the 'knows' statement before the 'wants' statement?",
                        user=True,
                        ctxt=ctxt,
                        loc=e.loc
                    )
                if isinstance(stmt, SShow):
                    ch_id = Id(ctxt.sym(stmt.target_id))
                    to_imagine.append(SGuess(name, ch_id, stmt.target_who, stmt.target_id, loc=stmt.loc))
                    stmt = dataclasses.replace(stmt, source_who=name, source_id=ch_id)
                to_imagine.append(dataclasses.replace(stmt, who=name))

    then_expr = EExpect(
        EWith(who=name, expr=expr, loc=expr.loc, static=False),
        reduction="expectation", warn=False, loc=expr.loc, static=False
    )
    eimagine_expr = EImagine(
        do=to_imagine,
        then=then_expr,
        loc=e.loc,
        static=then_expr.static
    )
    val = eval_expr(eimagine_expr, ctxt)
    return val

@eval_expr.register
def _(e: EUtil, ctxt: Context) -> Value:
    id = e.goal
    if id not in ctxt.frame.goals:
        raise MemoError(
            message=f"Unknown goal {id}",
            hint=f"Did you misspell {id}?",
            user=True,
            ctxt=ctxt,
            loc=e.loc
        )
    return eval_expr(EPredict(ctxt.frame.goals[id], loc=e.loc, static=False), ctxt)

def fresh_lls(ctxt: Context, f: Frame) -> None:
    if f.ll is not None:
        ll = ctxt.sym(f"{f.name}_ll")
        ctxt.emit(f"{ll} = {f.ll}")
        f.ll = ll
    for c in f.children.keys():
        fresh_lls(ctxt, f.children[c])

@singledispatch
def eval_stmt(s: Stmt, ctxt: Context) -> None:
    raise NotImplementedError

@eval_stmt.register
def _(s: SPass, ctxt: Context) -> None:
    pass

@eval_stmt.register
def _(s: SForAll, ctxt: Context) -> None:
    id, domain = s.id, s.domain
    assert ctxt.frame.name == ROOT_FRAME_NAME
    idx = ctxt.next_idx
    ctxt.next_idx += 1
    tag = ctxt.sym(f"forall_{id}")
    ctxt.emit(
        f"{tag} = jnp.array({domain}).reshape(*{(-1,) + tuple(1 for _ in range(idx))})"
    )
    ctxt.frame.choices[(Name("self"), id)] = Choice(
        tag, idx, True, domain
    )
    ctxt.forall_idxs.append((idx, id, domain))

@eval_stmt.register
def _(s: SChoose, ctxt: Context) -> None:
    who, choices, wpp = s.who, s.choices, s.wpp
    ctxt.frame.ensure_child(who)

    # briefly enter child's frame
    child_frame = ctxt.frame.children[who]
    old_frame = ctxt.frame
    ctxt.frame = child_frame

    idx_list = []
    for id, dom in choices:
        if (Name("self"), id) in ctxt.frame.choices:
            raise MemoError(
                "Repeated choice",
                hint=f"{who} has already chosen {id} earlier in this model! Pick a new name?",
                user=True,
                ctxt=ctxt,
                loc=s.loc
            )
        if id in ctxt.pctxt.static_parameters:
            raise MemoError(
                "Name conflict",
                hint=f"The name {id} is already being used as a parameter to the model.",
                user=True,
                ctxt=ctxt,
                loc=s.loc
            )
        idx = ctxt.next_idx
        ctxt.next_idx += 1
        idx_list.append(idx)
        tag = ctxt.sym(f"{who}_{id}")
        ctxt.emit(f"""# {who} choose {id}""")
        ctxt.emit(
            f"{tag} = jnp.array({dom}).reshape(*{(-1,) + tuple(1 for _ in range(idx))})"
        )

        ctxt.frame.choices[(Name("self"), id)] = Choice(
            tag, idx, True, dom
        )

    softmax = False
    match wpp:
        case EOp(Op.EXP, [logit]):
            wpp_val = eval_expr(logit, ctxt)
            softmax = True
        case _:
            wpp_val = eval_expr(wpp, ctxt)

    for id, dom in choices:
        if not wpp_val.known:
            raise MemoError(
                "Choice based on uncertain expression",
                hint=f"{who} is uncertain about the value of the expression (wpp/to_maximize) that {who} is using to choose {id}. Hence, {who} cannot compute the probabilities needed to make the choice. Perhaps you meant to take an expected value somewhere, using E[...]?",
                user=True,
                ctxt=ctxt,
                loc=s.loc
            )
        if (Name("self"), id) not in wpp_val.deps and not isinstance(wpp, ELit):
            warnings.warn(f"When {who} chooses {id}, the probability doesn't depend on the value of {id}. As a result, the choice will effectively be uniform over {id} after normalization. Are you sure this is what you want? (A uniform choice is more easily expressed with wpp=1.)")
    ctxt.frame = old_frame

    for id, dom in choices:
        child_choice = ctxt.frame.children[who].choices[Name("self"), id]
        ctxt.frame.choices[who, id] = Choice(child_choice.tag, child_choice.idx, False, dom)

    id_names_concat = '_'.join([id for id, dom in choices])
    id_ll = ctxt.sym(f"{id_names_concat}_ll")
    idx_tup = str(tuple(idx_list))
    shape_tup = ', '.join([f"{ctxt.frame.choices[who, id].tag}.shape" for id, dom in choices])
    ctxt.emit(f"{id_ll} = jnp.ones(jnp.broadcast_shapes({shape_tup}), dtype=jnp.float32) * {wpp_val.tag}")

    if softmax:
        ctxt.emit(f"{id_ll} = jnp.exp({id_ll} - maxx({id_ll}, {idx_tup}))")
    if s.reduction == "maximize":
        ctxt.emit(f"{id_ll} = 1.0 * ({id_ll} == maxx({id_ll}, {idx_tup}))")
    ctxt.emit(f"{id_ll} = jnp.nan_to_num({id_ll} / marg({id_ll}, {idx_tup}))")

    if ctxt.frame.ll is None:
        ctxt.frame.ll = ctxt.sym(f"{ctxt.frame.name}_ll")
        ctxt.emit(f"{ctxt.frame.ll} = 1.0")
    ctxt.emit(f"{ctxt.frame.ll} = {id_ll} * {ctxt.frame.ll}")

@eval_stmt.register
def _(s: SObserve, ctxt: Context) -> None:
    who, id = s.who, s.id
    if (who, id) not in ctxt.frame.choices:
        raise MemoError(
            "Observation of unmodeled choice",
            hint=f"{ctxt.frame.name} does not have {who}'s choice of {id} in their mental model. Perhaps you meant to write `{ctxt.frame.name}: thinks[ {who}: chooses({id} ...) ]` somewhere earlier in this memo?",
            user=True,
            ctxt=ctxt,
            loc=s.loc
        )
    ch = ctxt.frame.choices[(who, id)]
    if ch.known:
        raise MemoError(
            "Observation of already-known choice",
            hint=f"{ctxt.frame.name} already knows {who}'s choice of {id}. It doesn't make sense for {ctxt.frame.name} to re-observe that same choice again.",
            user=True,
            ctxt=ctxt,
            loc=s.loc
        )
    ch.known = True

    idxs = tuple([c.idx for _, c in ctxt.frame.choices.items() if not c.known])
    ctxt.emit(f"""# {ctxt.frame.name} observe {who}.{id}""")
    ctxt.emit(
        f"""{ctxt.frame.ll} = jnp.nan_to_num({ctxt.frame.ll} / marg({ctxt.frame.ll}, {idxs}))"""
    )

@eval_stmt.register
def _(s: SObserves, ctxt: Context) -> None:
    who, what, how = s.who, s.what, s.how
    ctxt.frame.ensure_child(who)
    old_frame = ctxt.frame
    ctxt.frame = ctxt.frame.children[who]
    if ctxt.frame.ll is None:
        ctxt.frame.ll = ctxt.sym(f"{ctxt.frame.name}_ll")
        ctxt.emit(f"{ctxt.frame.ll} = 1.0")
    what_val = eval_expr(what, ctxt)
    ctxt.emit(f"""# {ctxt.frame.name} factors""")
    if how == "boolean":
        ctxt.emit(f"""{what_val.tag} = jnp.bool({what_val.tag}) * 1.0""")
    ctxt.emit(f"""{ctxt.frame.ll} = {ctxt.frame.ll} * {what_val.tag}""")
    idxs = tuple([c.idx for _, c in ctxt.frame.choices.items() if not c.known])
    ctxt.emit(
        f"""{ctxt.frame.ll} = jnp.nan_to_num({ctxt.frame.ll} / marg({ctxt.frame.ll}, {idxs}))"""
    )
    ctxt.frame = old_frame

@eval_stmt.register
def _(s: SWith, ctxt: Context) -> None:
    who = s.who
    ctxt.frame.ensure_child(who)
    who, stmt = s.who, s.stmt
    f_old = ctxt.frame
    ctxt.frame = ctxt.frame.children[who]
    eval_stmt(stmt, ctxt)
    ctxt.frame = f_old

@eval_stmt.register
def _(s: SShow, ctxt: Context) -> None:
    who, target_who, target_id, source_who, source_id = (
        s.who, s.target_who, s.target_id, s.source_who, s.source_id
    )
    ctxt.emit(f"# telling {who} about {target_who}.{target_id}")
    ctxt.frame.ensure_child(who)
    if (target_who, target_id) not in ctxt.frame.children[who].choices:
        raise MemoError(
            "Observation of unmodeled choice",
            hint=f"{ctxt.frame.name} does not yet think that {who} is modeling {target_who}'s choice of {target_id}, so it doesn't make sense for {who} to observe that choice. Perhaps you meant to write `{who}: thinks[ {target_who}: chooses({target_id} ...) ]` somewhere earlier in {ctxt.frame.name}'s memo?",
            user=True,
            ctxt=ctxt,
            loc=s.loc
        )
    if (source_who, source_id) not in ctxt.frame.choices:
        raise MemoError(
            "Observation of unknown choice",
            hint=f"{ctxt.frame.name} does not yet think that {source_who} has chosen {source_id}, so cannot model {who} observing that value. Perhaps you misspelled {source_id}?",
            user=True,
            ctxt=ctxt,
            loc=s.loc
        )

    eval_stmt(
        SWith(who, SObserve(target_who, target_id, loc=None), loc=None), ctxt
    )
    target_addr = (target_who, target_id)
    source_addr = (source_who, source_id)
    target_dom = ctxt.frame.children[who].choices[target_addr].domain
    source_dom = ctxt.frame.choices[source_addr].domain
    if target_dom != source_dom:
        raise MemoError(
            "Domain mismatch",
            hint=f"{target_who}.{target_id} is from domain {target_dom}, while {source_who}.{source_id} is from domain {source_dom}.",
            user=True,
            ctxt=ctxt,
            loc=s.loc
        )
    ctxt.frame.children[who].conditions[target_addr] = source_addr
    tidx = ctxt.frame.children[who].choices[target_addr].idx
    sidx = ctxt.frame.choices[source_addr].idx
    ctxt.emit(
        f"{ctxt.frame.children[who].ll} = jnp.swapaxes(pad({ctxt.frame.children[who].ll}, {ctxt.next_idx}), -1-{sidx}, -1-{tidx})"
    )
    ctxt.frame.children[who].choices[target_addr].idx = sidx
    ctxt.frame.children[who].children[target_who].choices[(Name('self'), target_id)].idx = sidx
    ctxt.emit(
        f"{ctxt.frame.children[who].choices[target_addr].tag} = {ctxt.frame.choices[source_addr].tag}"
    )

@eval_stmt.register
def _(s: SKnows, ctxt: Context) -> None:
    who, source_who, source_id = s.who, s.source_who, s.source_id
    source_addr = (source_who, source_id)
    # out_addr = (ctxt.frame.name if source_who == "self" else source_who, source_id)
    ctxt.frame.ensure_child(who)
    if source_addr not in ctxt.frame.choices:
        raise MemoError(
            "Knowing unknown choice",
            hint=f"{ctxt.frame.name} does not yet model {source_who}'s choice of {source_id}. So, it doesn't make sense for {ctxt.frame.name} to model {who} as knowing that choice.",
            user=True,
            ctxt=ctxt,
            loc=s.loc
        )
    ctxt.frame.children[who].choices[source_addr] = dataclasses.replace(ctxt.frame.choices[source_addr], known=True)
    ctxt.frame.children[who].conditions[source_addr] = source_addr

    if source_who == "self":
        ctxt.frame.choices[(who, source_id)] = ctxt.frame.choices[source_addr]
        ctxt.frame.conditions[(who, source_id)] = (ctxt.frame.name, source_id)
    else:
        ctxt.frame.children[who].ensure_child(source_who)
        ctxt.frame.children[who].children[source_who].choices[(Name("self"), source_id)] = dataclasses.replace(ctxt.frame.choices[source_addr], known=True)
        ctxt.frame.children[who].children[source_who].conditions[(Name("self"), source_id)] = (source_who, source_id)
    ctxt.emit(f"pass  # {who} knows {source_who}.{source_id}")

@eval_stmt.register
def _(s: SSnapshot, ctxt: Context) -> None:
    who, alias = s.who, s.alias
    current_frame = ctxt.frame.children[who]
    future_frame = copy.deepcopy(current_frame)
    fresh_lls(ctxt, future_frame)
    future_frame.name = alias
    future_frame.parent = current_frame
    current_frame.children[alias] = future_frame

    # alice should be able to deal with all of the choices in future_alice's frame
    for name, id in list(future_frame.choices.keys()):
        if name == 'self':
            # alice should know about future_alice's own choices
            current_frame.choices[alias, id] = copy.deepcopy(future_frame.choices[name, id])
            # observer{ alice[future_alice.x]  -->  alice.x }
            current_frame.conditions[alias, id] = (who, id)
            # alice{ future_alice.x --> x }
            future_frame.conditions[name, id] = (Name("self"), id)
        else:
            # map everything else back to alice's own version of that
            future_frame.conditions[name, id] = (name, id)

@eval_stmt.register
def _(s: STrace, ctxt: Context) -> None:
    who = s.who
    import os, sys
    if s.loc is not None:
        loc_str = f"from {os.path.basename(s.loc.file)}, line {s.loc.line}, in @memo {s.loc.name}"
    else:
        loc_str = ""
    print(f"** Tracing {who} {loc_str}")
    if who not in ctxt.frame.children:
        print(f"-> From {ctxt.frame.name}'s perspective, there is not yet any model of {who}. The current agents currently modeled by {ctxt.frame.name} are: {', '.join(ctxt.frame.children.keys())}.")
        print()
        return

    f = ctxt.frame.children[who]
    print(f"-> {who} is tracking the following choices:")
    for key, val in f.choices.items():
        if key[0] == Name("self"):
            print(f"   - {key[1]}: {val.domain} ({'known' if val.known else 'uncertain'})")
        else:
            print(f"   - {key[0]}.{key[1]}: {val.domain} ({'known' if val.known else 'uncertain'})")
        if key in f.conditions:
            assert f.parent is not None
            print(f"     | (observed to be {f.parent.name}'s {f.conditions[key][0]}.{f.conditions[key][1]})")
    if len(f.children) == 0:
        print(f"-> {who} is not yet tracking any agents.")
    else:
        print(f"-> {who} is currently tracking the following agents:")
        for c in f.children:
            print(f"   + {c}")
    while f.parent is not None:
        print(f"-> {f.name} is being modeled by {f.parent.name}.")
        f = f.parent
    print()

@eval_stmt.register
def _(s: SWants, ctxt: Context) -> None:
    who, what, how = s.who, s.what, s.how
    ctxt.frame.ensure_child(who)
    assert what not in ctxt.frame.children[who].goals
    assert (Name("self"), what) not in ctxt.frame.choices
    ctxt.frame.children[who].goals[what] = EExpect(how, reduction='expectation', warn=False, loc=s.loc, static=False)

@eval_stmt.register
def _(s: SGuess, ctxt: Context) -> None:
    who, id, target_id, target_who = s.who, s.id, s.target_id, s.target_who
    dom = ctxt.frame.children[who].choices[target_who, target_id].domain
    eval_stmt(
        SChoose(
            who,
            [(id, dom)],
            EExpect(
                expr=EOp(
                    Op.EQ,
                    [
                        EChoice(
                            id,
                            loc=s.loc,
                            static=False
                        ),
                        EWith(
                            s.target_who,
                            EChoice(
                                s.target_id,
                                loc=s.loc,
                                static=False
                            ),
                            loc=s.loc,
                            static=False
                        )
                    ],
                    loc=s.loc,
                    static=False
                ),
                reduction='expectation',
                warn=False,
                loc=s.loc,
                static=False),
            reduction='normalize',
            loc=s.loc
        ),
        ctxt
    )