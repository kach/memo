from __future__ import annotations
from typing import NewType, Any, Tuple, Literal, Iterator

from contextlib import contextmanager
import itertools
from enum import Enum
import dataclasses
from dataclasses import dataclass, field
import copy

import textwrap
from io import StringIO

import warnings

try:
    from icecream import ic  # type: ignore
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


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
    wpp_deps: set[tuple[Name, Id]]


@dataclass
class Frame:
    name: Name
    choices: dict[tuple[Name, Id], Choice] = field(default_factory=dict)
    children: dict[Name, Frame] = field(default_factory=dict)
    conditions: dict[tuple[Name, Id], tuple[Name, Id]] = field(
        default_factory=dict
    )
    # key is a choice in this frame, val is a choice in the parent frame
    # used to create "aliases" in child's choices, e.g. in observe/knows
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
    args: list[Expr]
    ids: list[Tuple[Id, Name, Id]]


@dataclass(frozen=True)
class EChoice(ExprSyntaxNode):
    id: Id


@dataclass(frozen=True)
class EExpect(ExprSyntaxNode):
    expr: Expr
    reduction: Literal["expectation", "variance"]


@dataclass(frozen=True)
class EEntropy(ExprSyntaxNode):
    rvs: list[tuple[Name, Id]]


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


Expr = ELit | EOp | EFFI | EMemo | EChoice | EExpect | EEntropy | EWith | EImagine | ECost


@dataclass(frozen=True)
class SPass(SyntaxNode):
    pass


@dataclass(frozen=True)
class SChoose(SyntaxNode):
    who: Name
    id: Id
    domain: Dom
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
class SForAll(SyntaxNode):
    id: Id
    domain: Dom


@dataclass(frozen=True)
class SKnows(SyntaxNode):
    who: Name
    source_who: Name
    source_id: Id


Stmt = SPass | SChoose | SObserve | SWith | SShow | SForAll | SKnows


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
class Context:
    frame: Frame

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


def pprint_stmt(s: Stmt) -> str:
    match s:
        case SPass():
            return f""
        case SChoose(who, id, dom, wpp):
            wpp_str = pprint_expr(wpp)
            if len(wpp_str) > 10:
                wpp_str = "\n" + textwrap.indent(wpp_str, "  ")
            reduction_str = "wpp" if s.reduction == "normalize" else "to_maximize"
            return f"{who}: chooses({id} in {dom}, {reduction_str}={wpp_str})"
        case SObserve(who, id):
            return f"observe {who}.{id}"
        case SWith(who, stmt):
            return f"{who}: thinks[ {pprint_stmt(stmt)} ]"
        case SShow(who, target_who, target_id, source_who, source_id):
            if source_who == Name("self"):
                return f"{who}: observes [{target_who}.{target_id}] is {source_id}"
            else:
                return f"{who}: observes [{target_who}.{target_id}] is {source_who}.{source_id}"
        case SForAll(id, domain):
            return f"given: {id} in {domain}"
        case SKnows(who, source_who, source_id):
            return f"{who}: knows({source_who}.{source_id})"
    raise NotImplementedError(s)


def pprint_expr(e: Expr) -> str:
    match e:
        case ELit(a):
            return f"{a}"
        case EOp(op, args):
            match op:
                case Op.ADD:
                    return f"({pprint_expr(args[0])} + {pprint_expr(args[1])})"
                case Op.SUB:
                    return f"({pprint_expr(args[0])} - {pprint_expr(args[1])})"
                case Op.MUL:
                    return f"({pprint_expr(args[0])} * {pprint_expr(args[1])})"
                case Op.DIV:
                    return f"({pprint_expr(args[0])} / {pprint_expr(args[1])})"
                case Op.POW:
                    return f"({pprint_expr(args[0])} ** {pprint_expr(args[1])})"
                case Op.EQ:
                    return f"({pprint_expr(args[0])} == {pprint_expr(args[1])})"
                case Op.NEQ:
                    return f"({pprint_expr(args[0])} != {pprint_expr(args[1])})"
                case Op.LT:
                    return f"({pprint_expr(args[0])} < {pprint_expr(args[1])})"
                case Op.LTE:
                    return f"({pprint_expr(args[0])} <= {pprint_expr(args[1])})"
                case Op.GT:
                    return f"({pprint_expr(args[0])} > {pprint_expr(args[1])})"
                case Op.GTE:
                    return f"({pprint_expr(args[0])} >= {pprint_expr(args[1])})"
                case Op.AND:
                    return f"({pprint_expr(args[0])} & {pprint_expr(args[1])})"
                case Op.OR:
                    return f"({pprint_expr(args[0])} | {pprint_expr(args[1])})"
                case Op.XOR:
                    return f"({pprint_expr(args[0])} ^ {pprint_expr(args[1])})"
                case Op.EXP:
                    return f"exp({pprint_expr(args[0])})"
                case Op.ABS:
                    return f"abs({pprint_expr(args[0])})"
                case Op.LOG:
                    return f"log({pprint_expr(args[0])})"
                case Op.UADD:
                    return f"(+{pprint_expr(args[0])})"
                case Op.NEG:
                    return f"(-{pprint_expr(args[0])})"
                case Op.INV:
                    return f"(~{pprint_expr(args[0])})"
                case Op.ITE:
                    c_str = pprint_expr(args[0])
                    t_str = pprint_expr(args[1])
                    f_str = pprint_expr(args[2])
                    if len(t_str) + len(f_str) < 40:
                        return f"(if {c_str} then {t_str} else {f_str})"
                    return f"""\
(if {c_str} then
{textwrap.indent(t_str, '   ')}
 else
{textwrap.indent(f_str, '   ')})\
"""
        case EFFI(name, args):
            return f"{name}({', '.join(pprint_expr(arg) for arg in args)})"
        case EMemo(name, args, ids):
            return "EMEMO"
        case EChoice(id):
            return f"{id}"
        case EExpect(expr):
            return f"E[ {pprint_expr(expr)} ]"
        case EEntropy(rvs):
            return f"H[ {[z[0] + '.' + z[1] for z in rvs ]} ]"
        case EWith(who, expr):
            return f"{who}[ {pprint_expr(expr)} ]"
        case EImagine(do, then):
            stmts = "\n".join([pprint_stmt(s) for s in do] + [pprint_expr(then)])
            stmts_block = textwrap.indent(stmts, "  ")
            return f"""\
imagine [
{stmts_block}
]"""
        case ECost(name, args):
            return f"(cost @ {name}({', '.join(pprint_expr(arg) for arg in args)}))"
    raise NotImplementedError


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


def eval_expr(e: Expr, ctxt: Context) -> Value:
    match e:
        case ELit(val):
            with ctxt.hoist():
                out = ctxt.sym("lit")
                ctxt.emit(f"{out} = {val}")
            return Value(tag=out, known=True, deps=set())

        case EChoice(id):
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
            # out = ctxt.sym("ch")
            # ctxt.emit(f"{out} = {ch.tag}")
            return Value(
                tag=ch.tag, known=ch.known, deps=set([(Name("self"), id)])
            )

        case EFFI(name, args):
            args_out = []
            for arg in args:
                args_out.append(eval_expr(arg, ctxt))
            known = all(arg.known for arg in args_out)
            deps = set().union(*(arg.deps for arg in args_out))
            with ctxt.hoist(e.static):
                out = ctxt.sym(f"ffi_{name}")
                ctxt.emit(f'{out} = ffi({name}, {", ".join(arg.tag for arg in args_out)})')
                if e.static:
                    ctxt.emit(f'{out} = {out}.item()')
            return Value(tag=out, known=known, deps=deps)

        case ECost(name, args):
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
                ctxt.emit(f'_, {res} = {name}({assemble_tags([arg.tag for arg in args_out], compute_cost=True)})')
                ctxt.emit(f'{res} = {res}.cost')
                ctxt.dedent()
                ctxt.emit('else:')
                ctxt.indent()
                ctxt.emit(f'{res} = 0')
                ctxt.dedent()
            return Value(tag=res, known=True, deps=set())

        case EMemo(name, args, ids):
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
                ctxt.emit(f"""check_domains({name}._doms, {repr(tuple(str(d) for d in doms))})""")
                ctxt.emit(f'if {" and ".join(ctxt.path_condition) if len(ctxt.path_condition) > 0 else "True"}:')
                ctxt.indent()
                ctxt.emit(f'{res}, res_aux = {name}({assemble_tags([arg.tag for arg in args_out], return_aux=True, compute_cost='compute_cost')})')
                ctxt.emit(f"if compute_cost: aux.cost += res_aux.cost")
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
            # ctxt.emit(f"print({tuple(permuted_dims)})")
            # ctxt.emit(f"print({res}, {res}.shape)")

            return Value(
                tag=res,
                known=all(ctxt.frame.choices[sn, si].known for _, sn, si in ids),
                # deps=set.union(
                #     *(ctxt.frame.choices[sn, si].wpp_deps for _, sn, si in ids)
                # ),
                deps=set((sn, si) for _, sn, si in ids)
            )

        case EOp(op, args):
            if op in [
                Op.ADD,
                Op.SUB,
                Op.MUL,
                Op.DIV,
                Op.POW,
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

        case EExpect(expr, reduction):
            val_ = eval_expr(expr, ctxt)
            if all(ctxt.frame.choices[c].known for c in sorted(val_.deps)):
                warnings.warn(f"Redundant expectation {pprint_expr(e)}, not marginalizing")
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
            # deps = set.union(*[ctxt.frame.choices[c].wpp_deps for c in val_.deps]) | set.union(*[ctxt.frame.choices[c].wpp_deps for c in ctxt.frame.conditions.keys()])
            # ic(deps)
            return Value(
                tag=out,
                known=True,
                # deps={(name, id) for (name, id) in val_.deps if ctxt.frame.choices[(name, id)].known}
                deps=deps
            )

        case EEntropy(rvs):
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

        case EWith(who, expr):
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
            # ic(ctxt.frame.name, who, pprint_expr(e))
            # ic(ctxt.frame.children[who].conditions)
            # ic(val_.deps)
            for who_, id in val_.deps:
                if who_ == Name("self"):
                    deps.add((who, id))
                elif (who_, id) in ctxt.frame.children[who].conditions:
                    z_who, z_id = ctxt.frame.children[who].conditions[(who_, id)]
                    deps.add((z_who, z_id))
                else:
                    ic(ctxt.frame.name, who, pprint_expr(e), who_, id)
                    assert False  # should never happen
            # ic(deps)
            try:
                # "all" short-circuits!!!
                known = all(ctxt.frame.choices[(who_, id_)].known for (who_, id_) in reversed(sorted(deps)))
            except Exception:
                raise
            return Value(tag=val_.tag, known=known, deps=deps)

        case EImagine(do, then):
            ctxt.emit(f"# {ctxt.frame.name} imagines")

            future_name = Name(f"future_{ctxt.frame.name}")
            old_frame = ctxt.frame
            old_frame.children[future_name] = Frame(
                name=future_name,
                parent=old_frame
            )
            # this copy is for the "scratchpad"
            current_frame_copy = copy.deepcopy(ctxt.frame)
            fresh_lls(ctxt, current_frame_copy)

            # this copy is for the "inner" frame, representing the future self
            future_frame = copy.deepcopy(current_frame_copy)
            fresh_lls(ctxt, future_frame)
            future_frame.name = future_name
            future_frame.parent = current_frame_copy
            current_frame_copy.children[future_name] = future_frame
            # needed so that future_alice deps get correctly translated by EWith
            for k in future_frame.conditions.keys():
                future_frame.conditions[k] = k

            # alice should know all of future_alice's own choices
            for name, id in list(current_frame_copy.choices.keys()):
                if name != 'self':
                    continue
                current_frame_copy.choices[future_name, id] = copy.deepcopy(current_frame_copy.choices[name, id])
                current_frame_copy.conditions[future_name, id] = (name, id)

                # TODO: is this necessary?
                old_frame.conditions[future_name, id] = (name, id)

            ctxt.frame = current_frame_copy
            for stmt in do:
                eval_stmt(stmt, ctxt)
            val_ = eval_expr(then, ctxt)
            # assert val_.known  # TODO: error message? not sure if needed...

            # We only want to "translate" choices made by future self.
            # There must be a better way of doing this as well.
            new_deps = {
                ctxt.frame.conditions.get(d, d) if d[0] == future_name else d
                for d in val_.deps
            }
            # ic(new_deps)
            val_ = Value(  ## ??
                tag=val_.tag,
                known=val_.known,
                deps=new_deps
            )

            # if we do something like env.knows(a) within imagine, then
            # we need some way of "translating" that knowledge back into
            # the outside world. this is hack, not very robust because it
            # wouldn't survive multiple consecutive imagine statements, etc.
            for d in current_frame_copy.conditions.keys():
                old_frame.conditions[d] = current_frame_copy.conditions[d]

            ctxt.frame = old_frame
            return val_

    raise NotImplementedError


def fresh_lls(ctxt: Context, f: Frame) -> None:
    if f.ll is not None:
        ll = ctxt.sym(f"{f.name}_ll")
        ctxt.emit(f"{ll} = {f.ll}")
        f.ll = ll
    for c in f.children.keys():
        fresh_lls(ctxt, f.children[c])


def eval_stmt(s: Stmt, ctxt: Context) -> None:
    match s:
        case SPass():
            pass

        case SForAll(id, domain):
            assert ctxt.frame.name == ROOT_FRAME_NAME
            idx = ctxt.next_idx
            ctxt.next_idx += 1
            tag = ctxt.sym(f"forall_{id}")
            ctxt.emit(
                f"{tag} = jnp.array({domain}).reshape(*{(-1,) + tuple(1 for _ in range(idx))})"
            )
            ctxt.frame.choices[(Name("self"), id)] = Choice(
                tag, idx, True, domain, set()
            )
            ctxt.forall_idxs.append((idx, id, domain))

        case SChoose(who, id, domain, wpp):
            ctxt.frame.ensure_child(who)
            idx = ctxt.next_idx
            ctxt.next_idx += 1
            tag = ctxt.sym(f"{who}_{id}")
            ctxt.emit(f"""# {who} choose {id}""")
            ctxt.emit(
                f"{tag} = jnp.array({domain}).reshape(*{(-1,) + tuple(1 for _ in range(idx))})"
            )

            # briefly enter child's frame
            child_frame = ctxt.frame.children[who]
            old_frame = ctxt.frame
            ctxt.frame = child_frame
            ctxt.frame.choices[(Name("self"), id)] = Choice(
                tag, idx, True, domain, set()
            )
            wpp_val = eval_expr(wpp, ctxt)
            if not wpp_val.known:
                raise MemoError(
                    "Choice based on uncertain expression",
                    hint=f"{who} is uncertain about the value of the expression (wpp/to_maximize) that {who} is using to choose {id}. Hence, {who} cannot compute the probabilities needed to make the choice. Perhaps you meant to take an expected value somewhere, using E[...]?",
                    user=True,
                    ctxt=ctxt,
                    loc=s.loc
                )
            if (Name("self"), id) not in wpp_val.deps and not isinstance(wpp, ELit):
                warnings.warn(f"When {who} chooses {id}, the probability doesn't depend on the value of {id}, and is thus the same for all values of {id}. As a result, the choice will effectively be uniform after normalization. Are you sure this is what you want? (A uniform choice is more easily expressed with wpp=1.)")
            ctxt.frame.choices[(Name("self"), id)].wpp_deps = wpp_val.deps
            ctxt.frame = old_frame

            new_deps = set()
            for who_, id_ in wpp_val.deps:
                if who_ == Name("self"):
                    new_deps.add((who, id_))
                elif (who_, id_) in child_frame.conditions:
                    new_deps.add(child_frame.conditions[(who_, id_)])
                else:
                    ic(child_frame.name, child_frame.conditions)
                    assert False, f"Unexpected wpp_val.dep of {who_}.{id_} for choice {who}.{id}"
            ctxt.frame.choices[(who, id)] = Choice(tag, idx, False, domain, new_deps)
            id_ll = ctxt.sym(f"{id}_ll")
            ctxt.emit(
                f"{id_ll} = jnp.ones_like({tag}, dtype=jnp.float32) * {wpp_val.tag}"
            )
            if s.reduction == "normalize":
                ctxt.emit(
                    f"{id_ll} = jnp.nan_to_num({id_ll} / marg({id_ll}, ({idx},)))"
                )
            elif s.reduction == "maximize":
                argmax_tag = ctxt.sym(f"{id}_argmax")
                ctxt.emit(f"{argmax_tag} = jnp.argmax({id_ll}, {-1 - idx})")
                ctxt.emit(
                    f"{id_ll} = jnp.nan_to_num(jax.nn.one_hot({argmax_tag}, len({domain}), dtype=jnp.float32, axis={-1 - idx}))"
                )
            if ctxt.frame.ll is None:
                ctxt.frame.ll = ctxt.sym(f"{ctxt.frame.name}_ll")
                ctxt.emit(f"{ctxt.frame.ll} = 1.0")
            ctxt.emit(f"{ctxt.frame.ll} = {id_ll} * {ctxt.frame.ll}")


        case SObserve(who, id):
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

            for ch_addr, ch_val in ctxt.frame.choices.items():
                if not ch_val.known:
                    ch_val.wpp_deps.update(ctxt.frame.choices[(who, id)].wpp_deps)

            idxs = tuple([c.idx for _, c in ctxt.frame.choices.items() if not c.known])
            ctxt.emit(f"""# {ctxt.frame.name} observe {who}.{id}""")
            ctxt.emit(
                f"""{ctxt.frame.ll} = jnp.nan_to_num({ctxt.frame.ll} / marg({ctxt.frame.ll}, {idxs}))"""
            )

        case SWith(who, stmt):  # TODO: this could take many "who"s as input
            ctxt.frame.ensure_child(who)
            f_old = ctxt.frame
            ctxt.frame = ctxt.frame.children[who]
            eval_stmt(stmt, ctxt)
            ctxt.frame = f_old

        case SShow(who, target_who, target_id, source_who, source_id):
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
            ctxt.frame.children[who].choices[target_addr].idx = ctxt.frame.choices[
                source_addr
            ].idx
            ctxt.emit(
                f"{ctxt.frame.children[who].choices[target_addr].tag} = {ctxt.frame.choices[source_addr].tag}"
            )

        case SKnows(who, source_who, source_id):
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

        case _:
            raise NotImplementedError
