from __future__ import annotations
from typing import NewType, Any, Tuple

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
    static: bool


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
    )  # used to translate in EWith
    ll: str | None = None
    parent: Frame | None = None


@dataclass(frozen=True, kw_only=True)
class SyntaxNode:
    loc: SourceLocation | None


@dataclass(frozen=True)
class ELit(SyntaxNode):
    value: float | str


Op = Enum(
    "Op",
    [
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "EQ",
        "LT",
        "GT",
        "AND",
        "OR",
        "EXP",
        "NEG",
        "INV",
        "ITE",
    ],
)


@dataclass(frozen=True)
class EOp(SyntaxNode):
    op: Op
    args: list[Expr]


@dataclass(frozen=True)
class EFFI(SyntaxNode):
    name: str
    args: list[Expr]


@dataclass(frozen=True)
class EMemo(SyntaxNode):
    name: str
    args: list[Expr]
    ids: list[Tuple[Id, Name, Id]]


@dataclass(frozen=True)
class EChoice(SyntaxNode):
    id: Id


@dataclass(frozen=True)
class EExpect(SyntaxNode):
    expr: Expr


@dataclass(frozen=True)
class EWith(SyntaxNode):
    who: Name
    expr: Expr


@dataclass(frozen=True)
class EImagine(SyntaxNode):
    do: list[Stmt]
    then: Expr


Expr = ELit | EOp | EFFI | EMemo | EChoice | EExpect | EWith | EImagine


@dataclass(frozen=True)
class SPass(SyntaxNode):
    pass


@dataclass(frozen=True)
class SChoose(SyntaxNode):
    who: Name
    id: Id
    domain: Dom
    wpp: Expr


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


@dataclass
class Context:
    next_idx: int
    frame: Frame
    io: StringIO
    idx_history: list[str]
    _sym: int = -1

    tab_level: int = 0

    def emit(self: Context, line: str) -> None:
        print("    " * self.tab_level + line, file=self.io)

    def indent(self: Context) -> None:
        self.tab_level += 1

    def dedent(self: Context) -> None:
        self.tab_level -= 1

    def sym(self, hint: str = "") -> str:
        self._sym += 1
        return f"{hint}_{self._sym}"

    forall_idxs: list[tuple[int, Id, Dom]] = field(default_factory=list)


HEADER = """\
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
"""


def pprint_stmt(s: Stmt) -> str:
    match s:
        case SPass():
            return f""
        case SChoose(who, id, dom, wpp):
            wpp_str = pprint_expr(wpp)
            if len(wpp_str) > 10:
                wpp_str = "\n" + textwrap.indent(wpp_str, "  ")
            return f"{who}: chooses({id} in {dom}, wpp={wpp_str})"
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
    raise NotImplementedError


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
                case Op.EQ:
                    return f"({pprint_expr(args[0])} == {pprint_expr(args[1])})"
                case Op.LT:
                    return f"({pprint_expr(args[0])} < {pprint_expr(args[1])})"
                case Op.GT:
                    return f"({pprint_expr(args[0])} > {pprint_expr(args[1])})"
                case Op.AND:
                    return f"({pprint_expr(args[0])} & {pprint_expr(args[1])})"
                case Op.OR:
                    return f"({pprint_expr(args[0])} | {pprint_expr(args[1])})"
                case Op.EXP:
                    return f"exp({pprint_expr(args[0])})"
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
        case EChoice(id):
            return f"{id}"
        case EExpect(expr):
            return f"E[ {pprint_expr(expr)} ]"
        case EWith(who, expr):
            return f"{who}[ {pprint_expr(expr)} ]"
        case EImagine(do, then):
            stmts = "\n".join([pprint_stmt(s) for s in do] + [pprint_expr(then)])
            stmts_block = textwrap.indent(stmts, "  ")
            return f"""\
imagine [
{stmts_block}
]"""
    raise NotImplementedError


def eval_expr(e: Expr, ctxt: Context) -> Value:
    match e:
        case ELit(val):
            out = ctxt.sym("lit")
            # ctxt.emit(f"{out} = jnp.array({val})")
            ctxt.emit(f"{out} = {val}")
            return Value(tag=out, known=True, deps=set(), static=True)

        case EChoice(id):
            if (Name("self"), id) not in ctxt.frame.choices:
                print(ctxt.frame.choices)
                raise Exception(f"{ctxt.frame.name} has not yet chosen {id}")
            ch = ctxt.frame.choices[(Name("self"), id)]
            # out = ctxt.sym("ch")
            # ctxt.emit(f"{out} = {ch.tag}")
            return Value(
                tag=ch.tag, known=ch.known, deps=set([(Name("self"), id)]), static=False
            )

        case EFFI(name, args):
            args_out = []
            for arg in args:
                args_out.append(eval_expr(arg, ctxt))
            out = ctxt.sym(f"ffi_{name}")
            known = all(arg.known for arg in args_out)
            deps = set().union(*(arg.deps for arg in args_out))
            ctxt.emit(f'{out} = ffi({name}, {", ".join(arg.tag for arg in args_out)})')
            return Value(
                tag=out,
                known=known,
                deps=deps,
                static=all(arg.static for arg in args_out),
            )

        case EMemo(name, args, ids):
            args_out = []
            for arg in args:
                args_out.append(eval_expr(arg, ctxt))

            for arg_val, arg_node in zip(args_out, args):
                if not arg_val.static:
                    raise MemoError(
                        "parameter not statically known",
                        hint="""When calling a memo, you can only pass in parameters that are fixed ("static") values that memo can compute without reasoning about agents. Such values cannot depend on any agents' choices -- only on literal numeric values and other parameters. This constraint is what enables memo to help you fit/optimize parameters fast by gradient descent.""",
                        user=True,
                        ctxt=None,
                        loc=arg_node.loc,
                    )

            res = ctxt.sym(f"result_array")
            ctxt.emit(f'{res} = {name}({", ".join(arg.tag for arg in args_out)})')
            # ctxt.emit(f"print({res}, {res}.shape)")

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
                deps=set.union(
                    *(ctxt.frame.choices[sn, si].wpp_deps for _, sn, si in ids)
                ),
                static=False,
            )

        case EOp(op, args):
            out = ctxt.sym(f"op_{op.name.lower()}")
            if op in [
                Op.ADD,
                Op.SUB,
                Op.MUL,
                Op.DIV,
                Op.EQ,
                Op.LT,
                Op.GT,
                Op.AND,
                Op.OR,
            ]:
                assert len(args) == 2
                l = eval_expr(args[0], ctxt)
                r = eval_expr(args[1], ctxt)
                match op:
                    case Op.ADD:
                        ctxt.emit(f"{out} = {l.tag} + {r.tag}")
                    case Op.SUB:
                        ctxt.emit(f"{out} = {l.tag} - {r.tag}")
                    case Op.MUL:
                        ctxt.emit(f"{out} = {l.tag} * {r.tag}")
                    case Op.DIV:
                        ctxt.emit(f"{out} = {l.tag} / {r.tag}")
                    case Op.EQ:
                        ctxt.emit(f"{out} = jnp.equal({l.tag}, {r.tag})")
                    case Op.LT:
                        ctxt.emit(f"{out} = jnp.less({l.tag}, {r.tag})")
                    case Op.GT:
                        ctxt.emit(f"{out} = jnp.greater({l.tag}, {r.tag})")
                    case Op.AND:
                        ctxt.emit(f"{out} = {l.tag} & {r.tag}")
                    case Op.OR:
                        ctxt.emit(f"{out} = {l.tag} | {r.tag}")
                return Value(
                    tag=out,
                    known=l.known and r.known,
                    deps=l.deps | r.deps,
                    static=l.static and r.static,
                )
            elif op in [Op.EXP, Op.NEG, Op.INV]:
                assert len(args) == 1
                l = eval_expr(args[0], ctxt)
                match op:
                    case Op.EXP:
                        ctxt.emit(f"{out} = jnp.exp({l.tag})")
                    case Op.NEG:
                        ctxt.emit(f"{out} = -({l.tag})")
                    case Op.INV:
                        ctxt.emit(f"{out} = ~({l.tag})")
                return Value(tag=out, known=l.known, deps=l.deps, static=l.static)
            elif op == Op.ITE:
                assert len(args) == 3
                c = eval_expr(args[0], ctxt)
                if c.static:
                    ctxt.emit(f"if {c.tag}:")
                    ctxt.indent()
                    t = eval_expr(args[1], ctxt)
                    ctxt.emit(f"{out} = {t.tag}")
                    ctxt.dedent()
                    ctxt.emit("else:")
                    ctxt.indent()
                    f = eval_expr(args[2], ctxt)
                    ctxt.emit(f"{out} = {f.tag}")
                    ctxt.dedent()

                    # ctxt.emit(f"{out} = ({t.tag}) if ({c.tag}) else ({f.tag})")
                    return Value(
                        tag=out,
                        known=c.known and t.known and f.known,
                        deps=c.deps | t.deps | f.deps,
                        static=t.static and f.static,
                    )
                else:
                    t = eval_expr(args[1], ctxt)
                    f = eval_expr(args[2], ctxt)
                    ctxt.emit(f"{out} = jnp.where({c.tag}, {t.tag}, {f.tag})")
                    return Value(
                        tag=out,
                        known=c.known and t.known and f.known,
                        deps=c.deps | t.deps | f.deps,
                        static=False,
                    )
            else:
                raise NotImplementedError

        case EExpect(expr):
            val_ = eval_expr(expr, ctxt)
            idxs_to_marginalize = tuple(
                c.idx for _, c in ctxt.frame.choices.items() if not c.known
            )
            if (
                len(
                    [
                        ctxt.frame.choices[(name, id)].idx
                        for (name, id) in val_.deps
                        if not ctxt.frame.choices[(name, id)].known
                    ]
                )
                == 0
            ):
                warnings.warn(
                    f"Redundant expectation {pprint_expr(e)}, not marginalizing"
                )
                return val_
            ctxt.emit(f"# {ctxt.frame.name} expectation")
            #             ctxt.emit(f'print({ctxt.frame.ll}, {ctxt.frame.ll}.shape)')
            #             ctxt.emit(f'print({val_.tag}, {val_.tag}.shape)')
            #             ctxt.emit(f'print(list(reversed({ctxt.idx_history})))')
            #             ctxt.emit(f'print({idxs_to_marginalize}, {[ctxt.idx_history[i] for i in idxs_to_marginalize]})')
            #             ctxt.emit(f'''\
            # for k in range(3):
            #     for i in range(3):
            #         for j in range(3):
            #             print(i, j, "|", k, {ctxt.frame.ll}[..., i, j, k], {val_.tag}[..., i, j, :])
            # ''')

            out = ctxt.sym("exp")
            ctxt.emit(
                f"{out} = marg({ctxt.frame.ll} * {val_.tag}, {idxs_to_marginalize})"
            )
            deps = (
                {
                    c
                    for c, _ in ctxt.frame.choices.items()
                    if ctxt.frame.choices[c].known
                }
                # | set.union(*[ctxt.frame.choices[c].wpp_deps for c in val_.deps])
            )
            # ic({c: ctxt.frame.choices[c].wpp_deps for c in val_.deps})
            # ic(val_, deps)
            return Value(
                tag=out,
                known=True,
                # deps={(name, id) for (name, id) in val_.deps if ctxt.frame.choices[(name, id)].known}
                deps=deps,
                static=False,
            )

        case EWith(who, expr):
            if who == Name("self"):
                return eval_expr(expr, ctxt)
            if who not in ctxt.frame.children:
                print(who)
                print(expr)
                raise Exception(f"{ctxt.frame.name} asks, who is {who}?")

            old_frame = ctxt.frame
            ctxt.frame = ctxt.frame.children[who]
            val_ = eval_expr(expr, ctxt)
            ctxt.frame = old_frame
            if not val_.known:
                raise Exception(
                    f"{who} does not know {expr}. Did you mean to take {who}'s expected value?"
                )

            # ic(val_)
            deps = set()
            for who_, id in val_.deps:
                if who_ == Name("self"):
                    if who.startswith(
                        "future_"
                    ):  ## TODO: there is definitely a bug here
                        deps.add((Name("self"), id))
                        # ic(who, id)
                    else:
                        deps.add((who, id))
                elif (who_, id) in ctxt.frame.children[who].conditions:
                    deps.add(ctxt.frame.children[who].conditions[(who_, id)])
                else:
                    ic(ctxt.frame.name, who, val_, (who_, id))
                    raise Exception("??")  # should always be true
            try:
                known = all(ctxt.frame.choices[(who, id)].known for (who, id) in deps)
            except Exception as e__:
                print(val_)
                print(ctxt.frame.choices)
                print(who, id)
                raise e__
            return Value(tag=val_.tag, known=known, deps=deps, static=False)

        case EImagine(do, then):
            ctxt.emit(f"# {ctxt.frame.name} imagines")

            future_name = Name(f"future_{ctxt.frame.name}")
            old_frame = ctxt.frame
            old_frame.children[future_name] = Frame(name=future_name, parent=old_frame)
            current_frame_copy = copy.deepcopy(ctxt.frame)
            fresh_lls(ctxt, current_frame_copy)

            child_frame = copy.deepcopy(current_frame_copy)
            fresh_lls(ctxt, child_frame)
            child_frame.name = future_name
            child_frame.parent = current_frame_copy
            current_frame_copy.children[future_name] = child_frame
            k = list(current_frame_copy.choices.keys())
            for name, id in k:
                current_frame_copy.choices[future_name, id] = (
                    current_frame_copy.choices[name, id]
                )
                current_frame_copy.conditions[future_name, id] = (name, id)
                old_frame.conditions[future_name, id] = (name, id)

            ctxt.frame = current_frame_copy
            for stmt in do:
                eval_stmt(stmt, ctxt)
            val_ = eval_expr(then, ctxt)
            val_ = Value(  ## ??
                tag=val_.tag,
                known=val_.known,
                deps={ctxt.frame.conditions.get(d, d) for d in val_.deps},
                static=val_.static,
            )
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
            assert ctxt.frame.name == "root"
            idx = ctxt.next_idx
            ctxt.next_idx += 1
            ctxt.idx_history.append(f"forall {id}")
            tag = ctxt.sym(f"forall_{id}")
            ctxt.emit(
                f"{tag} = jnp.array({domain}).reshape(*{(-1,) + tuple(1 for _ in range(idx))})"
            )
            ctxt.frame.choices[(Name("self"), id)] = Choice(
                tag, idx, True, domain, set()
            )
            ctxt.forall_idxs.append((idx, id, domain))

        case SChoose(who, id, domain, wpp):
            if who not in ctxt.frame.children:
                ctxt.frame.children[who] = Frame(name=who, parent=ctxt.frame)
            idx = ctxt.next_idx
            ctxt.next_idx += 1
            ctxt.idx_history.append(f"{who}.{id}")
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
                raise Exception(
                    f"{ctxt.frame.name} does not know wpp when choosing {id}"
                )
            ctxt.frame.choices[(Name("self"), id)].wpp_deps = wpp_val.deps
            ctxt.frame = old_frame

            new_deps = set()
            for who_, id_ in wpp_val.deps:
                if who_ == Name("self"):
                    new_deps.add((who, id_))
                elif (who_, id_) in child_frame.conditions:
                    new_deps.add(child_frame.conditions[(who_, id_)])
                else:
                    ic(child_frame.conditions)
                    ic(wpp_val.deps)
                    ic(child_frame.name)
                    raise Exception(
                        f"Unexpected wpp_val.dep of {who_}.{id_} for choice {who}.{id}"
                    )  # should always be true
            ctxt.frame.choices[(who, id)] = Choice(tag, idx, False, domain, new_deps)
            id_ll = ctxt.sym(f"{id}_ll")
            ctxt.emit(
                f"{id_ll} = jnp.ones_like({tag}, dtype=jnp.float32) * {wpp_val.tag}"
            )
            ctxt.emit(f"{id_ll} = jnp.nan_to_num({id_ll} / marg({id_ll}, ({idx},)))")
            if ctxt.frame.ll is None:
                ctxt.frame.ll = ctxt.sym(f"{ctxt.frame.name}_ll")
                ctxt.emit(f"{ctxt.frame.ll} = 1.0")
            ctxt.emit(f"{ctxt.frame.ll} = {id_ll} * {ctxt.frame.ll}")

            # ctxt.emit(f'print("{id_ll}", {id_ll}.tolist(), {id_ll}.shape)')
            # ctxt.emit(f'print("{ctxt.frame.ll}", {ctxt.frame.ll}.tolist(), {ctxt.frame.ll}.shape); print()')

        case SObserve(who, id):
            if (who, id) not in ctxt.frame.choices:
                raise Exception(
                    f"{ctxt.frame.name} does not think that {who} chose {id}"
                )
            ch = ctxt.frame.choices[(who, id)]
            if ch.known:
                raise Exception(f"{ctxt.frame.name} already knows {who}.{id}")
            ch.known = True

            # ic(f'{ctxt.frame.name} observes {who}.{id}')
            for ch_addr, ch_val in ctxt.frame.choices.items():
                if not ch_val.known:
                    # ic('updating belief about', ch_addr)
                    # ic('previously depended on', ch_val.wpp_deps)
                    # ic('now adding', ctxt.frame.choices[(who, id)].wpp_deps)
                    ch_val.wpp_deps.update(ctxt.frame.choices[(who, id)].wpp_deps)
                    # ch_val.wpp_deps.add((who, id))

            idxs = tuple([c.idx for _, c in ctxt.frame.choices.items() if not c.known])
            ctxt.emit(f"""# {ctxt.frame.name} observe {who}.{id}""")
            ctxt.emit(
                f"""{ctxt.frame.ll} = jnp.nan_to_num({ctxt.frame.ll} / marg({ctxt.frame.ll}, {idxs}))"""
            )

        case SWith(who, stmt):  # TODO: this could take many "who"s as input
            if who not in ctxt.frame.children:
                ctxt.frame.children[who] = Frame(name=who, parent=ctxt.frame)
            f_old = ctxt.frame
            ctxt.frame = ctxt.frame.children[who]
            eval_stmt(stmt, ctxt)
            ctxt.frame = f_old

        case SShow(who, target_who, target_id, source_who, source_id):
            ctxt.emit(f"# telling {who} about {target_who}.{target_id}")
            if who not in ctxt.frame.children:
                raise Exception(f"{ctxt.frame.name} is not yet modeling {who}")
            if (target_who, target_id) not in ctxt.frame.children[who].choices:
                raise Exception(
                    f"{ctxt.frame.name} does not yet think {who} is modeling {target_who}.{target_id}"
                )
            if (source_who, source_id) not in ctxt.frame.choices:
                raise Exception(
                    f"{ctxt.frame.name} does not yet model {source_who}.{source_id}"
                )
            # TODO: assert domains match

            eval_stmt(
                SWith(who, SObserve(target_who, target_id, loc=None), loc=None), ctxt
            )
            target_addr = (target_who, target_id)
            source_addr = (source_who, source_id)
            assert (
                ctxt.frame.children[who].choices[target_addr].domain
                == ctxt.frame.choices[source_addr].domain
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
            if who not in ctxt.frame.children:
                ctxt.frame.children[who] = Frame(name=who, parent=ctxt.frame)
            assert source_addr in ctxt.frame.choices
            ctxt.frame.children[who].choices[source_addr] = ctxt.frame.choices[
                source_addr
            ]
            ctxt.frame.children[who].choices[source_addr].known = True
            ctxt.frame.choices[(who, source_id)] = ctxt.frame.choices[source_addr]
            # ic(who, source_id, ctxt.frame.name)
            ctxt.frame.conditions[(who, source_id)] = source_addr
            # ic(ctxt.frame.conditions)
            ctxt.emit(f"pass  # {who} knows {source_who}.{source_id}")

        case _:
            raise NotImplementedError
