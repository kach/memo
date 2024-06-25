from __future__ import annotations
from typing import NewType, Any

import itertools
from enum import Enum
import dataclasses
from dataclasses import dataclass, field
import copy

import textwrap
from io import StringIO

import warnings

from icecream import ic  # type: ignore

ic.configureOutput(includeContext=True)


Name = NewType("Name", str)
Id = NewType("Id", str)


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
    domain: list[float]
    wpp_deps: set[tuple[Name, Id]]


@dataclass
class Frame:
    name: Name
    choices: dict[tuple[Name, Id], Choice] = dataclasses.field(default_factory=dict)
    children: dict[Name, Frame] = dataclasses.field(default_factory=dict)
    conditions: dict[tuple[Name, Id], tuple[Name, Id]] = dataclasses.field(
        default_factory=dict
    )  # used to translate in EWith
    ll: str | None = None
    parent: Frame | None = None


@dataclass(frozen=True)
class ELit:
    value: float


Op = Enum(
    "Op", ["ADD", "SUB", "MUL", "DIV", "EQ", "LT", "GT", "AND", "OR", "EXP", "NEG", "INV", "ITE"]
)


@dataclass(frozen=True)
class EOp:
    op: Op
    args: list[Expr]


@dataclass(frozen=True)
class EChoice:  # TODO: alias given
    id: Id


@dataclass(frozen=True)
class EExpect:
    expr: Expr


@dataclass(frozen=True)
class EWith:
    who: Name
    expr: Expr


@dataclass(frozen=True)
class EImagine:
    do: list[Stmt]
    then: Expr


Expr = ELit | EOp | EChoice | EExpect | EWith | EImagine


@dataclass(frozen=True)
class SPass:
    pass


@dataclass(frozen=True)
class SChoose:
    who: Name
    id: Id
    domain: list[float]
    wpp: Expr


@dataclass(frozen=True)
class SObserve:
    who: Name
    id: Id


@dataclass(frozen=True)
class SWith:
    who: Name
    stmt: Stmt


@dataclass(frozen=True)
class SShow:
    who: Name
    target_who: Name
    target_id: Id
    source_who: Name
    source_id: Id


@dataclass(frozen=True)
class SForAll:
    id: Id
    domain: list[float]


Stmt = SPass | SChoose | SObserve | SWith | SShow | SForAll


@dataclass
class Context:
    next_idx: int
    frame: Frame
    io: StringIO
    idx_history: list[str]
    _sym: int = -1

    def emit(self: Context, line: str) -> None:
        print(line, file=self.io)

    def sym(self, hint: str = "") -> str:
        self._sym += 1
        return f"{hint}_{self._sym}"

    forall_idxs: list[int] = field(default_factory=list)


HEADER = """\
import jax.numpy as jnp
def marg(t, dims):
    if dims == ():
        return t
    # return t.sum(dim=tuple(-1 - d for d in dims), keepdims=True)
    return t.sum(axis=tuple(-1 - d for d in dims), keepdims=True)

def pad(t, total):
    count = total - len(t.shape)
    for _ in range(count):
        # t = t.unsqueeze(0)
        t = jnp.expand_dims(t, 0)
    return t
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
            if source_who == Name('self'):
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
            stmts = "\n".join(
                [pprint_stmt(s) for s in do] + [pprint_expr(then)]
            )
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
            ctxt.emit(f"{out} = jnp.array({val})")
            return Value(tag=out, known=True, deps=set())

        case EChoice(id):
            if (Name("self"), id) not in ctxt.frame.choices:
                raise Exception(f"{ctxt.frame.name} has not yet chosen {id}")
            ch = ctxt.frame.choices[(Name("self"), id)]
            # out = ctxt.sym("ch")
            # ctxt.emit(f"{out} = {ch.tag}")
            return Value(tag=ch.tag, known=ch.known, deps=set([(Name("self"), id)]))

        case EOp(op, args):
            out = ctxt.sym(f"op_{op.name.lower()}")
            if op in [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.EQ, Op.LT, Op.GT, Op.AND, Op.OR]:
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
                return Value(tag=out, known=l.known and r.known, deps=l.deps | r.deps)
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
                return Value(tag=out, known=l.known, deps=l.deps)
            elif op == Op.ITE:
                assert len(args) == 3
                c = eval_expr(args[0], ctxt)
                t = eval_expr(args[1], ctxt)
                f = eval_expr(args[2], ctxt)
                ctxt.emit(f"{out} = jnp.where({c.tag}, {t.tag}, {f.tag})")
                return Value(
                    tag=out,
                    known=c.known and t.known and f.known,
                    deps=c.deps | t.deps | f.deps,
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
            ctxt.emit(f"\n# {ctxt.frame.name} expectation")
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
            )

        case EWith(who, expr):
            if who == Name("self"):
                return eval_expr(expr, ctxt)
            if who not in ctxt.frame.children:
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
                    if who.startswith('future_'):  ## TODO: there is definitely a bug here
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
            return Value(tag=val_.tag, known=known, deps=deps)

        case EImagine(do, then):
            ctxt.emit(f"\n# {ctxt.frame.name} imagines")
            future_name = Name(ctxt.sym(f"future_{ctxt.frame.name}"))
            future_frame = copy.deepcopy(ctxt.frame)
            future_frame.name = future_name
            future_frame.parent = ctxt.frame
            if ctxt.frame.ll is not None:
                fresh_lls(ctxt, ctxt.frame)
                # ll = ctxt.sym(f"{ctxt.frame.name}_ll")
                # ctxt.emit(f"{ll} = {ctxt.frame.ll}")
                # future_frame.ll = ll
            ctxt.frame.children[future_name] = future_frame
            for stmt in do:
                eval_stmt(SWith(future_name, stmt), ctxt)
            val_ = eval_expr(EWith(future_name, then), ctxt)
            return val_

    raise NotImplementedError

def fresh_lls(ctxt: Context, f: Frame) -> None:
    if f.ll is not None:
        ll = ctxt.sym(f'{f.name}_ll')
        ctxt.emit(f'{ll} = {f.ll}')
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
            ctxt.forall_idxs.append(idx)

        case SChoose(who, id, domain, wpp):
            if who not in ctxt.frame.children:
                ctxt.frame.children[who] = Frame(name=who, parent=ctxt.frame)
            idx = ctxt.next_idx
            ctxt.next_idx += 1
            ctxt.idx_history.append(f"{who}.{id}")
            tag = ctxt.sym(f"{who}_{id}")
            ctxt.emit(f"""\n# {who} choose {id}""")
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
                    raise Exception("??")  # should always be true
            ctxt.frame.choices[(who, id)] = Choice(tag, idx, False, domain, new_deps)
            id_ll = ctxt.sym(f"{id}_ll")
            ctxt.emit(
                f"{id_ll} = jnp.ones_like({tag}, dtype=jnp.float32) * {wpp_val.tag}"
            )
            ctxt.emit(
                f"{id_ll} = jnp.nan_to_num({id_ll} / marg({id_ll}, ({idx},)))"
            )
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
            ctxt.emit(f"""\n# {ctxt.frame.name} observe {who}.{id}""")
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
            ctxt.emit(f"\n# telling {who} about {target_who}.{target_id}")
            if who not in ctxt.frame.children:
                raise Exception(f'{ctxt.frame.name} is not yet modeling {who}')
            if (target_who, target_id) not in ctxt.frame.children[who].choices:
                raise Exception(f'{ctxt.frame.name} does not yet think {who} is modeling {target_who}.{target_id}')
            if (source_who, source_id) not in ctxt.frame.choices:
                raise Exception(f'{ctxt.frame.name} does not yet model {source_who}.{source_id}')
            # TODO: assert domains match

            eval_stmt(SWith(who, SObserve(target_who, target_id)), ctxt)
            target_addr = (target_who, target_id)
            source_addr = (source_who, source_id)
            assert ctxt.frame.children[who].choices[target_addr].domain == ctxt.frame.choices[source_addr].domain
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
            # ctxt.emit(f'print("{ctxt.frame.children[who].ll}", {ctxt.frame.children[who].ll}.tolist(), {ctxt.frame.children[who].ll}.shape); print()')

        case _:
            raise NotImplementedError
