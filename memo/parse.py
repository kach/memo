from .core import *
import ast, inspect
from typing import Any, Callable
import textwrap
import os, sys
from io import StringIO
from dataclasses import dataclass


@dataclass
class ParsingContext:
    cast: list[str]
    static_parameters: list[str]
    loc_name: str
    loc_file: str


def parse_expr(expr: ast.expr, ctxt: ParsingContext) -> Expr:
    loc = SourceLocation(ctxt.loc_file, expr.lineno, expr.col_offset, ctxt.loc_name)
    match expr:
        case ast.Constant(value=val):
            assert isinstance(val, float) or isinstance(val, int)
            return ELit(value=val, loc=loc)

        case ast.Call(func=ast.Name(id="exp"), args=[e1]):
            return EOp(op=Op.EXP, args=[parse_expr(e1, ctxt)], loc=loc)

        case ast.Call(func=ast.Name(id=ffi_name), args=ffi_args):
            return EFFI(
                name=ffi_name, args=[parse_expr(arg, ctxt) for arg in ffi_args], loc=loc
            )

        # memo call single arg  TODO: make self optional here as well
        case ast.Call(
            func=ast.Subscript(
                value=ast.Name(id=f_name),
                slice=ast.Compare(
                    left=ast.Name(id=target_id),
                    ops=[ast.Is()],
                    comparators=[
                        ast.Attribute(value=ast.Name(id=source_name), attr=source_id)
                    ],
                ),
            ),
            args=args,
        ):
            return EMemo(
                name=f_name,
                args=[parse_expr(arg, ctxt) for arg in args],
                ids=[(Id(target_id), Name(source_name), Id(source_id))],
                loc=loc,
            )

        # memo call multi arg
        case ast.Call(
            func=ast.Subscript(value=ast.Name(id=f_name), slice=ast.Tuple(elts=elts)),
            args=args,
        ):
            ids = []
            for elt in elts:
                match elt:
                    case ast.Compare(
                        left=ast.Name(id=target_id),
                        ops=[ast.Is()],
                        comparators=[
                            ast.Attribute(
                                value=ast.Name(id=source_name), attr=source_id
                            )
                        ],
                    ):
                        ids.append((Id(target_id), Name(source_name), Id(source_id)))
                    case ast.Compare(
                        left=ast.Name(id=target_id),
                        ops=[ast.Is()],
                        comparators=[ast.Name(id=source_id)],
                    ):
                        ids.append((Id(target_id), Name("self"), Id(source_id)))
                    case ast.Attribute(value=ast.Name(id=source_name), attr=source_id):
                        ids.append((Id("..."), Name(source_name), Id(source_id)))
                    case ast.Name(id=source_id):
                        ids.append((Id("..."), Name("self"), Id(source_id)))
                    case _:
                        raise Exception()
            return EMemo(
                name=f_name,
                args=[parse_expr(arg, ctxt) for arg in args],
                ids=ids,
                loc=loc,
            )

        # operators
        case ast.Compare(left=e1, ops=[op], comparators=[e2]):
            return EOp(
                op={ast.Eq: Op.EQ, ast.Lt: Op.LT, ast.Gt: Op.GT}[op.__class__],
                args=[
                    parse_expr(e1, ctxt),
                    parse_expr(e2, ctxt),
                ],
                loc=loc,
            )

        case ast.UnaryOp(op=op, operand=operand):
            o_expr = parse_expr(operand, ctxt)
            return EOp(
                op={
                    ast.USub: Op.NEG,
                    ast.Invert: Op.INV,
                }[op.__class__],
                args=[o_expr],
                loc=loc,
            )

        case ast.BinOp(left=e1, op=op, right=e2):
            return EOp(
                op={
                    ast.Add: Op.ADD,
                    ast.Sub: Op.SUB,
                    ast.Mult: Op.MUL,
                    ast.Div: Op.DIV,
                }[op.__class__],
                args=[
                    parse_expr(e1, ctxt),
                    parse_expr(e2, ctxt),
                ],
                loc=loc,
            )

        case ast.BoolOp(op=op, values=values):
            if len(values) != 2:
                raise MemoError(
                    f"Incorrect number of arguments to logical operator {op}",
                    hint=None,
                    user=False,
                    ctxt=None,
                    loc=loc,
                )
            e1, e2 = values
            return EOp(
                op={ast.And: Op.AND, ast.Or: Op.OR}[op.__class__],
                args=[
                    parse_expr(e1, ctxt),
                    parse_expr(e2, ctxt),
                ],
                loc=loc,
            )

        case ast.IfExp(test=test, body=body, orelse=orelse):
            c_expr = parse_expr(test, ctxt)
            t_expr = parse_expr(body, ctxt)
            f_expr = parse_expr(orelse, ctxt)
            return EOp(op=Op.ITE, args=[c_expr, t_expr, f_expr], loc=loc)

        # literals
        case ast.Name(id=id):
            if id in ctxt.static_parameters:
                return ELit(id, loc=loc)
            return EChoice(id=Id(id), loc=loc)

        # expected value
        case ast.Subscript(value=ast.Name(id="E"), slice=slice):
            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EExpect(expr=parse_expr(slice, ctxt), loc=loc)

        # imagine
        case ast.Subscript(value=ast.Name("imagine"), slice=ast.Tuple(elts=elts)):
            stmts = []
            for elt in elts[:-1]:
                match elt:
                    case ast.Slice(
                        lower=ast.Name(id=who_), upper=expr_, step=None
                    ) if expr_ is not None:
                        stmts.extend(parse_stmt(expr_, who_, ctxt))
            assert not isinstance(elts[-1], ast.Slice)
            return EImagine(do=stmts, then=parse_expr(elts[-1], ctxt), loc=loc)

        # choice
        case ast.Subscript(value=ast.Name(id=who_id), slice=slice):
            if who_id not in ctxt.cast:
                raise MemoError(
                    f"agent `{who_id}` is not in the cast",
                    hint=f"Did you either misspell `{who_id}`, or forget to include `{who_id}` in the cast?",
                    user=True,
                    ctxt=None,
                    loc=loc,
                )
            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EWith(who=Name(who_id), expr=parse_expr(slice, ctxt), loc=loc)
        case ast.Attribute(value=ast.Name(id=who_id), attr=attr):
            if who_id not in ctxt.cast and who_id != "self":
                raise MemoError(
                    f"agent `{who_id}` is not in the cast",
                    hint=f"Did you either misspell `{who_id}`, or forget to include `{who_id}` in the cast?",
                    user=True,
                    ctxt=None,
                    loc=loc,
                )
            return EWith(who=Name(who_id), expr=EChoice(Id(attr), loc=loc), loc=loc)

        case _:
            raise MemoError(
                f"Unknown expression syntax",
                hint=f"The full expression is {ast.dump(expr)}",
                user=True,
                ctxt=None,
                loc=loc,
            )


def parse_stmt(expr: ast.expr, who: str, ctxt: ParsingContext) -> list[Stmt]:
    loc = SourceLocation(ctxt.loc_file, expr.lineno, expr.col_offset, ctxt.loc_name)
    match expr:
        case ast.Call(
            func=ast.Name(id="chooses" | "given"),
            args=[
                ast.Compare(
                    left=ast.Name(id=choice_id),
                    comparators=[ast.Name(id=dom_id)],
                    ops=[ast.In()],
                )
            ],
            keywords=[ast.keyword(arg="wpp", value=wpp_expr)],
        ):
            return [
                SChoose(
                    who=Name(who),
                    id=Id(choice_id),
                    domain=Dom(dom_id),
                    wpp=parse_expr(wpp_expr, ctxt),
                    loc=loc,
                )
            ]

        # knows with/without self
        case ast.Call(
            func=ast.Name(id="knows"),
            args=[ast.Name(id=source_id)],
            keywords=[],
        ):
            return [
                SKnows(
                    who=Name(who),
                    source_who=Name("self"),
                    source_id=Id(source_id),
                    loc=loc,
                )
            ]

        case ast.Call(
            func=ast.Name(id="knows"),
            args=[
                ast.Attribute(
                    value=ast.Name(id=source_who),
                    attr=source_id,
                )
            ],
            keywords=[],
        ):
            return [
                SKnows(
                    who=Name(who),
                    source_who=Name(source_who),
                    source_id=Id(source_id),
                    loc=loc,
                )
            ]

        # observes with/without self
        case ast.Compare(
            left=ast.Subscript(
                value=ast.Name(id="observes" | "sees" | "hears"),
                slice=ast.Attribute(
                    value=ast.Name(id=target_who),
                    attr=target_id,
                ),
            ),
            ops=[ast.Is()],
            comparators=[ast.Attribute(value=ast.Name(id=source_who), attr=source_id)],
        ):
            return [
                SShow(
                    who=Name(who),
                    target_who=Name(target_who),
                    target_id=Id(target_id),
                    source_who=Name(source_who),
                    source_id=Id(source_id),
                    loc=loc,
                )
            ]

        case ast.Compare(
            left=ast.Subscript(
                value=ast.Name(id="observes" | "sees" | "hears"),
                slice=ast.Attribute(
                    value=ast.Name(id=target_who),
                    attr=target_id,
                ),
            ),
            ops=[ast.Is()],
            comparators=[ast.Name(source_id)],
        ):
            return [
                SShow(
                    who=Name(who),
                    target_who=Name(target_who),
                    target_id=Id(target_id),
                    source_who=Name("self"),
                    source_id=Id(source_id),
                    loc=loc,
                )
            ]

        case ast.Subscript(
            value=ast.Name("thinks"), slice=ast.Slice(lower=ast.Name(who_), upper=expr_)
        ) if expr_ is not None:
            return [
                SWith(who=Name(who), stmt=s, loc=loc)
                for s in parse_stmt(expr_, who_, ctxt)
            ]
        case ast.Subscript(value=ast.Name("thinks"), slice=ast.Tuple(elts=elts)):
            stmts = []
            for elt in elts:
                match elt:
                    case ast.Slice(
                        lower=ast.Name(id=who_), upper=expr_, step=None
                    ) if expr_ is not None:
                        stmts.extend(parse_stmt(expr_, who_, ctxt))
                    case _:
                        raise Exception()
            return [SWith(who=Name(who), stmt=s, loc=s.loc) for s in stmts]
        case _:
            raise MemoError(
                f"Unknown statement syntax",
                hint=f"The full statement is {ast.dump(expr)}",
                user=True,
                ctxt=None,
                loc=loc,
            )


def memo_(f):  # type: ignore
    src = inspect.getsource(f)
    src_file = inspect.getsourcefile(f)
    assert src_file is not None
    lines, lineno = inspect.getsourcelines(f)
    tree = ast.parse(src, filename=src_file)
    ast.increment_lineno(tree, n=lineno - 1)

    cast = []
    static_parameters = []

    match tree:
        case ast.Module(body=[ast.FunctionDef(name=f_name) as f]):
            # print(ast.dump(f, include_attributes=True, indent=2))
            for arg in f.args.args:
                # assert isinstance(arg.annotation, ast.Name) and arg.annotation.id in ['float']
                # should always be true, see https://docs.python.org/3.8/library/ast.html#ast.parse
                assert arg.type_comment is None
                static_parameters.append(arg.arg)
            first_stmt = f.body[0]
            match first_stmt:
                case ast.AnnAssign(
                    target=ast.Name(id="cast"),
                    annotation=ast.List(elts=elts),
                    value=None,
                ):
                    for elt in elts:
                        assert isinstance(elt, ast.Name)
                        cast.append(elt.id)
                case _:
                    raise MemoError(
                        "No cast",
                        hint="The first line of a memo should always declare the cast of agents you will be working with. For example, to declare a memo which reasons about the two agents alice and bob, you would write `cast: [alice, bob]`.",
                        user=True,
                        ctxt=None,
                        loc=SourceLocation(
                            src_file, first_stmt.lineno, first_stmt.col_offset, f_name
                        ),
                    )
        case _:
            raise MemoError(
                "Unknown syntax error",
                hint=None,
                user=False,
                ctxt=None,
                loc=SourceLocation(src_file, f.lineno, f.col_offset, "??"),
            )

    pctxt = ParsingContext(
        cast=cast,
        static_parameters=static_parameters,
        loc_name=f_name,
        loc_file=src_file,
    )
    stmts: list[Stmt] = []
    retval = None

    for tp in f.type_params:
        assert isinstance(tp.bound, ast.Name)
        stmts.append(
            SForAll(
                id=Id(tp.name),
                domain=Dom(tp.bound.id),
                loc=None
            )
        )

    for stmt in f.body[1:]:
        match stmt:
            case ast.AnnAssign(
                target=ast.Name(id="forall"),
                annotation=ast.Compare(
                    left=ast.Name(id=choice_id),
                    comparators=[ast.Name(id=dom_id)],
                    ops=[ast.In()],
                ),
                value=None,
            ):
                assert choice_id not in static_parameters
                stmts.append(
                    SForAll(
                        id=Id(choice_id),
                        domain=Dom(dom_id),
                        loc=SourceLocation(
                            pctxt.loc_file, stmt.lineno, stmt.col_offset, pctxt.loc_name
                        ),
                    )
                )
            case ast.AnnAssign(target=ast.Name(id=who), annotation=expr, value=None):
                if who not in cast:
                    raise MemoError(
                        f"agent `{who}` is not in the cast",
                        hint=f"Did you either misspell `{who}`, or forget to include `{who}` in the cast?",
                        user=True,
                        ctxt=None,
                        loc=SourceLocation(
                            pctxt.loc_file, stmt.lineno, stmt.col_offset, pctxt.loc_name
                        ),
                    )
                stmts.extend(parse_stmt(expr, who, pctxt))
            case ast.Return(value=expr) if expr is not None:
                if retval is not None:
                    raise MemoError(
                        f"multiple return statements",
                        hint=f"A memo should only have one return statement, at the end",
                        user=True,
                        ctxt=None,
                        loc=SourceLocation(
                            pctxt.loc_file, stmt.lineno, stmt.col_offset, pctxt.loc_name
                        ),
                    )
                retval = parse_expr(expr, pctxt)
            case _:
                raise MemoError(
                    f"Unknown statement syntax",
                    hint=f"The full statement is {ast.dump(stmt)}",
                    user=True,
                    ctxt=None,
                    loc=SourceLocation(
                        pctxt.loc_file, stmt.lineno, stmt.col_offset, pctxt.loc_name
                    ),
                )

    if retval is None:
        raise MemoError(
            f"no return statement",
            hint=f"All memos should end with a return statement",
            user=True,
            ctxt=None,
            loc=SourceLocation(pctxt.loc_file, f.lineno, f.col_offset, pctxt.loc_name),
        )

    io = StringIO()
    ctxt = Context(next_idx=0, io=io, frame=Frame(name=Name("root")), idx_history=[])
    ctxt.emit(HEADER)
    for stmt_ in stmts:
        eval_stmt(stmt_, ctxt)
    val = eval_expr(retval, ctxt)
    squeeze_axes = [
        -1 - i
        for i in range(ctxt.next_idx)
        if i not in [z[0] for z in ctxt.forall_idxs]
    ]
    ctxt.emit(f"{val.tag} = jnp.array({val.tag})")
    ctxt.emit(
        f"{val.tag} = pad({val.tag}, {ctxt.next_idx}).squeeze(axis={tuple(squeeze_axes)})"
    )
    ctxt.emit(f"return {val.tag}")

    out = (
        ""
        + f"""def _out_{f_name}({", ".join(static_parameters)}):\n"""
        + textwrap.indent(io.getvalue(), "    ")
        + "\n\n"
        # + f"_out_{f_name}._foralls = ...\n"
        # + f"_out_{f_name}._memo = {repr([z[1:] for z in ctxt.forall_idxs])}\n"
        + f"{f_name} = _out_{f_name}\n"
    )

    # for s in stmts:
    #     print(pprint_stmt(s))
    # print(pprint_expr(retval))
    # for i, line in enumerate(out.splitlines()):
    #     print(f"{i + 1: 5d}  {line}")

    globals_of_caller = inspect.stack()[2].frame.f_globals
    retvals: dict[Any, Any] = {}
    exec(out, globals_of_caller, retvals)
    return retvals[f"_out_{f_name}"]


def memo(f):  # type: ignore
    try:
        return memo_(f)  # type: ignore
    except MemoError as e:
        if e.loc:
            # e.add_note(f"memo error: {e.message}")
            e.add_note(
                f"    at: {e.loc.name} in {os.path.basename(e.loc.file)} line {e.loc.line} column {e.loc.offset + 1}"
            )
        if e.hint is not None:
            for line in textwrap.wrap(
                e.hint, initial_indent="  hint: ", subsequent_indent="        "
            ):
                e.add_note(line)
        if e.ctxt:  # TODO
            e.add_note(repr(e.ctxt))
        if not e.user:
            e.add_note("")
            e.add_note(
                "[We think this may be a bug in memo: if you don't understand what is going on, please get in touch with us!]"
            )
        raise
