from .core import *

import ast, inspect, textwrap, re
from typing import Any, Callable, Literal
from dataclasses import dataclass

try:
    from icecream import ic  # type: ignore
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


@dataclass
class ParsingContext:
    cast: None | list[str]
    static_parameters: list[str]
    axes: list[tuple[str, str]]
    loc_name: str
    loc_dedent: int
    loc_file: str


def parse_expr(expr: ast.expr, ctxt: ParsingContext) -> Expr:
    loc = SourceLocation(ctxt.loc_file, expr.lineno, expr.col_offset + ctxt.loc_dedent, ctxt.loc_name)
    match expr:
        case ast.Constant(value=val):
            assert isinstance(val, float) or isinstance(val, int)
            return ELit(value=val, loc=loc, static=True)

        case ast.Call(func=ast.Name(id="exp"), args=[e1]):
            return EOp(op=Op.EXP, args=[parse_expr(e1, ctxt)], loc=loc, static=False)
        case ast.Call(func=ast.Name(id="abs"), args=[e1]):
            return EOp(op=Op.ABS, args=[parse_expr(e1, ctxt)], loc=loc, static=False)
        case ast.Call(func=ast.Name(id="log"), args=[e1]):
            return EOp(op=Op.LOG, args=[parse_expr(e1, ctxt)], loc=loc, static=False)

        case ast.BinOp(
            left=ast.Name(id="cost"),
            op=ast.MatMult(),
            right=ast.Call(func=ast.Name(id=f_name), args=args)
        ):
            return ECost(
                name=f_name,
                args=[parse_expr(arg, ctxt) for arg in args],
                loc=loc,
                static=False
            )

        case ast.Call(func=ast.Name(id=ffi_name), args=ffi_args):
            ffi_args_parsed = [parse_expr(arg, ctxt) for arg in ffi_args]
            return EFFI(
                name=ffi_name, args=ffi_args_parsed, loc=loc, static=all(arg.static for arg in ffi_args_parsed)
            )

        # memo call single arg
        case ast.Call(
            func=ast.Subscript(
                value=ast.Name(id=f_name),
                slice=ast.Attribute(value=ast.Name(id=source_name), attr=source_id)
            ),
            args=args,
        ):
            return EMemo(
                name=f_name,
                args=[parse_expr(arg, ctxt) for arg in args],
                ids=[(Id("..."), Name(source_name), Id(source_id))],
                loc=loc,
                static=False
            )

        case ast.Call(
            func=ast.Subscript(
                value=ast.Name(id=f_name),
                slice=ast.Name(id=source_id)
            ),
            args=args,
        ):
            return EMemo(
                name=f_name,
                args=[parse_expr(arg, ctxt) for arg in args],
                ids=[(Id("..."), Name("self"), Id(source_id))],
                loc=loc,
                static=False
            )

        # memo call multi arg
        case ast.Call(
            func=ast.Subscript(value=ast.Name(id=f_name), slice=ast.Tuple(elts=elts)),
            args=args,
        ):
            ids = []
            for elt in elts:
                match elt:
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
                static=False
            )

        # operators
        case ast.Compare(left=e1, ops=[op], comparators=[e2]):
            e1_ = parse_expr(e1, ctxt)
            e2_ = parse_expr(e2, ctxt)
            return EOp(
                op={
                    ast.Eq: Op.EQ,
                    ast.NotEq: Op.NEQ,
                    ast.Lt: Op.LT,
                    ast.LtE: Op.LTE,
                    ast.Gt: Op.GT,
                    ast.GtE: Op.GTE
                }[op.__class__],
                args=[e1_, e2_],
                loc=loc,
                static=e1_.static and e2_.static
            )

        case ast.UnaryOp(op=op, operand=operand):
            o_expr = parse_expr(operand, ctxt)
            return EOp(
                op={
                    ast.UAdd: Op.UADD,
                    ast.USub: Op.NEG,
                    ast.Invert: Op.INV,
                }[op.__class__],
                args=[o_expr],
                loc=loc,
                static=o_expr.static
            )

        case ast.BinOp(left=e1, op=op, right=e2):
            e1_ = parse_expr(e1, ctxt)
            e2_ = parse_expr(e2, ctxt)
            return EOp(
                op={
                    ast.Add: Op.ADD,
                    ast.Sub: Op.SUB,
                    ast.Mult: Op.MUL,
                    ast.Div: Op.DIV,
                    ast.Pow: Op.POW,
                    ast.BitXor: Op.XOR
                }[op.__class__],
                args=[e1_, e2_],
                loc=loc,
                static=e1_.static and e2_.static
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
            e1_ = parse_expr(e1, ctxt)
            e2_ = parse_expr(e2, ctxt)
            return EOp(
                op={ast.And: Op.AND, ast.Or: Op.OR}[op.__class__],
                args=[e1_, e2_],
                loc=loc,
                static=e1_.static and e2_.static
            )

        case ast.IfExp(test=test, body=body, orelse=orelse):
            c_expr = parse_expr(test, ctxt)
            t_expr = parse_expr(body, ctxt)
            f_expr = parse_expr(orelse, ctxt)
            return EOp(op=Op.ITE, args=[c_expr, t_expr, f_expr], loc=loc, static=c_expr.static and t_expr.static and f_expr.static)

        # literals
        case ast.Name(id=id):
            if id in ctxt.static_parameters:
                return ELit(id, loc=loc, static=True)
            return EChoice(id=Id(id), loc=loc, static=False)

        # expected value
        case ast.Subscript(value=ast.Name(id="E" | "Pr"), slice=rv_expr):
            assert not isinstance(rv_expr, ast.Slice)
            assert not isinstance(rv_expr, ast.Tuple)
            return EExpect(expr=parse_expr(rv_expr, ctxt), reduction="expectation", loc=loc, static=False)

        # entropy
        case ast.Subscript(value=ast.Name(id="H"), slice=rv_expr):
            assert not isinstance(rv_expr, ast.Slice)
            match rv_expr:
                case ast.Attribute(value=ast.Name(id=who_), attr=choice):
                    return EEntropy(rvs=[(Name(who_), Id(choice))], loc=loc, static=False)
                case ast.Tuple(elts=elts):
                    rvs = []
                    for elt in elts:
                        match elt:
                            case ast.Attribute(value=ast.Name(id=who_), attr=choice):
                                rvs.append((Name(who_), Id(choice)))
                            case _:
                                raise MemoError(
                                    f"Unexpected variable in H[...]",
                                    hint=f"You can only calculate the entropy of other agents' choices, e.g. alice.x",
                                    user=True,
                                    ctxt=None,
                                    loc=loc,
                                )
                    return EEntropy(rvs=rvs, loc=loc, static=False)
                case _:
                    raise MemoError(
                        f"Unexpected variable in H[...]",
                        hint=f"You can only calculate the entropy of other agents' choices, e.g. alice.x",
                        user=True,
                        ctxt=None,
                        loc=loc,
                    )

        # variance
        case ast.Subscript(value=ast.Name(id="Var"), slice=rv_expr):
            assert not isinstance(rv_expr, ast.Slice)
            assert not isinstance(rv_expr, ast.Tuple)
            return EExpect(expr=parse_expr(rv_expr, ctxt), reduction="variance", loc=loc, static=False)

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
            return EImagine(do=stmts, then=parse_expr(elts[-1], ctxt), loc=loc, static=False)

        case ast.Subscript(value=ast.Name("imagine"), slice=elt):
            return EImagine(do=[], then=parse_expr(elt, ctxt), loc=loc, static=False)

        # choice
        case ast.Subscript(value=ast.Name(id=who_id), slice=slice):
            if ctxt.cast is not None and who_id not in ctxt.cast:
                raise MemoError(
                    f"agent `{who_id}` is not in the cast",
                    hint=f"Did you either misspell `{who_id}`, or forget to include `{who_id}` in the cast?",
                    user=True,
                    ctxt=None,
                    loc=loc,
                )
            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EWith(who=Name(who_id), expr=parse_expr(slice, ctxt), loc=loc, static=False)
        case ast.Attribute(value=ast.Name(id=who_id), attr=attr):
            if ctxt.cast is not None and who_id not in ctxt.cast and who_id != "self":
                raise MemoError(
                    f"agent `{who_id}` is not in the cast",
                    hint=f"Did you either misspell `{who_id}`, or forget to include `{who_id}` in the cast?",
                    user=True,
                    ctxt=None,
                    loc=loc,
                )
            return EWith(who=Name(who_id), expr=EChoice(Id(attr), loc=loc, static=False), loc=loc, static=False)

        case _:
            raise MemoError(
                f"Unknown expression syntax",
                hint=f"The full expression is {ast.dump(expr)}",
                user=True,
                ctxt=None,
                loc=loc,
            )


def parse_stmt(expr: ast.expr, who: str, ctxt: ParsingContext) -> list[Stmt]:
    loc = SourceLocation(ctxt.loc_file, expr.lineno, expr.col_offset + ctxt.loc_dedent, ctxt.loc_name)
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
            keywords=kw
        ):
            kw_names = set(k.arg for k in kw)
            if len(kw) == 2 and kw_names == {"wpp", "to_maximize"}:
                raise MemoError(
                    f"cannot have both `wpp` and `to_maximize` in a chooses/given statement",
                    hint=f"please choose one or the other >:(",
                    user=True,
                    ctxt=None,
                    loc=loc
                )

            reduction: Literal["normalize", "maximize"]
            match kw:
                case [ast.keyword(arg="wpp", value=wpp_expr)]:
                    reduction = "normalize"
                case [ast.keyword(arg="to_maximize", value=wpp_expr)]:
                    reduction = "maximize"
                case _:
                    raise MemoError(
                        f"unknown argument(s) to chooses/given: {[k.arg for k in kw]}",
                        hint=f"expected either `wpp` or `to_maximize`",
                        user=True,
                        ctxt=None,
                        loc=loc
                    )
            return [
                SChoose(
                    who=Name(who),
                    id=Id(choice_id),
                    domain=Dom(dom_id),
                    wpp=parse_expr(wpp_expr, ctxt),
                    loc=loc,
                    reduction=reduction
                )
            ]

        # knows with/without self
        case ast.Call(
            func=ast.Name(id="knows"),
            args=args,
            keywords=[],
        ):
            stmts: list[Stmt] = []
            for arg in args:
                match arg:
                    case ast.Name(id=source_id):
                        stmts.append(
                            SKnows(
                                who=Name(who),
                                source_who=Name("self"),
                                source_id=Id(source_id),
                                loc=loc,
                            )
                        )

                    case ast.Attribute(
                        value=ast.Name(id=source_who),
                        attr=source_id,
                    ):
                        stmts.append(
                            SKnows(
                                who=Name(who),
                                source_who=Name(source_who),
                                source_id=Id(source_id),
                                loc=loc,
                            )
                        )
            return stmts

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


def parse_memo(f) -> tuple[ParsingContext, list[Stmt], Expr]:  # type: ignore
    try:
        rawsrc = inspect.getsource(f)
    except OSError:
        raise MemoError(
            "Python couldn't find your memo source code",
            hint="You cannot define a new @memo in the Python interactive REPL. Try writing your memo code to a file and running via `python filename.py`. If you really want an interactive experience, memo also works inside Jupyter notebooks.",
            user=True,
            ctxt=None,
            loc=None
        )
    src_file = inspect.getsourcefile(f)
    assert src_file is not None
    lines, lineno = inspect.getsourcelines(f)

    src = textwrap.dedent(rawsrc)  # borrowed from Exo's parser!
    lead_raw = re.match("^(.*)", rawsrc)
    lead_src = re.match("^(.*)", src)
    assert lead_raw is not None and lead_src is not None
    n_dedent = len(lead_raw.group()) - len(lead_src.group())
    tree = ast.parse(src, filename=src_file)
    ast.increment_lineno(tree, n=lineno - 1)

    cast = None
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
                    cast = []
                    for elt in elts:
                        assert isinstance(elt, ast.Name)
                        cast.append(elt.id)
                    rest_stmts = f.body[1:]
                case _:
                    rest_stmts = f.body[:]
        case _:
            raise MemoError(
                "Unknown syntax error",
                hint=None,
                user=False,
                ctxt=None,
                loc=SourceLocation(src_file, f.lineno, f.col_offset + n_dedent, "??"),
            )

    pctxt = ParsingContext(
        cast=cast,
        static_parameters=static_parameters,
        axes=[],
        loc_name=f_name,
        loc_dedent=n_dedent,
        loc_file=src_file,
    )
    stmts: list[Stmt] = []
    retval = None

    for tp in f.type_params:
        assert isinstance(tp, ast.TypeVar)
        if tp.bound is None:
            raise MemoError(
                f"Missing domain for {tp.name}",
                hint=f"Specify the domain for {tp.name} by writing `{pctxt.loc_name}[{tp.name}: ___, ...]`",
                user=True,
                ctxt=None,
                loc=SourceLocation(pctxt.loc_file, tp.lineno, tp.col_offset + pctxt.loc_dedent, pctxt.loc_name)
            )
        assert isinstance(tp.bound, ast.Name)
        stmts.append(
            SForAll(
                id=Id(tp.name),
                domain=Dom(tp.bound.id),
                loc=None
            )
        )
        pctxt.axes.append((tp.name, tp.bound.id))

    for stmt in rest_stmts:
        loc = SourceLocation(pctxt.loc_file, stmt.lineno, stmt.col_offset + pctxt.loc_dedent, pctxt.loc_name)
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
                        loc=loc,
                    )
                )
            case ast.AnnAssign(target=ast.Name(id=who), annotation=expr, value=None):
                if pctxt.cast is not None and who not in pctxt.cast:
                    raise MemoError(
                        f"agent `{who}` is not in the cast",
                        hint=f"Did you either misspell `{who}`, or forget to include `{who}` in the cast?",
                        user=True,
                        ctxt=None,
                        loc=loc
                    )
                stmts.extend(parse_stmt(expr, who, pctxt))
            case ast.Return(value=expr) if expr is not None:
                if retval is not None:
                    raise MemoError(
                        f"multiple return statements",
                        hint=f"A memo should only have one return statement, at the end",
                        user=True,
                        ctxt=None,
                        loc=loc
                    )
                retval = parse_expr(expr, pctxt)
            case _:
                raise MemoError(
                    f"Unknown statement syntax",
                    hint=f"The full statement is {ast.dump(stmt)}",
                    user=True,
                    ctxt=None,
                    loc=loc
                )

    if retval is None:
        raise MemoError(
            f"No return statement",
            hint=f"All memos should end with a return statement",
            user=True,
            ctxt=None,
            loc=SourceLocation(pctxt.loc_file, f.lineno, f.col_offset + pctxt.loc_dedent, pctxt.loc_name),
        )

    return pctxt, stmts, retval
