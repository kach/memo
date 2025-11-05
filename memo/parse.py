from .core import *

import ast, inspect, textwrap, re, builtins
from typing import Any, Callable, Literal
from dataclasses import dataclass

try:
    from icecream import ic  # type: ignore
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def ast_increment_colno(tree: ast.AST, n: int) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.expr) or isinstance(node, ast.stmt):
            node.col_offset += n
            if node.end_col_offset is not None:
                node.end_col_offset += n

def parse_args_list(args: list[ast.expr], ctxt: ParsingContext, loc: SourceLocation) -> list[Expr]:
    match args:
        case [ast.Constant(value=val)] if val is ...:
            return [ELit(value=param, loc=loc, static=True) for param in ctxt.static_parameters]
        case _:
            out: list[Expr] = []
            for arg in args:
                match arg:
                    case ast.Name(id=id) if id in ctxt.exotic_parameters:
                        loc_ = SourceLocation(ctxt.loc_file, arg.lineno, arg.col_offset, ctxt.loc_name)
                        out.append(ELit(value=id, loc=loc_, static=True))
                    case _:
                        out.append(parse_expr(arg, ctxt))
            return out

def parse_ememo(expr: ast.expr, ctxt: ParsingContext, loc: SourceLocation) -> Expr | None:
    match expr:
        case ast.Call(
            func=ast.Subscript(
                value=f,
                slice=axes
            ),
            args=args
        ):
            pass
        case _:
            return None

    match f:
        case ast.Name(id=f_name):
            which_retval = None
        case ast.Subscript(ast.Name(id=f_name), slice=ast.Constant(value=which_retval)) if isinstance(which_retval, int):
            pass
        case _:
            return None

    match axes:
        case ast.Tuple(elts=elts):
            pass
        case _:
            elts = [axes]

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
        which_retval=which_retval,
        args=parse_args_list(args, ctxt, loc),
        ids=ids,
        loc=loc,
        static=False
    )

def parse_expr(expr: ast.expr, ctxt: ParsingContext) -> Expr:
    loc = SourceLocation(ctxt.loc_file, expr.lineno, expr.col_offset, ctxt.loc_name)
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
            ffi_args_parsed = parse_args_list(ffi_args, ctxt, loc)
            return EFFI(
                name=ffi_name, args=ffi_args_parsed, loc=loc, static=all(arg.static for arg in ffi_args_parsed)
            )

        # memo call single arg
        case _ if ememo := parse_ememo(expr, ctxt, loc):
            return ememo

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
                    ast.Mod: Op.MOD,
                    ast.BitXor: Op.XOR
                }[op.__class__],
                args=[e1_, e2_],
                loc=loc,
                static=e1_.static and e2_.static
            )

        case ast.BoolOp(op=op, values=values):
            if len(values) < 2:
                raise MemoError(
                    f"Unexpectedly few arguments to logical operator {op}",
                    hint=None,
                    user=False,
                    ctxt=None,
                    loc=loc,
                )
            the_op = {ast.And: Op.AND, ast.Or: Op.OR}[op.__class__]
            out = parse_expr(values[-1], ctxt)
            for val in reversed(values[:-1]):
                val_ = parse_expr(val, ctxt)
                out = EOp(
                    op=the_op,
                    args=[val_, out],
                    loc=loc,
                    static=val_.static and out.static
                )
            return out

        case ast.IfExp(test=test, body=body, orelse=orelse):
            c_expr = parse_expr(test, ctxt)
            t_expr = parse_expr(body, ctxt)
            f_expr = parse_expr(orelse, ctxt)
            return EOp(op=Op.ITE, args=[c_expr, t_expr, f_expr], loc=loc, static=c_expr.static and t_expr.static and f_expr.static)

        # literals
        case ast.Name(id=id):
            if id in ctxt.static_parameters:
                if id in ctxt.exotic_parameters:
                    raise MemoError(
                        "Unexpected use of non-numeric parameter",
                        hint=f"{id} is labeled as a special (non-numeric / ...) parameter. Such parameters can only be used as direct arguments to function calls.",
                        user=True,
                        ctxt=None,
                        loc=loc
                    )
                return ELit(id, loc=loc, static=True)
            return EChoice(id=Id(id), loc=loc, static=False)

        case ast.Subscript(value=ast.Name(id="Predict"), slice=f_expr):
            if isinstance(f_expr, ast.Slice) or isinstance(f_expr, ast.Tuple):
                raise Exception()
            return EPredict(expr=parse_expr(f_expr, ctxt), loc=loc, static=False)

        case ast.Subscript(value=ast.Name(id="EU"), slice=ast.Name(id=gid)):
            return EUtil(goal=Id(gid), loc=loc, static=False)

        # joint probability
        case ast.Subscript(value=ast.Name(id="Pr"), slice=ast.Tuple(elts=elts)):
            rv_expr = parse_expr(elts[0], ctxt)
            for elt in elts[1:]:
                elt_ = parse_expr(elt, ctxt)
                rv_expr = EOp(
                    op=Op.AND,
                    args=[rv_expr, elt_],
                    loc=elt_.loc,
                    static=rv_expr.static and elt_.static
                )
            return EExpect(expr=rv_expr, reduction="expectation", loc=loc, static=False)

        # expected value
        case ast.Subscript(value=ast.Name(id="E" | "Pr"), slice=rv_expr):
            if isinstance(rv_expr, ast.Slice) or isinstance(rv_expr, ast.Tuple):
                raise MemoError(
                    "Incorrect syntax in E[...] or Pr[...]",
                    hint="Double-check that you don't have a stray comma or colon in your expressions.",
                    user=True,
                    ctxt=None,
                    loc=loc
                )
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

        # KL divergence
        case ast.Subscript(
            value=ast.Name(id='KL'),
            slice=ast.BinOp(
                left=ast.Attribute(value=ast.Name(id=p_who), attr=p_id),
                op=ast.BitOr(),
                right=ast.Attribute(value=ast.Name(id=q_who), attr=q_id)
            )
        ):
            return EKL(Name(p_who), Id(p_id), Name(q_who), Id(q_id), loc=loc, static=False)

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

        case ast.Set(elts):
            if len(elts) != 1:
                raise MemoError(
                    "Inlines {...} should only have one value, not multiple comma-separated ones.",
                    hint="Double-check for commas!",
                    user=True,
                    ctxt=None,
                    loc=loc
                )
            return EInline(val=ast.unparse(elts[0]), loc=loc, static=True)

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
        case ast.Call(func=ast.Name(id='INSPECT'), args=[]):
            return [STrace(who=Name(who), loc=loc)]

        case ast.Call(
            func=ast.Name(id="chooses" | "given" | "draws" | "assigned" | "guesses"),
            args=args,
            keywords=kw
        ):
            choices: list[tuple[Id, Dom]] = []
            if len(args) == 0:
                raise MemoError(
                    "`chooses` needs at least one choice, but none provided",
                    hint="Specify a choice as `name in Domain`",
                    user=True,
                    ctxt=None,
                    loc=loc
                )
            for arg in args:
                match arg:
                    case ast.Compare(
                        left=ast.Name(id=choice_id),
                        comparators=[ast.Name(id=dom_id)],
                        ops=[ast.In()],
                    ):
                        choices.append((Id(choice_id), Dom(dom_id)))
                    case _:
                        raise MemoError(
                            "Unexpected item in chooses",
                            hint="Expected a sequence of `name in domain` entries.",
                            user=True,
                            ctxt=None,
                            loc=loc
                        )

            reduction: Literal["normalize", "maximize"]
            match kw:
                case [ast.keyword(arg=reduction_name, value=wpp_expr)]:
                    wpp_expr_ = parse_expr(wpp_expr, ctxt)
                case _:
                    raise MemoError(
                        f"wrong number of keyword arguments to chooses",
                        hint="expected exactly one of: wpp, to_maximize, to_minimize",
                        user=True,
                        ctxt=None,
                        loc=loc
                    )

            if reduction_name == 'wpp':
                reduction = 'normalize'
            elif reduction_name == 'to_maximize':
                reduction = 'maximize'
            elif reduction_name == 'to_minimize':
                reduction = 'maximize'
                wpp_expr_ = EOp(
                    op=Op.NEG,
                    args=[wpp_expr_],
                    loc=wpp_expr_.loc,
                    static=wpp_expr_.static
                )
            else:
                raise MemoError(
                    f"unknown keyword argument to chooses: {reduction_name}",
                    hint="expected exactly one of: wpp, to_maximize, to_minimize",
                    user=True,
                    ctxt=None,
                    loc=loc
                )

            return [
                SChoose(
                    who=Name(who),
                    choices=choices,
                    wpp=wpp_expr_,
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

        case ast.Call(
            func=ast.Name(id="wants"),
            args=[],
            keywords=[ast.keyword(arg=what, value=how)]
        ):
            assert what is not None
            return [SWants(who=Name(who), what=Id(what), how=parse_expr(how, ctxt), loc=loc)]

        case ast.Call(
            func=ast.Name(id="snapshots_self_as"),
            args=args,
            keywords=[]
        ):
            stmts = []
            for arg in args:
                if not isinstance(arg, ast.Name):
                    raise MemoError(
                        "Inputs to snapshots_self_as() must be names of agents",
                        hint=f"`{ast.unparse(arg)}` is not a name",
                        user=True,
                        ctxt=None,
                        loc=loc
                    )
                stmts.append(
                    SSnapshot(
                        who=Name(who),
                        alias=Name(arg.id),
                        loc=loc
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

        case ast.Call(
            func=ast.Name("observes_event"),
            args=[],
            keywords=[ast.keyword(arg="wpp", value=e_)]
        ):
            return [SObserves(who=Name(who), what=parse_expr(e_, ctxt), how="probability", loc=loc)]

        case ast.Subscript(
            value=ast.Name("observes_that"),
            slice=e_
        ): return [SObserves(who=Name(who), what=parse_expr(e_, ctxt), how="boolean", loc=loc)]

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


def parse_memo(ff: Callable[..., Any]) -> tuple[ParsingContext, list[Stmt], list[Expr]]:
    try:
        rawsrc = inspect.getsource(ff)
    except OSError:
        raise MemoError(
            "Python couldn't find your memo source code",
            hint="You cannot define a new @memo in the Python interactive REPL. Try writing your memo code to a file and running via `python filename.py`. If you really want an interactive experience, memo also works inside Jupyter notebooks.",
            user=True,
            ctxt=None,
            loc=None
        )
    src_file = inspect.getsourcefile(ff)
    assert src_file is not None
    lines, lineno = inspect.getsourcelines(ff)

    src = textwrap.dedent(rawsrc)  # borrowed from Exo's parser!
    lead_raw = re.match("^(.*)", rawsrc)
    lead_src = re.match("^(.*)", src)
    assert lead_raw is not None and lead_src is not None
    n_dedent = len(lead_raw.group()) - len(lead_src.group())
    tree = ast.parse(src, filename=src_file).body[0]
    ast.increment_lineno(tree, n=lineno - 1)
    ast_increment_colno(tree, n_dedent)

    cast = None
    static_parameters: list[str] = []
    exotic_parameters: set[str] = set()
    static_defaults: list[None | str] = []

    match tree:
        case ast.FunctionDef(name=f_name) as f:
            # print(ast.dump(f, include_attributes=True, indent=2))
            num_required_args = len(f.args.args) - len(f.args.defaults)
            for arg_i, arg in enumerate(f.args.args):
                # assert isinstance(arg.annotation, ast.Name) and arg.annotation.id in ['float']
                # should always be true, see https://docs.python.org/3.8/library/ast.html#ast.parse
                match arg.annotation:
                    case ast.Constant(builtins.Ellipsis):
                        exotic_parameters.add(arg.arg)
                    case None:
                        pass
                    case _:
                        raise MemoError(
                            "Unexpected annotation on parameter",
                            hint=f"The parameter {arg.arg} was annotated with an unnexpected annotation. Currently, memo only supports the annotation `...`, which signals a non-numeric parameter.",
                            user=True,
                            ctxt=None,
                            loc=None
                        )
                assert arg.type_comment is None
                static_parameters.append(arg.arg)
                if arg_i < num_required_args:
                    static_defaults.append(None)
                else:
                    static_defaults.append(ast.unparse(f.args.defaults[arg_i - num_required_args]))
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
                loc=SourceLocation(src_file, tree.lineno, tree.col_offset, "??"),
            )

    pctxt = ParsingContext(
        cast=cast,
        static_parameters=static_parameters,
        exotic_parameters=exotic_parameters,
        static_defaults=static_defaults,
        axes=[],
        loc_name=f_name,
        loc_file=src_file,
        qualname=ff.__qualname__
    )
    stmts: list[Stmt] = []
    retvals = []

    for tp in f.type_params:
        assert isinstance(tp, ast.TypeVar)
        if tp.bound is None:
            raise MemoError(
                f"Missing domain for {tp.name}",
                hint=f"Specify the domain for {tp.name} by writing `{pctxt.loc_name}[{tp.name}: ___, ...]`",
                user=True,
                ctxt=None,
                loc=SourceLocation(pctxt.loc_file, tp.lineno, tp.col_offset, pctxt.loc_name)
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
        loc = SourceLocation(pctxt.loc_file, stmt.lineno, stmt.col_offset, pctxt.loc_name)
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
                retvals.append(parse_expr(expr, pctxt))
            case ast.Expr(value=ast.Constant(value=docstr)) if (
                isinstance(docstr, str) and pctxt.doc is None
            ):
                pctxt.doc = docstr
            case _:
                raise MemoError(
                    f"Unknown statement syntax",
                    hint=f"The full statement is {ast.dump(stmt)}",
                    user=True,
                    ctxt=None,
                    loc=loc
                )

    if len(retvals) == 0:
        raise MemoError(
            f"No return statement",
            hint=f"All memos should end with a return statement",
            user=True,
            ctxt=None,
            loc=SourceLocation(pctxt.loc_file, f.lineno, f.col_offset, pctxt.loc_name),
        )

    return pctxt, stmts, retvals
