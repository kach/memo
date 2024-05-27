from memo import *
import ast, inspect
import typing

from icecream import ic  # type: ignore
ic.configureOutput(includeContext=True)


def parse_expr(expr : ast.expr) -> Expr:
    match expr:
        case ast.Constant(value=val):
            assert isinstance(val, float) or isinstance(val, int)
            return ELit(value=val)

        case ast.Call(
            func=ast.Name(id='exp'),
            args=[e1]
        ):
            return EOp(
                op=Op.EXP, args=[parse_expr(e1)]
            )

        case ast.Compare(
            left=e1,
            ops=[op],
            comparators=[e2]
        ):
            return EOp(
                op={
                    ast.Eq: Op.EQ
                }[op.__class__],
                args=[parse_expr(e1), parse_expr(e2)]
            )

        case ast.BinOp(
            left=e1,
            op=op,
            right=e2
        ):
            return EOp(
                op={
                    ast.Add: Op.ADD,
                    ast.Sub: Op.SUB,
                    ast.Mult: Op.MUL,
                    ast.Div: Op.DIV
                }[op.__class__],
                args=[parse_expr(e1), parse_expr(e2)]
            )

        case ast.Name(id=id):
            return EChoice(id=Id(id))

        case ast.Subscript(
            value=ast.Name(id='E'),
            slice=slice
        ):
            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EExpect(expr=parse_expr(slice))

        case ast.Subscript(
                value=ast.Name("imagine"),
                slice=ast.Tuple(
                    elts=elts
                )
        ):
            stmts = []
            for elt in elts[:-1]:
                match elt:
                    case ast.Slice(
                        lower=ast.Name(id=who_),
                        upper=expr_,
                        step=None
                    ):
                        stmts.extend(parse_stmt(expr_, who_))
            # print(ast.dump(elts[-1], include_attributes=True, indent=2))
            assert not isinstance(elts[-1], ast.Slice)
            return EImagine(do=stmts, then=parse_expr(elts[-1]))

        case ast.Subscript(
            value=ast.Name(id=who_id),
            slice=slice
        ):
            # print(ast.dump(expr, include_attributes=True, indent=2))

            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EWith(who=Name(who_id), expr=parse_expr(slice))

        case _:
            raise Exception(f"Unknown expression {expr} at line {expr.lineno}")


def parse_stmt(expr : ast.expr, who : str) -> list[Stmt]:
    match expr:
        case ast.Call(
            func=ast.Name(id='chooses'),
            args=[
                ast.Compare(
                    left=ast.Name(id=choice_id),
                    comparators=[ast.Name(id=dom_id)],
                    ops=[ast.In()]
                )
            ],
            keywords=[ast.keyword(arg='wpp', value=wpp_expr)]
        ):
            return [SChoose(
                who=Name(who),
                id=Id(choice_id),
                domain=dom_id,
                wpp=parse_expr(wpp_expr)
            )]
        case ast.Call(
            func=ast.Name(id='observes'),
            args=[
                ast.Compare(
                    left=ast.Attribute(value=ast.Name(id=target_who), attr=target_id),
                    comparators=[ast.Attribute(value=ast.Name(id=source_who), attr=source_id)],
                    ops=[ast.Is()]
                )
            ]
        ):
            return [SShow(
                who=Name(who),
                target_who=Name(target_who),
                target_id=Id(target_id),
                source_who=Name(source_who),
                source_id=Id(source_id)
            )]
        case ast.Subscript(  # TODO: handle plural variant
            value=ast.Name('thinks'),
            slice=ast.Slice(
                lower=ast.Name(who_),
                upper=expr_
            )
        ):
            return [SWith(
                who=Name(who),
                stmt=s
            ) for s in parse_stmt(expr_, who_)]
        case ast.Subscript(
            value=ast.Name('thinks'),
            slice=ast.Tuple(
                elts=elts
            )
        ):
            stmts = []
            for elt in elts:
                match elt:
                    case ast.Slice(
                            lower=ast.Name(id=who_),
                            upper=expr_,
                            step=None
                        ):
                        stmt = parse_stmt(expr_, who_)
                        stmts.extend(stmt)
                    case _:
                        raise Exception()
            return [SWith(who=Name(who), stmt=s) for s in stmts]
        case _:
            raise Exception()


def memo(f) -> None:
    src = inspect.getsource(f)
    lines, lineno = inspect.getsourcelines(f)
    tree = ast.parse(src)
    ast.increment_lineno(tree, n=lineno - 1)

    cast = []

    match tree:
        case ast.Module(body=[ast.FunctionDef(_) as f]):
            assert f.args.args == []
            first_stmt = f.body[0]
            match first_stmt:
                case ast.AnnAssign(
                    target=ast.Name(id='cast'),
                    annotation=ast.List(
                        elts=elts
                    ),
                    value=None
                ):
                    for elt in elts:
                        assert isinstance(elt, ast.Name)
                        cast.append(elt.id)
                case _:
                    raise Exception()

            stmts = []
            retval = None
            for stmt in f.body[1:]:
                # print(ast.dump(stmt, include_attributes=True, indent=2))
                match stmt:
                    case ast.AnnAssign(
                        target=ast.Name(id='given'),
                        annotation=ast.Compare(
                            left=ast.Name(id=choice_id),
                            comparators=[ast.Name(id=dom_id)],
                            ops=[ast.In()]
                        ),
                        value=None
                    ):
                        stmt = SForAll(
                            id=Id(choice_id),
                            domain=dom_id
                        )
                        stmts.append(stmt)
                    case ast.AnnAssign(
                        target=ast.Name(id=who),
                        annotation=expr,
                        value=None
                    ):
                        assert who in cast
                        stmts.extend(parse_stmt(expr, who))
                    case ast.Return(
                        value=expr
                    ):
                        if retval is not None:
                            raise Exception()
                        retval = parse_expr(expr)
                    case _:
                        raise Exception()
        case _:
            raise Exception()

    for s in stmts:
        print(pprint_stmt(s))
    print(pprint_expr(retval))
    run_memo(stmts, retval)


def run_memo(stmts: list[Stmt], retval: Expr):
    io = StringIO()
    ctxt = Context(next_idx=0, io=io, frame=Frame(name=Name("root")), idx_history=[])
    ctxt.emit(HEADER)
    for stmt in stmts:
        eval_stmt(stmt, ctxt)
    val = eval_expr(retval, ctxt)
    ctxt.emit(f'retval = {val.tag}')
    print(io.getvalue())

    retvals: dict[Any, Any] = {}
    exec(io.getvalue(), globals(), retvals)
    print(retvals['retval'])
    # print(retvals['exp_27'], retvals['exp_27'].shape)
    # print(retvals['u_ll_28'], retvals['u_ll_28'].shape)
    # print(retvals['speaker_r_0'], retvals['speaker_r_0'].shape)


R = [2, 3] # 10 -> hat, 11 -> glasses + hat
U = [2, 3] # 10 -> hat, 01 -> glasses

# @memo
# def literal_speaker():
#     cast: [speaker]
#     speaker: chooses(r in R, wpp=1)
#     speaker: chooses(u in U, wpp=(1 - 1.0 * ( r == 2 ) * ( u == 3) ))
#     return E[ speaker[u] == speaker[r] ]

# @memo
# def l1_listener():
#     cast: [listener]
#     given: u in U
#     listener: thinks[
#         speaker: chooses(r in R, wpp=1),
#         speaker: chooses(u in U, wpp=(1 - 1.0 * ( r == 2 ) * ( u == 3) ))
#     ]
#     listener: observes(speaker.u is self.u)
#     return E[ listener[ E[ speaker[r] ] ] ]



X = [0, 1]
Y = [0, 1, 2]
Z = [0, 1, 2, 3]

# type: ignore
# @memo
# def l2_speaker():
#     cast: [speaker]
#     speaker: chooses(x in X, wpp=1)
#     speaker: chooses(y in Y, wpp=(
#         imagine[
#             listener: chooses(z in Z, wpp=1),
#             E[ listener[z] ]
#         ]
#     ))
#     return E[speaker[x]]

@memo
def l2_speaker():
    cast: [speaker]
    speaker: chooses(r in R, wpp=1)
    # speaker: thinks[
    # ]
    given: u_ in U
    given: r_ in R
    speaker: chooses(u in U, wpp=imagine[
        listener: thinks[
            speaker: chooses(r in R, wpp=1),
            speaker: chooses(u in U, wpp=(1 - 1. * (r == 2) * (u == 3) ))
        ],
        listener: observes(speaker.u is self.u),
        listener: chooses(r_ in R, wpp=(E[speaker[r] == r_])),
        exp(10. * E[listener[r_] == r])
    ])
    return E[(speaker[u] == u_) * (speaker[r] == r_)]

# @memo
# def speaker():
#     cast: [observer, alice]
#     given: a in R
#     observer: thinks[alice: chooses(r in R, wpp=1)]
    # observer: thinks[alice: chooses(s in R, wpp=1 - 1.0 * (r == s))]
    # observer: observes(alice.s is self.a)
    # return observer[ E[alice[r]] ]