from memo import *
import ast, inspect


def parse_expr(expr : ast.expr, static_parameters: list[str]) -> Expr:
    match expr:
        case ast.Constant(value=val):
            assert isinstance(val, float) or isinstance(val, int)
            return ELit(value=val)

        case ast.Call(
            func=ast.Name(id='exp'),
            args=[e1]
        ):
            return EOp(
                op=Op.EXP, args=[parse_expr(e1, static_parameters)]
            )

        case ast.Compare(
            left=e1,
            ops=[op],
            comparators=[e2]
        ):
            return EOp(
                op={
                    ast.Eq: Op.EQ,
                    ast.Lt: Op.LT,
                    ast.Gt: Op.GT
                }[op.__class__],
                args=[parse_expr(e1, static_parameters), parse_expr(e2, static_parameters)]
            )

        case ast.UnaryOp(
            op=op,
            operand=operand
        ):
            o_expr = parse_expr(operand, static_parameters)
            return EOp(
                op={
                    ast.USub: Op.NEG,
                    ast.Invert: Op.INV,
                }[op.__class__],
                args=[o_expr]
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
                args=[parse_expr(e1, static_parameters), parse_expr(e2, static_parameters)]
            )

        case ast.BoolOp(
            op=op,
            values=values
        ):
            if len(values) != 2:
                raise Exception(f"Incorrect number of arguments to logical operator {op}")
            e1, e2 = values
            return EOp(
                op={ast.And: Op.AND, ast.Or: Op.OR}[op.__class__],
                args = [parse_expr(e1, static_parameters),
                        parse_expr(e2, static_parameters)]
            )

        case ast.IfExp(
                test=test,
                body=body,
                orelse=orelse
        ):
            c_expr = parse_expr(test, static_parameters)
            t_expr = parse_expr(body, static_parameters)
            f_expr = parse_expr(orelse, static_parameters)
            return EOp(op=Op.ITE, args=[c_expr, t_expr, f_expr])


        case ast.Name(id=id):
            if id in static_parameters:
                return ELit(id)
            return EChoice(id=Id(id))

        case ast.Subscript(
            value=ast.Name(id='E'),
            slice=slice
        ):
            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EExpect(expr=parse_expr(slice, static_parameters))

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
                    ) if expr_ is not None:
                        stmts.extend(parse_stmt(expr_, who_, static_parameters))
            assert not isinstance(elts[-1], ast.Slice)
            return EImagine(do=stmts, then=parse_expr(elts[-1], static_parameters))

        case ast.Subscript(
            value=ast.Name(id=who_id),
            slice=slice
        ):
            assert not isinstance(slice, ast.Slice)
            assert not isinstance(slice, ast.Tuple)
            return EWith(who=Name(who_id), expr=parse_expr(slice, static_parameters))
        case ast.Attribute(
                value = ast.Name(id=who_id),
                attr=attr
                ):
            return EWith(who=Name(who_id), expr=EChoice(attr))

        case _:
            raise Exception(f"Unknown expression {expr} at line {expr.lineno}")


def parse_stmt(expr : ast.expr, who : str, static_parameters: list[str]) -> list[Stmt]:
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
                wpp=parse_expr(wpp_expr, static_parameters)
            )]
        case ast.Compare(
            left=ast.Subscript(
                value=ast.Name(id="observes"),
                slice=ast.Attribute(
                    value=ast.Name(id=target_who),
                    attr=target_id,
                )
            ),
            ops=[ast.Is()],
            comparators=[
                ast.Attribute(
                    value=ast.Name(id=source_who),
                    attr=source_id
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
        case ast.Subscript(
            value=ast.Name('thinks'),
            slice=ast.Slice(
                lower=ast.Name(who_),
                upper=expr_
            )
        ) if expr_ is not None:
            return [SWith(
                who=Name(who),
                stmt=s
            ) for s in parse_stmt(expr_, who_, static_parameters)]
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
                        ) if expr_ is not None:
                        stmts.extend(parse_stmt(expr_, who_, static_parameters))
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
    static_parameters = []

    match tree:
        case ast.Module(body=[ast.FunctionDef(_) as f]):
            for arg in f.args.args:
                # assert isinstance(arg.annotation, ast.Name) and arg.annotation.id in ['float']
                # assert arg.type_comment is None
                static_parameters.append(arg.arg)
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
        case _:
            raise Exception()

    stmts: list[Stmt] = []
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
                assert choice_id not in static_parameters
                stmts.append(SForAll(
                    id=Id(choice_id),
                    domain=dom_id
                ))
            case ast.AnnAssign(
                target=ast.Name(id=who),
                annotation=expr,
                value=None
            ):
                assert who in cast
                stmts.extend(parse_stmt(expr, who, static_parameters))
            case ast.Return(
                value=expr
            ) if expr is not None:
                if retval is not None:
                    raise Exception()
                retval = parse_expr(expr, static_parameters)
            case _:
                raise Exception()

    if retval is None:
        raise Exception()
    # for s in stmts:
        # print(pprint_stmt(s))
    # print(pprint_expr(retval))

    io = StringIO()
    ctxt = Context(next_idx=0, io=io, frame=Frame(name=Name("root")), idx_history=[])
    ctxt.emit(HEADER)
    for stmt in stmts:
        eval_stmt(stmt, ctxt)
    val = eval_expr(retval, ctxt)
    ctxt.emit(f'return {val.tag}')

    out = 'def _out(' + ', '.join(static_parameters) + '):\n' + textwrap.indent(io.getvalue(), '    ')

    # for i, line in enumerate(out.splitlines()):
        # print(f"{i + 1: 5d}  {line}")

    retvals: dict[Any, Any] = {}
    exec(out, globals(), retvals)
    return retvals['_out']

    # print(retvals)
    # print(retvals['listener_ll_9'], retvals['listener_ll_9'].shape)
    # print(ctxt.idx_history)
    # print(retvals['u_ll_28'], retvals['u_ll_28'].shape)
    # print(retvals['speaker_r_0'], retvals['speaker_r_0'].shape)




R = [2, 3] # 10 -> hat, 11 -> glasses + hat
U = [2, 3] # 10 -> hat, 01 -> glasses

@memo
def literal_speaker():
    cast: [speaker]
    given: u_ in U
    given: r_ in R

    speaker: chooses(r in R, wpp=1)
    speaker: chooses(u in U, wpp=(0. if (r == 2 and u == 3) else 1.))
    return E[(speaker.u == u_) and (speaker.r == r_)]

@memo
def l1_listener():
    cast: [listener]
    given: u in U
    given: r_ in R

    listener: thinks[
        speaker: chooses(r in R, wpp=1),
        speaker: chooses(u in U, wpp=(0. if (r == 2 and u == 3) else 1.))
    ]
    listener: observes [speaker.u] is self.u
    listener: chooses(r_ in R, wpp=E[speaker.r == r_])
    return E[ listener.r_ == r_ ]
ic(l1_listener())

@memo
def l2_speaker(beta):
    cast: [speaker, listener]
    given: u_ in U
    given: r_ in R

    speaker: thinks[
        listener: thinks[
            speaker: chooses(r in R, wpp=1),
            speaker: chooses(u in U, wpp=(0. if (r == 2 and u == 3) else 1.))
        ]
    ]

    speaker: chooses(r in R, wpp=1)
    speaker: chooses(u in U, wpp=imagine[
        listener: observes [speaker.u] is self.u,
        listener: chooses(r_ in R, wpp=E[speaker.r == r_]),
        exp(beta * E[listener.r_ == r])
    ])
    return E[(speaker.u == u_) and (speaker.r == r_)]
ic(l2_speaker(3.))

