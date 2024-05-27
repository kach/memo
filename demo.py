def demo() -> None:
    io = StringIO()
    ctxt = Context(next_idx=0, io=io, frame=Frame(name=Name("root")), idx_history=[])

    DOORS: list[float] = [1, 2, 3]
    e = EImagine(
        do=[
            SWith(
                Name("alice"),
                SChoose(Name("monty"), Id("prize"), DOORS, wpp=ELit(1)),
            ),
            SPass(),
            SChoose(Name("alice"), Id("initial_pick"), DOORS, wpp=ELit(1)),
            SWith(
                Name("alice"),
                SWith(
                    Name("monty"),
                    SChoose(Name("alice"), Id("initial_pick"), DOORS, wpp=ELit(1)),
                ),
            ),
            SWith(
                Name("alice"),
                SShow(
                    Name("monty"),
                    Name("alice"),
                    Id("initial_pick"),
                    Name("self"),
                    Id("initial_pick"),
                ),
            ),
            SPass(),
            SForAll(Id("revealed_door"), DOORS),
            SWith(
                Name("alice"),
                SChoose(
                    Name("monty"),
                    Id("open"),
                    DOORS,
                    wpp=EOp(
                        Op.ITE,
                        [
                            EOp(
                                Op.OR,
                                [
                                    EOp(
                                        Op.EQ,
                                        [
                                            EChoice(Id("open")),
                                            EWith(
                                                Name("alice"),
                                                EChoice(Id("initial_pick")),
                                            ),
                                        ],
                                    ),
                                    EOp(
                                        Op.EQ,
                                        [EChoice(Id("open")), EChoice(Id("prize"))],
                                    ),
                                ],
                            ),
                            ELit(0),
                            ELit(1),
                        ],
                    ),
                ),
            ),
            SShow(Name("alice"), Name("monty"), Id("open"), Name("self"), Id("revealed_door")),
            SPass(),
            SChoose(
                Name("alice"),
                Id("final_pick"),
                DOORS,
                wpp=EOp(
                    Op.ITE,
                    [
                        EOp(
                            Op.EQ,
                            [
                                EChoice(Id("final_pick")),
                                EWith(Name("monty"), EChoice(Id("open"))),
                            ],
                        ),
                        ELit(0),
                        EOp(
                            Op.EXP,
                            [
                                EExpect(
                                    EOp(
                                        Op.ITE,
                                        [
                                            EOp(
                                                Op.EQ,
                                                [
                                                    EChoice(Id("final_pick")),
                                                    EWith(
                                                        Name("monty"),
                                                        EChoice(Id("prize")),
                                                    ),
                                                ],
                                            ),
                                            ELit(5),
                                            ELit(-5),
                                        ],
                                    )
                                )
                            ],
                        ),
                    ],
                ),
            ),
            SPass(),
        ],
        then=
        # EWith(
        #     Name("alice"),
        #     EExpect(
        #         EOp(
        #             Op.MUL, [ELit(1), EOp(
        #             Op.EQ,
        #             [EChoice(Id("pick")), EWith(Name("monty"), EChoice(Id("prize")))],
        #         )])
        #         # EOp(Op.EQ, [EWith(Name("monty"), EChoice(Id("open"))), EWith(Name("monty"), EChoice(Id("prize")))])
        #     ),
        # )
        # EWith(
        #     Name("alice"),
        #     EExpect(
        #         EOp(
        #             Op.MUL,
        #             [
        #                 ELit(1),
        #                 EOp(
        #                     Op.EQ,
        #                     [
        #                         EChoice(Id("final")),
        #                         EWith(Name("monty"), EChoice(Id("prize"))),
        #                     ],
        #                 ),
        #             ],
        #         )
        #     ),
        # )
        EExpect(
            EWith(
                Name("alice"),
                EOp(Op.EQ, [EChoice(Id("initial_pick")), EChoice(Id("final_pick"))]),
            )
        ),
    )
    print(pprint_expr(e))

    ctxt.emit(HEADER)
    for s in e.do:
        eval_stmt(s, ctxt)
    val = eval_expr(e.then, ctxt)
    ctxt.emit(f"retval = {val.tag}")
    assert val.known

    print()
    print('# Compiled code')
    for i, line in enumerate(io.getvalue().splitlines()):
        print(f"{i + 1: 5d}  {line}")

    retvals: dict[Any, Any] = {}
    exec(io.getvalue(), globals(), retvals)
    # for k, v in retvals.items():
    #     if k in ["torch", "marg"] or k.startswith("lit_"):
    #         continue
    #     ic(k, v.tolist(), v.shape)
    # ic(val, retvals["retval"].tolist(), retvals["retval"].shape)
    # print()

    # print('Output:')

    print()
    deps = val.deps
    for d in deps:
        print(d[0] + "." + d[1], end="\t")
    print("result")
    print("-" * 50)

    for tup in itertools.product(
        *[range(len(ctxt.frame.choices[d].domain)) for d in deps]
    ):
        z: list[Any] = [slice(None) for _ in range(len(retvals["retval"].shape))]
        for d, t in zip(deps, tup):
            print(str(ctxt.frame.choices[d].domain[t]) + "       ", end="\t")
            i = ctxt.frame.choices[d].idx
            z[-1 - i] = t
        assert retvals["retval"][z].numel() == 1
        print(retvals["retval"][z].item())


if __name__ == "__main__":
    demo()


"""
MEMO -> (AGENT : ID(...))... + STMT...

STMT ->
| self/nature.choose(ID=DOM, wpp=EXPR)
| AGENT.show(ID, EXPR)  # dual to "observe"
| observe(ID=AGENT.ID)  # dual to "show"
| condition(EXPR)
| reward(EXPR)
| time(LABEL)

EXPR ->
| LIT | OP(EXPR, ...)
| imagine(STMT..., EXPR)
| self/nature.ID
| expectation(EXPR)  # TODO: probability desugars via indicator
| AGENT[EXPR]        # TODO: AGENT.ID desugars to AGENT[self.ID]
| self.reward        # TODO: in general, variables
| computation_cost(EXPR)
"""