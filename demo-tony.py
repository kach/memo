from memoparse import memo, ic

R = [2, 3]
U = [2, 3]
T = [2, 2, 2]

@memo
def test():
    cast: [alice, bob]

    alice: chooses(t in T, wpp=1)
    alice: thinks[
        bob: chooses(r in R, wpp=1),
        # uniform if r == 2, always 2 if r == 3
        bob: chooses(u in U, wpp=0 if r == 3 and u == 3 else 1)
    ]
    alice: observes [bob.u] is alice.t
    alice: chooses(r in R, wpp=E[bob.r])
    return E[alice.r]
ic(test())