from memo import memo
from icecream import ic

X = [0, 1, 2, 3]

@memo
def f():
    cast: []
    forall: x in X
    forall: y in X
    return x + y

@memo
def g():
    cast: [alice]
    alice: given(x in X, wpp=1)
    alice: chooses(a in X, wpp=1)
    alice: chooses(b in X, wpp=1)
    alice: chooses(c in X, wpp=1)
    alice: chooses(y in X, wpp=f[x is self.x, y is self.y]())
    return E[alice.y]
ic(g())


@memo
def h(t):
    cast: []
    forall: x in X
    return x if t == 0 else h[x is self.x](t - 1) * 2
ic(h(4))
