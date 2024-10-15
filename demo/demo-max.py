from functools import partial
from memo import memo

X = [1, 2]
Y = [0, 1, 2]


@partial(memo, debug_print_compiled=True)
def f():
    cast: [alice]
    alice: chooses(x in X, to_maximize=x)
    return E[alice.x]


@partial(memo, debug_print_compiled=True, nojit=True)
def g():
    cast: [alice]
    alice: chooses(x in X, wpp=x)
    alice: chooses(y in Y, to_maximize=self.x * y)
    return E[alice.y]


@partial(memo, debug_print_compiled=True, nojit=True)
def h():
    cast: [alice]
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in Y, to_maximize=1.0 * (self.x + y < 4) * (self.x + y))
    return E[alice.y]

@partial(memo, debug_print_compiled=True, nojit=True)
def k():
    cast: [alice]
    alice: chooses(y in Y, to_maximize=1.0 * (y == 1) + 1.0 * (y == 2))
    return E[alice.y]

f()
# g()
# h()
