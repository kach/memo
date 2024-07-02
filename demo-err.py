from memo import memo
from icecream import ic

X = [1, 2, 3]

@memo
def f():
    cast: [alice]
    alice: chooses(x in X, wpp=1)
    # return g[a is alice.x](alice.x)
    # alce: chooses(x in X, wpp=1)
    # alice: burns()
    # return a.b.c
    # alice: thinks[
    #     bob: (yield from f)
    # ]
    # return 1.0
    return 1.0
    # return bob.x
ic(f())