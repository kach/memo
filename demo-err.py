from memo import memo
X = [1, 2, 3]
Y = [2, 3]

@memo
def f[y: X]():
    cast: [alice, bob]
    bob: thinks [ alice: chooses(x in X, wpp=1) ]
    bob: observes [alice.x] is y
    return bob[ alice.x ]

# @memo
# def f():
#     cast: [alice, bob]
#     alice: chooses(x in X, wpp=1)
#     bob: thinks [ alice: chooses(x in X, wpp=1) ]
#     alice: thinks [ bob: knows(y) ]
#     return 1

# @memo
# def f():
#     cast: [alice, bob]
#     alice: chooses(x in X, wpp=1)
#     alice: thinks[
#         bob: chooses(y in Y, wpp=1)
#     ]
#     alice: observes [bob.y] is alice.x
#     return E[alice.x - alice[bob.y]]
# print(f())