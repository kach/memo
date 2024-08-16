from memo import memo
X = [1, 2, 3]
Y = [2, 3]

@memo
def f[x: X]():
    # cast: [alice, bob]
    alice: knows(x)
    alice: thinks[ bob: knows(x) ]
    return alice[bob[x]]
print(f())

@memo
def g[y: X]():
    cast: [alice, bob]
    bob: thinks [ alice: chooses(x in X, wpp=1) ]
    bob: observes [alice.x] is y
    return bob[ alice.x ]
print(g())

# @memo
# def h():
#     cast: [alice, bob, future_alice]
#     alice: chooses(z in z, wpp=1)
#     alice: thinks[ bob: chooses(y in Y, wpp=1) ]

#     alice: chooses(x in X, wpp=
#                    E[
#         imagine[
#             future_alice: chooses(x in X, wpp=z),
#             bob: knows(z),
#             E[future_alice.x]
#         ]
#                    ]
#     )
#     return E[ alice[ bob.z ] ]
# print(h())