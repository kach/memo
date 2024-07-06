from memo import memo
X = [1, 2, 3]
Y = [2, 3]

@memo
def f[x: X]():
    cast: [alice, bob]
    alice: knows(x)
    alice: thinks[ bob: knows(x) ]
    return alice[bob[x]]
print(f())

# @memo
# def f[y: X]():
#     cast: [alice, bob]
#     bob: thinks [ alice: chooses(x in X, wpp=1) ]
#     bob: observes [alice.x] is y
#     return bob[ alice.x ]
