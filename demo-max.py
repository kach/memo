from memo import memo

X = [1, 2]
Y = [0, 1, 2]

@memo
def f():
    cast: [alice]
    alice: chooses(x in X, to_maximize=x)
    return E[alice.x]

@memo
def g():
    cast: [alice]
    alice: chooses(x in X, wpp=x)
    alice: chooses(y in Y, to_maximize=self.x * y)
    return E[alice.y]

@memo
def h():
    cast: [alice]
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in Y, to_maximize=1.0 * (self.x + y < 4) * (self.x + y))
    return E[alice.y]