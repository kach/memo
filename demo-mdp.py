from memoparse import *
import jax
import jax.numpy as np
from functools import cache

W, H = 10, 10
N = W * H

S = np.arange(N)
A = [+10, +1, 0, -1, -10]

@jax.jit
def Tr(s, a, s_):
    return 1. * (np.clip(s + a, 0, N - 1) == s_)

@jax.jit
def R(s, a):
    return 1. * (s == N - 1)

@cache
@memo
def V(t):
    cast: [alice]
    forall: s in S
    alice: knows(self.s)
    alice: chooses(a in A, wpp=Q[s is self.s, a is self.a](t))
    alice: given(s_ in S, wpp=Tr(s, a, s_))
    return E[R(s, alice.a) + (0. if t < 0 else 0.9 * V[s is alice.s_](t - 1))]

@cache
@memo
def Q(t):
    cast: [alice]
    forall: s in S
    forall: a in A

    alice: knows(self.s)
    alice: chooses(
        a in A,
        wpp=exp(2.0 * (
            R(s, a) + (0. if t < 0 else 0.9 * imagine[
                future_alice: given(s_ in S, wpp=Tr(s, a, s_)),
                E[V[s is future_alice.s_](t - 1)]
            ])
        ))
    )
    return E[alice.a == a]

V(200)
V(400)
V(600)
V(800)
V(1000)

import timeit
print(timeit.timeit(lambda: V(1000), number=250))
ic(V(1000))
ic(Q(1000))