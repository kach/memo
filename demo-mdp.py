from memoparse import *
import jax
import jax.numpy as np

S = [0, 1, 2, 3, 4]
A = [+1, -1]

@jax.jit
def Tr(s, a, s_):
    return 1. * (np.clip(s + a, 0, 4) == s_)

@jax.jit
def R(s, a):
    return s

@memo
def V(t):
    cast: [alice]
    forall: s in S  ## TODO: alice "knows" self.s
    alice: given(s in S, wpp=1)
    alice: chooses(a in A, wpp=Q[s is self.s, a is self.a](t))
    alice: given(s_ in S, wpp=Tr(s, a, s_))
    return E[R(alice.s, alice.a) + (0. if t < 0 else 0.99 * V[s is alice.s_](t - 1))]

@memo
def Q(t):
    cast: [alice]
    forall: s in S
    forall: a in A

    alice: given(s in S, wpp=1)
    alice: chooses(
        a in A,
        wpp=exp(
            R(s, a) + (0. if t < 0 else 0.99 * imagine[
                future_alice: given(s_ in S, wpp=Tr(s, a, s_)),
                E[V[s is future_alice.s_](t - 1)]
            ])
        )
    )
    return E[alice.s == s and alice.a == a]

ic(V(3))