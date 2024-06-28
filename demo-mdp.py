from memo import memo
from icecream import ic
import jax
import jax.numpy as np
from functools import cache

N = 100
S = np.arange(N)  # state space is a number line from 0 to N
A = [-10, -1, 0, +1, +10]  # action space is motions left/right of various increments

@jax.jit
def Tr(s, a, s_):
    '''Move by a along the number line, clamp to edges.'''
    return 1. * (np.clip(s + a, 0, N - 1) == s_)

@jax.jit
def R(s, a):
    '''Reward of 1 for being on the final state.'''
    return 1. * (s == N - 1)

@cache
@memo
def V(t):
    cast: [alice]
    forall: s in S
    alice: knows(self.s)

    # alice chooses her action based on her policy
    alice: chooses(a in A, wpp=π[s is self.s, a is self.a](t))
    alice: given(s_ in S, wpp=Tr(s, a, s_))

    # her value depends on the expected V-function at the next state
    return E[R(s, alice.a) + (0. if t < 0 else 0.9 * V[s is alice.s_](t - 1))]

@cache
@memo
def π(t):
    cast: [alice]
    forall: s in S
    forall: a in A

    alice: knows(self.s)

    # alice chooses her action based on a softmax over future value
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

ic(V(200))
ic(π(200))