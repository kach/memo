from memo import memo
from icecream import ic
import jax
import jax.numpy as np

U = [0, 1]
R = [0, 1]

@jax.jit
def denotes(u, r):
    return np.array([
        [0, 1],
        [1, 1]
    ])[r, u]

@memo
def speaker[u: U, r: R](beta, t):
    speaker: knows(r)
    speaker: chooses(u in U, wpp=imagine[
        listener: knows(u),
        listener: chooses(r in R, wpp=listener[u, r](beta, t)),
        exp(beta * Pr[listener.r == r])
    ])
    return Pr[speaker.u == u]

@memo
def listener[u: U, r: R](beta, t):
    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=speaker[u, r](beta, t - 1) if t > 0 else denotes(u, r))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

beta = 3.
ic(4, listener(beta, 4, compute_cost=True))