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
def literal_speaker[u: U, r: R]():
    cast: [speaker]
    speaker: knows(r)
    speaker: chooses(u in U, wpp=denotes(u, r))
    return Pr[ speaker.u == u ]

@memo
def l1_listener[u: U, r: R]():
    cast: [listener, speaker]

    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=1. * denotes(u, r))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[ listener.r == r ]

@memo
def l2_speaker[u: U, r: R](beta):
    cast: [speaker, listener]
    speaker: knows(r)

    speaker: thinks[
        listener: thinks[
            speaker: given(r in R, wpp=1),
            speaker: chooses(u in U, wpp=1. * denotes(u, r))
        ]
    ]

    speaker: chooses(u in U, wpp=imagine[
        listener: observes [speaker.u] is u,
        listener: chooses(r in R, wpp=Pr[speaker.r == r]),
        exp(beta * Pr[listener.r == r])
    ])
    return Pr[speaker.u == u]

ic(literal_speaker())
ic(l1_listener())
ic(l2_speaker(3.))
