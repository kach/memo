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
def literal_speaker():
    cast: [speaker]
    forall: u_ in U
    forall: r in R
    speaker: knows(r)
    speaker: chooses(u in U, wpp=denotes(u, r))
    return E[ speaker.u == u_ ]

@memo
def l1_listener():
    cast: [listener, speaker]
    forall: u in U
    forall: r_ in R

    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=1. * denotes(u, r))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r_ in R, wpp=E[speaker.r == r_])
    return E[ listener.r_ == r_ ]

@memo
def l2_speaker(beta):
    cast: [speaker, listener]
    forall: u_ in U
    forall: r in R
    speaker: knows(r)

    speaker: thinks[
        listener: thinks[
            speaker: given(r in R, wpp=1),
            speaker: chooses(u in U, wpp=1. * denotes(u, r))
        ]
    ]

    speaker: chooses(u in U, wpp=imagine[
        listener: observes [speaker.u] is u,
        listener: chooses(r_ in R, wpp=E[speaker.r == r_]),
        exp(beta * E[listener.r_ == r])
    ])
    return E[speaker.u == u_]

ic(literal_speaker())
ic(l1_listener())
ic(l2_speaker(3.))
