from memoparse import memo, ic
import jax
import jax.numpy as np

U = [0, 1]
R = [0, 1]

@jax.jit
def denotes(u, r):
    return np.array([
        [False, True ],
        [True , True ]
    ])[u, r]

@memo
def literal_speaker():
    cast: [speaker]
    forall: u_ in U
    forall: r_ in R

    speaker: chooses(r in R, wpp=1)
    speaker: chooses(u in U, wpp=(1. if denotes(u, r) else 0.))
    return E[(speaker.u == u_) and (speaker.r == r_)]

@memo
def l1_listener():
    cast: [listener]
    forall: u in U
    forall: r_ in R

    listener: thinks[
        speaker: chooses(r in R, wpp=1),
        speaker: chooses(u in U, wpp=(1. if denotes(u, r) else 0.))
    ]
    listener: observes [speaker.u] is self.u
    listener: chooses(r_ in R, wpp=E[speaker.r == r_])
    return E[ listener.r_ == r_ ]
ic(l1_listener())

@memo
def l2_speaker(beta):
    cast: [speaker, listener]
    forall: u_ in U
    forall: r_ in R

    speaker: thinks[
        listener: thinks[
            speaker: given(r in R, wpp=1),
            speaker: chooses(u in U, wpp=(1. if denotes(u, r) else 0.))
        ]
    ]

    speaker: chooses(r in R, wpp=1)
    speaker: chooses(u in U, wpp=imagine[
        listener: observes [speaker.u] is self.u,
        listener: chooses(r_ in R, wpp=E[speaker.r == r_]),
        exp(beta * E[listener.r_ == r])
    ])
    return E[(speaker.u == u_) and (speaker.r == r_)]
ic(l2_speaker(3.))

@jax.value_and_grad
def f(beta):
    return l2_speaker(beta)[0, 0]
ic(f(3.))

