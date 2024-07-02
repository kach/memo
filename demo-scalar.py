from memo import memo
from icecream import ic
import jax
import jax.numpy as np

N = [0, 1, 2, 3]  # number of nice people
U = [0, 1, 2]     # utterance: {none, some, all} of the people are nice

@jax.jit
def meaning(n, u):
    return np.array([
        n == 0,  # none
        n > 0,   # some
        n == 3   # all
    ])[u]

@memo
def scalar():
    cast: [speaker, listener]
    forall: n in N
    forall: u in U

    listener: thinks[
        speaker: given(n in N, wpp=1),

        # "literal" speaker
        # speaker: chooses(u in U, wpp=meaning(n, u))

        # "pragmatic" speaker, as on webppl.org
        speaker: chooses(u in U, wpp=imagine[
            listener: knows(self.u),
            listener: chooses(n in N, wpp=meaning(n, self.u)),
            E[listener.n == self.n]
        ])
    ]
    listener: observes [speaker.u] is self.u
    listener: chooses(n in N, wpp=E[speaker.n == n])

    return E[listener.n == n]
ic(scalar())