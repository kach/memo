from memo import memo
import jax
import jax.numpy as np

N = [0, 1, 2, 3]  # number of nice people
U = [0, 1, 2]     # utterance: {none, some, all} of the people are nice

@jax.jit
def meaning(n, u):  # (none)  (some)  (all)
    return np.array([ n == 0, n > 0, n == 3 ])[u]

@memo
def scalar[n: N, u: U]():
    cast: [speaker, listener]
    listener: thinks[
        speaker: given(n in N, wpp=1),
        speaker: chooses(u in U, wpp=imagine[
            listener: knows(u),
            listener: chooses(n in N, wpp=meaning(n, u)),
            E[listener.n == n]
        ])
    ]
    listener: hears [speaker.u] is u
    listener: chooses(n in N, wpp=E[speaker.n == n])
    return E[listener.n == n]

print(scalar())