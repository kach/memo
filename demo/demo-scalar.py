from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

## Scalar implicature

NN = 10_000

N = np.arange(NN + 1)  # number of nice people
class U(IntEnum):
    NONE = 0
    SOME = 1
    ALL = 2

@jax.jit
def meaning(n, u):  # (none)  (some)  (all)
    return np.array([ n == 0, n > 0, n == NN ])[u]

@memo
def scalar[n: N, u: U]():
    listener: thinks[
        speaker: chooses(n in N, wpp=1),
        speaker: chooses(u in U, wpp=imagine[
            listener: knows(u),
            listener: chooses(n in N, wpp=meaning(n, u)),
            Pr[listener.n == n]
        ])
    ]
    listener: observes [speaker.u] is u
    listener: chooses(n in N, wpp=E[speaker.n == n])
    return Pr[listener.n == n]

scalar()  # warm up JIT

import time
t_s = time.time()
scalar()
print(time.time() - t_s)
