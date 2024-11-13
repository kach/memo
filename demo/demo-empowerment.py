from memo import memo, domain, make_module
import jax
import jax.numpy as np
from enum import IntEnum

from matplotlib import pyplot as plt

"""
**Inspired by:** Klyubin, A. S., Polani, D., & Nehaniv, C. L. (2005, September).
All else being equal be empowered. In European Conference on Artificial Life
(pp. 744-753). Berlin, Heidelberg: Springer Berlin Heidelberg.

This example shows how to use memo to compute an agent's empowerment in a gridworld.
The particular example is inspired by Figure 3a in Klyubin et al (2005).
"""

## This is a little memo module that implements the Blahut-Arimoto algorithm for empowerment
# See: https://www.comm.utoronto.ca/~weiyu/ab_isit04.pdf
def make_blahut_arimoto(X, Y, Z, p_Y_given_X):
    m = make_module('blahut_arimoto')
    m.X = X
    m.Y = Y
    m.Z = Z
    m.p_Y_given_X = p_Y_given_X

    @memo(install_module=m.install)
    def q[x: X, z: Z](t):
        alice: knows(z)
        alice: chooses(x in X, wpp=imagine[
            bob: knows(x, z),
            bob: chooses(y in Y, wpp=p_Y_given_X(y, x, z)),
            # exp(E[ log(Q[x, bob.y, z](t - 1) if t > 0 else 1) ])
            bob: thinks[
                charlie: knows(y, z),
                charlie: chooses(x in X, wpp=Q[x, y, z](t - 1) if t > 0 else 1)
            ],
            exp(E[bob[H[charlie.x]]])
        ])
        return Pr[alice.x == x]

    @memo(install_module=m.install)
    def Q[x: X, y: Y, z: Z](t):
        alice: knows(x, y, z)
        alice: thinks[
            bob: knows(x, z),
            bob: chooses(x in X, wpp=q[x, z](t)),
            bob: chooses(y in Y, wpp=p_Y_given_X(y, x, z))
        ]
        alice: observes [bob.y] is y
        return alice[Pr[bob.x == x]]

    @memo(install_module=m.install)
    def C[z: Z](t):
        alice: knows(z)
        alice: chooses(x in X, wpp=q[x, z](t))
        alice: chooses(y in Y, wpp=p_Y_given_X(y, x, z))
        return (H[alice.x] + H[alice.y] - H[alice.x, alice.y]) / log(2)  # convert to bits

    return m

# # Sanity check: a channel that drops messages with probability 0.1 should have capacity 0.9 bits.
# X = [0, 1]
# Y = [0, 1, 2]
# @jax.jit
# def p_Y_given_X(y, x, z):
#     return np.array([
#         [0.9, 0.1, 1e-10],
#         [1e-10, 0.1, 0.9]
#     ])[x, y]
# m = make_blahut_arimoto(X, Y, np.array([0]), p_Y_given_X)
# print(m.q(10))
# print(m.C(10))


## Setting up a gridworld...
world = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

X = np.arange(world.shape[0])
Y = np.arange(world.shape[1])
S = domain(x=len(X), y=len(Y))

class A(IntEnum):
    N = 0
    S = 1
    W = 2
    E = 3
    O = 4
Ax = domain(
    a1=len(A),
    a2=len(A),
    a3=len(A),
    a4=len(A),
)

@jax.jit
def Tr1(s, a):
    x = S.x(s)
    y = S.y(s)
    z = np.array([
        [x, y - 1],
        [x, y + 1],
        [x - 1, y],
        [x + 1, y],
        [x, y]
    ])[a]
    x_ = np.clip(z[0], 0, len(X) - 1)
    y_ = np.clip(z[1], 0, len(Y) - 1)
    return np.where(world[x_, y_], s, S(x_, y_))


@jax.jit
def Tr(s_, ax, s):
    for a in Ax._tuple(ax):
        s = Tr1(s, a)
    return s == s_

# ...and computing 4-step empowerment in the gridworld!
m = make_blahut_arimoto(X=Ax, Y=S, Z=S, p_Y_given_X=Tr)
m.Z = S
@memo(install_module=m.install, debug_trace=True)
def empowerment[s: Z](t):
    return C[s](t)

emp = m.empowerment(5).block_until_ready()
emp = emp.reshape(len(X), len(Y))
emp = emp * (1 - world)
plt.colorbar(plt.imshow(emp.reshape(len(X), len(Y)) * (1 - world), cmap='gray'))
plt.savefig('out.png')