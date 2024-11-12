from memo import memo
import jax
import jax.numpy as np
from icecream import ic
from functools import cache
from enum import IntEnum

''' Baby POMDP '''
# https://algorithmsbook.com/files/appendix-f.pdf

class S(IntEnum):
    Hungry = 0
    Sated = 1

class A(IntEnum):
    Feed = 0
    Sing = 1
    Ignore = 2

class O(IntEnum):
    Crying = 0
    Quiet = 1

B = np.linspace(0, 1, 50)  # P(hungry)

@jax.jit
def get_belief(b, s):
    return np.array([b, 1 - b])[s]

@jax.jit
def Tr(s, a, s_):
    z = np.array([  # P(hungry | s, a)
        [0.0, 1.0, 1.0],  # if hungry
        [0.0, 0.1, 0.1]   # if sated
    ])[s, a]
    return np.array([z, 1 - z])[s_]

@jax.jit
def Obs(o, s, a):
    z = np.array([  # P(cry | s, a)
        [0.8, 0.9, 0.8],  # if hungry
        [0.1, 0.0, 0.1]   # if sated
    ])[s, a]
    return np.array([z, 1 - z])[o]

@jax.jit
def R(s, a):
    return (
        np.array([-10, 0])[s] +
        np.array([-5, -0.5, 0])[a]
    )

@cache
@memo
def V[b: B](t):
    cast: [alice, env, future_alice]
    alice: knows(b)

    alice: thinks[
        env: knows(b),
        env: chooses(s in S, wpp=get_belief(b, s))
    ]

    alice: chooses(a in A, wpp=π[b, a](t))

    alice: thinks[
        env: knows(a),
        env: chooses(s_ in S, wpp=Tr(s, a, s_)),
        env: chooses(o in O, wpp=Obs(o, s_, a))
    ]

    return E[ alice[
        E[ R(env.s, a) ] + (0.0 if t <= 0 else 0.9 * imagine[
            future_alice: observes [env.o] is env.o,
            future_alice: chooses(b_ in B, wpp=exp(-100.0 * abs(E[env.s_ == 0] - b_))),
            E[ future_alice[ V[b_](t - 1) ] ]
        ])
    ] ]

@cache
@memo
def π[b: B, a: A](t):
    cast: [alice, env, future_alice]
    alice: knows(b)

    alice: thinks[
        env: knows(b),
        env: chooses(s in S, wpp=get_belief(b, s))
    ]

    alice: chooses(
        a in A,
        to_maximize=(
            (E[ R(env.s, a) ] + (0.0 if t <= 0 else 0.9 * imagine[
                        env: knows(a),
                        env: chooses(s_ in S, wpp=Tr(s, a, s_)),
                        env: chooses(o in O, wpp=Obs(o, s_, a)),
                        future_alice: thinks[
                            env: knows(a),
                            env: chooses(s_ in S, wpp=Tr(s, a, s_)),
                            env: chooses(o in O, wpp=Obs(o, s_, a))
                        ],
                        future_alice: observes [env.o] is env.o,
                        future_alice: chooses(b_ in B, wpp=exp(-100.0 * abs(E[env.s_ == 0] - b_))),
                        E[V[future_alice.b_](t - 1)],
                    ]
                )
            )
        ),
    )
    return E[ alice.a == a ]


@memo  # not used above, just an example of how you would implement this
def belief_update[b: B, b_: B, a: A, o: O]():
    alice: knows(b, a)
    alice: thinks[
        env: knows(b, a),
        env: chooses(s in S, wpp=get_belief(b, s)),
        env: chooses(s_ in S, wpp=Tr(s, a, s_)),
        env: chooses(o in O, wpp=Obs(o, s_, a))
    ]
    alice: observes [env.o] is o
    alice: chooses(b_ in B, wpp=exp(-100.0 * abs(E[env.s == 0] - b_)))
    return E[alice.b_ == b_]


from matplotlib import pyplot as plt
plt.figure(figsize=(3, 2))
v = V(10)
p = π(10)
plt.plot(B[p[:, 0] == 1], v[p[:, 0] == 1], label='feed')
plt.plot(B[p[:, 1] == 1], v[p[:, 1] == 1], ':', label='sing')
plt.plot(B[p[:, 2] == 1], v[p[:, 2] == 1], '--', label='ignore')
plt.legend()
plt.xlabel('Belief state, P(hungry)')
plt.ylabel('Long-term reward')
plt.title('Crying baby POMDP solution')
plt.tight_layout()
plt.savefig('../paper/fig/pomdp.pdf')
