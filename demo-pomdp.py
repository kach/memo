from memo import memo
import jax
import jax.numpy as np
from icecream import ic

''' Baby POMDP '''
# https://algorithmsbook.com/files/appendix-f.pdf


S = [0, 1]     # hungry, sated
A = [0, 1, 2]  # feed, sing, ignore
O = [0, 1]     # crying, quiet
B = np.linspace(0, 1, 25)  # P(hungry)

@jax.jit
def get_belief(b, s):
    return np.array([b, 1 - b])[s]

@jax.jit
def Tr(s, a, s_):
    z = np.array([
        [0.0, 1.0, 1.0],
        [0.0, 0.1, 0.1]
    ])[s, a]
    return np.array([z, 1 - z])[s_]

@jax.jit
def Obs(o, s, a):
    z = np.array([
        [0.8, 0.9, 0.8],
        [0.1, 0.0, 0.1]
    ])[s, a]
    return np.array([z, 1 - z])[o]

@jax.jit
def R(s, a):
    return np.array([0, -10])[s] + np.array([0, -5, -0.5])[a]

@memo
def V[b: B](t):
    cast: [alice, env, future_alice]
    alice: knows(b)

    alice: thinks[
        env: knows(b),
        env: chooses(s in S, wpp=get_belief(b, s))
    ]
    # alice chooses her action based on her policy
    # alice: chooses(a in A, wpp=π[b, a](t))
    alice: chooses(a in A, wpp=1)

    # alice gets the next state
    alice: thinks[
        env: knows(a),
        env: chooses(s_ in S, wpp=Tr(s, a, s_)),
        env: chooses(o in O, wpp=Obs(o, s_, a))
    ]

    return alice[
        E[ R(env.s, a) ] + (0.0 if t < 0 else 0.9 * imagine[
            future_alice: observes [env.o] is env.o,
            future_alice: chooses(b_ in B, wpp=exp(-2.0 * (E[env.s] - b_) * (E[env.s] - b_))),
            E[ future_alice[ V[b is self.b_](t - 1) ] ]
        ])
    ]

ic(V(0))

# @cache
# @memo
# def π[b: B, a: A](t):
#     cast: [alice]
#     alice: knows(b)

#     # alice chooses her action based on a softmax over future value
#     alice: chooses(
#         a in A,
#         wpp=exp(
#             2.0 * (R(s, a, g) + (0.0 if t < 0 else ( 0.0 if is_terminating(s, g) else
#                 0.9 * imagine[
#                             future_alice: given(s_ in S, wpp=Tr(s, a, s_)),
#                             E[V[future_alice.s_, future_alice.g](t - 1)],
#                         ]
#                     )
#                 )
#             )
#         ),
#     )
#     return E[ alice.a == a ]