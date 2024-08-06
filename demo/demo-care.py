from memo import memo, domain
import jax.numpy as np
import jax
from icecream import ic
from functools import cache
from enum import IntEnum

class A(IntEnum):
    N = 0; E = 1; S = 2; W = 3; X = 4

maze = '''\
*--*--*--*--*--*
|     |        |
*--*  *  *--*  *
|  |     |     |
*  *--*  *  *  *
|           |  |
*--*--*--*--*--*
'''

maze = maze.splitlines()
H = (len(maze) - 1) // 2
W = (len(maze[0]) - 1) // 3

motion = np.zeros((W, H, 4))
for x in range(W):
    for y in range(H):
        for a in A:
            assert maze[y * 2][x * 3] == '*'
            if a == A.N:
                z = maze[y * 2 + 0][x * 3 + 1] == '-'
            if a == A.E:
                z = maze[y * 2 + 1][x * 3 + 3] == '|'
            if a == A.S:
                z = maze[y * 2 + 2][x * 3 + 1] == '-'
            if a == A.W:
                z = maze[y * 2 + 1][x * 3 + 0] == '|'
            motion = motion.at[x, y, a].set(z)

class Mode(IntEnum):
    GOOD = 0
    EVIL = 1

class Horizon(IntEnum):
    FIN = 0
    INF = 1

Hid = np.array(Mode)
Loc = domain(x=W, y=H)
S = domain(player=len(Loc), portal=len(Loc), horizon=len(Horizon))
print("State space size:", len(S))

S_init = Loc(0, 0)
S_goal = Loc(W - 1, H - 1)
S_good = S_goal
S_evil = S_init

@jax.jit
def Tr(h, s, a, s_):
    sxy, pxy, hor = S._tuple(s)
    sxy_, pxy_, hor_ = S._tuple(s_)

    sx, sy = Loc._tuple(sxy)
    nxy = Loc(
        np.array([0, 1, 0, -1, 0])[a] + sx,
        np.array([-1, 0, 1, 0, 0])[a] + sy
    )

    return np.select(
        [
            # regular move
            (a != A.X) & (s_ == np.where(motion[sx, sy, a], s, S(nxy, pxy, hor))),
            # goal -> reset and randomize portal
            (a == A.X) & (sxy == S_goal) & (sxy_ == S_init) & (hor == hor_),
            # on portal -> goto mode
            (a == A.X) & (sxy == pxy) & (h == Mode.GOOD) & (s_ == S._update(s, player=S_good)),
            (a == A.X) & (sxy == pxy) & (h == Mode.GOOD) & (s_ == S._update(s, player=S_evil)),
            (a == A.X) & (sxy == pxy) & (h == Mode.EVIL) & (s_ == S._update(s, player=S_good)),
            (a == A.X) & (sxy == pxy) & (h == Mode.EVIL) & (s_ == S._update(s, player=S_evil)),
        ],
        [1, 1,  1, 0, 0, 1], 0
    )

@jax.jit
def R(s, a):
    sxy, _, _ = S._tuple(s)
    return 1. * ((sxy == S_goal) & (a == A.X)) - 0.03

@jax.jit
def term(s, a):
    sxy, _, hor = S._tuple(s)
    return (sxy == S_goal) & (a == A.X) & (hor == Horizon.FIN)

B = np.linspace(0.01, 0.99, 5)  # P(good)
@jax.jit
def get_belief(b, h):
    return np.array([b, 1 - b])[h]

@jax.jit
def gamma():
    return 1.0

@jax.jit
def beta():
    return 7


@cache
@memo
def V[b: B, s: S](t):
    cast: [alice, env, future_alice]
    alice: knows(b)
    alice: knows(s)

    alice: thinks[
        env: knows(b),
        env: knows(s),
        env: chooses(h in Hid, wpp=get_belief(b, h))
    ]

    alice: chooses(a in A, to_maximize=π[b, s, a](t))

    alice: thinks[
        env: knows(a),
        env: chooses(s_ in S, wpp=Tr(h, s, a, s_))
    ]

    return E[ alice[
        R(s, a) + (0.0 if t <= 0 else 0.0 if term(s, a) else gamma() * imagine[
            future_alice: observes [env.s_] is env.s_,
            future_alice: chooses(b_ in B, wpp=exp(-10.0 * abs(E[env.h == 0] - get_belief(b_, 0)))),
            E[ future_alice[ V[b_, env.s_](t - 1) ] ]
        ])
    ] ]
ic("Compiled V")

@cache
@memo
def π[b: B, s: S, a: A](t):
    cast: [alice, env, future_alice]
    alice: knows(b)
    alice: knows(s)

    alice: thinks[
        env: knows(b),
        env: knows(s),
        env: chooses(h in Hid, wpp=get_belief(b, h))
    ]

    alice: chooses(
        a in A,
        wpp=exp(
            beta() * (R(s, a) + (0.0 if t <= 0 else 0.0 if term(s, a) else gamma() * imagine[
                        env: knows(a),
                        env: chooses(s_ in S, wpp=Tr(h, s, a, s_)),
                        future_alice: thinks[
                            env: knows(a),
                            env: chooses(s_ in S, wpp=Tr(h, s, a, s_))
                        ],
                        future_alice: observes [env.s_] is env.s_,
                        future_alice: chooses(b_ in B, wpp=exp(-10.0 * abs(E[env.h == 0] - get_belief(b_, 0)))),
                        E[V[future_alice.b_, env.s_](t - 1)],
                    ]
                )
            )
        ),
    )
    return E[ alice.a == a ]
ic('Compiled π')

@cache
@memo
def V_veridical[h: Hid, b: B, s: S](t):
    cast: [alice, env, future_alice]
    alice: knows(b)
    alice: knows(s)
    env: knows(h)
    env: knows(s)

    alice: thinks[
        env: knows(b),
        env: knows(s),
        env: chooses(h in Hid, wpp=get_belief(b, h))
    ]
    alice: chooses(a in A, wpp=π[b, s, a](t))

    alice: thinks[
        env: knows(a),
        env: chooses(s_ in S, wpp=Tr(h, s, a, s_))
    ]
    env: thinks[ alice: chooses(a in A, wpp=1) ]
    env: observes [alice.a] is alice.a
    env: chooses(s_ in S, wpp=Tr(h, s, alice.a, s_))

    alice: observes [env.s_] is env.s_
    alice: chooses(b_ in B, wpp=exp(-10.0 * abs(E[env.h == 0] - get_belief(b_, 0))))

    return E[
        R(s, alice.a)
        + (0.0 if t <= 0 else 0.0 if term(s, alice.a) else gamma() * V_veridical[h, alice.b_, env.s_](t - 1))
    ]
ic('Compiled V_veridical')

@jax.jit
def make_teacher_state(l, hor):
    return S(Loc(0, 0), l, hor)

@jax.jit
def is_valid_pxy(l):
    return (l != S_goal) & (l != S_init)

@cache
@memo
def teacher[h: Hid, hor: Horizon, b: B, l: Loc](t):
    cast: [teacher, env]
    teacher: knows(h)
    teacher: knows(hor)
    teacher: knows(b)
    teacher: chooses(l in Loc, wpp=is_valid_pxy(l) * beta() * exp(
        imagine[
            env: knows(l),
            env: knows(hor),
            env: chooses(s in S, wpp=s == make_teacher_state(l, hor)),
            E[V_veridical[h, b, env.s](t)]
        ]))
    return E[teacher.l == l]
ic('Compiled teacher')

@cache
@memo
def student[h: Hid, hor: Horizon, b: B, l: Loc](t):
    cast: [student, teacher]
    student: knows(hor)
    student: knows(b)
    student: thinks[
        teacher: given(h in Hid, wpp=1),
        teacher: knows(hor),
        teacher: knows(b),
        teacher: chooses(l in Loc, wpp=teacher[h, hor, b, l](t))
    ]
    student: observes[teacher.l] is l
    student: chooses(h in Hid, wpp=E[teacher.h == h])
    return E[student.h == h]
ic('Compiled student')

import sys
if len(sys.argv) > 1:
    v = V(50)
    ic('got V')
    np.save('aux-care/v.npy', v)

    vv = V_veridical(50)
    ic('got Vv')
    np.save('aux-care/vv.npy', vv)

    p = π(50)
    ic('got π')
    np.save('aux-care/p.npy', p)

    t = teacher(50)
    ic('got t')
    np.save('aux-care/t.npy', t)

    s = student(50)
    ic('got s')
    np.save('aux-care/s.npy', s)





# from jax.experimental import sparse
# M = sparse.BCOO.fromdense(np.array(
#     [0, 0, 0, 1, 0, 0, 0, 0, 2]
# ))
# print((M * M[..., None]))