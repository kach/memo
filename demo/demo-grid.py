from functools import cache

from icecream import ic
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(1_000_000)

from memo import memo

H = 21
W = 21
S = np.arange(H * W)
G = np.array([0, H * W - 1])
A = np.array([0, 1, 2, 3])  # left, right, up, down

coord_actions = np.array([
    [-1, 0],
    [+1, 0],
    [0, -1],
    [0, +1],
])

maze_raw = np.array(1 - plt.imread('../paper/fig/logo-maze.png'), dtype=int); maze = maze_raw.reshape(-1)
# maze = np.zeros(H * W)

@jax.jit
def Tr(s, a, s_):
    x, y = s % W, s // W
    next_coords = np.array([x, y]) + coord_actions[a]
    next_state = (
        + 1 * np.clip(next_coords[0], 0, W - 1)
        + W * np.clip(next_coords[1], 0, H - 1)
    )
    return (
        + 1.0 * ((next_state == s_) & (maze[next_state] == 0))
        + 1.0 * ((maze[next_state] == 1) & (s == s_))
    )

@jax.jit
def R(s, a, g):
    return 1.0 * (s == g) - 0.1

@jax.jit
def is_terminating(s, g):
    return s == g

@jax.jit
def gamma():
    return 1.0

@cache
@memo
def Q[s: S, a: A, g: G](t):
    alice: knows(s, a, g)
    alice: given(s_ in S, wpp=Tr(s, a, s_))
    alice: chooses(a_ in A, to_maximize=0.0 if t < 0 else Q[s_, a_, g](t - 1))
    return E[
        R(s, a, g) + (0.0 if t < 0 else 0.0 if is_terminating(s, g) else gamma() * Q[alice.s_, alice.a_, g](t - 1))
    ]

@memo
def invplan[s: S, a: A, g: G](t):
    observer: knows(a, s, g)
    observer: thinks[
        alice: chooses(g in G, wpp=1),
        alice: knows(s),
        alice: chooses(a in A, wpp=exp(2 * Q[s, a, g](t))),
    ]
    observer: observes [alice.a] is a
    return observer[E[alice.g == g]]

if len(sys.argv) > 1:
    import time
    t = int(sys.argv[1])
    print('t =', t)
    Q(0);
    t0 = time.time()
    v = Q(t).max(axis=1)
    print(time.time() - t0)
    exit()
