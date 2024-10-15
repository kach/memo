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

t = 100

plt.subplot(1, 2, 1)
q_fn = Q(t).transpose().reshape((2, 4, H, W))[0]
value_fn = q_fn.max(axis=0)

ic(value_fn)

p = plt.imshow(
    value_fn * (1 - maze.reshape((H, W))),
    origin="upper",
    cmap="PuRd_r",
)
plt.colorbar(p)

policy = q_fn.argmax(axis=0)

directions = coord_actions[policy]
plt.quiver(
    np.arange(W),
    np.arange(H),
    directions[:, :, 0],
    -directions[:, :, 1],
    color="red",
)
plt.axis("off")


plt.subplot(1, 2, 2)
posterior = invplan().transpose()
plt.imshow(
    1 - maze.reshape((H, W)),
    origin="upper",
    cmap="gray",
)

plt.quiver(np.arange(W), np.arange(H), -np.ones((H, W)), np.zeros((H, W)), posterior[0].reshape((H, W)), cmap="coolwarm", clim=(0, 1))
plt.quiver(np.arange(W), np.arange(H), np.ones((H, W)), np.zeros((H, W)), posterior[1].reshape((H, W)), cmap="coolwarm", clim=(0, 1))
plt.quiver(np.arange(W), np.arange(H), np.zeros((H, W)), np.ones((H, W)), posterior[2].reshape((H, W)), cmap="coolwarm", clim=(0, 1))
plt.quiver(np.arange(W), np.arange(H), np.zeros((H, W)), -np.ones((H, W)), posterior[3].reshape((H, W)), cmap="coolwarm", clim=(0, 1))

plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill="tab:gray", ec="black", linewidth=5, alpha=0.25))

plt.axis("off")
plt.savefig('out.png')
