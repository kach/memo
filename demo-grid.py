from functools import cache

from icecream import ic
import jax
import jax.numpy as np
import matplotlib.pyplot as plt

from memo import memo

H = 10
W = 5
S = np.arange(H * W)
G = np.array([0, 4])
A = np.array([0, 1, 2, 3])  # left, right, up, down

coord_actions = np.array([
    [-1, 0],
    [+1, 0],
    [0, -1],
    [0, +1],
])

# fmt: off
maze = np.array([
    0, 0, 0, 0, 0,
    0, 1, 0, 1, 0,
    0, 1, 1, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 1, 0,
    0, 1, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 0, 0, 0,
])
# fmt: on


@jax.jit
def Tr(s, a, s_):
    x = s % W
    y = s // W

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
    """Reward of 1 for being on the goal state."""
    return 1.0 * (s == g)


@jax.jit
def is_terminating(s, g):
    return False  # s == g


@cache
@memo
def V(t):
    cast: [alice]
    forall: s in S
    forall: g in G

    alice: knows(s)
    alice: knows(g)

    # alice chooses her action based on her policy
    alice: chooses(a in A, wpp=π[s, a, g](t))

    # alice gets the next state
    alice: given(s_ in S, wpp=Tr(s, a, s_))

    # her value depends on the expected V-function at the next state
    return E[
        R(s, alice.a, g) +
        (0.0 if t < 0 else (0.0 if is_terminating(s, g) else 0.9 * V[alice.s_, g](t - 1)))
    ]


@cache
@memo
def π(t):
    cast: [alice]
    forall: s in S
    forall: a in A
    forall: g in G

    alice: knows(s)
    alice: knows(g)

    # alice chooses her action based on a softmax over future value
    alice: chooses(
        a in A,
        wpp=exp(
            2.0 * (R(s, a, g) + (0.0 if t < 0 else ( 0.0 if is_terminating(s, g) else
                0.9 * imagine[
                            future_alice: given(s_ in S, wpp=Tr(s, a, s_)),
                            E[V[future_alice.s_, future_alice.g](t - 1)],
                        ]
                    )
                )
            )
        ),
    )
    return E[ alice.a == a ]


@memo
def invplan():
    cast: [observer, alice]
    forall: s in S
    forall: a in A

    observer: knows(a)
    observer: knows(s)

    observer: thinks[
        alice: chooses(g in G, wpp=1),
        alice: knows(s),
        alice: chooses(a in A, wpp=π[s, a, g](200)),
    ]
    observer: observes [alice.a] is a
    return observer[E[alice.g == 0]]


plt.subplot(1, 2, 1)
value_fn = (V(200)).reshape((2, H, W))[0]
p = plt.imshow(
    value_fn,
    origin="upper",
    cmap="PuRd_r",
)
plt.colorbar(p)

policy = (π(200)).reshape(2, 4, H, W)[0]
policy = policy.argmax(axis=0)

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
posterior = invplan()
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