from memo import memo
import jax
import jax.numpy as np

Bar = np.arange(2)
@jax.jit
def prior(b): return np.array([0.55, 0.45])[b]

@memo
def alice[b: Bar](depth):
    alice: thinks[ bob: chooses(b in Bar, wpp=bob[b](depth - 1)) ]
    alice: chooses(b in Bar, wpp=prior(b) * Pr[b == bob.b])
    return Pr[alice.b == b]

@memo
def bob[b: Bar](depth):
    bob: thinks[ alice: chooses(b in Bar, wpp=alice[b](depth) if depth > 0 else 1) ]
    bob: chooses(b in Bar, wpp=prior(b) * Pr[b == alice.b])
    return Pr[bob.b == b]

for i in range(1, 10):
    print(f'alice({i}) =', alice(i))
    print(f'bob({i}) =', bob(i))


@memo
def alice_confidence(depth):
    alice: thinks[ bob: chooses(b in Bar, wpp=bob[b](depth - 1)) ]
    alice: chooses(b in Bar, wpp=prior(b) * Pr[b == bob.b])
    return E[alice[Pr[b == bob.b]]]

@memo
def obs_confidence(depth):
    alice: chooses(b in Bar, wpp=alice[b](depth))
    bob: chooses(b in Bar, wpp=bob[b](depth))
    return Pr[alice.b == bob.b]

for i in range(1, 10):
    print(alice_confidence(i), obs_confidence(i))