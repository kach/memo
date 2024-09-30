from itertools import combinations
import math

import jax
import jax.numpy as jnp


def belief_discretize(S, n=3):
    # takes in a specification of a pomdp and returns:
    # - a discretized belief space B
    # the belief transition function p(b' | b, a, o)
    # the belief state mapping function giving the probability of a state under a belief

    # beliefs = jnp.zeros((math.comb(n + len(S) - 1, len(S) - 1), len(S)))
    _beliefs = []
    for i, bars in enumerate(combinations(range(n + len(S) - 1), len(S) - 1)):
        # the gaps between the bars are the beliefs
        belief = []
        prev = -1
        for b in bars:
            belief.append(b - prev - 1)
            prev = b
        belief.append(n - prev - 1)
        z = sum(belief)
        belief = [b / z for b in belief]
        _beliefs.append(belief)

    beliefs = jnp.array(_beliefs)
    print(beliefs)

    @jax.jit
    def get_belief(b, s):
        return beliefs[b][s]

    @jax.jit
    def belief_filter(b_, a, o):
        pass

    return jnp.arange(math.comb(n + len(S) - 1, len(S) - 1)), get_belief, belief_filter


if __name__ == "__main__":
    b, gb, _ = belief_discretize([0, 1, 2, 3], 5)