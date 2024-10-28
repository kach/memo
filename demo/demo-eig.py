"""
Question-asking based on expected information gain (EIG)

Bob rolls a red die and a blue die.
Alice gets to ask one yes-no question about the sum.
What is the most informative question she could ask,
in order to learn the most about the two die rolls?

We'll compute the EIG of various questions...
"""

from memo import memo
import jax
import jax.numpy as np

is_prime  = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
is_square = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
is_pow_2  = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
Qs = [
    lambda n: n == 7,
    lambda n: n == 12,
    lambda n: n > 10,
    lambda n: n > 8,
    lambda n: n > 6,
    lambda n: n > 5,
    lambda n: n % 2 == 0,  # even??
    lambda n: n % 2 == 1,  # odd??
    lambda n: n % 3 == 0,
    lambda n: n % 4 == 0,
    lambda n: n % 5 == 0,
    lambda n: is_prime[n],
    lambda n: is_square[n],
    lambda n: is_pow_2[n],
]

N = np.arange(1, 6 + 1)  # single die's outcomes
Q = np.arange(len(Qs))   # questions
A = np.array([0, 1])     # answers (yes/no)

@jax.jit
def respond(q, a, n):
    return np.array([q_(n) for q_ in Qs])[q] == a

@memo
def eig[q: Q]():
    alice: knows(q)
    alice: thinks[
        # bob rolls dice...
        bob: chooses(n_red in N, wpp=1),
        bob: chooses(n_blu in N, wpp=1),

        # bob answers question...
        bob: knows(q),
        bob: chooses(a in A, wpp=respond(q, a, n_red + n_blu))
    ]
    return alice[ imagine[
        # if I were to get the answer...
        future_alice: observes [bob.a] is bob.a,
        # EIG = entropy minus conditional entropy
        H[bob.n_red, bob.n_blu] - E[future_alice[ H[bob.n_red, bob.n_blu] ]]
    ] ]

z = eig()
## print questions and EIGs in sorted order
print('EIG     Question')
print('---     ---')
import inspect
q_names = [inspect.getsource(q_).strip()[10:-1] for q_ in Qs]
for eig_, q_ in reversed(sorted(list(zip(z, q_names)))):
    print(f'{eig_:0.5f}', q_)