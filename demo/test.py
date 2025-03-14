from memo import memo, memo_test, make_module

mod = make_module('test_suite')
mod.install('''
import jax
import jax.numpy as np
X = np.arange(3)
Y = np.arange(3)
N = 5

@jax.jit
def f(n):
    return n + 1

Z = np.arange(1000)
R = np.linspace(-10, 10, 1000)

@jax.jit
def normpdf(x, mu, sigma):
    return jax.scipy.stats.norm.pdf(x, mu, sigma)
''')

@memo(install_module=mod.install)
def test_[x: X](t):
    return x

@memo_test(mod, expect='ce')
def chooses_multiple():
    bob: chooses(x in X, wpp=1)
    bob: chooses(x in X, wpp=1)
    return 1

@memo_test(mod)
def observes_call[x: X]():
    a: thinks[ b: chooses(x in X, wpp=1) ]
    a: observes [b.x] is x
    return a[ b[ test_[x](0) ] ]

@memo_test(mod)
def observes_other():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    charlie: chooses(x in X, wpp=1)
    alice: observes [bob.x] is charlie.x
    return E[alice[bob.x]]

@memo_test(mod)
def observes_other_imagine():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    charlie: chooses(x in X, wpp=1)
    alice: observes [bob.x] is charlie.x
    return E[alice[
        imagine[
            env: knows(bob.x),
            bob.x + 1
        ]
    ]]

@memo_test(mod)
def imagine_ok():
    return alice[
        imagine[
            bob: chooses(y in X, wpp=1),
            E[bob.y]
        ]
    ]

@memo_test(mod)
def inline():
    return {N}

@memo_test(mod)
def inline_call():
    return f({N})

@memo_test(mod)
def inline_memo[x: X]():
    return test_[x]({N})

@memo_test(mod)
def memo_call_ellipsis(t=0):
    alice: chooses(x in X, wpp=test_[x](...))
    return E[alice.x]

@memo_test(mod)
def imagine_ok():
    return alice[
        imagine[
            bob: chooses(y in X, wpp=1),
            E[bob.y]
        ]
    ]

@memo_test(mod, expect='ce')
def imagine_unknown_err():
    return alice[
        imagine[
            bob: chooses(y in X, wpp=1),
            bob.y
        ]
    ]

@memo_test(mod, expect='ce')
def imagine_unknown_err_expect():
    return alice[
        E[imagine[
            bob: chooses(y in X, wpp=1),
            bob.y
        ]]
    ]

@memo_test(mod, expect='ce')
def imagine_unknown_err_expect_future():
    return alice[
        E[imagine[
            future_alice: chooses(y in X, wpp=1),
            future_alice.y
        ]]
    ]

@memo_test(mod)
def imagine_knows():
    alice: chooses(x in X, wpp=1)
    return E[alice[
        imagine[
            world: knows(x),
            world.x
        ]
    ]]

@memo_test(mod)
def imagine_knows_other[z: X]():
    alice: chooses(x in X, wpp=1)
    alice: thinks[ bob: chooses(z in X, wpp=1) ]
    alice: observes [bob.z] is z
    return alice[
        imagine[
            world: knows(bob.z),
            world[bob.z]
        ]
    ]

@memo_test(mod)
def imagine_future_stress():
    alice: chooses(x in X, wpp=1)
    alice: thinks[ bob: chooses(z in X, wpp=1) ]
    return E[alice[
        imagine[
            world: knows(x, bob.z),
            world: chooses(z in X, wpp=1),
            future_alice: chooses(y in X, wpp=x + y),
            future_alice: thinks[ world: chooses(z in X, wpp=1) ],
            future_alice: observes [world.z] is world.z,
            E[future_alice.y + future_alice[world.z + y] + world.z + world[bob.z]]
        ]
    ]]

@memo_test(mod)
def imagine_toplevel():
    return imagine[
        alice: chooses(x in X, wpp=1),
        E[alice.x]
    ]

mod.install('''
@jax.jit
def returns_scalar(x):
    return np.cos(x) + np.array([0, 1, 2])[x]
@jax.jit
def returns_scalar_no_arg():
    return np.cos(3.14)
@jax.jit
def returns_nonscalar0():
    return np.array([0, 1])
@jax.jit
def returns_nonscalar1(x):
    return np.array([0, 1])
''')

@memo_test(mod)
def ffi_ok():
    alice: chooses(x in X, wpp=1)
    return E[returns_scalar(alice.x)] + 12

@memo_test(mod)
def ffi_ok_no_arg():
    return returns_scalar_no_arg() + 15

@memo_test(mod, expect='ce')
def ffi_scalar0():
    return returns_nonscalar0()

@memo_test(mod, expect='ce')
def ffi_scalar1():
    return returns_nonscalar1(1.0)

@memo_test(mod)
def observes_const():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes_that [bob.x == 0]
    return alice[E[bob.x]]

@memo_test(mod)
def observes_const_float():
    alice: thinks[ bob: chooses(x in X, wpp=1) ]
    alice: observes_event(wpp=bob.x / 3)
    return alice[E[bob.x]]

@memo_test(mod)
def observes_const_void_choose():
    alice: chooses(x in X, wpp=1)
    alice: observes_event(wpp=x / 3)
    return E[alice.x]

@memo_test(mod)
def observes_const_void():
    alice: observes_event(wpp=3.14)
    return alice[2]

@memo_test(mod)
def pr_joint():
    alice: chooses(x in X, wpp=1)
    alice: chooses(y in X, wpp=1)
    return Pr[alice.x == 0, alice.y == 0]

@memo_test(mod)
def choose_many():
    alice: chooses(x in X, y in Y, wpp=1)
    return Pr[alice.x == 0, alice.y == 0]

@memo_test(mod)
def choose_max():
    alice: chooses(x in X, y in Y, to_maximize=x + y)
    return Pr[alice.x == 0, alice.y == 0]

@memo_test(mod)
def choose_min():
    alice: chooses(x in X, y in Y, to_minimize=x + y)
    return Pr[alice.x == 0, alice.y == 0]

@memo_test(mod, expect='ce')
def choose_err():
    alice: chooses(x in X, y in Y, to_eat=x + y)
    return Pr[alice.x == 0, alice.y == 0]

@memo_test(mod)  # crashes without post optim
def post_optim[z1: Z, z2: Z]():
    alice: chooses(z1 in Z, wpp=1)
    alice: chooses(z2 in Z, wpp=1)
    return Pr[z1 == alice.z1, alice.z2 == z2]

@memo_test(mod)  # https://stackoverflow.com/a/22348885
def post_optim_distinctness[z: Z]():
    alice: chooses(z1 in Z, wpp=1)
    alice: chooses(z2 in Z, wpp=1)
    return Pr[z == alice.z1, z == alice.z2]

from math import log
@memo_test(mod, item=log(2/1) + 1/8 - 1/2)
def kl():
    alice: chooses(p in R, wpp=normpdf(p, 0, 1))
    alice: chooses(q in R, wpp=normpdf(q, 0, 2))
    return KL[alice.p | alice.q]

@memo_test(mod, expect='ce')
def kl_fail_unknown():
    alice: chooses(p in R, wpp=normpdf(p, 0, 1))
    alice: chooses(q in R, wpp=normpdf(q, 0, 2))
    return KL[alice.r | alice.q]

@memo_test(mod, expect='ce')
def kl_fail_known[r: R]():
    alice: knows(r)
    alice: chooses(q in R, wpp=normpdf(q, 0, 2))
    return KL[alice.r | alice.q]

@memo_test(mod, expect='ce')
def kl_fail_dom():
    alice: chooses(p in Z, wpp=normpdf(p, 0, 1))
    alice: chooses(q in R, wpp=normpdf(q, 0, 2))
    return KL[alice.p | alice.q]

@memo_test(mod)
def kl_victor[x: X]():
    bob: thinks[
        alice: given(x in X, wpp=1),
        env: chooses(x in X, wpp=1)
    ]
    return bob[KL[alice.x | env.x]]
