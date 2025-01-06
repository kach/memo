from memo import memo, memo_test, make_module

mod = make_module('test_suite')
mod.install('''
import jax
import jax.numpy as np
X = np.arange(3)
N = 5

@jax.jit
def f(n):
    return n + 1
''')

@memo(install_module=mod.install)
def test_[x: X](t):
    return x

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
def returns_nonscalar0():
    return np.array([0, 1])
@jax.jit
def returns_nonscalar1(x):
    return np.array([0, 1])
''')

@memo_test(mod, expect='ce')
def ffi_scalar0():
    return returns_nonscalar0()

@memo_test(mod, expect='ce')
def ffi_scalar1():
    return returns_nonscalar1(1.0)