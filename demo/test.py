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
def imagine_unknown_err():
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