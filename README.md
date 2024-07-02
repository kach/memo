# memo

memo is a new probabilistic programming language for expressing computational cognitive models involving **sophisticated recursive reasoning**, and for performing **fast enumerative inference** on such models. memo inherits from the tradition of WebPPL-based Bayesian modeling (see [probmods](http://probmods.org/), [agentmodels](https://agentmodels.org/), and [problang](https://www.problang.org/)), but aims to make models **easier to write and run** by taking advantage of modern programming language techniques and hardware capabilities.

## Installing memo

1. memo is based on Python. Before installing memo, make sure you have Python 3.12 or higher installed. You can check this by running `python --version`. (As of writing, Python 3.12 is the latest version of Python, and memo depends on several of its powerful new features.)
2. Next, install [JAX](https://github.com/google/jax), a Python module that memo uses to produce fast, differentiable, GPU-enabled code. If you don't have a GPU, then running `pip install jax` should be enough. Otherwise, please consult the JAX website for installation instructions. You can check if JAX is installed by running `import jax` in Python.
3. Finally, install memo by running `pip install memo-lang`. You can check if memo is installed by running `from memo import memo` in Python.
