![memo's logo](./assets/memo.png)

memo is a new probabilistic programming language for expressing computational cognitive models involving **sophisticated recursive reasoning**, and for performing **fast enumerative inference** on such models. memo inherits from the tradition of WebPPL-based Bayesian modeling (see [probmods](http://probmods.org/), [agentmodels](https://agentmodels.org/), and [problang](https://www.problang.org/)), but aims to make models **easier to write and run** by taking advantage of modern programming language techniques and hardware capabilities.

memo stands for: mental modeling, memoized matrix operations, model-expressed-model-optimized, and metacognitive memos.

## Installing memo

1. memo is based on Python. Before installing memo, make sure you have Python 3.12 or higher installed. You can check this by running `python --version`. (As of writing, Python 3.12 is the latest version of Python, and memo depends on several of its powerful new features.)
2. Next, install [JAX](https://github.com/google/jax), a Python module that memo uses to produce fast, differentiable, GPU-enabled code. If you don't have a GPU, then running `pip install jax` should be enough. Otherwise, please consult the JAX website for installation instructions. You can check if JAX is installed by running `import jax` in Python.
3. Finally, install memo by running `pip install memo-lang`. You can check if memo is installed by running `from memo import memo` in Python.

## Getting started

Once you have installed memo, take a look at the [Memonomicon](./Memonomicon.ipynb) for a tour of the language!

This repository also includes several classical examples of recursive reasoning models implemented in memo:
- [demo-scalar.py](./demo-scalar.py) shows scalar implicature, analogous to the example on the front page of WebPPL.org.
- [demo-rsa.py](./demo-rsa.py) shows Rational Speech Acts with the recursion explicitly unrolled.
- [demo-rsa-recursive.py](./demo-rsa-recursive.py) shows Rational Speech Acts with recursive calls.
- [demo-grid.py](./demo-grid.py) shows planning and inverse planning in a grid-world MDP.
- [demo-pomdp.py](./demo-pomdp.py) shows belief-space planning in a POMDP.