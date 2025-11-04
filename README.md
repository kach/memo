![memo's logo](https://github.com/kach/memo/raw/main/assets/logo.png)

memo is a probabilistic programming language for expressing computational cognitive models involving **recursive reasoning about reasoning**. memo inherits from the tradition of WebPPL-based Bayesian modeling (see [probmods](http://probmods.org/), [agentmodels](https://agentmodels.org/), and [problang](https://www.problang.org/)), but aims to make models **easier to write and run** by taking advantage of modern programming language techniques and hardware capabilities (including GPUs!). As a result, models are often significantly simpler to express (we've seen codebases shrink by a **factor of 3x or more**), and dramatically faster to execute and fit to data (we've seen **speedups of 3,000x or more**). In idiomatic memo, a POMDP solver is 15 lines of code, and is just as fast as a hand-optimized solver written in 200 lines of code.

memo stands for: mental modeling, memoized matrix operations, model-expressed-model-optimized, and metacognitive memos.

## Installing memo

1. memo is based on Python. Before installing memo, make sure you have Python 3.12 or higher installed. You can check this by running `python --version`.
2. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) by running `python -m venv venv`, and then `. venv/bin/activate`. (Your command prompt should change to be prefixed by "venv", indicating that you have successfully created the virtual environment.)
3. Next, install [JAX](https://github.com/google/jax), a Python module that memo uses to produce fast, differentiable, GPU-enabled code. If you don't have a GPU, then running `pip install jax` should be enough. Otherwise, please consult the JAX website for GPU-specific installation instructions. You can check if JAX is installed by running `import jax` in Python.
4. Finally, install memo by running `pip install memo-lang`. You can check if memo is installed by running `from memo import memo` in Python. (Important: Make sure to install `memo-lang`, not `memo`! The latter is a different package, unrelated to this project.)

## Learning memo

There are many resources available for learning memo.
1. The [Memonomicon](./demo/Memonomicon.ipynb) gives a brief tour of the language, and an example of how to build a model and fit it to data by parallel grid search and/or gradient descent.
2. You can watch a [video tutorial](https://www.dropbox.com/scl/fi/c3jjup1lheowfppbz41zr/memo-live-tutorial.mp4?rlkey=ce7reeadff2nh2ktqh3tubbik&st=lai8yx1h&dl=0) that covers similar material. You can also check out a [talk given at LAFI '25](https://www.youtube.com/live/RLEFVgx2UWk?t=12500s) that offers a bigger-picture overview of memo.
3. The [Handbook](./Handbook.pdf) is a complete reference for memo's syntactic constructs.
4. This repository includes over a dozen classic examples of recursive reasoning models implemented in memo, which you can find in the [demo directory](./demo/).
6. I am happy to give a short hands-on tutorial on memo in your lab. Just email me to ask!

You may also be looking for general resources on the theory behind memo modeling.
1. For background on the theory of decision making under uncertainty, e.g. MDPs and POMDPs, we recommending consulting _Decision Making Under Uncertainty_ as a reference. You can read the entire book for free online [here](https://algorithmsbook.com/decisionmaking/).
2. For background on Bayesian models of theory of mind, we recommend consulting chapter 14 of _Bayesian Models of Cognition_ as a reference. You can read the published version [here](https://mitpress.ublish.com/ebook/bayesian-models-of-cognition-reverse-engineering-the-mind-preview/12799/341) and a PDF preprint [here](https://www.tomerullman.org/papers/BBB_chapter14.pdf).
3. Dae Houlihan (Dartmouth University) taught a winter '25 [course](https://comosoco.daeh.info) on computational models of social cognition using memo.
4. Robert Hawkins (Stanford University) is teaching a summer '25 seminar on pragmatics using memo. Here is the work-in-progress [textbook](https://hawkrobe.github.io/probLang-memo/), a version of problang that has been adapted to use memo.

## The memo community

Here are some ways to engage with the memo community.

1. For updates on memo's development, we _strongly_ encourage you to subscribe to our low-traffic monthly announcements mailing list [here](https://lists.csail.mit.edu/mailman/listinfo/memo-lang).
2. To ask questions about memo, and to get help from other memo users, use [Github Discussions](https://github.com/kach/memo/discussions). Note that you will need a Github account to participate.
3. For live support, we host memOH (**memo office hours**) every Tuesday at 2pm ET. Email Kartik for the zoom link!

## The memo on memo

The authoritative paper on memo's design and implementation is available [here](https://dl.acm.org/doi/10.1145/3763078). If you use memo in your work, you are invited to cite this paper via this BibTeX citation:

```bibtex
@article{chandra2025memo,
author = {Chandra, Kartik and Chen, Tony and Tenenbaum, Joshua B. and Ragan-Kelley, Jonathan},
title = {A Domain-Specific Probabilistic Programming Language for Reasoning about Reasoning (Or: A Memo on memo)},
year = {2025},
issue_date = {October 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {9},
number = {OOPSLA2},
url = {https://doi.org/10.1145/3763078},
doi = {10.1145/3763078},
abstract = {The human ability to think about thinking ("theory of mind") is a fundamental object of study in many disciplines. In recent decades, researchers across these disciplines have converged on a rich computational paradigm for modeling theory of mind, grounded in recursive probabilistic reasoning. However, practitioners often find programming in this paradigm challenging: first, because thinking-about-thinking is confusing for programmers, and second, because models are slow to run. This paper presents memo, a new domain-specific probabilistic programming language that overcomes these challenges: first, by providing specialized syntax and semantics for theory of mind, and second, by taking a unique approach to inference that scales well on modern hardware via array programming. memo enables practitioners to write dramatically faster models with much less code, and has already been adopted by several research groups.},
journal = {Proc. ACM Program. Lang.},
month = oct,
articleno = {300},
numpages = {31},
keywords = {probabilistic programming}
}
```

or this reference:

> Kartik Chandra, Tony Chen, Joshua B. Tenenbaum, and Jonathan Ragan-Kelley. 2025. A Domain-Specific Probabilistic Programming Language for Reasoning about Reasoning (Or: A Memo on memo). Proc. ACM Program. Lang. 9, OOPSLA2, Article 300 (October 2025), 31 pages. https://doi.org/10.1145/3763078

I would love to hear about any research using memo. Please don't hesitate to share your work with me!

## As seen on…

**Papers/projects using memo**  
- People use theory of mind to craft lies exploiting audience desires (Sterling, Berke, Chandra, & Jara-Ettinger, CogSci '25, SPP '25)
- Solving strategic social coordination via Bayesian learning (Lamba, Houlihan, & Saxe, CogSci '25)
- Empathy in Explanation (Collins, Chandra, Weller, Ragan-Kelley, & Tenenbaum, CogSci '25, SPP '25)
- Minding the Politeness Gap in Cross-cultural Communication (Machino, Siegel, & Hawkins, CogSci '25)
- Preparing a learner for an independent future (Sundar, Chandra, & Kleiman-Weiner, CogSci '25)
- A Computational Theory of Dignity (Chandra, Tenenbaum, & Saxe, SPP '25)
- Theories of Mind as Languages of Thought for Thought about Thought (Chandra, Ragan-Kelley & Tenenbaum, CogSci '25)

_(Email me to have your work listed here!)_

**Talks on memo**  
- New England PL/Systems Summit (NEPLS) (2024)
- Languages For Inference (LAFI @ POPL) (2025)

**Courses using memo**  
- Semester-long course at Dartmouth College (2025)
- Summer seminar at Stanford University (2025)
- Tutorial at CogSci conference in San Francisco (2025)
- Tutorial at COSMOS summer school in Tokyo (2025)


## FAQ

<details><summary>How do I capitalize memo? Is it Memo? MEMO? MeMo?</summary>

"memo," all-lowercase.
</details>

<details><summary>When should I use memo rather than Gen or WebPPL?</summary>

memo's core competence is fast tabular/enumerative inference on models with recursive reasoning about reasoning. That covers a wide range of common models: from RSA, to POMDP planning (value iteration = tabular operations), to inverse planning. In general, if you are making nested queries, we recommend using memo.

There are however two particular cases where you may prefer another PPL:
1. If you are interested specifically in modeling a sophisticated inference scheme, such as MCMC, particle filters, or variational inference, then we recommend trying Gen. _(But make sure you really need those tools — the fast enumerative inference provided by memo is often sufficient for many common kinds of models!)_
2. If you are performing inference over an unbounded domain of hypotheses with varied structure, such as programs generated by a grammar, then we recommend trying Gen or WebPPL because memo's tabular enumerative inference can only handle probability distributions with finite support. _(But if you are okay with inference over a "truncated" domain, e.g. the top 1,000,000 shortest programs, then memo can do that! Similarly, memo can handle continuous domains by discretizing finely.)_

The aforementioned cases are explicitly out of scope for memo. By specializing memo to a particular commonly-used class of models and inference strategies, we can produce extremely fast code that is difficult for general-purpose PPLs to produce.
</details>

<details><summary>Okay, so how does memo produce such fast code?</summary>

memo compiles enumerative inference to JAX array programs, which can be run extremely fast. The reason for this is that array programs are inherently very easy to execute in parallel (by performing operations on each element of the array independently). Modern hardware is particularly good at parallel processing.
</details>

<details><summary>What exactly is JAX?</summary>

[JAX](https://github.com/google/jax) is a library developed by Google that takes Python array programs (similar to NumPy) and compiles them to very fast code that can run on CPUs and GPUs, taking advantage of modern hardware functionality. JAX supports a lot of Google's deep learning, because neural networks involve a lot of array operations. memo compiles your probabilistic models into JAX array programs, and JAX further compiles those array programs into machine code.

Note that JAX has some unintuitive behaviors. We recommend reading [this guide](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) to get a sense of its "sharp edges."
</details>

<details><summary>I'm used to thinking of probabilistic programming as sampling execution traces from generative models. Should I think of memo the same way?</summary>

One way to think about memo is that it simulates _all_ possible traces at the same time. There is no need for sampling, because we always have access to the full posterior distribution.
</details>

<details><summary>Is memo a research prototype, or a mature software product? Should I invest in learning memo?</summary>

memo is stable software that is being used by many labs around the world, and has led to several published papers. memo will be supported for a long time to come, and you should feel confident in using memo for your own projects.
</details>

---

<details>
<summary>I installed memo but importing memo gives an error.</summary>

Did you accidentally pip-install the (unrelated) package [memo](https://pypi.org/project/memo/) instead of [memo-lang](https://pypi.org/project/memo-lang/)?
</details>

<details>
<summary>I installed memo on my Mac, but running models gives a weird JAX error about "AVX".</summary>

The common cause of this is that you have a modern Mac (with an ARM processor), but an old version of Python (compiled for x86). We recommend the following installation strategy on ARM-based Macs:
1. Do not use conda.
2. Install Homebrew. Make sure you have the ARM version of brew: `brew --prefix` should be `/opt/homebrew`, and `brew config` should say `Rosetta 2: false`. If this is not the case, you have the x86 version of brew, which you should uninstall.
3. Install Python via `brew install python3`. Ensure that `python3 --version` works as expected, and that `which python3` points to something in `/opt/homebrew/bin/`.
4. In your project directory, create a virtual environment via `python3 -m venv venv`.
5. Activate the virtual environment via `. venv/bin/activate`. Your prompt should now begin with `(venv)`.
6. Install memo via `pip install memo-lang`.
</details>

<details><summary>How do I use memo with a GPU?</summary>

Assuming you have [installed JAX with GPU support](https://jax.readthedocs.io/en/latest/installation.html), all you have to do is plug in your GPU!
</details>

<details><summary>Can I run memo on Apple's "metal" platform?</summary>

Yes! However, JAX on Metal is not very well-supported by Apple, so we cannot guarantee that everything will work perfectly. See this issue for details: https://github.com/kach/memo/issues/66
</details>

---

<details><summary>VS Code underlines all my memo code in red. It's a bloodbath out there!</summary>
This is because your editor is trying to interpret your memo code as regular python
code while checking it for problems. To disable this behavior, you can annotate your 
memo models with `@no_type_check` and (if you are using `ruff`) `# ruff: noqa`.
For example, if you have memo code that looks like:

```python
@memo
def model[u: U, r: R]():
   # ...
```

You can disable the checks by adding the following to the top of your file:

```python
from typing import no_type_check
```

Then, annotate your model function like so:

```python
# ruff: noqa
@no_type_check
@memo
def model[u: U, r: R]():
   # ...
```
</details>

<details><summary>Sometimes my model returns 0 in unexpected places, often at the edges/extreme values of distributions.</summary>

This can be caused by numerical stability errors. For example, if a `wpp=` expression gets too big, then it might "overflow" to infinity, and wreak havoc downstream. Similarly, if a `wpp=` expression returns 0 for all possible choices, then normalizing that distribution causes a division-by-zero error that wreaks havoc downstream. This havoc usually comes in the form of calculations being unexpectedly clipped to 0.

So, if you are seeing unexpected 0s, we recommend inspecting your `wpp=` expressions to see whether they could be returning very large or very small values. Often, you can fix the problem by adding a little epsilon value (e.g. `wpp=f(x) + 1e-5`).
</details>

<details><summary>Some of my output array's dimensions are unexpectedly of size 1.</summary>

memo attempts to minimize redundant computation. If the output of your model doesn't depend on an input axis, then instead of repeating the computation along that axis, memo will set that axis to size 1. The idea is that [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) will keep the array compatible with downstream computations.

As an example, consider the following models:

```python
X = np.arange(10)

@memo
def f[a: X, b: X]():
    return a
f().shape  # (10, 1) because output is independent of b

@memo
def f[a: X, b: X]():
    return b
f().shape  # (1, 10) because output is independent of a

@memo
def f[a: X, b: X]():
    return a + b
f().shape  # (10, 10) because output depends on a and b

@memo
def f[a: X, b: X]():
    return 999
f().shape  # (1, 1) because output depends on neither a nor b
```
</details>

<details><summary>How can I visualize what's going on with my model in "comic-book" format?</summary>

Use `@memo(save_comic="filename")` instead of just `@memo`. memo will produce a [Graphviz](https://graphviz.org/) `filename.dot` file that you can [render online](https://dreampuf.github.io/GraphvizOnline/). If you have Graphviz installed, memo will also automatically render a `filename.png` file for you.

</details>

<details><summary>How can I get model outputs in pandas/xarray format?</summary>
Pass in the <code>return_pandas=True</code> or <code>return_xarray=True</code> keyword arguments to your model. Your model will then return a tuple: the first argument will be the raw array, and the second argument will have a <code>.pandas</code> or <code>.xarray</code> property, respectively.
</details>

<details><summary>How can I sample from common distributions, e.g. normal or Beta?</summary>
You can import and call the pdfs and pmfs defined in <code>jax.scipy.stats</code> (see https://docs.jax.dev/en/latest/jax.scipy.html). For example, to choose from a Bernoulli distribution with p=0.9, you can write:

```python
from jax.scipy.stats.bernoulli import pmf as ber_pmf

...

alice: chooses(x in Bool, wpp=ber_pmf(x, 0.9))
```
</details>