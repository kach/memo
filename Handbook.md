# The memo Handbook (v1.2.9)

This handbook provides a complete reference for the memo programming language syntax and semantics.

## How This Guide Is Organized

The handbook is divided into four main sections:

1. **[Anatomy of a memo](#anatomy-of-a-memo)** — The overall structure of a memo model: how to declare axes, parameters, agents, and return values.

2. **[Statements](#statements)** — The building blocks of agent behavior. Statements define the choices agents make (`chooses`), their beliefs about other agents (`thinks`), their observations (`observes`), and more.

3. **[Expressions](#expressions)** — How to compute values within a memo. Expressions include literals, operators, probabilistic queries (`E`, `Pr`, `Var`), information-theoretic measures (`H`, `KL`), and hypotheticals (`imagine`).

4. **[Running and Configuring Memos](#running-and-configuring-memos)** — How to execute memo models, configure options like caching and tracing, and use automatic differentiation for model fitting.

---

## Anatomy of a memo

A memo model is a decorated Python function that defines agents, their choices, and beliefs. Here is the general structure:

```python
@memo
def model[x: X, y: Y](a, b, c=3.14):
    ''' This is a model of ____! '''
    alice: …
    bob: …
    return …
```

The components of a memo model are:

| Component | Description |
|-----------|-------------|
| `model` | The name of the model function |
| `[x: X, y: Y]` | The axes of the array to compute, where each axis variable iterates over a domain |
| `(a, b, c=3.14)` | Scalar free parameters, with optional default values |
| `''' ... '''` | An optional docstring for documentation |
| `alice: …` / `bob: …` | A sequence of **statements** that define agent behavior |
| `return …` | An **expression** whose value is computed for each cell in the returned array |

**Advanced usage:** You can also pass arrays as parameters. Array-valued parameters must be declared with an ellipsis annotation, like `a: ...`.

**Multiple return values:** You can have models with multiple return values by writing multiple `return` statements.

---

# Statements

Statements define agent behavior: the choices they make, what they think about other agents, and what they observe.

## `chooses`

The `chooses` statement declares that an agent makes a probabilistic choice from a domain.

```python
bob: chooses(a in Actions, wpp=exp(β*utility(a)))
```

This statement has the following components:

- **Agent making the choice:** `bob`
- **Name of the choice variable:** `a`
- **Domain of the choice:** `Actions`, which should be a Python list, enum, or JAX array
- **Probability weighting:** The `wpp=` parameter specifies "with probability proportional to" the given expression
  - For softmax decision-making, use `wpp=exp(…)`
  - For a uniform distribution, use `wpp=1`

You can also use [JAX's built-in probability distributions](https://docs.jax.dev/en/latest/jax.scipy.html). For example, to sample from a Bernoulli distribution with p=0.9:

```python
from jax.scipy.stats.bernoulli import pmf as ber_pmf
...
alice: chooses(x in Bool, wpp=ber_pmf(x, 0.9))
```

### Making multiple choices simultaneously

An agent can make multiple choices at once by listing them in the same statement:

```python
bob: chooses(x in X, y in Y, wpp=joint(x, y))
```

### Deterministic choice with `to_maximize` and `to_minimize`

For argmax behavior (choosing the option that maximizes a utility), use `to_maximize`:

```python
bob: chooses(a in Actions, to_maximize=utility(a))
```

Similarly, for argmin behavior, use `to_minimize`.

### Aliases for `chooses`

Several aliases exist for `chooses` that have identical behavior but don't imply agency or goal-orientation. These can make your models easier to read:

```python
bob: given(r in Roles, wpp=1)
bob: draws(r in Roles, wpp=1)
bob: assigned(r in Roles, wpp=1)
bob: guesses(r in Roles, wpp=1)
```

## `thinks`

The `thinks` statement defines what an agent believes about other agents. This creates nested reasoning—an agent reasoning about the reasoning of others.

```python
bob: thinks[
    alice: chooses(...),
    charlie: chooses(...),
    ...
]
```

- **Agent doing the thinking:** `bob`
- **Contents of the agent's mental model:** The statements inside the brackets define what `bob` believes about `alice`, `charlie`, and any other agents

Note that statements inside `thinks` are separated by commas, not newlines.

## `observes`

The `observes` statement updates an agent's beliefs based on an observation.

```python
bob: observes [alice.x] is y
```

- **Agent making the observation:** `bob`
- **Choice being observed:** `[alice.x]` (the square brackets indicate that this is another agent's choice)
- **Observed value:** `y`—the name of a choice (either the observer's own choice or another agent's choice)

This mechanism can be used to create false beliefs by setting the observed value to something different from the actual value.

The observed value can also be another agent's choice:

```python
bob: observes [alice.x] is charlie.y
```

### Advanced observation constructs

For conditioning on boolean expressions, use `observes_that`:

```python
bob: observes_that [coin.bias > 0.5]
```

This is analogous to `condition(…)` in WebPPL. The agent observes that the boolean expression is true, which conditions their beliefs accordingly.

For soft evidence (factor statements), use `observes_event`:

```python
bob: observes_event(wpp=coin.bias)
```

This is analogous to `factor(log(…))` in WebPPL. The observation has probability proportional to the given expression.

*Most models should only need the basic `observes` construct. These advanced forms are rarely necessary.*

## `knows`

The `knows` statement is a convenient shorthand for pushing variables into an agent's mental model.

```python
bob: knows(x, alice.y)
```

- **Agent who knows:** `bob`
- **Values that are known:** `x` and `alice.y`

This is roughly equivalent to the following expanded form:

```python
bob: thinks[ alice: chooses(y in Y, wpp=...) ]
bob: observes [alice.y] is alice.y
```

The `knows` statement is useful when you want an agent to have correct beliefs about certain variables without writing out the full `thinks` and `observes` pattern.

## `snapshots_self_as`

The `snapshots_self_as` statement allows agents to remember "snapshots" of their past selves. This is useful for modeling counterfactuals and hypotheticals, especially in combination with `imagine` expressions.

```python
alice: snapshots_self_as(past_alice, …)
```

- **Agent taking the snapshot:** `alice`
- **Alias(es) for the snapshot:** `past_alice` (and any additional aliases separated by commas)

Example usage:

```python
alice: observes [bob.x] is x
return alice[ past_alice[ E[bob.x] ] ]
#             ↑ The past_alice snapshot is not affected by the "observe" statement
```

## `wants` and `EU`

The `wants` statement defines a goal or utility function that depends on future states of the world. You can reference choices that haven't been made yet. The expected value of a goal is evaluated using the `EU` expression.

```python
alice: wants(win=score > bob.score)
alice: chooses(a in A, to_maximize=EU[win])
alice: given(score in N, wpp=…)
```

- **Agent with the goal:** `alice`
- **Goal name:** `win`
- **Goal expression:** `score > bob.score`, which will be evaluated once `score` is defined
- **Evaluating expected utility:** `EU[win]` computes the expected value of the goal

Even though `score` is defined after the `wants` statement, this code is valid. Conceptually, the `EU` expression is evaluated at the point where all referenced variables are defined.

## `INSPECT`

The `INSPECT` statement prints the current state of an agent at compile time. This is helpful for debugging complex models.

```python
alice: INSPECT()
```

---

# Expressions

Expressions compute values within a memo model. They can reference agent choices, apply operators, and use probabilistic reasoning.

## Literals and Parameters

Numeric literals are floating-point numbers:

```python
3.14
```

You can also reference declared free parameters by name:

```python
a, b, c, …
```

## Operators

memo supports a variety of Python operators and built-in functions.

### Binary operators

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `**` | Exponentiation |
| `%` | Modulo |

### Comparison operators

| Operator | Description |
|----------|-------------|
| `==` | Equal to |
| `!=` | Not equal to |
| `<` | Less than |
| `<=` | Less than or equal to |
| `>` | Greater than |
| `>=` | Greater than or equal to |

### Logical operators

| Operator | Description |
|----------|-------------|
| `and` | Logical AND |
| `or` | Logical OR |
| `^` | Logical XOR |

### Unary operators

| Operator | Description |
|----------|-------------|
| `-x` | Negation |
| `+x` | Unary plus |
| `~x` | Bitwise inversion |

### Built-in functions

| Function | Description |
|----------|-------------|
| `exp(x)` | Exponential (e^x) |
| `log(x)` | Natural logarithm |
| `abs(x)` | Absolute value |

### Conditional expressions

memo supports Python's ternary conditional expression:

```python
a if condition else b
```

### Calling custom JAX functions

You can also call any JAX-compatible function decorated with `@jax.jit`:

```python
@jax.jit
def f(x):
    return np.cos(x)
```

This enables integration with deep learning models and other JAX ecosystem libraries.

**Note:** Custom functions can only take scalar inputs and return a single scalar output.

## Choice Expressions

When an agent makes a choice, you can reference that choice in subsequent expressions.

```python
alice: chooses(x in X, wpp=1)
alice: chooses(y in Y, wpp=f(x, y))
```

Within an agent's scope, you can refer to their own choices as simple variables (e.g., `x`, `y` above).

To reference another agent's choices, use dot notation:

```python
alice.x + alice.y
```

This is equivalent to either of these bracket notations:

```python
alice[x] + alice[y]
alice[x + y]
```

## Probabilistic Operators

memo provides operators for computing statistics over probability distributions.

**Expectation:** Compute the expected value of an expression:

```python
E[alice.x + bob.z]
```

**Variance:** Compute the variance of an expression:

```python
Var[alice.y * 2]
```

**Probability:** Compute the probability that a condition holds:

```python
Pr[alice.y >= 0]
```

For joint probabilities, separate conditions with commas:

```python
Pr[a.x > 0, b.y < 2]
```

## Information-Theoretic Operators

memo supports entropy and divergence calculations.

**Entropy:** Compute the (mutual) entropy between choices:

```python
H[alice.x, bob.y, …]
```

**KL divergence:** Compute the Kullback-Leibler divergence between distributions:

```python
KL[alice.x | bob.y]
```

## Queries

You can query an agent's mental model using square brackets. This evaluates an expression from that agent's perspective:

```python
Var[alice[abs(x) * 2]]
alice[bob.y == 7]
```

## Hypotheticals with `imagine`

The `imagine` expression sets up a hypothetical world and evaluates an expression within it:

```python
imagine[
    bob: chooses(y in Y, wpp=1),
    alice: observes [bob.y] is bob.y,
    alice[Pr[bob.x == 7]]
]
```

The statements inside `imagine` modify the world, and the final line is the expression to evaluate in that hypothetical world.

## Calling Other Memos

One memo can reference another. Be sure to pass all required parameters:

```python
@memo
def f[x: X](a, b, c): …

@memo
def g():
    alice: chooses(x in X, wpp=f[x](1.0, 0.0, 3.1))
```

**Shorthand for forwarding parameters:** Use `...` to pass all of the caller's parameters:

```python
@memo
def g(a, b, c):
    alice: chooses(x in X, wpp=f[x](...))
```

In this case, `...` expands to `(a, b, c)`.

**Accessing specific return values:** When calling a model with multiple return values, specify which one you want using an index:

```python
g[0][x, y](3.14)
```

## Cost Reflection

You can query the computational cost (in FLOPs) needed to evaluate a memo. Note that you only pass parameters, not axes:

```python
@memo def f[…](a, b, c): …

cost @ f(3, 4, 5)
```

## Referencing Python Variables

Use curly braces to reference global Python variables within a memo expression:

```python
class Action(IntEnum): WAIT = 0; …

@memo def f[…](…):
    return {Action.WAIT}
```

You can also write arbitrary Python expressions inside the braces, including references to model parameters:

```python
@memo def f[…](x):
    return {x[0] << 4}
```

**Important:** You cannot reference memo constructs (agents or their choices) inside curly braces.

---

# Running and Configuring Memos

## Executing a Memo

Call a memo like a regular Python function with its parameters. It returns an array with the prescribed axes:

```python
f(a, b)
```

To pretty-print the results as a table:

```python
f(a, b, print_table=True)
```

To get output in different formats:

```python
f(a, b, return_pandas=True)
f(a, b, return_xarray=True)
```

To generate a "comic book" visualization of the model using Graphviz:

```python
f(a, b, save_comic="file")
```

## Decorator Options

Enable caching of results (keyed by scalar parameters):

```python
@memo(cache=True)
```

Enable execution tracing with timing information:

```python
@memo(debug_trace=True)
```

## Automatic Differentiation

memo models are differentiable via JAX. This is useful for fitting models to data using gradient descent:

```python
@memo
def f[…](a, b): …

jax.value_and_grad(f)(a, b)
```

This returns a tuple containing the model's output value and the gradient with respect to the parameters `a` and `b`.
