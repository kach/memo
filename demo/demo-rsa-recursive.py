from memo import memo
import jax
import jax.numpy as np
from enum import IntEnum

class U(IntEnum):  # utterance space
    GREEN  = 0b0001
    PINK   = 0b0010
    SQUARE = 0b0100
    ROUND  = 0b1000

class R(IntEnum):  # referent space
    GREEN_SQUARE = U.GREEN | U.SQUARE
    GREEN_CIRCLE = U.GREEN | U.ROUND
    PINK_CIRCLE  = U.PINK  | U.ROUND

@jax.jit
def denotes(u, r):
    return (u & r) != 0

@memo  # recursive RSA model
def L[u: U, r: R](beta, t):
    listener: thinks[
        speaker: given(r in R, wpp=1),
        speaker: chooses(u in U, wpp=
            denotes(u, r) * (1 if t == 0 else exp(beta * L[u, r](beta, t - 1))))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in R, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

beta = 1.
print(L(beta, 0))
print(L(beta, 1))


## Fitting the model to data...
Y = np.array([65, 115, 0]) / 180  # data from Qing & Franke 2015
@jax.jit
def loss(beta):
    return np.mean((L(beta, 1)[0] - Y) ** 2)

from matplotlib import pyplot as plt
plt.figure(figsize=(5, 4))

## Best model fit vs. data
beta = 1.74
plt.subplot(2, 1, 2)
X = np.array([0, 1, 2])
plt.bar(X - 0.25, Y, width=0.25, yerr=2 * np.sqrt(Y * (1 - Y) / 180), capsize=2, label='humans')
plt.bar(X + 0.00, L(beta, 1)[0], width=0.25, label='model, t=1')
plt.bar(X + 0.25, L(beta, 0)[0], width=0.25, label='model, t=0')
plt.xticks([0, 1, 2], ['green\nsquare', 'green\ncircle', 'pink\ncircle'])
plt.xlabel('Inferred referent r')
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.title('Final model fit')

## Fitting by grid search!
plt.subplot(2, 2, 1)
beta = np.linspace(0, 3, 100)
plt.plot(beta, jax.vmap(loss)(beta))
plt.xlabel('beta')
plt.ylabel('MSE (%)')
plt.yticks([0, 0.02], [0, 2])
plt.xticks([0, 1, 2, 3])
plt.title('Grid search')

## Fitting by gradient descent!
vg = jax.value_and_grad(loss)
plt.subplot(2, 2, 2)
losses = []
beta = 0.
for _ in range(26):
    l, dbeta = vg(beta)
    losses.append(l)
    beta = beta - dbeta * 12.
plt.plot(np.arange(len(losses)), losses)
plt.ylabel('MSE (%)')
plt.xlabel('Step #')
plt.yticks([0, 0.02], [0, 2])
plt.title('Gradient descent')

plt.tight_layout()
plt.savefig('../paper/fig/rsa-fit.pdf')