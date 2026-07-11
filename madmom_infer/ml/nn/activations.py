"""Neural-network activation functions -- verbatim port of
madmom.ml.nn.activations. Every function here doubles as a pickle target:
madmom's own `.pkl` model files reference these exact names
(`madmom.ml.nn.activations.sigmoid`/`tanh`/`softmax`, ...) as an
`activation_fn` attribute on a `layers.Layer` instance, so the restricted
unpickler (`madmom_infer/ml/nn/unpickle.py`) maps class paths onto THIS
module's functions by name -- the function names/signatures below are not
just an API choice, they are load-bearing pickle-compatibility surface.

Pickletools inspection of all 8 `downbeats_blstm_[1-8].pkl` files (the
Phase-2 target ensemble) found exactly three activation functions in use:
`sigmoid` (gates), `tanh` (cell/output), `softmax` (final feed-forward
layer). `linear`/`relu`/`elu` are ported anyway (cheap, verbatim, and other
madmom models phase-2 does not yet target reference them) but are not
exercised by this project's own golden fixtures.

Reads: numpy, scipy.special.expit; read by: madmom_infer/ml/nn/layers.py,
unpickled `.pkl` model files (via madmom_infer/ml/nn/unpickle.py).
"""

import numpy as np
from scipy.special import expit as _sigmoid


def linear(x, out=None):
    """Linear (identity) function. Port of `activations.linear`
    (`madmom-upstream/madmom/ml/nn/activations.py:14-34`)."""
    if out is None or x is out:
        return x
    out[:] = x
    return out


def tanh(x, out=None):
    """Hyperbolic tangent. Port of `activations.tanh`
    (`activations.py:37-56`). A thin wrapper around `np.tanh` so pickled
    models only ever depend on this module, not directly on numpy."""
    return np.tanh(x, out)


def sigmoid(x, out=None):
    """Logistic sigmoid. Port of `activations.sigmoid`
    (`activations.py:108-128`). Original madmom falls back to a hand-rolled
    `0.5 * (1 + tanh(0.5 * x))` implementation for scipy < 0.14 (a bug-
    workaround, see `activations.py:59-105`); this project requires a modern
    scipy (no such bug), so it always uses `scipy.special.expit` directly."""
    return _sigmoid(x, out)


def relu(x, out=None):
    """Rectified linear unit. Port of `activations.relu`
    (`activations.py:131-148`)."""
    return np.maximum(x, 0, out)


def elu(x, out=None):
    """Exponential linear unit. Port of `activations.elu`
    (`activations.py:151-179`)."""
    if out is None:
        out = x.copy()
    elif out is not x:
        out[:] = x[:]
    m = x < 0
    out[m] = np.exp(x[m]) - 1
    return out


def softmax(x, out=None):
    """Softmax (row-wise, over the last/class axis). Port of
    `activations.softmax` (`activations.py:182-209`)."""
    tmp = np.amax(x, axis=1, keepdims=True)
    if out is None:
        out = np.exp(x - tmp)
    else:
        np.exp(x - tmp, out=out)
    np.sum(out, axis=1, keepdims=True, out=tmp)
    out /= tmp
    return out
