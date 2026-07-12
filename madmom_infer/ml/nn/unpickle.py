"""Restricted unpickler for madmom's own pretrained `.pkl` model files --
the safe-unpickle discipline this project uses INSTEAD of madmom's own
`Processor.load` (`madmom-upstream/madmom/processors.py:36-67`), which is a
bare `pickle.load(f, encoding='latin1')` with NO restriction on what classes
get instantiated. Unpickling is inherently code execution (`find_class` can
be asked to import and call anything the module could resolve); a `.pkl`
sourced from a git-cloned model repo (see `madmom_infer/models.py`) is a
lower-trust artifact than this project's own source, so this module allows
ONLY the exact class/function paths the target models are known to
reference, and raises loudly on anything else.

**How the allowlist below was derived**: not by reading source and guessing,
but by running `pickletools.dis()` over the target `.pkl` files and
collecting every `GLOBAL`/`STACK_GLOBAL` opcode's `(module, name)` pair.

Phase 2 target: all 8 `downbeats_blstm_[1-8].pkl` files (madmom's
`DOWNBEATS_BLSTM` ensemble, `madmom_infer/models.py`). All 8 reference the
IDENTICAL set (same architecture, different trained weights):

| pickled path (madmom original)             | mapped to (madmom_infer)                    |
|---------------------------------------------|----------------------------------------------|
| `madmom.ml.nn.NeuralNetwork`                | `madmom_infer.ml.nn.NeuralNetwork`            |
| `madmom.ml.nn.layers.FeedForwardLayer`      | `madmom_infer.ml.nn.layers.FeedForwardLayer`  |
| `madmom.ml.nn.layers.BidirectionalLayer`    | `madmom_infer.ml.nn.layers.BidirectionalLayer`|
| `madmom.ml.nn.layers.LSTMLayer`             | `madmom_infer.ml.nn.layers.LSTMLayer`         |
| `madmom.ml.nn.layers.Gate`                  | `madmom_infer.ml.nn.layers.Gate`              |
| `madmom.ml.nn.layers.Cell`                  | `madmom_infer.ml.nn.layers.Cell`              |
| `madmom.ml.nn.activations.sigmoid`          | `madmom_infer.ml.nn.activations.sigmoid`      |
| `madmom.ml.nn.activations.tanh`             | `madmom_infer.ml.nn.activations.tanh`         |
| `madmom.ml.nn.activations.softmax`          | `madmom_infer.ml.nn.activations.softmax`      |
| `numpy.core.multiarray._reconstruct`        | (unchanged -- numpy's own array-rebuild hook) |
| `numpy.ndarray`                             | (unchanged)                                   |
| `numpy.dtype`                               | (unchanged)                                   |

4a target: `key/2018/key_cnn.pkl` (madmom's `KEY_CNN`,
`madmom_infer/models.py`'s `key_cnn()`). `pickletools.dis()` against the
actual file (not guessed from reading `features/key.py`) found exactly:

| pickled path (madmom original)              | mapped to (madmom_infer)                       |
|----------------------------------------------|-------------------------------------------------|
| `madmom.ml.nn.layers.ConvolutionalLayer`      | `madmom_infer.ml.nn.layers.ConvolutionalLayer`   |
| `madmom.ml.nn.layers.MaxPoolLayer`            | `madmom_infer.ml.nn.layers.MaxPoolLayer`         |
| `madmom.ml.nn.layers.BatchNormLayer`          | `madmom_infer.ml.nn.layers.BatchNormLayer`       |
| `madmom.ml.nn.layers.PadLayer`                | `madmom_infer.ml.nn.layers.PadLayer`             |
| `madmom.ml.nn.layers.AverageLayer`            | `madmom_infer.ml.nn.layers.AverageLayer`         |
| `madmom.ml.nn.activations.elu`                | (already allowed, Phase 2 pre-emptive addition)  |
| `madmom.ml.nn.activations.linear`             | (already allowed, Phase 2 pre-emptive addition)  |

(plus the same `NeuralNetwork`/numpy-array-reconstruction entries above --
`key_cnn.pkl` does not reference `FeedForwardLayer` directly even though
`ConvolutionalLayer` subclasses it: pickle only needs a `GLOBAL` for the
leaf class actually instantiated, base-class behavior comes along via
ordinary Python attribute resolution once `find_class` returns the leaf.)

4b target: `onsets_rnn_[1-8].pkl`/`onsets_brnn_[1-8].pkl` (madmom's
`ONSETS_RNN`/`ONSETS_BRNN`) reference only already-allowed globals
(`NeuralNetwork`, `FeedForwardLayer`, `RecurrentLayer`, plus
`BidirectionalLayer` for the BRNN family, `sigmoid`/`tanh`) -- no table
changes needed. `onsets_cnn.pkl` (`ONSETS_CNN`) needs one new class entry
plus one new numpy global, both found by the same `pickletools.dis()`
technique:

| pickled path (madmom original)                | mapped to (madmom_infer)                        |
|-------------------------------------------------|---------------------------------------------------|
| `madmom.ml.nn.layers.StrideLayer`                | `madmom_infer.ml.nn.layers.StrideLayer`            |
| `numpy.core.multiarray.scalar`                   | (unchanged -- numpy's own 0-d-array-reconstruction hook, needed because `onsets_cnn.pkl`'s `BatchNormLayer.beta`/`.gamma` are pickled as bare numpy scalars, not 1-element arrays) |

`RecurrentLayer`, `relu` are allowed pre-emptively (they are `Gate`'s/
`Cell`'s own base class, and a cheap sibling activation function
respectively) even though no target `.pkl` file happens to reference them
directly, since a future model reusing the same layer family plausibly
would; anything NOT in this table (arbitrary builtins, `os`, `subprocess`,
`eval`, or ANY class outside madmom_infer's own `ml.nn` package plus numpy's
array-reconstruction primitives) is rejected with a loud
`pickle.UnpicklingError`, not silently allowed.

Reads: pickle (stdlib), numpy, madmom_infer.ml.nn.{NeuralNetwork},
madmom_infer.ml.nn.layers.*, madmom_infer.ml.nn.activations.*; read by:
madmom_infer/ml/nn/__init__.py (NeuralNetwork.load), madmom_infer/models.py
(cache-then-load flow).
"""

import io
import pickle

import numpy
import numpy.core.multiarray as _np_multiarray

from . import NeuralNetwork
from . import activations as _activations
from . import layers as _layers

# -- the full, closed allowlist (module, name) -> object -------------------
ALLOWED_GLOBALS = {
    ("madmom.ml.nn", "NeuralNetwork"): NeuralNetwork,
    ("madmom.ml.nn.layers", "Layer"): _layers.Layer,
    ("madmom.ml.nn.layers", "FeedForwardLayer"): _layers.FeedForwardLayer,
    ("madmom.ml.nn.layers", "RecurrentLayer"): _layers.RecurrentLayer,
    ("madmom.ml.nn.layers", "BidirectionalLayer"): _layers.BidirectionalLayer,
    ("madmom.ml.nn.layers", "Gate"): _layers.Gate,
    ("madmom.ml.nn.layers", "Cell"): _layers.Cell,
    ("madmom.ml.nn.layers", "LSTMLayer"): _layers.LSTMLayer,
    ("madmom.ml.nn.layers", "ConvolutionalLayer"): _layers.ConvolutionalLayer,
    ("madmom.ml.nn.layers", "MaxPoolLayer"): _layers.MaxPoolLayer,
    ("madmom.ml.nn.layers", "BatchNormLayer"): _layers.BatchNormLayer,
    ("madmom.ml.nn.layers", "PadLayer"): _layers.PadLayer,
    ("madmom.ml.nn.layers", "AverageLayer"): _layers.AverageLayer,
    ("madmom.ml.nn.layers", "StrideLayer"): _layers.StrideLayer,
    ("madmom.ml.nn.activations", "linear"): _activations.linear,
    ("madmom.ml.nn.activations", "tanh"): _activations.tanh,
    ("madmom.ml.nn.activations", "sigmoid"): _activations.sigmoid,
    ("madmom.ml.nn.activations", "relu"): _activations.relu,
    ("madmom.ml.nn.activations", "elu"): _activations.elu,
    ("madmom.ml.nn.activations", "softmax"): _activations.softmax,
    # numpy's own array-reconstruction primitives -- required to unpickle
    # any numpy array (every weight/bias matrix in the model), unchanged
    # (not remapped into madmom_infer, there is no madmom_infer numpy fork).
    ("numpy.core.multiarray", "_reconstruct"): _np_multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): _np_multiarray.scalar,
    ("numpy", "ndarray"): numpy.ndarray,
    ("numpy", "dtype"): numpy.dtype,
}


class SafeUnpickler(pickle.Unpickler):
    """A `pickle.Unpickler` that only resolves classes/functions present in
    `ALLOWED_GLOBALS` -- every other `GLOBAL`/`STACK_GLOBAL` opcode raises.

    This is the standard "safe unpickling" pattern (override `find_class`,
    documented in the stdlib `pickle` module itself as the recommended
    mitigation for untrusted pickle data): both legacy `GLOBAL` opcodes
    (protocol <= 2, what madmom's own `.pkl` files use) and modern
    `STACK_GLOBAL` (protocol >= 4) route through this same method, so one
    override covers both.
    """

    def find_class(self, module, name):
        key = (module, name)
        try:
            return ALLOWED_GLOBALS[key]
        except KeyError:
            raise pickle.UnpicklingError(
                "SafeUnpickler: refusing to unpickle disallowed global "
                "%r.%r -- only madmom's own NN-layer/activation classes "
                "(remapped to madmom_infer.ml.nn.*) and numpy's array-"
                "reconstruction primitives are permitted. If this is a "
                "legitimate madmom model using a layer/activation type "
                "this project hasn't ported yet, see madmom_infer/ml/nn/"
                "unpickle.py's ALLOWED_GLOBALS table." % (module, name)
            ) from None


def load_model(infile):
    """Load a single madmom `NeuralNetwork` from a pickled `.pkl` file path,
    file handle, or in-memory `bytes`, via `SafeUnpickler`.

    Matches the *effect* of madmom's own `Processor.load`
    (`madmom-upstream/madmom/processors.py:36-67`) -- `encoding='latin1'`
    (needed because these pickles were originally written under Python 2;
    `latin1` is the encoding `pickle` itself recommends for py2->py3 bytes/
    str compatibility) -- but via the restricted `SafeUnpickler` instead of
    bare `pickle.load`.
    """
    if isinstance(infile, (bytes, bytearray)):
        fh = io.BytesIO(infile)
        return SafeUnpickler(fh, encoding="latin1").load()
    if hasattr(infile, "read"):
        return SafeUnpickler(infile, encoding="latin1").load()
    with open(infile, "rb") as fh:
        return SafeUnpickler(fh, encoding="latin1").load()
