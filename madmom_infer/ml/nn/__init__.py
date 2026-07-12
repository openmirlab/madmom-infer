"""Neural-network runtime core -- port of madmom.ml.nn's `NeuralNetwork`
(a plain forward-pass `Processor` over a list of `layers.Layer` instances)
and `NeuralNetworkEnsemble` (a `ParallelProcessor` of `NeuralNetwork`s,
averaged). This is Phase 2's centerpiece: madmom's own NN layers are already
forward-inference-only (no `backward`/`train`/`fit`/`grad` anywhere in
`ml/nn/*`, confirmed by grep against `madmom-upstream`, see docs/DESIGN.md
B.4), so porting them is "just" reproducing the forward math and (this
module's real complexity) the pickled-model class-path surface unpickled
`.pkl` files reference -- see `madmom_infer/ml/nn/unpickle.py`.

Deliberately NOT ported here (this project never ships `Processor.load`'s
unrestricted `pickle.load` -- see `unpickle.py`'s restricted, class-
allowlisted `SafeUnpickler` instead): `NeuralNetwork`/`NeuralNetworkEnsemble`
still expose a `.load()` classmethod, but it calls into `unpickle.py`, not
`pickle.load` directly. `add_arguments()` (argparse plumbing) is skipped, per
CLAUDE.md.

Wave 4c fix: `average_predictions`'s `avg()` helper has an explicit
`np.ndim(pred[0]) == 0` branch not present upstream -- a genuine numpy-2.x-
vs-1.23.5 DTYPE divergence, found via `RNNBarProcessor`'s GRU ensembles
(`features/downbeats.py`), which can be fed a single-frame input, making
each per-network prediction a 0-DIMENSIONAL array. Real madmom's own
`sum(pred) / len(pred)` upcasts a list of 0-d float32 arrays to float64 on
numpy < 2.0 (0-d "scalar-kind" arrays follow different value-based-casting
rules than N-d ones there) but NOT on numpy >= 2.0 (NEP 50 unifies the two
paths, keeping float32) -- confirmed empirically against the reference
venv. The fix reproduces the old (real-madmom-recorded) float64 dtype on
every numpy version; N-d predictions (every other model family this
project ports, including the already-cross-BLAS-proven `DOWNBEATS_BLSTM`
ensemble) are unaffected -- `sum(pred) / len(pred)` already stayed float32
on both numpy versions for those, confirmed by the full suite staying
green after this fix.

Reads: madmom_infer.processors (Processor, ParallelProcessor),
madmom_infer.ml.nn.unpickle (restricted model loading); read by:
madmom_infer/features/downbeats.py (RNNDownBeatProcessor, RNNBarProcessor),
madmom_infer/features/beats.py (RNNBeatProcessor).
"""

import numpy as np

from ...processors import ParallelProcessor, Processor, SequentialProcessor


def average_predictions(predictions):
    """Average all predictions.

    Port of `madmom.ml.nn.average_predictions`
    (`madmom-upstream/madmom/ml/nn/__init__.py:18-59`). If `predictions[0]`
    is a tuple (a multi-task network's output), each tuple position is
    averaged separately.
    """
    if len(predictions) == 1:
        return predictions[0]

    def avg(pred):
        # NOTE: the `np.ndim(pred[0]) == 0` branch (not in upstream, which
        # is just `sum(pred) / len(pred)` on every input shape) -- found
        # and fixed a genuine numpy-2.x-vs-1.23.5 divergence, same class as
        # docs/DESIGN.md C.1 (see `features/onsets.py`'s
        # `normalized_weighted_phase_deviation` for the precedent), this
        # time in DTYPE rather than value. Confirmed empirically against
        # the reference venv: `sum(pred) / len(pred)` over a list of 0-
        # DIMENSIONAL float32 arrays (e.g. a `NeuralNetworkEnsemble` fed a
        # single-frame/single-beat-window input, as `RNNBarProcessor`'s GRU
        # ensembles can be) upcasts to float64 on numpy < 2.0 (its value-
        # based-casting rules treat 0-d "scalar-kind" arrays differently
        # from N-d ones -- `int + 0d_float32_array` promotes, `int +
        # 1d_float32_array` does not), but NOT on numpy >= 2.0 (NEP 50
        # unifies the two paths, keeping float32 in both cases). For N-d
        # (ndim >= 1) predictions -- every OTHER model family this project
        # ports, including the already-cross-BLAS-proven DOWNBEATS_BLSTM
        # ensemble -- `sum(pred) / len(pred)` already stays float32 on
        # BOTH numpy versions, so this branch is a no-op there; only the
        # 0-d case needed an explicit fix to reproduce real madmom's
        # actual (float64) recorded dtype on every numpy version.
        if np.ndim(pred[0]) == 0:
            return np.float64(sum(float(p) for p in pred) / len(pred))
        return sum(pred) / len(pred)

    if isinstance(predictions[0], tuple):
        avg_pred = []
        for pred in list(zip(*predictions)):
            avg_pred.append(avg(pred))
        return tuple(avg_pred)
    return avg(predictions)


class NeuralNetwork(Processor):
    """A feed-forward/recurrent neural network: a plain list of `Layer`s,
    activated in sequence.

    Port of `madmom.ml.nn.NeuralNetwork`
    (`madmom-upstream/madmom/ml/nn/__init__.py:62-133`). Unpickled `.pkl`
    model files (`madmom_infer/ml/nn/unpickle.py`) construct this class
    directly via pickle's `NEWOBJ` + attribute-dict restore (bypassing
    `__init__` entirely) -- see `unpickle.py`'s module header for why this
    means `NeuralNetwork` needs no custom `__reduce__`/`__setstate__` of its
    own to be a valid unpickle target, only the right attribute name
    (`layers`).
    """

    def __init__(self, layers):
        self.layers = layers

    def process(self, data, reset=True, **kwargs):
        """Process `data` (shape `(num_frames, num_inputs)`) through every
        layer in order, returning the final layer's (squeezed) output.

        Matches `madmom.ml.nn.NeuralNetwork.process`
        (`ml/nn/__init__.py:95-124`).
        """
        # make data at least 2d (required by NN layers); `copy=None` (not
        # `copy=False`) for numpy-2.x compatibility -- see docs/DESIGN.md
        # C.1 (this exact call site, `ml/nn/__init__.py:114`, is one of the
        # three numpy-2.x incompatibilities documented there).
        if isinstance(data, np.ndarray) and data.ndim < 2:
            data = np.array(data, subok=True, copy=None, ndmin=2)
        for layer in self.layers:
            data = layer(data, reset=reset)
        try:
            return data.squeeze()
        except AttributeError:
            # multi-task networks have multiple outputs and return lists
            return tuple(d.squeeze() for d in data)

    def reset(self):
        """Reset every layer to its initial state."""
        for layer in self.layers:
            layer.reset()

    @classmethod
    def load(cls, infile):
        """Load a single `NeuralNetwork` from a pickled madmom model file.

        Delegates to `unpickle.load_model` (restricted class-allowlisted
        unpickler), NOT `pickle.load` -- see this module's header.
        """
        from .unpickle import load_model

        return load_model(infile)


class NeuralNetworkEnsemble(SequentialProcessor):
    """An ensemble of `NeuralNetwork`s, run in parallel and averaged.

    Port of `madmom.ml.nn.NeuralNetworkEnsemble`
    (`madmom-upstream/madmom/ml/nn/__init__.py:135-225`, minus
    `add_arguments` -- argparse plumbing, out of scope per CLAUDE.md).
    `RNNDownBeatProcessor` builds one of these from `DOWNBEATS_BLSTM`, madmom's
    8-network downbeat BLSTM ensemble (`madmom_infer/models.py`).
    """

    def __init__(self, networks, ensemble_fn=average_predictions,
                 num_threads=None, **kwargs):
        # pylint: disable=unused-argument
        networks_processor = ParallelProcessor(networks, num_threads=num_threads)
        super().__init__((networks_processor, ensemble_fn))

    @classmethod
    def load(cls, nn_files, **kwargs):
        """Instantiate a `NeuralNetworkEnsemble` from a list of model files.

        Matches `madmom.ml.nn.NeuralNetworkEnsemble.load`
        (`ml/nn/__init__.py:176-195`).
        """
        networks = [NeuralNetwork.load(f) for f in nn_files]
        return cls(networks, **kwargs)
