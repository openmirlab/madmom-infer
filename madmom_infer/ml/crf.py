"""Linear-chain Conditional Random Field -- pure-numpy Viterbi decode, port of
`madmom.ml.crf.ConditionalRandomField`. Wave 4d's chord-decoding backend:
`DeepChromaChordRecognitionProcessor`/`CRFChordRecognitionProcessor`
(`madmom_infer/features/chords.py`) both decode a chroma/CNN-feature
observation sequence into a most-probable major/minor-chord label sequence
by loading one of these (`CHORDS_DCCRF`/`CHORDS_CFCRF`, `madmom_infer/
models.py`) and calling it.

Forward-inference only (this project never trains a CRF, matching the
inference-only scope of every other `ml/*` module) -- verbatim port of
`madmom.ml.crf.ConditionalRandomField.process`
(`madmom-upstream/madmom/ml/crf.py:76-119`), a textbook matrix-formulation
Viterbi decode (`argmax_y P(Y=y|X=x)`), no training/`fit`/gradient code
exists upstream to port either.

`ConditionalRandomField.load` (new here, not in upstream -- upstream's CRF
inherits `Processor.load`'s bare `pickle.load`, out of scope per
`madmom_infer/processors.py`'s header) delegates to `unpickle.load_model`
instead, same pattern as `NeuralNetwork.load`/`NeuralNetworkEnsemble.load`
(`madmom_infer/ml/nn/__init__.py`). `pickletools`-walking both real target
pickles (`chords_dccrf.pkl`, `chords_cnncrf.pkl`) confirms they pickle a
`ConditionalRandomField` via `NEWOBJ` (empty-tuple constructor args) plus a
direct `__dict__` restore under the exact attribute names this class's own
`__init__` uses (`pi`, `tau`, `c`, `A`, `W`) -- no custom
`__getstate__`/`__setstate__` needed, same "attribute names, not
constructor-argument-perfect `__init__`s" shape as `ml/nn/layers.py`'s
pickled layer classes (see that module's header).

Reads: numpy, madmom_infer.processors (Processor), madmom_infer.ml.nn.unpickle
(load_model, for `.load()`); read by: madmom_infer/features/chords.py
(DeepChromaChordRecognitionProcessor, CRFChordRecognitionProcessor).
"""

import numpy as np

from ..processors import Processor


class ConditionalRandomField(Processor):
    """Linear-chain Conditional Random Field, matrix-based definition:

    .. math::
        P(Y|X) = exp[E(Y,X)] / Sum_{Y'}[E(Y', X)]

        E(Y,X) = Sum_{i=1}^{N} [y_{n-1}^T A y_n + y_n^T c + x_n^T W y_n] +
                y_0^T pi + y_N^T tau,

    where Y is a sequence of labels in one-hot encoding and X are the
    observed features.

    Verbatim port of `madmom.ml.crf.ConditionalRandomField`
    (`madmom-upstream/madmom/ml/crf.py:12-119`).

    Parameters
    ----------
    initial : numpy array
        Initial potential (pi) of the CRF. Also defines the number of states.
    final : numpy array
        Potential (tau) of the last variable of the CRF.
    bias : numpy array
        Label bias potential (c).
    transition : numpy array
        Matrix defining the transition potentials (A), where the rows are
        the 'from' dimension, and columns the 'to' dimension.
    observation : numpy array
        Matrix defining the observation potentials (W), where the rows are
        the 'observation' dimension, and columns the 'state' dimension.

    Examples
    --------
    Create a CRF that emulates a simple hidden markov model. This means that
    the bias and final potential will be constant and thus have no effect
    on the predictions.

    >>> eta = np.spacing(1)  # for numerical stability
    >>> initial = np.log(np.array([0.7, 0.2, 0.1]) + eta)
    >>> final = np.ones(3)
    >>> bias = np.ones(3)
    >>> transition = np.log(np.array([[0.6, 0.2, 0.2],
    ...                               [0.1, 0.7, 0.2],
    ...                               [0.1, 0.1, 0.8]]) + eta)
    >>> observation = np.log(np.array([[0.9, 0.5, 0.1],
    ...                                [0.1, 0.5, 0.1]]) + eta)
    >>> crf = ConditionalRandomField(initial, final, bias,
    ...                              transition, observation)

    We can now decode the most probable state sequence given an observation
    sequence. Since we are emulating a discrete HMM, the observation
    sequence needs to be observation ids in one-hot encoding.

    The following observation sequence corresponds to "0, 0, 1, 0, 1, 1":

    >>> obs = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]])

    Now we can find the most likely state sequence:

    >>> crf.process(obs)
    array([0, 0, 1, 1, 1, 1], dtype=uint32)
    """

    def __init__(self, initial, final, bias, transition, observation):
        self.pi = initial
        self.tau = final
        self.c = bias
        self.A = transition
        self.W = observation

    def process(self, observations, **kwargs):
        """Determine the most probable configuration of Y given the state
        sequence x:

        .. math::
            y^* = argmax_y P(Y=y|X=x)

        Parameters
        ----------
        observations : numpy array
            Observations (x) to decode the most probable state sequence for.

        Returns
        -------
        y_star : numpy array
            Most probable state sequence.
        """
        # pylint: disable=unused-argument
        num_observations = len(observations)
        num_states = len(self.pi)
        bt_pointers = np.empty((num_observations, num_states), dtype=np.uint32)
        viterbi = self.pi.copy()
        y_star = np.empty(num_observations, dtype=np.uint32)

        for i in range(num_observations):
            all_trans = self.A + viterbi[:, np.newaxis]
            best_trans = np.max(all_trans, axis=0)
            bt_pointers[i] = np.argmax(all_trans, axis=0)
            viterbi = self.c + np.dot(observations[i], self.W) + best_trans

        viterbi += self.tau

        y_star[-1] = np.argmax(viterbi)
        for i in range(len(y_star) - 1)[::-1]:
            y_star[i] = bt_pointers[i + 1, y_star[i + 1]]

        return y_star

    @classmethod
    def load(cls, infile):
        """Load a single `ConditionalRandomField` from a pickled madmom
        model file.

        Delegates to `unpickle.load_model` (restricted class-allowlisted
        unpickler), NOT `pickle.load` -- see this module's header and
        `madmom_infer/ml/nn/unpickle.py`.
        """
        from .nn.unpickle import load_model

        return load_model(infile)
