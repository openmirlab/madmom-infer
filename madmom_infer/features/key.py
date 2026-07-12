"""Reimplementation of madmom.features.key -- `CNNKeyRecognitionProcessor`,
the CNN-based global key recognizer, PLUS its two small decoding helpers
(`key_prediction_to_label`, `add_axis`) and the `KEY_LABELS` class-index
table. This is Wave 4a's end-to-end acceptance target: audio in, a
24-class (12 major + 12 minor) key-probability vector AND a decoded
human-readable label ("E major") out.

Port of `madmom-upstream/madmom/features/key.py` (98 lines total, the
smallest `features/*.py` module in the audit -- no HMM/DBN decode stage,
just a spectrogram frontend feeding a CNN ensemble followed by two cheap
post-processing functions). `key_prediction_to_label`/`add_axis`/
`KEY_LABELS` are verbatim ports (`key.py:14-41`); `KEY_LABELS`' ordering
(`argmax()` index -> label) IS the model's own class-index convention, not
an arbitrary choice -- it must match `key_cnn.pkl`'s training-time label
order exactly, which this project cannot re-derive independently, only
preserve byte-for-byte from upstream.

`CNNKeyRecognitionProcessor.__init__` (`key.py:77-100`) builds:

1. `SignalProcessor(num_channels=1, sample_rate=44100)` -- downmix and (if
   needed) resample to 44.1kHz mono. Same resampling caveat as
   `RNNDownBeatProcessor` (`madmom_infer/features/downbeats.py`'s header):
   this project has no ffmpeg-backed resample, callers must supply 44.1kHz
   audio.
2. `FramedSignalProcessor(frame_size=8192, fps=5)` -- long frames (~186ms),
   sparse hop (200ms/frame) relative to the beat-tracking cascade's
   1024-4096-sample frames; key is a near-static, whole-clip property, so
   this doesn't need beat-rate time resolution.
3. `ShortTimeFourierTransformProcessor()` (defaults, caching FFT window --
   same caching-gotcha shape documented in `audio/stft.py`'s header).
4. `FilteredSpectrogramProcessor(num_bands=24, fmin=65, fmax=2100,
   unique_filters=True)` -> `LogarithmicSpectrogramProcessor(mul=1, add=1)`
   (i.e. `LOG`/`MUL`/`ADD` defaults) -- **two separate stages, not
   upstream's fused `LogarithmicFilteredSpectrogramProcessor`
   convenience class**, deliberately (see `audio/spectrogram.py`'s module
   header: that fused class is out of this port's scope, every call site
   composes the same effect from the two already-ported stages instead,
   exactly like `RNNDownBeatProcessor` already does). Numerically
   identical to the fused class -- `LogarithmicFilteredSpectrogramProcessor.
   process()` itself just constructs `LogarithmicFilteredSpectrogram`,
   which is literally `LogarithmicSpectrogram(FilteredSpectrogram(...))`
   under the hood (`madmom-upstream/madmom/audio/spectrogram.py:664-704`).
5. `NeuralNetworkEnsemble.load(KEY_CNN)` (`madmom_infer/models.py`'s
   `key_cnn()`) -- a size-1 ensemble (`KEY_CNN` resolves to exactly one
   file, `key/2018/key_cnn.pkl`, unlike `DOWNBEATS_BLSTM`'s 8-network
   ensemble; `NeuralNetworkEnsemble`/`average_predictions` degrade
   correctly for a length-1 list, see `ml/nn/__init__.py`). Its final
   layer is an `AverageLayer(axis=(0, 1))` (`ml/nn/layers.py`) --
   global-average-pooling the CNN's per-frame, per-bin 24-class map down
   to one 24-vector per clip, with a `linear` activation on the preceding
   `ConvolutionalLayer` (softmax is deliberately NOT baked into the model
   itself, see next step).
6. `add_axis` -- reshape the size-1-ensemble's `(24,)` vector back up to
   `(1, 24)` (`NeuralNetwork.process()` already `.squeeze()`s it down to
   1D, since there's only one "row"; the class-probability convention
   downstream, and every `nn_files=[...]` custom-model caller, expects a
   2D `(num_clips, 24)` array).
7. `softmax` (`ml/nn/activations.py`) -- applied HERE, outside the network,
   over the already-averaged, already-ensemble-averaged logits. This is
   why the CNN's own last-layer `activation_fn` is `linear`, not
   `softmax`: averaging raw probabilities across models/frames would not
   equal the average of pre-softmax logits passed through softmax once
   (softmax doesn't commute with averaging), so real madmom defers the
   softmax to the very end of the pipeline, after all averaging is done --
   this port must do the same, not "helpfully" move it earlier.

Reads: madmom_infer/audio/{signal,stft,spectrogram}.py (the pre-processing
cascade), madmom_infer/ml/nn/__init__.py (NeuralNetworkEnsemble),
madmom_infer/ml/nn/activations.py (softmax), madmom_infer/models.py
(key_cnn download), madmom_infer/processors.py (SequentialProcessor);
read by: nothing yet (Wave 4a's own end-to-end target).
"""

import numpy as np

from madmom_infer.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
)
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor
from madmom_infer.ml.nn import NeuralNetworkEnsemble
from madmom_infer.ml.nn.activations import softmax
from madmom_infer.processors import SequentialProcessor

KEY_LABELS = ['A major', 'Bb major', 'B major', 'C major', 'Db major',
              'D major', 'Eb major', 'E major', 'F major', 'F# major',
              'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
              'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
              'F minor', 'F# minor', 'G minor', 'G# minor']


def key_prediction_to_label(prediction):
    """Convert a key-class probability vector (or batch of them) to a
    human-readable key name.

    Verbatim port of `key.key_prediction_to_label`
    (`madmom-upstream/madmom/features/key.py:21-37`). Only the FIRST row's
    argmax is used even if `prediction` has multiple rows (matches
    upstream exactly -- not a bug this port introduces).
    """
    prediction = np.atleast_2d(prediction)
    return KEY_LABELS[prediction[0].argmax()]


def add_axis(x):
    """Prepend a length-1 axis. Verbatim port of `key.add_axis`
    (`madmom-upstream/madmom/features/key.py:40-41`)."""
    return x[np.newaxis, ...]


class CNNKeyRecognitionProcessor(SequentialProcessor):
    """Recognise the global key of a musical piece using a Convolutional
    Neural Network, as described in [1]_.

    Port of `madmom.features.key.CNNKeyRecognitionProcessor`
    (`madmom-upstream/madmom/features/key.py:44-100`) -- see this module's
    header for the full stage-by-stage breakdown, including why the
    fused-vs-composed filtered-log-spectrogram split is a no-op numerically
    and why `softmax` is applied outside the network.

    Parameters
    ----------
    nn_files : list, optional
        List of trained CNN model files. Defaults to `None`, which loads
        madmom's own pretrained `KEY_CNN` ensemble (downloaded at runtime,
        sha256-verified, via `madmom_infer.models.key_cnn()` -- CC BY-NC-SA
        4.0, non-commercial use only, see `madmom_infer/models.py`).

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer, "Genre-Agnostic Key
           Classification with Convolutional Neural Networks", In
           Proceedings of the 19th International Society for Music
           Information Retrieval Conference (ISMIR), Paris, France, 2018.

    Examples
    --------
    >>> from madmom_infer.features.key import (
    ...     CNNKeyRecognitionProcessor, key_prediction_to_label,
    ... )
    >>> proc = CNNKeyRecognitionProcessor()
    >>> prediction = proc('track.wav')  # doctest: +SKIP
    >>> key_prediction_to_label(prediction)  # doctest: +SKIP
    'E major'
    """

    def __init__(self, nn_files=None, **kwargs):
        from madmom_infer.models import key_cnn

        # spectrogram computation
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=8192, fps=5)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=24, fmin=65, fmax=2100, unique_filters=True)
        log = LogarithmicSpectrogramProcessor(mul=1, add=1)

        # neural network
        nn_files = nn_files or key_cnn()
        nn = NeuralNetworkEnsemble.load(nn_files)

        # create processing pipeline
        super().__init__((sig, frames, stft, filt, log, nn, add_axis, softmax))
