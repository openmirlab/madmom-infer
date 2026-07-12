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

4c target: `beats_lstm_[1-8].pkl`/`beats_blstm_[1-8].pkl` (madmom's
`BEATS_LSTM`/`BEATS_BLSTM`) reference only already-allowed globals (same
architecture family as `downbeats_blstm_*.pkl`) -- no table changes needed.
All 12 `downbeats_bgru_{harmonic,rhythmic}_[0-5].pkl` files (`DOWNBEATS_BGRU`)
need exactly one new class pair, found by the same `pickletools.dis()`
technique (both `downbeats_bgru_harmonic_0.pkl` and
`downbeats_bgru_rhythmic_0.pkl` walked -- identical global set):

| pickled path (madmom original)         | mapped to (madmom_infer)                |
|------------------------------------------|--------------------------------------------|
| `madmom.ml.nn.layers.GRUCell`             | `madmom_infer.ml.nn.layers.GRUCell`         |
| `madmom.ml.nn.layers.GRULayer`            | `madmom_infer.ml.nn.layers.GRULayer`        |

**Found empirically, not guessed**: all 12 `downbeats_bgru_*.pkl` files are
themselves OLDER-format pickles than `downbeats_blstm_*.pkl`/`key_cnn.pkl`/
the onset models -- loading one with real madmom actually emits a
`RuntimeWarning` ("Please update your GRU models...", see
`madmom_infer/ml/nn/layers.py`'s `GRULayer.__setstate__`), and their
`pickletools.dis()` walk additionally references two GENERIC old-style-class
reconstruction globals neither `downbeats_blstm_*.pkl` nor any other target
`.pkl` in this project needs:

| pickled path (madmom original)  | mapped to (madmom_infer)                                |
|------------------------------------|-------------------------------------------------------------|
| `copy_reg._reconstructor`          | `copyreg._reconstructor` (stdlib, Py2 `copy_reg` -> Py3 `copyreg`) |
| `__builtin__.object`               | `builtins.object` (stdlib, Py2 `__builtin__` -> Py3 `builtins`)    |

Both are standard-library object-reconstruction primitives (the same
category of "safe, mechanical, no arbitrary code execution" as numpy's own
`_reconstruct`/`scalar` entries below) -- `copy_reg._reconstructor(cls, base,
state)` is Python 2's pre-`__reduce_ex__`-protocol-2 old-style-class
rebuilding helper, still present in Python 3's `copyreg` module for
backwards-compatible unpickling of exactly this kind of legacy pickle.

4d target: `chroma/2016/chroma_dnn.pkl` (`CHROMA_DNN`) references only
already-allowed globals (`NeuralNetwork`, `FeedForwardLayer`, `relu`,
`sigmoid`) -- no table changes needed, confirming the 4.0 audit's own
prediction. `chords/2016/chords_cnnfeat.pkl` (`CHORDS_CNN_FEAT`) likewise
references only already-allowed globals (`NeuralNetwork`,
`ConvolutionalLayer`, `BatchNormLayer`, `MaxPoolLayer`, `linear`, `relu` --
4a's CNN layer set). `chords/2016/chords_dccrf.pkl`/`chords_cnncrf.pkl`
(`CHORDS_DCCRF`/`CHORDS_CFCRF`) each need exactly one new class, found by
the same `pickletools.dis()` technique (both walked -- identical single-class
global set beyond numpy's own array-reconstruction primitives):

| pickled path (madmom original)         | mapped to (madmom_infer)                     |
|-------------------------------------------|--------------------------------------------------|
| `madmom.ml.crf.ConditionalRandomField`    | `madmom_infer.ml.crf.ConditionalRandomField`      |

4e target: `notes/2013/notes_brnn.pkl` (`NOTES_BRNN`) references only
already-allowed globals (`NeuralNetwork`, `BidirectionalLayer`,
`FeedForwardLayer`, `RecurrentLayer`, `tanh`, `linear`) -- no table changes
needed, confirming the 4.0 audit's own prediction. `notes/2019/notes_cnn.pkl`
(`NOTES_CNN`) -- plus `notes/2018/notes_cnn_{1,2}.pkl` (`NOTES_CNN_MIREX`,
walked for completeness even though no ported processor loads them, same
"real but unused, no registry entry" status as 4b's `ONSETS_BRNN_PP`) --
is a REAL SURPRISE, found only by actually `pickletools.dis()`-walking the
file (not guessed, not assumable from `key_cnn.pkl`'s precedent): it does
**not** pickle a bare `NeuralNetwork` the way every other target `.pkl` in
this project does. It pickles an entire
`madmom.processors.SequentialProcessor`/`ParallelProcessor` OBJECT GRAPH
directly -- the model's own multi-task (note/onset/offset) branch-and-merge
structure is baked straight into the pickle, not built by
`CNNPianoNoteProcessor.__init__` the way `CNNKeyRecognitionProcessor`
builds its pipeline around a bare `NeuralNetwork`. Confirmed by actually
loading the file with real madmom (`pickle.load`, reference venv) and
inspecting `type(obj)`/`obj.__dict__` recursively: `SequentialProcessor([
BatchNormLayer, ConvolutionalLayer x3, ParallelProcessor([3x
SequentialProcessor(ConvolutionalLayer, TransposeLayer, ReshapeLayer,
FeedForwardLayer)]), numpy.dstack])` -- the 3 parallel branches are the
note/onset/offset heads, `numpy.dstack` is the final multi-task merge. This
needs 8 new allowlist entries, not 2:

| pickled path (madmom original)             | mapped to (madmom_infer)                          |
|------------------------------------------------|--------------------------------------------------------|
| `madmom.ml.nn.layers.ReshapeLayer`              | `madmom_infer.ml.nn.layers.ReshapeLayer`                |
| `madmom.ml.nn.layers.TransposeLayer`            | `madmom_infer.ml.nn.layers.TransposeLayer`              |
| `madmom.processors.SequentialProcessor`         | `madmom_infer.processors.SequentialProcessor`           |
| `madmom.processors.ParallelProcessor`           | `madmom_infer.processors.ParallelProcessor`             |
| `numpy.dstack`                                  | (unchanged -- plain function reference; `notes_cnn.pkl` itself pickles it as `('numpy', 'dstack')`) |
| `numpy.lib.shape_base.dstack`                   | `numpy.dstack` (same function; `notes_cnn_{1,2}.pkl`, an older pickle, spells its module path differently -- both resolve to numpy's one real `dstack`) |
| `_codecs.encode`                                | (unchanged -- stdlib, part of an old-format numpy `ndarray`/`dtype` byte-payload reconstruction, same "safe, mechanical, no arbitrary code execution" category as `numpy.core.multiarray._reconstruct`/`scalar` above) |
| `itertools.imap`                                | Python 3's builtin `map` |

The last one is its own small finding: Python 3's stdlib `pickle.Unpickler.
find_class` normally auto-remaps a handful of Python-2-only module paths via
`pickle._compat_pickle.NAME_MAPPING` (e.g. `('itertools', 'imap') ->
('builtins', 'map')`) whenever the pickle's protocol is < 4 -- which is
exactly how real madmom's own bare `pickle.load` transparently resolves
this old `ParallelProcessor.__init__`'s `self.map = it.imap` (an older
madmom that imported `itertools as it`, before it simplified to plain
`self.map = map`) into today's builtin `map`. `SafeUnpickler.find_class`
does its own raw `(module, name)` lookup and does NOT consult
`_compat_pickle` at all, so this remapping needs an explicit allowlist
entry to reproduce -- same shape of gap 4c's `copy_reg._reconstructor`/
`__builtin__.object` entries already closed for the older-format
`downbeats_bgru_*.pkl` files. (This unpickled `self.map` value is inert in
this port either way -- `madmom_infer.processors.ParallelProcessor.process`
never reads `self.map`, always runs its sub-processors with a plain list
comprehension, see `processors.py`'s module header -- but the allowlist
entry is still required for the unpickle to succeed at all.)

4f target: `patterns/2013/ballroom_pattern_{3,4}_4.pkl` (`PATTERNS_BALLROOM`)
-- `pickletools.dis()`-walked both directly (not guessed): each is a plain
`dict` (`{'gmms': [...], 'num_beats': int, 'time_signature': (...)}`, no
`madmom.processors.*`/NN globals at all) whose `'gmms'` list elements are
`madmom.ml.gmm.GMM` instances, restored via `NEWOBJ` + `BUILD` (i.e.
`GMM.__setstate__`, ALREADY handled by the standard `pickle.Unpickler`
protocol this module's `SafeUnpickler` inherits unmodified -- only
`find_class` is overridden, see this module's header). Needs exactly ONE
new allowlist entry, found by the same technique:

| pickled path (madmom original)  | mapped to (madmom_infer)         |
|-------------------------------------|--------------------------------------|
| `madmom.ml.gmm.GMM`                 | `madmom_infer.ml.gmm.GMM`            |

Reads: pickle (stdlib), numpy, madmom_infer.ml.nn.{NeuralNetwork},
madmom_infer.ml.nn.layers.*, madmom_infer.ml.nn.activations.*,
madmom_infer.ml.crf.ConditionalRandomField, madmom_infer.ml.gmm.GMM,
madmom_infer.processors.{SequentialProcessor,ParallelProcessor}; read by:
madmom_infer/ml/nn/__init__.py (NeuralNetwork.load), madmom_infer/ml/crf.py
(ConditionalRandomField.load), madmom_infer/features/downbeats.py
(PatternTrackingProcessor.__init__, loading PATTERNS_BALLROOM pattern
files), madmom_infer/models.py (cache-then-load flow).
"""

import _codecs
import copyreg
import io
import pickle

import numpy
import numpy.core.multiarray as _np_multiarray

from . import NeuralNetwork
from . import activations as _activations
from . import layers as _layers
from ..crf import ConditionalRandomField
from ..gmm import GMM
from ...processors import ParallelProcessor, SequentialProcessor

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
    ("madmom.ml.nn.layers", "GRUCell"): _layers.GRUCell,
    ("madmom.ml.nn.layers", "GRULayer"): _layers.GRULayer,
    ("madmom.ml.nn.layers", "ReshapeLayer"): _layers.ReshapeLayer,
    ("madmom.ml.nn.layers", "TransposeLayer"): _layers.TransposeLayer,
    ("madmom.ml.crf", "ConditionalRandomField"): ConditionalRandomField,
    # 4f: PATTERNS_BALLROOM's pattern .pkl files (each a plain dict of
    # {'gmms': [...], 'num_beats': int, 'time_signature': ...}) pickle a
    # list of madmom.ml.gmm.GMM instances -- see madmom_infer/ml/gmm.py's
    # module header for the pickletools-confirmed finding (old-format
    # pickle, GMM.__setstate__'s legacy weights_/means_/covars_ rename
    # branch fires on both target files).
    ("madmom.ml.gmm", "GMM"): GMM,
    # 4e: notes_cnn.pkl pickles a whole SequentialProcessor/ParallelProcessor
    # graph, not a bare NeuralNetwork -- see this module's header.
    ("madmom.processors", "SequentialProcessor"): SequentialProcessor,
    ("madmom.processors", "ParallelProcessor"): ParallelProcessor,
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
    # 4e: notes_cnn.pkl's final multi-task merge stage is a plain function
    # reference to numpy's own `dstack` -- two module-path spellings appear
    # across the target pickles (see this module's header), both the same
    # real function.
    ("numpy", "dstack"): numpy.dstack,
    ("numpy.lib.shape_base", "dstack"): numpy.dstack,
    # 4e: `_codecs.encode` -- stdlib, part of an old-format numpy array/dtype
    # byte-payload reconstruction inside notes_cnn.pkl, same "safe,
    # mechanical" category as numpy's own _reconstruct/scalar above.
    ("_codecs", "encode"): _codecs.encode,
    # old-style-class reconstruction primitives needed only by the
    # older-format DOWNBEATS_BGRU pickles (4c) -- see this module's header.
    ("copy_reg", "_reconstructor"): copyreg._reconstructor,
    ("__builtin__", "object"): object,
    # 4e: notes_cnn.pkl's ParallelProcessor.__init__ pickles `self.map` as
    # `itertools.imap` (an older madmom that used `itertools as it`) --
    # Python 3's own pickle.Unpickler transparently remaps this via
    # `_compat_pickle.NAME_MAPPING` for protocol < 4; SafeUnpickler doesn't
    # consult that table, so this needs an explicit entry -- see this
    # module's header. Inert either way (this port's own ParallelProcessor
    # never reads `self.map`).
    ("itertools", "imap"): map,
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
