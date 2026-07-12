"""Golden-fixture A/B tests for `RNNDownBeatProcessor`
(`madmom_infer/features/downbeats.py`) chained into the already-ported
`DBNDownBeatTrackingProcessor` -- Phase 2's end-to-end acceptance target:
real audio in, beat/downbeat times out, using madmom's OWN pretrained
`DOWNBEATS_BLSTM` weights (downloaded via `madmom_infer/models.py`).

**Read this before "fixing" any near-miss below -- same discipline as
`test_spectrogram.py`.** The pre-processing cascade (multi-frame-size
spectrogram + `SpectrogramDifference`) inherits `test_spectrogram.py`'s
proven BLAS-non-associativity bound; the NN forward pass ADDS to that bound
by re-running `np.dot` dozens of times per ensemble member (3 stacked BLSTM
layers x 2 directions x 4 gates, x8 ensemble networks) -- so activation-level
ULP drift compounds well past the raw spectrogram stage's ~12-ULP worst
case, empirically measured here at up to 190 ULP (`float32` view-as-`int32`
bit-pattern distance) across the 3 usable test-wav cases. **The proof this
project's philosophy requires (CLAUDE.md: never label "approximately right"
as bit-identical) is `test_full_pipeline_is_exact_under_original_blas`
below**: this project's OWN code, executed under the reference
venv's numpy/scipy (`madmom-reference/.venv`, rebuilt 2026-07-12 to the same
recorded versions -- numpy 1.23.5, scipy 1.15.3 -- as the original,
now-gone `all-in-one-fix/.venv` that the committed fixtures were recorded
from) -- not just a
single matmul, the ENTIRE `RNNDownBeatProcessor` -> `DBNDownBeatTrackingProcessor`
pipeline -- reproduces real madmom's own recorded activations AND decoded
beat times with ZERO differing elements. That is the direct, executable
analogue of `test_spectrogram.py`'s
`test_filtered_spectrogram_algorithm_is_exact_under_original_blas`, scaled
up to the whole Phase-2 target.

Only 44.1kHz-native test wavs are used (`mono_44100`, `stereo_44100`,
`float32_44100`) -- `RNNDownBeatProcessor` hard-codes 44.1kHz and this
project has no resampling (`madmom_infer/audio/signal.py`'s header);
`stereo_48000.wav` is out of scope here for that reason, same as
`tools/generate_phase2_fixtures.py`.

**A FOURTH caching gotcha, found empirically while writing these tests
(same shape of bug as `test_spectrogram.py`'s documented STFT-window-caching
and filterbank-caching gotchas -- not a new bug, the SAME two Phase-1 bugs,
now visible one level up).** `RNNDownBeatProcessor.__init__` builds ONE
`ShortTimeFourierTransformProcessor()` and ONE `FilteredSpectrogramProcessor`
PER FRAME-SIZE BRANCH, but if ONE `RNNDownBeatProcessor` INSTANCE is reused
across multiple calls with DIFFERING-dtype input (e.g. `mono_44100.wav`,
int16, then `float32_44100.wav`, float32), those per-branch processors
silently keep the FIRST call's dtype-scaled FFT window / sample-rate-scoped
filterbank on every later call -- exactly the two bugs `test_stft.py`/
`test_filters.py` already pin, just triggered here via `RNNDownBeatProcessor`
reuse instead of a bare `ShortTimeFourierTransformProcessor`/
`FilteredSpectrogramProcessor` reuse. Confirmed empirically: a FRESH
`RNNDownBeatProcessor()` per wav gives a wildly different `float32_44100`
activation (max abs diff ~0.14, nowhere near ULP-scale) than a SHARED
instance processing `mono_44100` -> `stereo_44100` -> `float32_44100` in
that order. **`tools/generate_phase2_fixtures.py` reuses ONE
`RNNDownBeatProcessor`/`DBNDownBeatTrackingProcessor` pair across all 3
cases in exactly that order** (matching real madmom's own behavior, bug
included, per this project's golden-fixture mandate) -- so every test below
MUST replicate that exact call order/instance-reuse to compare against the
right numbers, the same discipline `test_spectrogram.py` already documents
for `FILTERBANK_CHAIN_CASES`. A fresh-processor-per-case test (the more
"obviously correct"-looking shape) would silently compare against the wrong
numbers.

`DBNDownBeatTrackingProcessor` decode is an integer/rational-domain
argmax-over-frames operation -- the task's stated expectation ("decode is
integer-domain, should absorb ULP noise") holds here empirically: despite
up to 190 ULP of activation-level drift, every decoded beat/downbeat time
in every case tested is EXACT (`np.array_equal`), including the empty-
decode `float32_44100` case (no beats above threshold in either
environment).

Reads: madmom_infer/features/downbeats.py (RNNDownBeatProcessor,
DBNDownBeatTrackingProcessor), madmom_infer/models.py,
tests/fixtures/rnn_downbeat.npz, tests/fixtures/wavs/*.wav
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.features.downbeats import (
    DBNDownBeatTrackingProcessor, RNNDownBeatProcessor,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

DBN_PARAMS = dict(beats_per_bar=[3, 4], fps=100)

# 44.1kHz-native cases only -- see module header (no resampling support).
RNN_CASES = ("mono_44100", "stereo_44100", "float32_44100")

# generous (~2.7x the worst observed, 190) but not unlimited -- see header
MAX_ULP = 512


@pytest.fixture(scope="module")
def _downbeats_blstm_ready():
    """Triggers the DOWNBEATS_BLSTM download/cache once per module, only
    when a test that actually needs it runs. Deliberately NOT module-level
    eager code: a network call at import time would run during test
    COLLECTION regardless of any `-m 'not network'` deselection, since
    pytest imports every test module before applying marker filters.
    Failure is a clean `pytest.skip`, never a collection error."""
    from madmom_infer.models import downbeats_blstm

    try:
        downbeats_blstm()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download DOWNBEATS_BLSTM weights: {exc}")


@pytest.fixture(scope="module")
def rnn_downbeat_fixture():
    return np.load(FIXTURES_DIR / "rnn_downbeat.npz")


@pytest.mark.network
def test_rnn_downbeat_activations_match_fixture_within_ulp(
    rnn_downbeat_fixture, _downbeats_blstm_ready
):
    """Deliberately ONE shared `RNNDownBeatProcessor` instance, processing
    all 3 cases IN ORDER -- see module header's "fourth caching gotcha":
    a fresh-instance-per-case version would silently compare against the
    wrong (uncontaminated-by-the-real-bug) numbers for `float32_44100`."""
    rnn = RNNDownBeatProcessor()
    for case in RNN_CASES:
        act = rnn(str(WAVS_DIR / f"{case}.wav"))
        expected = rnn_downbeat_fixture[f"{case}_activations"]
        assert act.shape == expected.shape, case
        assert act.dtype == expected.dtype, case
        np.testing.assert_array_max_ulp(act, expected, maxulp=MAX_ULP)


@pytest.mark.network
def test_end_to_end_beat_times_are_exact(rnn_downbeat_fixture, _downbeats_blstm_ready):
    """Despite activation-level ULP drift (previous test), the DECODED
    beat/downbeat times must be EXACT -- an integer-domain argmax-over-
    frames operation absorbs float32-ULP-scale input noise. Same shared-
    instance-in-order requirement as the activations test above."""
    rnn = RNNDownBeatProcessor()
    dbn = DBNDownBeatTrackingProcessor(**DBN_PARAMS)
    for case in RNN_CASES:
        act = rnn(str(WAVS_DIR / f"{case}.wav"))
        beats = np.asarray(dbn(act))
        expected_beats = rnn_downbeat_fixture[f"{case}_beat_times"]
        assert np.array_equal(beats, expected_beats), case


def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_full_pipeline_is_exact_under_original_blas(rnn_downbeat_fixture):
    """THE proof: this project's own `RNNDownBeatProcessor` ->
    `DBNDownBeatTrackingProcessor` code, run under the ORIGINAL reference
    venv's numpy/scipy build (the same environment real madmom's own
    recorded fixture came from), reproduces both the activations AND the
    decoded beat times with ZERO differing elements, for all 3 cases --
    proving the divergence measured by the two tests above is caused
    entirely by (already-known, Phase-1-proven) BLAS-library non-
    associativity, not by any algorithmic difference in this port's NN
    runtime, spectrogram cascade, or unpickling. Runs all 3 cases through
    ONE shared `rnn`/`dbn` pair, in order, in a SINGLE subprocess -- same
    instance-reuse requirement as the two in-process tests above (see
    module header).
    """
    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in RNN_CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.features.downbeats import (
    DBNDownBeatTrackingProcessor, RNNDownBeatProcessor,
)

cases = {list(RNN_CASES)!r}
wav_paths = [{case_paths}]
rnn = RNNDownBeatProcessor()
dbn = DBNDownBeatTrackingProcessor(**{DBN_PARAMS!r})
fixture = np.load({str(FIXTURES_DIR / "rnn_downbeat.npz")!r})

for case, wav_path in zip(cases, wav_paths):
    act = rnn(wav_path)
    beats = np.asarray(dbn(act))
    expected_act = fixture[case + "_activations"]
    expected_beats = fixture[case + "_beat_times"]
    assert np.array_equal(act, expected_act), f"{{case}}: activations differ"
    assert np.array_equal(beats, expected_beats), f"{{case}}: beat times differ"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout


# ---------------------------------------------------------------------------
# Wave 4c: SyncronizeFeaturesProcessor + RNNBarProcessor's GRU ensembles
# (GRULayer/GRUCell, DOWNBEATS_BGRU) -- see madmom_infer/features/
# downbeats.py's module header for why RNNBarProcessor's fixture is
# INTERMEDIATE (perc_synced/harm_synced), not full audio-in: this port's
# own RNNBarProcessor cannot be instantiated until Wave 4d ports
# CLPChromaProcessor, so what's proven bit-exact here is the actual "does
# the GRU port work" question -- the DOWNBEATS_BGRU NeuralNetworkEnsemble
# forward pass itself, fed real madmom's own captured intermediate
# beat-synchronized features.
# ---------------------------------------------------------------------------
UPSTREAM_DOWNBEATS_DIR = (
    REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "downbeats" / "2016"
)


@pytest.fixture(scope="module")
def sync_features_fixture():
    return np.load(FIXTURES_DIR / "sync_features.npz")


def test_syncronize_features_processor_matches_fixture(sync_features_fixture):
    """Pure numpy, no NN weights -- fully offline (see
    madmom_infer/features/downbeats.py's SyncronizeFeaturesProcessor
    docstring)."""
    from madmom_infer.features.downbeats import SyncronizeFeaturesProcessor

    proc = SyncronizeFeaturesProcessor(4, fps=100)
    out = proc((sync_features_fixture["sync_features_input"],
                sync_features_fixture["sync_beats_input"]))
    np.testing.assert_array_equal(
        out, sync_features_fixture["sync_features_output"])


def test_syncronize_features_processor_empty_beats():
    from madmom_infer.features.downbeats import SyncronizeFeaturesProcessor

    proc = SyncronizeFeaturesProcessor(4, fps=100)
    features, beats = proc((np.zeros((10, 3)), np.array([])))
    assert features.size == 0
    assert beats.size == 0


@pytest.fixture(scope="module")
def bgru_intermediate_fixture():
    return np.load(FIXTURES_DIR / "downbeats_bgru_intermediate.npz")


@pytest.fixture(scope="module")
def _downbeats_bgru_ready():
    """Downloads (or reuses the local cache for) both DOWNBEATS_BGRU
    ensembles. Deliberately NOT module-level eager code -- see this file's
    `_downbeats_blstm_ready` fixture for why."""
    from madmom_infer.models import downbeats_bgru

    try:
        return downbeats_bgru()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download DOWNBEATS_BGRU weights: {exc}")


@pytest.mark.network
def test_downbeats_bgru_ensembles_match_fixture_within_ulp(
    bgru_intermediate_fixture, _downbeats_bgru_ready
):
    """Feed real madmom's own captured `perc_synced`/`harm_synced`
    intermediate features directly into THIS PORT's `perc_nn`/`harm_nn`
    `NeuralNetworkEnsemble`s (built from this wave's new `GRULayer`/
    `GRUCell`) and compare to real madmom's own `perc_nn_out`/`harm_nn_out`
    -- proves the GRU forward pass itself, independent of the still-missing
    `CLPChromaProcessor` harmonic-feature extraction (see module header)."""
    from madmom_infer.ml.nn import NeuralNetworkEnsemble

    bgru = _downbeats_bgru_ready
    perc_nn = NeuralNetworkEnsemble.load(bgru[0])
    harm_nn = NeuralNetworkEnsemble.load(bgru[1])

    perc_synced = bgru_intermediate_fixture["perc_synced"]
    harm_synced = bgru_intermediate_fixture["harm_synced"]
    perc_out = perc_nn(perc_synced.reshape((len(perc_synced), -1)))
    harm_out = harm_nn(harm_synced.reshape((len(harm_synced), -1)))

    expected_perc = bgru_intermediate_fixture["perc_nn_out"]
    expected_harm = bgru_intermediate_fixture["harm_nn_out"]
    assert perc_out.shape == expected_perc.shape
    assert perc_out.dtype == expected_perc.dtype == np.float64
    assert harm_out.shape == expected_harm.shape
    assert harm_out.dtype == expected_harm.dtype == np.float64
    # NOTE: relative tolerance, not `assert_array_max_ulp`, for THIS
    # particular fixture -- see madmom_infer/ml/nn/__init__.py's
    # `average_predictions` module header: `perc_synced`/`harm_synced` here
    # happen to be a single beat-sync WINDOW (mono_44100.wav is short, only
    # 2 beats detected), so each ensemble's per-network outputs are 0-d
    # scalars, and real madmom's own `average_predictions` genuinely
    # promotes THOSE to float64 (a real, confirmed quirk -- see that
    # module's header) even though every underlying per-network forward
    # pass was computed at float32 precision throughout. Comparing float64
    # bit-patterns via `assert_array_max_ulp` on a value whose actual
    # numerical CONTENT is only float32-precision is the wrong lens (a
    # float64-view ULP count of ~4e8 corresponds to a relative error of
    # ~7e-8, i.e. approximately ONE float32 ULP) -- `rtol` scoped to
    # float32 precision is the honest comparison. The exact-equality claim
    # is instead made by `test_downbeats_bgru_ensembles_are_exact_under_
    # original_blas` below (reference-venv subprocess, ZERO differing
    # elements, not a tolerance).
    np.testing.assert_allclose(perc_out, expected_perc, rtol=1e-5)
    np.testing.assert_allclose(harm_out, expected_harm, rtol=1e-5)


def _upstream_downbeats_bgru_available():
    return (
        UPSTREAM_DOWNBEATS_DIR.exists()
        and (UPSTREAM_DOWNBEATS_DIR / "downbeats_bgru_rhythmic_0.pkl").exists()
        and (UPSTREAM_DOWNBEATS_DIR / "downbeats_bgru_harmonic_0.pkl").exists()
    )


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
@pytest.mark.skipif(
    not _upstream_downbeats_bgru_available(),
    reason="local ../madmom-upstream/madmom/models/downbeats checkout not "
           "found; the cross-BLAS proof needs it (no network required this "
           "way, direct .pkl paths)",
)
def test_downbeats_bgru_ensembles_are_exact_under_original_blas():
    """THE GRU proof: this port's own `GRULayer`/`GRUCell` (inside a
    `NeuralNetworkEnsemble` built from the real `DOWNBEATS_BGRU` weights),
    run under the ORIGINAL reference venv's numpy/scipy build, reproduces
    real madmom's `perc_nn`/`harm_nn` outputs on the SAME captured
    intermediate features with ZERO differing elements. Uses the local
    `../madmom-upstream` `.pkl` copies directly, no network needed.
    """
    rhythmic_paths = [
        str(UPSTREAM_DOWNBEATS_DIR / f"downbeats_bgru_rhythmic_{i}.pkl")
        for i in range(6)
    ]
    harmonic_paths = [
        str(UPSTREAM_DOWNBEATS_DIR / f"downbeats_bgru_harmonic_{i}.pkl")
        for i in range(6)
    ]
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.ml.nn import NeuralNetworkEnsemble

fixture = np.load({str(FIXTURES_DIR / "downbeats_bgru_intermediate.npz")!r})
perc_nn = NeuralNetworkEnsemble.load({rhythmic_paths!r})
harm_nn = NeuralNetworkEnsemble.load({harmonic_paths!r})

perc_synced = fixture["perc_synced"]
harm_synced = fixture["harm_synced"]
perc_out = perc_nn(perc_synced.reshape((len(perc_synced), -1)))
harm_out = harm_nn(harm_synced.reshape((len(harm_synced), -1)))

assert np.array_equal(perc_out, fixture["perc_nn_out"]), "perc_nn differs"
assert np.array_equal(harm_out, fixture["harm_nn_out"]), "harm_nn differs"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout


# ---------------------------------------------------------------------------
# GRULayer/GRUCell unpickling structural correctness (network -- needs the
# real, older-format .pkl bytes, see madmom_infer/ml/nn/unpickle.py's
# module header for the copy_reg._reconstructor/__builtin__.object finding)
# ---------------------------------------------------------------------------
def digest_layer(layer):
    """Independent reimplementation of
    tools/generate_beat_tempo_fixtures.py's digest_layer -- deliberately
    not imported from tools/, same discipline as this repo's other
    *_structurally_matches_real_madmom tests."""
    import hashlib

    def _arr_digest(arr):
        arr = np.ascontiguousarray(arr)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }

    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if t in ("Gate", "Cell", "GRUCell", "RecurrentLayer"):
        if hasattr(layer, "recurrent_weights"):
            d["recurrent_weights"] = _arr_digest(layer.recurrent_weights)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    if t == "BidirectionalLayer":
        d["fwd_layer"] = digest_layer(layer.fwd_layer)
        d["bwd_layer"] = digest_layer(layer.bwd_layer)
    elif t == "GRULayer":
        d["reset_gate"] = digest_layer(layer.reset_gate)
        d["update_gate"] = digest_layer(layer.update_gate)
        d["cell"] = digest_layer(layer.cell)
    return d


@pytest.fixture(scope="module")
def bgru_structural_digest_fixture():
    import json

    with open(FIXTURES_DIR / "downbeats_bgru_structural_digest.json") as fh:
        return json.load(fh)


@pytest.mark.network
def test_unpickled_downbeats_bgru_structurally_matches_real_madmom(
    bgru_structural_digest_fixture, _downbeats_bgru_ready
):
    import warnings

    from madmom_infer.ml.nn.unpickle import load_model

    bgru = _downbeats_bgru_ready
    with warnings.catch_warnings():
        # this port's own GRULayer.__setstate__ also warns on this legacy
        # pickle format, matching real madmom exactly (see
        # madmom_infer/ml/nn/layers.py's module header) -- expected here.
        warnings.simplefilter("ignore")
        nn_rhythmic = load_model(bgru[0][0])
        nn_harmonic = load_model(bgru[1][0])
    ours_rhythmic = [digest_layer(l) for l in nn_rhythmic.layers]
    ours_harmonic = [digest_layer(l) for l in nn_harmonic.layers]
    assert ours_rhythmic == bgru_structural_digest_fixture["downbeats_bgru_rhythmic_0"]
    assert ours_harmonic == bgru_structural_digest_fixture["downbeats_bgru_harmonic_0"]
