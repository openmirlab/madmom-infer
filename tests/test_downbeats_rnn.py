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
below**: this project's OWN code, executed under the ORIGINAL reference
venv's numpy/scipy (`all-in-one-fix/.venv`, numpy 1.23.5) -- not just a
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
    "/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python"
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
    reason="reference madmom install (all-in-one-fix/.venv) not found on "
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
