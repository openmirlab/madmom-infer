"""Golden-fixture tests for `madmom_infer.audio.cepstrogram` -- Wave 4g's
port of `madmom.audio.cepstrogram` (`Cepstrogram`, `CepstrogramProcessor`,
`MFCC`, `MFCCProcessor`). Fixtures recorded by `tools/generate_leftovers_
fixtures.py` from real (compiled) madmom on a real spectrogram of
`mono_44100.wav`.

**`Cepstrogram` is proven EXACTLY equal (`np.array_equal`), both in-process
AND cross-BLAS** -- confirmed empirically: `scipy.fftpack.dct` (the default
transform) produces bit-identical output across this project's dev venv
(numpy 2.4.6, scipy 1.17.1) and the reference venv (numpy 1.23.5, scipy
1.15.3).

**`MFCC` is proven exact CROSS-BLAS (`np.array_equal`, zero differing
elements, confirmed directly against the reference venv), but shows small
IN-PROCESS drift** (up to ~3.8e-6 absolute on a roughly [-25, 25]-range
output, max 8192 "ULP" via int32 bit-pattern view) when run under this
project's OWN numpy/scipy build -- root-caused to the compounding `np.dot`
(filterbank) -> `np.log10` -> `dct` chain amplifying ordinary float32
last-bit rounding differences between numpy/scipy builds, same class of
finding as `audio/spectrogram.py`'s `SemitoneBandpassSpectrogram` (Wave 4d)
and `features/notes.py`'s raw (non-probability) RNN activations (Wave 4e) --
NOT a logic bug in this port (the cross-BLAS test, run under the exact
numpy/scipy build that recorded the fixture, is bit-identical). The in-
process test therefore asserts a documented absolute tolerance
(`atol=1e-5`, ~2.6x the ~3.8e-6 observed) rather than an unstable near-zero
ULP metric -- most of `MFCC`'s output floats near zero, where a handful of
sign-crossing elements would otherwise dominate a raw ULP-distance metric,
exactly the class of instability documented for the two prior waves' own
"raw, not-a-probability" outputs.

**Major real, confirmed upstream bug in `MFCC` -- ported bug-for-bug, not
fixed**: see `madmom_infer/audio/cepstrogram.py`'s module header. `MFCC`
can only be constructed from an ALREADY-`FilteredSpectrogram` instance;
every other input (a plain `Spectrogram`, a `LogarithmicSpectrogram`, or a
raw wav path/array) raises `AttributeError` in real madmom 0.17.dev0,
confirmed directly against the reference venv. This is pinned by
`pytest.raises`, not a golden-output fixture, and this file's fixture-
exactness tests use the one input shape that actually works.

Reads: madmom_infer/audio/cepstrogram.py, tests/fixtures/cepstrogram.npz.
"""

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.cepstrogram import (
    MFCC, Cepstrogram, CepstrogramProcessor, MFCCProcessor,
)
from madmom_infer.audio.filters import LogarithmicFilterbank
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor, SpectrogramProcessor,
)
from madmom_infer.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
WAV_PATH = FIXTURES_DIR / "wavs" / "mono_44100.wav"
REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)
MFCC_ATOL = 1e-5


@pytest.fixture(scope="module")
def fixture():
    return np.load(FIXTURES_DIR / "cepstrogram.npz")


@pytest.fixture(scope="module")
def spectrogram():
    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=2048, fps=100)
    stft = ShortTimeFourierTransformProcessor()
    spec = SpectrogramProcessor()
    return spec(stft(frames(sig(str(WAV_PATH)))))


@pytest.fixture(scope="module")
def filtered_spectrogram(spectrogram):
    return FilteredSpectrogramProcessor(
        filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
        norm_filters=True,
    )(spectrogram)


# ---------------------------------------------------------------------------
# structural / hand-written sanity checks
# ---------------------------------------------------------------------------
def test_cepstrogram_array_interop(spectrogram):
    cep = Cepstrogram(spectrogram)
    assert cep.bin_frequencies is None
    assert cep.num_frames == spectrogram.num_frames
    assert cep.num_bins == spectrogram.num_bins
    assert len(cep) == len(spectrogram)
    np.testing.assert_array_equal(np.asarray(cep)[0], cep[0])


def test_cepstrogram_processor_matches_direct_construction(spectrogram):
    proc = CepstrogramProcessor()
    out = proc.process(spectrogram)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(Cepstrogram(spectrogram)))


def test_mfcc_rejects_non_filterbank_type(filtered_spectrogram):
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MFCC(filtered_spectrogram, filterbank=object())


def test_mfcc_from_plain_spectrogram_raises_attribute_error(spectrogram):
    """Faithful bug-for-bug reproduction: a plain `Spectrogram` has no
    `.filterbank` attribute -- confirmed against the reference venv, see
    this module's header."""
    with pytest.raises(AttributeError):
        MFCC(spectrogram)


def test_mfcc_from_wav_path_raises_attribute_error():
    """Same bug as above: a raw wav-path argument builds a plain
    `Spectrogram` internally, hitting the identical `AttributeError`."""
    with pytest.raises(AttributeError):
        MFCC(str(WAV_PATH), sample_rate=44100, num_channels=1)


def test_mfcc_from_filtered_spectrogram_warns_and_recomputes(filtered_spectrogram):
    """The one input shape that doesn't crash: an already-`FilteredSpectrogram`
    trips the "redo calculation" warning, matching upstream."""
    with pytest.warns(UserWarning, match="already"):
        mfcc = MFCC(filtered_spectrogram)
    assert mfcc.shape[1] == 30  # MFCC_BANDS default


def test_mfcc_processor_ignores_stored_transform(filtered_spectrogram):
    """Verbatim-ported upstream oversight: `MFCCProcessor.process()` never
    forwards its own stored `self.transform` to `MFCC(...)` -- see this
    module's header."""
    custom_transform_calls = []

    def spy_transform(data):
        custom_transform_calls.append(data)
        return data * 2

    proc = MFCCProcessor(transform=spy_transform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc.process(filtered_spectrogram)
    assert custom_transform_calls == []  # never invoked -- inert, as upstream


# ---------------------------------------------------------------------------
# real-madmom-fixture exactness
# ---------------------------------------------------------------------------
def test_spectrogram_matches_fixture_exactly(fixture, spectrogram):
    np.testing.assert_array_equal(
        np.asarray(spectrogram), fixture["spectrogram_input"])


def test_cepstrogram_matches_fixture_exactly(fixture, spectrogram):
    cep = Cepstrogram(spectrogram)
    np.testing.assert_array_equal(np.asarray(cep), fixture["cepstrogram_default"])


def test_mfcc_default_matches_fixture(fixture, filtered_spectrogram):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = MFCC(filtered_spectrogram)
    np.testing.assert_allclose(
        np.asarray(mfcc), fixture["mfcc_default"], atol=MFCC_ATOL, rtol=0)


def test_mfcc_custom_matches_fixture(fixture, filtered_spectrogram):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = MFCC(filtered_spectrogram, num_bands=13, fmin=20.0,
                    fmax=8000.0, norm_filters=False)
    np.testing.assert_allclose(
        np.asarray(mfcc), fixture["mfcc_custom"], atol=MFCC_ATOL, rtol=0)


# ---------------------------------------------------------------------------
# cross-BLAS exactness -- MFCC is proven BIT-IDENTICAL here (see module
# header for why the in-process test above uses a tolerance instead).
# ---------------------------------------------------------------------------
def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine",
)
def test_cepstrogram_and_mfcc_are_exact_under_original_blas():
    """This port's own `Cepstrogram`/`MFCC`, run under the reference venv's
    numpy/scipy build, reproduce real madmom's fixture values with ZERO
    differing elements -- confirms the in-process MFCC drift above is a
    numpy/scipy-build artifact, not a logic difference."""
    script = f"""
import sys, warnings
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.cepstrogram import MFCC, Cepstrogram
from madmom_infer.audio.filters import LogarithmicFilterbank
from madmom_infer.audio.spectrogram import (
    FilteredSpectrogramProcessor, SpectrogramProcessor,
)
from madmom_infer.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom_infer.audio.stft import ShortTimeFourierTransformProcessor

fixture = np.load({str(FIXTURES_DIR / "cepstrogram.npz")!r})
sig = SignalProcessor(num_channels=1, sample_rate=44100)
frames = FramedSignalProcessor(frame_size=2048, fps=100)
stft = ShortTimeFourierTransformProcessor()
spec_proc = SpectrogramProcessor()
spec = spec_proc(stft(frames(sig({str(WAV_PATH)!r}))))
assert np.array_equal(np.asarray(spec), fixture["spectrogram_input"])

cep = Cepstrogram(spec)
assert np.array_equal(np.asarray(cep), fixture["cepstrogram_default"])

filt_spec = FilteredSpectrogramProcessor(
    filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
    norm_filters=True,
)(spec)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mfcc_default = MFCC(filt_spec)
    mfcc_custom = MFCC(filt_spec, num_bands=13, fmin=20.0, fmax=8000.0,
                       norm_filters=False)
assert np.array_equal(np.asarray(mfcc_default), fixture["mfcc_default"])
assert np.array_equal(np.asarray(mfcc_custom), fixture["mfcc_custom"])
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
