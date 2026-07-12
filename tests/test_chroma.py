"""Golden-fixture tests for Wave 4d's `madmom_infer.audio.chroma` module --
all three chroma paths: classic (`PitchClassProfile`/
`HarmonicPitchClassProfile`), CLP (`CLPChroma`/`CLPChromaProcessor`, via
`audio/spectrogram.py`'s `SemitoneBandpassSpectrogram`), and DNN
(`DeepChromaProcessor`), all recorded by `tools/generate_chroma_chord_fixtures.py`
from real (compiled) madmom.

**Precision claims, measured not assumed** (see `audio/chroma.py`'s module
header for the full story):

- Classic chroma (PCP/HPCP): pure linear filterbank ops on top of an
  already-golden-fixture-proven `Spectrogram` -- in-process ULP drift
  measured at up to 5 ULP (`float32` view-as-`int32` bit-pattern distance)
  across the 3 usable test-wav cases; asserted here with a 16-ULP margin
  (~3x observed), same convention as `test_key.py`/`test_onsets.py`.
- CLP chroma: NOT bit-identical, NOT mere ULP drift -- `scipy.signal.
  filtfilt`'s recursive nature amplifies tiny per-scipy-version `ellip()`
  filter-coefficient differences (this project's dev venv: scipy 1.17.1;
  the reference venv: scipy 1.15.3) into an absolute difference measured up
  to ~1e-5 across the 3 cases; asserted here with `atol=1e-4` (~10x
  observed), `rtol=0` (chroma values include near-zero bins where a
  relative bound would be meaningless). Downstream, `RNNBarProcessor`'s
  DECODED bar-relative activation still matches to within ~4e-8 (see
  `tests/test_downbeats_rnn.py`'s "RNNBarProcessor end-to-end (Wave 4d)"
  section) -- this level of upstream feature noise turns out not to move
  the needle on the actual decoded numbers for the cases tested.
- DeepChromaProcessor: a real NN forward pass -- in-process ULP drift
  measured at up to 24 ULP across the 3 cases; asserted here with a
  128-ULP margin (~5x observed), same order of magnitude as `test_key.py`'s
  CNN margin.

Reads: madmom_infer/audio/chroma.py, madmom_infer/audio/spectrogram.py
(SemitoneBandpassSpectrogram), tests/fixtures/chroma_classic.npz,
tests/fixtures/clp_chroma.npz, tests/fixtures/chroma_dnn_structural_digest.json,
tests/fixtures/chroma_dnn_activations.npz
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from madmom_infer.audio.chroma import (
    CLPChroma, CLPChromaProcessor, DeepChromaProcessor,
    HarmonicPitchClassProfile, PitchClassProfile,
)
from madmom_infer.audio.signal import Signal
from madmom_infer.audio.spectrogram import SemitoneBandpassSpectrogram, Spectrogram

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"
REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PYTHON = Path(
    "/home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python"
)

# 44.1kHz-native cases only -- see module header (no file-load resampling).
CASES = ("mono_44100", "stereo_44100", "float32_44100")

CLASSIC_MAX_ULP = 16
DNN_MAX_ULP = 128
CLP_ATOL = 1e-4


def _ulp(a, b):
    """`float32` view-as-int32 signed-magnitude bit-pattern distance --
    same helper this project's other cross-numpy-version tests use."""
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(0x80000000) - ai, ai)
    bi = np.where(bi < 0, np.int64(0x80000000) - bi, bi)
    return np.max(np.abs(ai - bi))


# ---------------------------------------------------------------------------
# 1. Classic chroma (PitchClassProfile / HarmonicPitchClassProfile) --
#    offline (no model weights, only the shared test wavs)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chroma_classic_fixture():
    return np.load(FIXTURES_DIR / "chroma_classic.npz")


@pytest.mark.parametrize("case", CASES)
def test_pitch_class_profile_matches_fixture(chroma_classic_fixture, case):
    sig = Signal(str(WAVS_DIR / f"{case}.wav"), num_channels=1, sample_rate=44100)
    spec = Spectrogram(sig, frame_size=2048, fps=100)
    pcp = PitchClassProfile(spec)
    expected = chroma_classic_fixture[f"{case}_pcp"]
    assert np.asarray(pcp).shape == expected.shape
    assert np.asarray(pcp).dtype == expected.dtype
    np.testing.assert_array_max_ulp(np.asarray(pcp), expected, maxulp=CLASSIC_MAX_ULP)


@pytest.mark.parametrize("case", CASES)
def test_harmonic_pitch_class_profile_matches_fixture(chroma_classic_fixture, case):
    sig = Signal(str(WAVS_DIR / f"{case}.wav"), num_channels=1, sample_rate=44100)
    spec = Spectrogram(sig, frame_size=2048, fps=100)
    hpcp = HarmonicPitchClassProfile(spec)
    expected = chroma_classic_fixture[f"{case}_hpcp"]
    assert np.asarray(hpcp).shape == expected.shape
    assert np.asarray(hpcp).dtype == expected.dtype
    np.testing.assert_array_max_ulp(np.asarray(hpcp), expected, maxulp=CLASSIC_MAX_ULP)


def test_pitch_class_profile_warns_on_already_filtered_spectrogram():
    from madmom_infer.audio.spectrogram import FilteredSpectrogram

    sig = Signal(str(WAVS_DIR / "mono_44100.wav"), num_channels=1,
                sample_rate=44100)
    spec = Spectrogram(sig, frame_size=2048, fps=100)
    filtered = FilteredSpectrogram(spec, num_bands=12)
    with pytest.warns(RuntimeWarning, match="should not be filtered"):
        PitchClassProfile(filtered)


def test_pitch_class_profile_fref_none_raises_not_implemented():
    sig = Signal(str(WAVS_DIR / "mono_44100.wav"), num_channels=1,
                sample_rate=44100)
    spec = Spectrogram(sig, frame_size=2048, fps=100)
    with pytest.raises(NotImplementedError):
        PitchClassProfile(spec, fref=None)


# ---------------------------------------------------------------------------
# 2. CLP chroma (SemitoneBandpassSpectrogram / CLPChroma / CLPChromaProcessor)
#    -- offline, needs ffmpeg on PATH (see audio/signal.py's resample header)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clp_chroma_fixture():
    return np.load(FIXTURES_DIR / "clp_chroma.npz")


def _ffmpeg_available():
    import shutil

    return shutil.which("ffmpeg") is not None


pytestmark_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg binary not found on PATH -- required by "
           "SemitoneBandpassFilterbank's internal resample, see "
           "madmom_infer/audio/signal.py's module header",
)


@pytestmark_ffmpeg
@pytest.mark.parametrize("case", CASES)
def test_semitone_bandpass_spectrogram_matches_fixture(clp_chroma_fixture, case):
    sbs = SemitoneBandpassSpectrogram(str(WAVS_DIR / f"{case}.wav"), fps=50,
                                      fmin=27.5, fmax=4200.0)
    expected = clp_chroma_fixture[f"{case}_semitone_bandpass"]
    expected_bins = clp_chroma_fixture[f"{case}_semitone_bin_frequencies"]
    assert np.asarray(sbs).shape == expected.shape
    np.testing.assert_allclose(np.asarray(sbs), expected, rtol=0, atol=CLP_ATOL)
    np.testing.assert_array_equal(sbs.bin_frequencies, expected_bins)


@pytestmark_ffmpeg
@pytest.mark.parametrize("case", CASES)
def test_clp_chroma_matches_fixture(clp_chroma_fixture, case):
    clp = CLPChromaProcessor(fps=50, fmin=27.5, fmax=4200.0,
                             compression_factor=100, norm=True,
                             threshold=0.001)
    out = clp(str(WAVS_DIR / f"{case}.wav"))
    expected = clp_chroma_fixture[f"{case}_clp_chroma"]
    assert np.asarray(out).shape == expected.shape
    np.testing.assert_allclose(np.asarray(out), expected, rtol=0, atol=CLP_ATOL)
    assert out.bin_labels == ["C", "C#", "D", "D#", "E", "F", "F#", "G",
                              "G#", "A", "A#", "B"]


@pytestmark_ffmpeg
def test_clp_chroma_accepts_prebuilt_semitone_bandpass_spectrogram():
    """`CLPChroma(data, ...)` must not re-run `SemitoneBandpassSpectrogram`
    if `data` already is one -- verbatim port of upstream's own `isinstance`
    short-circuit (`chroma.py:335-338`)."""
    sbs = SemitoneBandpassSpectrogram(str(WAVS_DIR / "mono_44100.wav"), fps=50,
                                      fmin=27.5, fmax=4200.0)
    clp_direct = CLPChroma(sbs, fps=50, fmin=27.5, fmax=4200.0)
    clp_indirect = CLPChroma(str(WAVS_DIR / "mono_44100.wav"), fps=50,
                             fmin=27.5, fmax=4200.0)
    np.testing.assert_array_equal(np.asarray(clp_direct), np.asarray(clp_indirect))


def test_resample_requires_ffmpeg_binary(monkeypatch):
    """`resample()` raises a clear, actionable `RuntimeError` (not a
    confusing `FileNotFoundError` from a failed subprocess spawn) when
    `ffmpeg` isn't on PATH -- see audio/signal.py's module header."""
    import shutil

    from madmom_infer.audio.signal import Signal as _Signal
    from madmom_infer.audio.signal import resample

    monkeypatch.setattr(shutil, "which", lambda name: None)
    sig = _Signal(np.zeros(100, dtype=np.int16), sample_rate=44100)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        resample(sig, 22050)


def test_resample_same_rate_is_a_noop():
    from madmom_infer.audio.signal import Signal as _Signal
    from madmom_infer.audio.signal import resample

    sig = _Signal(np.zeros(100, dtype=np.int16), sample_rate=44100)
    assert resample(sig, 44100) is sig


# ---------------------------------------------------------------------------
# 3. DeepChromaProcessor -- structural digest + end-to-end activations
#    (network -- needs the real chroma_dnn.pkl bytes)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_paths():
    from madmom_infer.models import chroma_dnn

    try:
        return chroma_dnn()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download CHROMA_DNN weights: {exc}")


def _arr_digest(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer):
    """Independent reimplementation of
    tools/generate_chroma_chord_fixtures.py's digest_layer -- deliberately
    not imported from tools/, same discipline as this repo's other
    *_structurally_matches_real_madmom tests."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    return d


@pytest.fixture(scope="module")
def structural_digest_fixture():
    with open(FIXTURES_DIR / "chroma_dnn_structural_digest.json") as fh:
        return json.load(fh)


@pytest.mark.network
def test_unpickled_chroma_dnn_structurally_matches_real_madmom(
    structural_digest_fixture, model_paths
):
    from madmom_infer.ml.nn.unpickle import load_model

    assert len(model_paths) == 1
    nn = load_model(model_paths[0])
    ours = [digest_layer(l) for l in nn.layers]
    expected = structural_digest_fixture["chroma_dnn"]
    assert ours == expected


@pytest.fixture(scope="module")
def _chroma_dnn_ready():
    from madmom_infer.models import chroma_dnn

    try:
        chroma_dnn()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download CHROMA_DNN weights: {exc}")


@pytest.fixture(scope="module")
def chroma_dnn_activations_fixture():
    return np.load(FIXTURES_DIR / "chroma_dnn_activations.npz")


@pytest.mark.network
@pytest.mark.parametrize("case", CASES)
def test_deep_chroma_processor_matches_fixture_within_ulp(
    chroma_dnn_activations_fixture, _chroma_dnn_ready, case
):
    proc = DeepChromaProcessor()
    chroma = proc(str(WAVS_DIR / f"{case}.wav"))
    expected = chroma_dnn_activations_fixture[f"{case}_chroma"]
    assert chroma.shape == expected.shape
    assert chroma.dtype == expected.dtype
    np.testing.assert_array_max_ulp(chroma, expected, maxulp=DNN_MAX_ULP)


def _reference_python_available():
    return REFERENCE_PYTHON.exists()


@pytest.mark.skipif(
    not _reference_python_available(),
    reason="reference madmom install (madmom-reference/.venv) not found on "
           "this machine; the cross-BLAS proof requires it",
)
def test_deep_chroma_processor_is_exact_under_original_blas():
    """This port's own `DeepChromaProcessor`, run under the ORIGINAL
    reference venv's numpy/scipy build, reproduces real madmom's chroma
    activations with ZERO differing elements, for all 3 cases. Uses the
    local `chroma_dnn.pkl` copy under `../madmom-upstream` directly, no
    network needed."""
    upstream_chroma_dnn = (
        REPO_ROOT.parent / "madmom-upstream" / "madmom" / "models" / "chroma"
        / "2016" / "chroma_dnn.pkl"
    )
    if not upstream_chroma_dnn.exists():
        pytest.skip(f"local chroma_dnn.pkl not found at {upstream_chroma_dnn}")

    case_paths = ", ".join(repr(str(WAVS_DIR / f"{c}.wav")) for c in CASES)
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
import numpy as np
from madmom_infer.audio.chroma import DeepChromaProcessor

cases = {list(CASES)!r}
wav_paths = [{case_paths}]
fixture = np.load({str(FIXTURES_DIR / "chroma_dnn_activations.npz")!r})

for case, wav_path in zip(cases, wav_paths):
    proc = DeepChromaProcessor(models=[{str(upstream_chroma_dnn)!r}])
    chroma = proc(wav_path)
    expected = fixture[case + "_chroma"]
    assert np.array_equal(chroma, expected), f"{{case}}: chroma differs"
print("EXACT_MATCH")
"""
    proc = subprocess.run(
        [str(REFERENCE_PYTHON), "-c", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EXACT_MATCH" in proc.stdout
