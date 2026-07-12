"""Wave-4f golden-fixture generator: `features/beats_crf.py`'s numpy CRF
Viterbi port (function-level fixtures + `CRFBeatDetectionProcessor`
end-to-end), `BeatTrackingProcessor`/`BeatDetectionProcessor` end-to-end
decoded beat times, `ml/gmm.py`'s `GMM` (score/posterior fixtures against
the real `PATTERNS_BALLROOM` GMMs), and `PatternTrackingProcessor` end-to-end
(audio -> beat+downbeat positions) -- the 4f sibling of `tools/
generate_beat_tempo_fixtures.py`, same conventions (own file, own fixture
files, independently regenerable without touching prior waves' already-
committed fixtures).

HOW TO RUN -- same real-madmom reference venv as every prior wave
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_crf_pattern_fixtures.py

Reuses the same 44.1kHz-native test-wav subset established by prior waves
(`mono_44100`, `stereo_44100`, `float32_44100`) for the model-dependent
(RNN activation + decoded beat/downbeat position) fixtures. The CRF/GMM
function-level fixtures are self-contained (record both a representative
INPUT and real madmom's OUTPUT), matching `generate_beat_tempo_fixtures.py`'s
own "several 4c targets are pure-numpy functions with no dependency on this
project's own DSP chain" precedent -- `tests/test_beats_crf.py`/
`tests/test_gmm.py` can run those cases fully offline.

**Same "shared-instance-in-order" caching-gotcha discipline as prior waves'
generators** (see e.g. `generate_beat_tempo_fixtures.py`'s module header):
`RNNBeatProcessor` builds one `ShortTimeFourierTransformProcessor`/
`FilteredSpectrogramProcessor` PER FRAME-SIZE BRANCH inside `__init__` and
reuses them across calls -- this generator builds ONE `RNNBeatProcessor
(online=False)` instance and processes all 3 cases through it, IN ORDER
(`mono_44100` -> `stereo_44100` -> `float32_44100`); `tests/test_beats.py`-
style end-to-end tests below (and `tests/test_beats_crf.py`) must replicate
that exact call order/instance reuse.

Reads: real `madmom` (features.beats, features.beats_crf, features.downbeats,
ml.gmm, models.BEATS_LSTM/BEATS_BLSTM/PATTERNS_BALLROOM, audio.spectrogram),
numpy. Writes: tests/fixtures/beats_crf_functions.npz, tests/fixtures/
beats_crf_end_to_end.npz, tests/fixtures/beat_tracking_end_to_end.npz,
tests/fixtures/gmm_scores.npz, tests/fixtures/patterns_structural_digest.json,
tests/fixtures/pattern_tracking_end_to_end.npz. Read by: tests/
test_beats_crf.py, tests/test_beats.py, tests/test_gmm.py,
tests/test_patterns.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

# 44.1kHz-native cases only -- see module header (no resampling support).
BEAT_CASES = {
    "mono_44100": "mono_44100.wav",
    "stereo_44100": "stereo_44100.wav",
    "float32_44100": "float32_44100.wav",
}
DSP_CASE_WAV = "mono_44100.wav"


def _arr_digest(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def generate_beats_crf_function_fixtures() -> dict:
    """Direct function-level fixtures for `initial_distribution`,
    `transition_distribution`, `normalisation_factors`, `best_sequence` --
    fed a REAL beat activation function (`RNNBeatProcessor(online=False)`'s
    output on `mono_44100.wav`, same technique
    `generate_beat_tempo_fixtures.py`'s comb-filter fixtures use), for
    several representative `(interval, interval_sigma)` combinations."""
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.beats_crf import (
        best_sequence, initial_distribution, normalisation_factors,
        transition_distribution,
    )

    act = RNNBeatProcessor(online=False)(str(WAVS_DIR / DSP_CASE_WAV))
    out = {"crf_input_activations": act}

    for interval in (20, 35, 55):
        sigma = 0.18
        init = initial_distribution(act.shape[0], interval)
        trans = transition_distribution(interval, sigma)
        norm_fact = normalisation_factors(act, trans)
        out[f"init_dist_interval{interval}"] = init
        out[f"trans_dist_interval{interval}"] = trans
        out[f"norm_factors_interval{interval}"] = norm_fact
        contiguous_act = np.ascontiguousarray(act, dtype=np.float32)
        path, log_prob = best_sequence(contiguous_act, interval, sigma)
        out[f"best_sequence_path_interval{interval}"] = path
        out[f"best_sequence_log_prob_interval{interval}"] = np.asarray(
            log_prob, dtype=np.float64)

    return out


def generate_beats_end_to_end_fixtures() -> dict:
    """End-to-end `BeatTrackingProcessor`/`BeatDetectionProcessor`/
    `CRFBeatDetectionProcessor` decoded beat times, all 3 cases,
    shared-`RNNBeatProcessor`-instance-in-order (see module header)."""
    from madmom.features.beats import (
        BeatDetectionProcessor, BeatTrackingProcessor,
        CRFBeatDetectionProcessor, RNNBeatProcessor,
    )

    rnn = RNNBeatProcessor(online=False)
    bt = BeatTrackingProcessor(fps=100)
    bd = BeatDetectionProcessor(fps=100)
    crf = CRFBeatDetectionProcessor(fps=100)

    out = {}
    for case, wav_name in BEAT_CASES.items():
        wav_path = str(WAVS_DIR / wav_name)
        act = rnn(wav_path)
        out[f"{case}_activations"] = act
        out[f"{case}_beat_tracking"] = np.asarray(bt(act))
        out[f"{case}_beat_detection"] = np.asarray(bd(act))
        out[f"{case}_crf"] = np.asarray(crf(act))
    return out


def generate_gmm_fixtures() -> dict:
    """Self-contained `GMM.score`/`GMM.score_samples` fixtures against the
    real `PATTERNS_BALLROOM` GMMs -- records each GMM's parameters
    (means/covars/weights/covariance_type/n_components) plus a fixed random
    query array and real madmom's score/responsibilities output, so
    `tests/test_gmm.py` needs neither the model download nor unpickling."""
    import pickle

    from madmom.models import PATTERNS_BALLROOM

    rng = np.random.RandomState(20260713)
    out = {}
    for p_idx, pattern_file in enumerate(PATTERNS_BALLROOM):
        with open(pattern_file, "rb") as fh:
            pattern = pickle.load(fh, encoding="latin1")
        gmms = pattern["gmms"]
        # exercise a handful of GMMs per pattern file (not all -- cheap,
        # representative subset, same economy as other waves' fixture
        # tools), always including the first and last for coverage.
        indices = sorted(set([0, len(gmms) - 1] +
                             list(rng.choice(len(gmms),
                                            size=min(4, len(gmms)),
                                            replace=False))))
        for g_idx in indices:
            gmm = gmms[g_idx]
            x = rng.randn(15, gmm.means.shape[1]).astype(np.float64)
            log_prob, responsibilities = gmm.score_samples(x)
            key = f"pattern{p_idx}_gmm{g_idx}"
            out[f"{key}_means"] = gmm.means
            out[f"{key}_covars"] = gmm.covars
            out[f"{key}_weights"] = gmm.weights
            out[f"{key}_n_components"] = np.asarray(gmm.n_components)
            out[f"{key}_x"] = x
            out[f"{key}_log_prob"] = log_prob
            out[f"{key}_responsibilities"] = responsibilities
        out[f"pattern{p_idx}_covariance_type"] = np.asarray(
            gmms[0].covariance_type)
    return out


def generate_patterns_structural_digest() -> dict:
    """Structural digest of both `PATTERNS_BALLROOM` pattern files (num_beats
    + per-GMM means/covars/weights digests) -- proves this port's
    `SafeUnpickler`-based loading matches real madmom's own bare
    `pickle.load` exactly."""
    import pickle

    from madmom.models import PATTERNS_BALLROOM

    digest = {}
    for p_idx, pattern_file in enumerate(PATTERNS_BALLROOM):
        with open(pattern_file, "rb") as fh:
            pattern = pickle.load(fh, encoding="latin1")
        digest[f"pattern{p_idx}"] = {
            "num_beats": pattern["num_beats"],
            "num_gmms": len(pattern["gmms"]),
            "gmms": [
                {
                    "n_components": gmm.n_components,
                    "covariance_type": gmm.covariance_type,
                    "means": _arr_digest(gmm.means),
                    "covars": _arr_digest(gmm.covars),
                    "weights": _arr_digest(gmm.weights),
                }
                for gmm in pattern["gmms"]
            ],
        }
    return digest


def generate_pattern_tracking_fixtures() -> dict:
    """End-to-end `PatternTrackingProcessor` fixture: audio -> multi-band
    spectral-flux features -> decoded (down-)beat positions + beat numbers,
    all 3 cases -- the same pre-processing chain
    `PatternTrackingProcessor`'s own docstring example uses (`Logarithmic
    SpectrogramProcessor` -> `SpectrogramDifferenceProcessor(positive_diffs
    =True)` -> `MultiBandSpectrogramProcessor(crossover_frequencies=
    [270])`), fresh instances per case (this pipeline has no per-frame-size-
    branch caching gotcha the way the RNN pipelines do, but fresh-per-case
    is the safe default regardless, matching Wave 4d's own precedent)."""
    from madmom.audio.signal import SignalProcessor
    from madmom.audio.spectrogram import (
        LogarithmicSpectrogramProcessor, MultiBandSpectrogramProcessor,
        SpectrogramDifferenceProcessor,
    )
    from madmom.features.downbeats import PatternTrackingProcessor
    from madmom.models import PATTERNS_BALLROOM
    from madmom.processors import SequentialProcessor

    out = {}
    for case, wav_name in BEAT_CASES.items():
        wav_path = str(WAVS_DIR / wav_name)
        pre = SequentialProcessor([
            SignalProcessor(num_channels=1, sample_rate=44100),
            LogarithmicSpectrogramProcessor(),
            SpectrogramDifferenceProcessor(positive_diffs=True),
            MultiBandSpectrogramProcessor(crossover_frequencies=[270]),
        ])
        feat = pre(wav_path)
        out[f"{case}_features"] = np.asarray(feat)
        pt = PatternTrackingProcessor(PATTERNS_BALLROOM, fps=50)
        out[f"{case}_decoded"] = np.asarray(pt(feat))
    return out


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its BEATS_LSTM/BEATS_BLSTM/"
            "PATTERNS_BALLROOM pretrained model/pattern files. Run it with "
            "the madmom-reference venv's interpreter -- see this file's "
            "module docstring for the exact command.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if not WAVS_DIR.exists() or not any(WAVS_DIR.glob("*.wav")):
        print(
            "ERROR: tests/fixtures/wavs/ is empty -- run "
            "tools/generate_fixtures.py first (Phase-1 script, generates "
            "the shared test wavs this script reuses).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Using madmom {madmom.__version__} from {madmom.__file__}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("4f-1: beats_crf function fixtures ...")
        crf_fn_fixtures = generate_beats_crf_function_fixtures()
        np.savez_compressed(FIXTURES_DIR / "beats_crf_functions.npz",
                             **crf_fn_fixtures)

        print("4f-2: BeatTracking/BeatDetection/CRF end-to-end fixtures ...")
        beats_e2e_fixtures = generate_beats_end_to_end_fixtures()
        np.savez_compressed(FIXTURES_DIR / "beat_tracking_end_to_end.npz",
                             **beats_e2e_fixtures)

        print("4f-3: GMM score/posterior fixtures ...")
        gmm_fixtures = generate_gmm_fixtures()
        np.savez_compressed(FIXTURES_DIR / "gmm_scores.npz", **gmm_fixtures)

        print("4f-4: PATTERNS_BALLROOM structural digest ...")
        patterns_digest = generate_patterns_structural_digest()
        (FIXTURES_DIR / "patterns_structural_digest.json").write_text(
            json.dumps(patterns_digest, indent=2, sort_keys=True) + "\n"
        )

        print("4f-5: PatternTrackingProcessor end-to-end fixture ...")
        pattern_tracking_fixtures = generate_pattern_tracking_fixtures()
        np.savez_compressed(FIXTURES_DIR / "pattern_tracking_end_to_end.npz",
                             **pattern_tracking_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
