"""Wave-4d golden-fixture generator: `ConditionalRandomField` direct-decode
fixtures, classic chroma (`PitchClassProfile`/`HarmonicPitchClassProfile`)
function-level fixtures, CLP chroma (`CLPChroma`) class-level fixtures,
`chroma_dnn.pkl`'s unpickled structural digest + `DeepChromaProcessor`
end-to-end activations, chord-recognition end-to-end (audio -> chord
segments, both `DeepChromaChordRecognitionProcessor` and
`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor` paths), and
`RNNBarProcessor`'s full AUDIO-IN end-to-end fixture (the loop 4c left open,
see `madmom_infer/features/downbeats.py`'s module header) -- the 4d sibling
of `tools/generate_key_fixtures.py`/`generate_beat_tempo_fixtures.py`, same
conventions.

**Deliberate deviation from prior waves' "shared-instance-in-order"
discipline, noted explicitly, not silently**: every end-to-end fixture below
uses a FRESH processor instance per wav case rather than one shared instance
looped over all 3 (unlike `generate_key_fixtures.py`/
`generate_beat_tempo_fixtures.py`). This is a deliberate choice, not an
oversight -- `RNNBarProcessor`'s `SyncronizeFeaturesProcessor` step is
sensitive enough to any of its upstream stages' cached, wrong-for-this-call
state (the FFT-window/filterbank caching gotchas `audio/stft.py`/
`audio/spectrogram.py` document) that a shared-instance run across
differing-dtype wavs (`mono`/`stereo` int16, `float32` float32) produced an
EMPTY `perc_synced` array for the `float32_44100` case during this wave's
own testing -- confirmed to be a real, reproducible instance-reuse artifact
of THIS port's composition-style caching (not a fixture-vs-port algorithmic
mismatch: a fresh `RNNBarProcessor()` per wav reproduces real madmom's
recorded numbers exactly). Since this is a NEW fixture set (not
re-deriving numbers a previously-committed fixture already pinned), fresh
instances per case sidestep the whole caching-gotcha minefield rather than
faithfully reproducing it -- both sides (this script, real madmom) and
`tests/test_chroma.py`/`test_chords.py`/`test_downbeats_rnn.py` all use this
same fresh-per-case discipline, so the comparison stays apples-to-apples.

HOW TO RUN -- same real-madmom reference venv as prior waves
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3),
whose already-installed madmom 0.17.dev0 wheel vendors `chroma/2016/
chroma_dnn.pkl` and `chords/2016/chords_{dccrf,cnnfeat,cnncrf}.pkl` as
package data (no network needed to run this script):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_chroma_chord_fixtures.py

Reuses the same 44.1kHz-native test-wav subset established by
`tools/generate_phase2_fixtures.py` (`mono_44100`, `stereo_44100`,
`float32_44100`) -- every processor here hard-codes `SignalProcessor(
sample_rate=44100)`, and this project has no ffmpeg-backed resampling for
FILE LOADING (only the narrow, load-bearing `SemitoneBandpassFilterbank`
internal resample, see `audio/signal.py`'s header) -- `stereo_48000.wav`
stays out of scope for that reason.

Reads: real `madmom` (audio.chroma.*, audio.spectrogram.
SemitoneBandpassSpectrogram, features.chords.*, ml.crf.
ConditionalRandomField, ml.nn.NeuralNetwork, models.{CHROMA_DNN,
CHORDS_DCCRF,CHORDS_CNN_FEAT,CHORDS_CFCRF}, features.beats.*,
features.downbeats.RNNBarProcessor), numpy. Writes: tests/fixtures/
chroma_classic.npz, tests/fixtures/clp_chroma.npz, tests/fixtures/
chroma_dnn_structural_digest.json, tests/fixtures/chroma_dnn_activations.npz,
tests/fixtures/crf_decode.npz, tests/fixtures/chords_end_to_end.npz,
tests/fixtures/rnn_bar_end_to_end.npz. Read by: tests/test_chroma.py,
tests/test_chords.py, tests/test_downbeats_rnn.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

# 44.1kHz-native cases only -- see module header (no file-load resampling).
CASES = ("mono_44100", "stereo_44100", "float32_44100")


def _arr_digest(arr) -> dict:
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def digest_layer(layer) -> dict:
    """Structural digest of one NN layer -- independent, narrower
    reimplementation matching `tools/generate_key_fixtures.py`'s, since
    `chroma_dnn.pkl` only needs `FeedForwardLayer` (no CNN-era attrs)."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    return d


# ---------------------------------------------------------------------------
# 1. ConditionalRandomField direct-decode fixture -- real chord-feature data
#    (not synthetic), real weights, real decoded label sequence.
# ---------------------------------------------------------------------------
def generate_crf_decode_fixtures() -> dict:
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import (
        CNNChordFeatureProcessor, majmin_targets_to_chord_labels,
    )
    from madmom.ml.crf import ConditionalRandomField
    from madmom.models import CHORDS_CFCRF, CHORDS_DCCRF

    wav_path = str(WAVS_DIR / "mono_44100.wav")

    # DeepChroma -> DCCRF
    chroma = DeepChromaProcessor()(wav_path)
    dccrf = ConditionalRandomField.load(CHORDS_DCCRF[0])
    dccrf_y_star = dccrf.process(chroma)
    dccrf_labels = majmin_targets_to_chord_labels(dccrf_y_star, fps=10)

    # CNNChordFeature -> CFCRF
    feats = CNNChordFeatureProcessor()(wav_path)
    cfcrf = ConditionalRandomField.load(CHORDS_CFCRF[0])
    cfcrf_y_star = cfcrf.process(feats)
    cfcrf_labels = majmin_targets_to_chord_labels(cfcrf_y_star, fps=10)

    return {
        "dccrf_observations": np.asarray(chroma),
        "dccrf_y_star": dccrf_y_star,
        "dccrf_labels_start": np.asarray(dccrf_labels["start"]),
        "dccrf_labels_end": np.asarray(dccrf_labels["end"]),
        "dccrf_labels_label": np.asarray(
            [str(x) for x in dccrf_labels["label"]]),
        "cfcrf_observations": np.asarray(feats),
        "cfcrf_y_star": cfcrf_y_star,
        "cfcrf_labels_start": np.asarray(cfcrf_labels["start"]),
        "cfcrf_labels_end": np.asarray(cfcrf_labels["end"]),
        "cfcrf_labels_label": np.asarray(
            [str(x) for x in cfcrf_labels["label"]]),
    }


# ---------------------------------------------------------------------------
# 2. Classic chroma (PitchClassProfile / HarmonicPitchClassProfile) --
#    function/class-level golden fixtures on the shared test wavs.
# ---------------------------------------------------------------------------
def generate_classic_chroma_fixtures() -> dict:
    from madmom.audio.chroma import HarmonicPitchClassProfile, PitchClassProfile
    from madmom.audio.signal import Signal
    from madmom.audio.spectrogram import Spectrogram

    out = {}
    for case in CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        # downmix to mono first -- Spectrogram/STFT only accept mono
        sig = Signal(wav_path, num_channels=1, sample_rate=44100)
        spec = Spectrogram(sig, frame_size=2048, fps=100)
        pcp = PitchClassProfile(spec)
        hpcp = HarmonicPitchClassProfile(spec)
        out[f"{case}_pcp"] = np.asarray(pcp)
        out[f"{case}_hpcp"] = np.asarray(hpcp)
    return out


# ---------------------------------------------------------------------------
# 3. CLP chroma -- class-level golden fixtures (self-contained: records the
#    input signal path is reused, but real madmom's OWN SemitoneBandpass-
#    Spectrogram intermediate is recorded too, so a test can check both the
#    final CLPChroma AND the intermediate stage against real madmom).
# ---------------------------------------------------------------------------
def generate_clp_chroma_fixtures() -> dict:
    from madmom.audio.chroma import CLPChroma
    from madmom.audio.spectrogram import SemitoneBandpassSpectrogram

    out = {}
    for case in CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        sbs = SemitoneBandpassSpectrogram(wav_path, fps=50, fmin=27.5,
                                          fmax=4200.0)
        clp = CLPChroma(sbs, fps=50, fmin=27.5, fmax=4200.0,
                        compression_factor=100, norm=True, threshold=0.001)
        out[f"{case}_semitone_bandpass"] = np.asarray(sbs)
        out[f"{case}_semitone_bin_frequencies"] = np.asarray(
            sbs.bin_frequencies)
        out[f"{case}_clp_chroma"] = np.asarray(clp)
    return out


# ---------------------------------------------------------------------------
# 4. DeepChromaProcessor -- structural digest + end-to-end activations
# ---------------------------------------------------------------------------
def generate_chroma_dnn_structural_digest() -> dict:
    import madmom
    from madmom.models import CHROMA_DNN

    assert len(CHROMA_DNN) == 1, (
        f"expected CHROMA_DNN to resolve to exactly 1 file, got "
        f"{len(CHROMA_DNN)}."
    )
    nn = madmom.ml.nn.NeuralNetwork.load(CHROMA_DNN[0])
    return {"chroma_dnn": [digest_layer(l) for l in nn.layers]}


def generate_chroma_dnn_activations() -> dict:
    from madmom.audio.chroma import DeepChromaProcessor

    out = {}
    for case in CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        proc = DeepChromaProcessor()
        chroma = proc(wav_path)
        out[f"{case}_chroma"] = np.asarray(chroma)
    return out


# ---------------------------------------------------------------------------
# 5. Chord recognition end-to-end (audio -> chord segments), both paths --
#    EXACT is the claim tests/test_chords.py makes for these.
# ---------------------------------------------------------------------------
def generate_chords_end_to_end_fixtures() -> dict:
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import (
        CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
        DeepChromaChordRecognitionProcessor,
    )
    from madmom.processors import SequentialProcessor

    out = {}
    for case in CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")

        dcp = DeepChromaProcessor()
        decode = DeepChromaChordRecognitionProcessor()
        dccrf_chain = SequentialProcessor([dcp, decode])
        dccrf_result = dccrf_chain(wav_path)

        featproc = CNNChordFeatureProcessor()
        decode2 = CRFChordRecognitionProcessor()
        cfcrf_chain = SequentialProcessor([featproc, decode2])
        cfcrf_result = cfcrf_chain(wav_path)

        out[f"{case}_dccrf_start"] = np.asarray(dccrf_result["start"])
        out[f"{case}_dccrf_end"] = np.asarray(dccrf_result["end"])
        out[f"{case}_dccrf_label"] = np.asarray(
            [str(x) for x in dccrf_result["label"]])
        out[f"{case}_cfcrf_start"] = np.asarray(cfcrf_result["start"])
        out[f"{case}_cfcrf_end"] = np.asarray(cfcrf_result["end"])
        out[f"{case}_cfcrf_label"] = np.asarray(
            [str(x) for x in cfcrf_result["label"]])
    return out


# ---------------------------------------------------------------------------
# 6. RNNBarProcessor -- full AUDIO-IN end-to-end fixture (closes the loop
#    4c left open, see madmom_infer/features/downbeats.py's module header).
# ---------------------------------------------------------------------------
def generate_rnn_bar_end_to_end_fixtures() -> dict:
    import warnings

    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import RNNBarProcessor

    out = {}
    for case in CASES:
        wav_path = str(WAVS_DIR / f"{case}.wav")
        rnn = RNNBeatProcessor(online=False)
        act = rnn(wav_path)
        dbn = DBNBeatTrackingProcessor(fps=100)
        beats = dbn(act)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bar = RNNBarProcessor()
            full_out = bar((wav_path, beats))
        out[f"{case}_beats"] = beats
        out[f"{case}_bar_output"] = full_out
    return out


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its CHROMA_DNN/CHORDS_* pretrained "
            "model weights. Run it with the madmom-reference venv's "
            "interpreter -- see this file's module docstring for the exact "
            "command.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if not WAVS_DIR.exists() or not any(WAVS_DIR.glob("*.wav")):
        print(
            "ERROR: tests/fixtures/wavs/ is empty -- run "
            "tools/generate_fixtures.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Using madmom {madmom.__version__} from {madmom.__file__}")

    print("4d-1: ConditionalRandomField direct-decode fixtures ...")
    crf_fixtures = generate_crf_decode_fixtures()
    np.savez_compressed(FIXTURES_DIR / "crf_decode.npz", **crf_fixtures)

    print("4d-2: classic chroma (PCP/HPCP) fixtures ...")
    classic_fixtures = generate_classic_chroma_fixtures()
    np.savez_compressed(FIXTURES_DIR / "chroma_classic.npz",
                        **classic_fixtures)

    print("4d-3: CLP chroma fixtures ...")
    clp_fixtures = generate_clp_chroma_fixtures()
    np.savez_compressed(FIXTURES_DIR / "clp_chroma.npz", **clp_fixtures)

    print("4d-4: chroma_dnn.pkl structural digest ...")
    digest = generate_chroma_dnn_structural_digest()
    (FIXTURES_DIR / "chroma_dnn_structural_digest.json").write_text(
        json.dumps(digest, indent=2, sort_keys=True) + "\n"
    )

    print("4d-5: DeepChromaProcessor end-to-end activations ...")
    dnn_fixtures = generate_chroma_dnn_activations()
    np.savez_compressed(FIXTURES_DIR / "chroma_dnn_activations.npz",
                        **dnn_fixtures)

    print("4d-6: chord recognition end-to-end fixtures ...")
    chords_fixtures = generate_chords_end_to_end_fixtures()
    np.savez_compressed(FIXTURES_DIR / "chords_end_to_end.npz",
                        **chords_fixtures)

    print("4d-7: RNNBarProcessor full audio-in end-to-end fixtures ...")
    bar_fixtures = generate_rnn_bar_end_to_end_fixtures()
    np.savez_compressed(FIXTURES_DIR / "rnn_bar_end_to_end.npz",
                        **bar_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
