"""Wave-4c golden-fixture generator: comb-filter function fixtures,
`beats_lstm_1`/`beats_blstm_1`/`downbeats_bgru_{rhythmic,harmonic}_0`
unpickled structural digests, end-to-end `RNNBeatProcessor`(online=False/
True) activations + `DBNBeatTrackingProcessor` decoded beat times,
`MultiModelSelectionProcessor`'s self-contained selection fixture, per-mode
(`acf`/`comb`/`dbn`) tempo histogram + `TempoEstimationProcessor` tempi
fixtures, `SyncronizeFeaturesProcessor`'s self-contained fixture, and
`RNNBarProcessor`'s GRU-ensemble intermediate-feature fixture -- the 4c
sibling of `tools/generate_onset_fixtures.py`/`generate_key_fixtures.py`,
same conventions (own file, own fixture files, independently regenerable
without touching prior waves' already-committed fixtures).

**Most fixtures here are self-contained (input array + real-madmom output
array), not just output-only** -- unlike `generate_onset_fixtures.py`'s
DSP-function fixtures (which could rely on this project's own already-
golden-fixture-proven Phase-1 DSP chain to reconstruct inputs), several 4c
targets are pure-numpy functions with NO dependency on this project's own
DSP chain at all (`feed_forward_comb_filter`, `SyncronizeFeaturesProcessor`,
the tempo histogram functions) -- so their fixtures record BOTH the input
array (synthetic or captured from a real activation function) and real
madmom's output, letting `tests/test_*.py` run them fully OFFLINE with zero
madmom_infer-side reconstruction needed.

**`RNNBarProcessor`'s fixture is intermediate, not full audio-in.** Real
madmom's `RNNBarProcessor.process()` is exercised (perc/harm feature
extraction, beat-sync, GRU ensembles), but what gets RECORDED is the
INTERMEDIATE (perc_synced, harm_synced) arrays and the two GRU ensembles'
own outputs (perc_nn_out, harm_nn_out) -- not because the audio-in path
doesn't work in real madmom (it does), but because THIS PORT's own
`RNNBarProcessor` cannot reconstruct `perc_synced`/`harm_synced` from raw
audio until Wave 4d ports `CLPChromaProcessor` (see `madmom_infer/features/
downbeats.py`'s module header). Recording the intermediate arrays lets
`tests/test_downbeats_rnn.py` feed them DIRECTLY into this port's own
`perc_nn`/`harm_nn` `NeuralNetworkEnsemble`s (built from this wave's new
`GRULayer`/`GRUCell`) and prove THAT forward pass is bit-identical --
which is the actual "does the GRU port work" question -- without needing
the still-missing harmonic-feature extractor.

HOW TO RUN -- same real-madmom reference venv as Phase 1/2/4a/4b
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_beat_tempo_fixtures.py

Reuses the same 44.1kHz-native test-wav subset established by
`tools/generate_phase2_fixtures.py`/`generate_key_fixtures.py`/
`generate_onset_fixtures.py` (`mono_44100`, `stereo_44100`, `float32_44100`)
for the model-dependent (RNN activation + decoded beat time) fixtures --
`RNNBeatProcessor`/`RNNBarProcessor` hard-code `SignalProcessor(sample_rate=
44100)` exactly like their siblings (no ffmpeg-backed resampling in this
port, see `audio/signal.py`'s module header). Model-independent fixtures
(comb filters, tempo histograms given a fixed activation array,
`SyncronizeFeaturesProcessor`) only need ONE representative wav
(`mono_44100.wav`), matching the economy `generate_key_fixtures.py`/
`generate_onset_fixtures.py` already established.

**Same "shared-instance-in-order" caching-gotcha discipline as
`test_downbeats_rnn.py`/`test_onsets.py`** (see those files' module
headers): `RNNBeatProcessor` builds ONE `ShortTimeFourierTransformProcessor`/
`FilteredSpectrogramProcessor` PER FRAME-SIZE BRANCH inside `__init__` and
reuses them across calls -- this generator builds ONE `RNNBeatProcessor
(online=False)` and ONE `RNNBeatProcessor(online=True)` instance and
processes all 3 cases through each, IN ORDER (`mono_44100` ->
`stereo_44100` -> `float32_44100`); `tests/test_beats.py` must replicate
that exact call order/instance-reuse or it would silently compare against
the wrong numbers.

Reads: real `madmom` (audio.comb_filters, features.beats,
features.downbeats, features.tempo, models.BEATS_LSTM/BEATS_BLSTM/
DOWNBEATS_BGRU, ml.nn.NeuralNetwork), numpy. Writes: tests/fixtures/
beats_comb_filters.npz, tests/fixtures/beats_structural_digest.json,
tests/fixtures/beats_activations.npz, tests/fixtures/
beats_multimodel_selection.npz, tests/fixtures/tempo_histograms.npz,
tests/fixtures/sync_features.npz, tests/fixtures/
downbeats_bgru_structural_digest.json, tests/fixtures/
downbeats_bgru_intermediate.npz. Read by: tests/test_beats.py,
tests/test_tempo.py, tests/test_comb_filters.py, tests/test_downbeats_rnn.py.
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


def digest_layer(layer) -> dict:
    """Structural digest of one NN layer -- extends `tools/
    generate_phase2_fixtures.py`'s `digest_layer` with the 4c `GRULayer`/
    `GRUCell` recursion (`reset_gate`/`update_gate`/`cell`, alongside the
    already-established `BidirectionalLayer`/`LSTMLayer`/`Gate`/`Cell`
    handling)."""
    t = type(layer).__name__
    d = {"type": t}
    if hasattr(layer, "weights"):
        d["weights"] = _arr_digest(layer.weights)
    if hasattr(layer, "bias"):
        d["bias"] = _arr_digest(layer.bias)
    if t in ("Gate", "Cell", "GRUCell", "RecurrentLayer"):
        if hasattr(layer, "recurrent_weights"):
            d["recurrent_weights"] = _arr_digest(layer.recurrent_weights)
    if t == "Gate" and getattr(layer, "peephole_weights", None) is not None:
        d["peephole_weights"] = _arr_digest(layer.peephole_weights)
    if getattr(layer, "activation_fn", None) is not None:
        d["activation_fn"] = layer.activation_fn.__name__
    if t == "BidirectionalLayer":
        d["fwd_layer"] = digest_layer(layer.fwd_layer)
        d["bwd_layer"] = digest_layer(layer.bwd_layer)
    elif t == "LSTMLayer":
        d["input_gate"] = digest_layer(layer.input_gate)
        d["forget_gate"] = digest_layer(layer.forget_gate)
        d["cell"] = digest_layer(layer.cell)
        d["output_gate"] = digest_layer(layer.output_gate)
    elif t == "GRULayer":
        d["reset_gate"] = digest_layer(layer.reset_gate)
        d["update_gate"] = digest_layer(layer.update_gate)
        d["cell"] = digest_layer(layer.cell)
    return d


def generate_beats_structural_digest() -> dict:
    import madmom
    from madmom.models import BEATS_BLSTM, BEATS_LSTM

    nn_lstm = madmom.ml.nn.NeuralNetwork.load(BEATS_LSTM[0])
    nn_blstm = madmom.ml.nn.NeuralNetwork.load(BEATS_BLSTM[0])
    return {
        "beats_lstm_1": [digest_layer(l) for l in nn_lstm.layers],
        "beats_blstm_1": [digest_layer(l) for l in nn_blstm.layers],
    }


def generate_downbeats_bgru_structural_digest() -> dict:
    import warnings

    import madmom
    from madmom.models import DOWNBEATS_BGRU

    with warnings.catch_warnings():
        # real madmom's own GRULayer.__setstate__ warns (legacy pickle
        # format, see madmom_infer/ml/nn/layers.py's module header) --
        # expected, not an error, silence it for a clean generator run.
        warnings.simplefilter("ignore")
        nn_rhythmic = madmom.ml.nn.NeuralNetwork.load(DOWNBEATS_BGRU[0][0])
        nn_harmonic = madmom.ml.nn.NeuralNetwork.load(DOWNBEATS_BGRU[1][0])
    return {
        "downbeats_bgru_rhythmic_0": [digest_layer(l) for l in nn_rhythmic.layers],
        "downbeats_bgru_harmonic_0": [digest_layer(l) for l in nn_harmonic.layers],
    }


def generate_comb_filter_fixtures() -> dict:
    """Direct function-level fixtures for `feed_forward_comb_filter`/
    `feed_backward_comb_filter`/`comb_filter`, fed a REAL beat activation
    function (not synthetic noise) -- `RNNBeatProcessor(online=False)`'s
    output on `mono_44100.wav`, the same kind of signal
    `CombFilterTempoHistogramProcessor` actually filters in practice."""
    from madmom.audio.comb_filters import (
        comb_filter, feed_backward_comb_filter, feed_forward_comb_filter,
    )
    from madmom.features.beats import RNNBeatProcessor

    act = RNNBeatProcessor(online=False)(str(WAVS_DIR / DSP_CASE_WAV))
    act_2d = np.tile(act[:, None], (1, 2)).astype(float)

    out = {"comb_filter_input_1d": act, "comb_filter_input_2d": act_2d}
    # alpha=0.79 (this module's own default `ALPHA`) is NOT exactly
    # representable in float32 -- deliberately exercises the
    # feed_backward_comb_filter float32-alpha-rounding quirk documented in
    # madmom_infer/audio/comb_filters.py's module header.
    for tau in (5, 17, 43):
        out[f"feed_forward_tau{tau}"] = feed_forward_comb_filter(act, tau, 0.79)
        out[f"feed_backward_tau{tau}"] = feed_backward_comb_filter(act, tau, 0.79)
    out["feed_backward_2d_tau17"] = feed_backward_comb_filter(act_2d, 17, 0.79)
    out["comb_filter_bank_forward"] = comb_filter(
        act, feed_forward_comb_filter, [5, 17, 43], [0.79, 0.79, 0.79])
    out["comb_filter_bank_backward"] = comb_filter(
        act, feed_backward_comb_filter, [5, 17, 43], [0.79, 0.79, 0.79])
    return out


def generate_beats_fixtures() -> "tuple[dict, dict]":
    """End-to-end `RNNBeatProcessor`(online=False/True) activations +
    `DBNBeatTrackingProcessor` decoded beat times (all 3 cases, shared-
    instance-in-order -- see module header), plus `MultiModelSelectionProcessor`'s
    self-contained fixture (the 8 per-network `BEATS_BLSTM` predictions for
    `mono_44100.wav` + real madmom's own selection output)."""
    from madmom.features.beats import (
        DBNBeatTrackingProcessor, MultiModelSelectionProcessor,
        RNNBeatProcessor,
    )

    blstm_proc = RNNBeatProcessor(online=False)
    lstm_proc = RNNBeatProcessor(online=True)
    dbn = DBNBeatTrackingProcessor(fps=100)

    out = {}
    for case, wav_name in BEAT_CASES.items():
        wav_path = str(WAVS_DIR / wav_name)
        act_blstm = blstm_proc(wav_path)
        act_lstm = lstm_proc(wav_path)
        out[f"{case}_blstm_activations"] = act_blstm
        out[f"{case}_lstm_activations"] = act_lstm
        out[f"{case}_blstm_beat_times"] = np.asarray(dbn(act_blstm))
        out[f"{case}_lstm_beat_times"] = np.asarray(dbn(act_lstm))

    mm_proc = RNNBeatProcessor(online=False, post_processor=None)
    predictions = mm_proc(str(WAVS_DIR / DSP_CASE_WAV))
    mm = MultiModelSelectionProcessor(num_ref_predictions=None)
    selected = mm(predictions)
    mm_fixtures = {
        f"prediction_{i}": p for i, p in enumerate(predictions)
    }
    mm_fixtures["selected"] = selected
    return out, mm_fixtures


def generate_tempo_fixtures() -> dict:
    """Per-mode (`acf`/`comb`/`dbn`) tempo histogram + `TempoEstimationProcessor`
    tempi fixtures, self-contained (records the input activation array too,
    real `RNNBeatProcessor(online=False)` output on `mono_44100.wav`, so
    `tests/test_tempo.py` needs neither the RNN model nor a network call)."""
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor

    act = RNNBeatProcessor(online=False)(str(WAVS_DIR / DSP_CASE_WAV))
    out = {"tempo_input_activations": act}
    for method in ("acf", "comb", "dbn"):
        proc = TempoEstimationProcessor(method=method, fps=100)
        histogram = proc.interval_histogram(act.astype(float)
                                            if method != "dbn" else act)
        bins, delays = histogram
        out[f"{method}_histogram_bins"] = np.asarray(bins)
        out[f"{method}_histogram_delays"] = np.asarray(delays)
        out[f"{method}_tempi"] = proc(act)
    return out


def generate_sync_features_fixture() -> dict:
    """Self-contained `SyncronizeFeaturesProcessor` fixture: a synthetic,
    deterministic (features, beats) pair + real madmom's output -- pure
    numpy, no NN weights, no dependency on any of this project's own DSP
    chain (see module header)."""
    from madmom.features.downbeats import SyncronizeFeaturesProcessor

    rng = np.random.RandomState(20260712)
    features = rng.rand(500, 6).astype(np.float32)
    beats = np.sort(rng.uniform(0.1, 4.9, 8))
    proc = SyncronizeFeaturesProcessor(4, fps=100)
    out_features = proc((features, beats))
    return {
        "sync_features_input": features,
        "sync_beats_input": beats,
        "sync_features_output": out_features,
    }


def generate_downbeats_bgru_intermediate_fixture() -> dict:
    """`RNNBarProcessor`'s GRU-ensemble intermediate-feature fixture -- see
    module header for why this is intermediate (perc_synced/harm_synced),
    not full audio-in. Beat positions are real madmom's own
    `RNNBeatProcessor` -> `DBNBeatTrackingProcessor` output on
    `mono_44100.wav` (a genuine, non-arbitrary beat sequence), not
    hand-picked."""
    import warnings

    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.downbeats import RNNBarProcessor

    beat_act = RNNBeatProcessor(online=False)(str(WAVS_DIR / DSP_CASE_WAV))
    beats = DBNBeatTrackingProcessor(fps=100)(beat_act)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = RNNBarProcessor()
        wav_path = str(WAVS_DIR / DSP_CASE_WAV)
        perc = proc.perc_feat(wav_path)
        harm = proc.harm_feat(wav_path)
        perc_synced = proc.perc_beat_sync((perc, beats))
        harm_synced = proc.harm_beat_sync((harm, beats))
        perc_nn_out = proc.perc_nn(
            perc_synced.reshape((len(perc_synced), -1)))
        harm_nn_out = proc.harm_nn(
            harm_synced.reshape((len(harm_synced), -1)))
        full_downbeat_act = proc((wav_path, beats))

    return {
        "beats": beats,
        "perc_synced": perc_synced,
        "harm_synced": harm_synced,
        "perc_nn_out": perc_nn_out,
        "harm_nn_out": harm_nn_out,
        "full_downbeat_activation": full_downbeat_act,
    }


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer), including its BEATS_LSTM/BEATS_BLSTM/"
            "DOWNBEATS_BGRU pretrained model weights. Run it with the "
            "madmom-reference venv's interpreter -- see this file's module "
            "docstring for the exact command.",
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

    print("4c-1: comb-filter function fixtures ...")
    comb_fixtures = generate_comb_filter_fixtures()
    np.savez_compressed(FIXTURES_DIR / "beats_comb_filters.npz",
                         **comb_fixtures)

    print("4c-2: beats_lstm_1/beats_blstm_1 structural digest ...")
    beats_digest = generate_beats_structural_digest()
    (FIXTURES_DIR / "beats_structural_digest.json").write_text(
        json.dumps(beats_digest, indent=2, sort_keys=True) + "\n"
    )

    print("4c-3: downbeats_bgru_{rhythmic,harmonic}_0 structural digest ...")
    bgru_digest = generate_downbeats_bgru_structural_digest()
    (FIXTURES_DIR / "downbeats_bgru_structural_digest.json").write_text(
        json.dumps(bgru_digest, indent=2, sort_keys=True) + "\n"
    )

    print("4c-4: RNNBeatProcessor + DBNBeatTrackingProcessor + "
          "MultiModelSelectionProcessor fixtures ...")
    beats_fixtures, mm_fixtures = generate_beats_fixtures()
    np.savez_compressed(FIXTURES_DIR / "beats_activations.npz",
                         **beats_fixtures)
    np.savez_compressed(FIXTURES_DIR / "beats_multimodel_selection.npz",
                         **mm_fixtures)

    print("4c-5: tempo histogram + TempoEstimationProcessor fixtures ...")
    tempo_fixtures = generate_tempo_fixtures()
    np.savez_compressed(FIXTURES_DIR / "tempo_histograms.npz",
                         **tempo_fixtures)

    print("4c-6: SyncronizeFeaturesProcessor fixture ...")
    sync_fixtures = generate_sync_features_fixture()
    np.savez_compressed(FIXTURES_DIR / "sync_features.npz", **sync_fixtures)

    print("4c-7: RNNBarProcessor GRU-ensemble intermediate fixture ...")
    bgru_fixtures = generate_downbeats_bgru_intermediate_fixture()
    np.savez_compressed(FIXTURES_DIR / "downbeats_bgru_intermediate.npz",
                         **bgru_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
