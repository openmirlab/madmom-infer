"""Golden-fixture generator: runs REAL (compiled) madmom and records its exact
outputs as .npz files under tests/fixtures/, so every future madmom_infer port
of Signal/FramedSignal/STFT/filterbank/log-spectrogram/HMM/DBN code has a
bit-for-bit acceptance standard to match, the same philosophy as the sibling
all-in-one-infer project's NATTEN golden fixtures (see this repo's CLAUDE.md
"Dual-backend + golden-fixture testing philosophy"). This script is
standalone -- it imports only numpy/scipy/madmom, never madmom_infer -- so it
can run today, before any port code exists (docs/DESIGN.md section C.4).

Every random input is seeded (see the *_SEED constants below) and generation
is a straight-line sequence of independent, seed-scoped phases, so re-running
this script reproduces byte-identical .npz files and .wav inputs -- this is
verified by the harness itself (run twice, hash-compare) rather than assumed.

HOW TO RUN (this script needs the real `madmom` package, which is not, and
will never be, a madmom_infer dependency -- see this repo's README "What this
project will NEVER bundle"). A working madmom 0.17.dev0 install exists in
the reference venv at `madmom-reference/.venv` (Python 3.10.18, numpy
1.23.5, scipy 1.15.3), rebuilt 2026-07-12 (Wave 4.0) from
`../madmom-upstream` after the original `all-in-one-fix/.venv` this
package's fixtures were first recorded from stopped existing on this
machine -- see tests/fixtures/README.md for full provenance.
Run this script with THAT interpreter, from anywhere (paths below are
resolved relative to this file, not the cwd):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_fixtures.py

For a fully-reproducible from-scratch environment (no dependency on that
checkout existing), see docs/DESIGN.md section C.4 for the pinned
`uv venv --python 3.10 ... numpy==1.23.5 scipy==1.15.3 cython
'git+https://github.com/CPJKU/madmom'` recipe.

Reads: real `madmom` (audio.signal, audio.stft, audio.filters,
audio.spectrogram, processors, ml.hmm, features.beats_hmm,
features.downbeats), numpy, scipy.io.wavfile. Writes: tests/fixtures/*.npz,
tests/fixtures/wavs/*.wav, tests/fixtures/manifest.json. Read by:
tests/test_fixtures_exist.py (existence/shape smoke test) and, eventually,
every madmom_infer port module's own golden-fixture test.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile

# --------------------------------------------------------------------------
# Paths (resolved relative to this file, so the script works from any cwd).
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

# --------------------------------------------------------------------------
# Phase-1 API surface parameters -- copied verbatim from all-in-one-infer's
# build_spec_processor() (all-in-one-fix/src/allin1_infer/spectrogram.py:27-40)
# and DBNDownBeatTrackingProcessor call site, per docs/DESIGN.md C.4.
# --------------------------------------------------------------------------
FRAME_SIZE = 2048
FPS = 100
NUM_BANDS = 12
FMIN = 30
FMAX = 17000
NORM_FILTERS = True
LOG_MUL = 1
LOG_ADD = 1

DBN_PARAMS = dict(
    beats_per_bar=[3, 4],
    min_bpm=55.0,
    max_bpm=215.0,
    num_tempi=60,
    transition_lambda=100,
    observation_lambda=16,
    fps=100,
)

# Independent seeds per generation phase (not one running RNG stream) so
# adding/removing an earlier phase never perturbs a later phase's numbers.
WAV_SEED = 1234
HMM_TOY_SEED = 1235
DBN_ACTIVATION_SEED = 1236

WAV_DURATION_S = 1.5  # short on purpose -- keeps fixtures small (C.4 constraint)


def sha256_of_array(arr: np.ndarray) -> str:
    """Stable content hash of an array's bytes, used to fingerprint "all
    frames"/"all bins" without paying to store them all in the .npz."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ==========================================================================
# Phase 0: synthesize deterministic test wavs
# ==========================================================================

def make_test_wavs() -> dict:
    """Write 4 short (1.5s), deterministic wavs covering the dtype/channel/
    sample-rate combinations all-in-one-infer actually feeds madmom with:
    mono int16 44.1kHz, stereo int16 44.1kHz, stereo int16 48kHz, float32
    44.1kHz. Returns a dict of case-name -> wav Path. Re-running this
    function (fixed seed) reproduces byte-identical wav files."""
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(WAV_SEED)
    paths = {}

    def sine_mix(sr, n, freqs_amps, noise_scale):
        t = np.arange(n) / sr
        sig = np.zeros(n)
        for freq, amp in freqs_amps:
            sig += amp * np.sin(2 * np.pi * freq * t)
        sig += rng.normal(0, noise_scale, size=n)
        return sig

    # mono int16 44.1kHz
    sr = 44100
    n = int(sr * WAV_DURATION_S)
    sig = sine_mix(sr, n, [(220, 0.3), (880, 0.1)], 0.01)
    mono_i16 = np.clip(sig * 32767, -32768, 32767).astype(np.int16)
    p = WAVS_DIR / "mono_44100.wav"
    wavfile.write(p, sr, mono_i16)
    paths["mono_44100"] = p

    # stereo int16 44.1kHz -- L/R deliberately different so the mono downmix
    # (mean-then-truncate-toward-zero, see remix() in madmom/audio/signal.py)
    # exercises genuinely different fractional-average cases.
    sr = 44100
    n = int(sr * WAV_DURATION_S)
    left = sine_mix(sr, n, [(196, 0.35), (784, 0.08)], 0.012)
    right = sine_mix(sr, n, [(233, 0.25), (932, 0.15)], 0.012)
    left_i16 = np.clip(left * 32767, -32768, 32767).astype(np.int16)
    right_i16 = np.clip(right * 32767, -32768, 32767).astype(np.int16)
    stereo_i16 = np.stack([left_i16, right_i16], axis=-1)
    p = WAVS_DIR / "stereo_44100.wav"
    wavfile.write(p, sr, stereo_i16)
    paths["stereo_44100"] = p

    # stereo int16 48kHz
    sr = 48000
    n = int(sr * WAV_DURATION_S)
    left = sine_mix(sr, n, [(330, 0.3), (660, 0.1)], 0.01)
    right = sine_mix(sr, n, [(311, 0.28), (622, 0.12)], 0.01)
    left_i16 = np.clip(left * 32767, -32768, 32767).astype(np.int16)
    right_i16 = np.clip(right * 32767, -32768, 32767).astype(np.int16)
    stereo48_i16 = np.stack([left_i16, right_i16], axis=-1)
    p = WAVS_DIR / "stereo_48000.wav"
    wavfile.write(p, sr, stereo48_i16)
    paths["stereo_48000"] = p

    # float32 mono 44.1kHz, values in [-1, 1]
    sr = 44100
    n = int(sr * WAV_DURATION_S)
    sig = sine_mix(sr, n, [(440, 0.3), (1320, 0.1)], 0.01)
    f32 = np.clip(sig, -1.0, 1.0).astype(np.float32)
    p = WAVS_DIR / "float32_44100.wav"
    wavfile.write(p, sr, f32)
    paths["float32_44100"] = p

    return paths


# ==========================================================================
# Phase 1: Signal fixtures
# ==========================================================================

def generate_signal_fixtures(wav_paths: dict) -> dict:
    from madmom.audio.signal import Signal

    out = {}
    for case, path in wav_paths.items():
        sig_raw = Signal(path)
        out[f"{case}_raw"] = np.asarray(sig_raw)
        out[f"{case}_raw_sample_rate"] = np.array(sig_raw.sample_rate)

        # Signal(path, num_channels=1): the int16 truncating-mean downmix for
        # stereo files; a no-op passthrough for already-mono files (recorded
        # anyway, to confirm it really is a no-op).
        sig_mono = Signal(path, num_channels=1)
        out[f"{case}_mono"] = np.asarray(sig_mono)
        out[f"{case}_mono_sample_rate"] = np.array(sig_mono.sample_rate)

        # Signal(ndarray, sample_rate=...): from-array construction directly
        # from the already-loaded raw array (a plain ndarray, not a Signal).
        plain_arr = np.asarray(sig_raw).copy()
        sig_from_arr = Signal(plain_arr, sample_rate=sig_raw.sample_rate)
        out[f"{case}_fromarray"] = np.asarray(sig_from_arr)
        out[f"{case}_fromarray_sample_rate"] = np.array(sig_from_arr.sample_rate)

    return out


# ==========================================================================
# Phase 2: FramedSignalProcessor fixtures (on the *raw* per-wav signals, so
# multi-channel framing shape is covered too, not just the mono downmix).
# ==========================================================================

def generate_framing_fixtures(wav_paths: dict) -> dict:
    from madmom.audio.signal import FramedSignalProcessor, Signal

    out = {}
    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    for case, path in wav_paths.items():
        sig = Signal(path)
        framed = frames_proc(sig)
        num_frames = len(framed)
        all_frames = np.stack([np.asarray(framed[i]) for i in range(num_frames)])

        out[f"{case}_num_frames"] = np.array(num_frames)
        out[f"{case}_frame_size"] = np.array(framed.frame_size)
        out[f"{case}_hop_size"] = np.array(framed.hop_size)
        out[f"{case}_frame0"] = np.asarray(framed[0])
        out[f"{case}_frame1"] = np.asarray(framed[1])
        out[f"{case}_frame_last"] = np.asarray(framed[-1])
        out[f"{case}_all_frames_sha256"] = np.array(sha256_of_array(all_frames))

    return out


# ==========================================================================
# Phase 3: ShortTimeFourierTransformProcessor fixtures -- mono-compatible
# variant of each test wav (STFT rejects multi-channel FramedSignal input;
# see the "surprises" note in tests/fixtures/README.md), plus an explicit
# demonstration of a real madmom gotcha: a *reused* STFT processor instance
# silently keeps a stale, wrongly-scaled cached window when fed a different
# signal dtype on a later call.
# ==========================================================================

def _mono_compatible_signals(wav_paths: dict) -> dict:
    """Return {case_name: mono Signal} for every test wav, downmixing the
    stereo ones. This mirrors how all-in-one-infer always feeds madmom: one
    mono channel per stem, never raw multi-channel."""
    from madmom.audio.signal import Signal

    return {
        "mono_44100": Signal(wav_paths["mono_44100"]),
        "stereo_44100_mono": Signal(wav_paths["stereo_44100"], num_channels=1),
        "stereo_48000_mono": Signal(wav_paths["stereo_48000"], num_channels=1),
        "float32_44100": Signal(wav_paths["float32_44100"]),
    }


def generate_stft_fixtures(wav_paths: dict) -> dict:
    from madmom.audio.signal import FramedSignalProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor

    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    mono_signals = _mono_compatible_signals(wav_paths)
    mono_frames = {case: frames_proc(sig) for case, sig in mono_signals.items()}

    out = {}
    for case, framed in mono_frames.items():
        # Fresh processor instance per case -- this is the *correct* usage
        # and is what the golden numbers below represent.
        stft_proc = ShortTimeFourierTransformProcessor()
        stft_out = stft_proc(framed)
        all_stft = np.asarray(stft_out)

        out[f"{case}_stft_frame0"] = all_stft[0]
        out[f"{case}_stft_frame1"] = all_stft[1]
        out[f"{case}_stft_frame_last"] = all_stft[-1]
        out[f"{case}_stft_all_sha256"] = np.array(sha256_of_array(all_stft))

    # --- window-caching gotcha demonstration ---
    # ShortTimeFourierTransformProcessor caches its scaled `fft_window` the
    # first time it's called (scaled for *that* call's signal dtype, per the
    # int16-vs-window-scaling convention in madmom/audio/stft.py:339-349).
    # Reusing the *same instance* across a later call with a *different*
    # dtype does NOT recompute the window scale -- it silently reuses the
    # stale one, producing wrong numbers with no error or warning. This
    # never bites all-in-one-infer today (it always feeds mono int16 stems
    # through one reused instance -- same dtype every call -- see
    # build_spec_processor() in all-in-one-fix/src/allin1_infer/
    # spectrogram.py:27-40), but it's a real, silent correctness trap for
    # any future caller that mixes dtypes through one processor instance.
    shared_stft = ShortTimeFourierTransformProcessor()
    _ = shared_stft(mono_frames["mono_44100"])  # int16 call -- caches int16-scaled window
    reused_output = np.asarray(shared_stft(mono_frames["float32_44100"]))  # BUG: stale window

    fresh_stft = ShortTimeFourierTransformProcessor()
    fresh_output = np.asarray(fresh_stft(mono_frames["float32_44100"]))

    out["window_caching_reused_output"] = reused_output
    out["window_caching_fresh_output"] = fresh_output
    out["window_caching_max_abs_diff"] = np.array(
        np.abs(reused_output - fresh_output).max(), dtype=np.float64
    )

    return out


# ==========================================================================
# Phase 4: Filterbank fixtures -- the fixed matrix itself, plus filtered
# spectrogram outputs for two representative (sample-rate-distinct) chains.
# ==========================================================================

FILTERBANK_CHAIN_CASES = ("mono_44100", "stereo_48000_mono")


def generate_filterbank_fixtures(wav_paths: dict) -> dict:
    from madmom.audio.signal import FramedSignalProcessor
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor

    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    mono_signals = _mono_compatible_signals(wav_paths)
    filt_proc = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )

    out = {}
    seen_sample_rates = set()
    for case in FILTERBANK_CHAIN_CASES:
        sig = mono_signals[case]
        framed = frames_proc(sig)
        stft_out = ShortTimeFourierTransformProcessor()(framed)
        filtered = filt_proc(stft_out)

        sr = sig.sample_rate
        if sr not in seen_sample_rates:
            seen_sample_rates.add(sr)
            out[f"filterbank_matrix_{sr}"] = np.asarray(filtered.filterbank)

        all_filtered = np.asarray(filtered)
        out[f"{case}_filtered_frame0"] = all_filtered[0]
        out[f"{case}_filtered_frame1"] = all_filtered[1]
        out[f"{case}_filtered_frame_last"] = all_filtered[-1]
        out[f"{case}_filtered_all_sha256"] = np.array(sha256_of_array(all_filtered))

    return out


# ==========================================================================
# Phase 5: LogarithmicSpectrogramProcessor fixtures (mul=1, add=1), applied
# to the same two filtered-spectrogram chains as phase 4.
# ==========================================================================

def generate_logspec_fixtures(wav_paths: dict) -> dict:
    from madmom.audio.signal import FramedSignalProcessor
    from madmom.audio.spectrogram import (
        FilteredSpectrogramProcessor,
        LogarithmicSpectrogramProcessor,
    )
    from madmom.audio.stft import ShortTimeFourierTransformProcessor

    frames_proc = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    mono_signals = _mono_compatible_signals(wav_paths)
    filt_proc = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )
    log_proc = LogarithmicSpectrogramProcessor(mul=LOG_MUL, add=LOG_ADD)

    out = {}
    for case in FILTERBANK_CHAIN_CASES:
        sig = mono_signals[case]
        framed = frames_proc(sig)
        stft_out = ShortTimeFourierTransformProcessor()(framed)
        filtered = filt_proc(stft_out)
        logspec = log_proc(filtered)
        all_log = np.asarray(logspec)

        out[f"{case}_logspec_frame0"] = all_log[0]
        out[f"{case}_logspec_frame1"] = all_log[1]
        out[f"{case}_logspec_frame_last"] = all_log[-1]
        out[f"{case}_logspec_all_sha256"] = np.array(sha256_of_array(all_log))

    return out


# ==========================================================================
# Phase 6: full-chain (top-level) integration fixture -- exactly
# all-in-one-infer's build_spec_processor(): SequentialProcessor([frames,
# stft, filt, spec]), for each mono-compatible test wav.
# ==========================================================================

def build_spec_processor():
    from madmom.audio.signal import FramedSignalProcessor
    from madmom.audio.spectrogram import (
        FilteredSpectrogramProcessor,
        LogarithmicSpectrogramProcessor,
    )
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.processors import SequentialProcessor

    frames = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    stft = ShortTimeFourierTransformProcessor()
    filt = FilteredSpectrogramProcessor(
        num_bands=NUM_BANDS, fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS
    )
    spec = LogarithmicSpectrogramProcessor(mul=LOG_MUL, add=LOG_ADD)
    return SequentialProcessor([frames, stft, filt, spec])


def generate_full_chain_fixtures(wav_paths: dict) -> dict:
    mono_signals = _mono_compatible_signals(wav_paths)

    out = {}
    for case, sig in mono_signals.items():
        # fresh SequentialProcessor per case -- avoids the window-caching
        # gotcha from phase 3 contaminating this golden reference.
        chain = build_spec_processor()
        result = np.asarray(chain(sig))

        out[f"{case}_frame0"] = result[0]
        out[f"{case}_frame1"] = result[1]
        out[f"{case}_frame_last"] = result[-1]
        out[f"{case}_num_frames"] = np.array(result.shape[0])
        out[f"{case}_all_sha256"] = np.array(sha256_of_array(result))

    return out


def check_stereo_full_chain_error(wav_paths: dict) -> dict:
    """Confirms (does not silently assume) that feeding a *raw*, un-downmixed
    stereo Signal through the standard chain raises -- ShortTimeFourierTransform
    requires 2D (frames, samples) input and rejects the 3D (frames, samples,
    channels) shape a multi-channel FramedSignal produces. Recorded as text
    (error type + message), not array data, and surfaced in the manifest so a
    port's test suite can assert the same failure mode rather than silently
    "succeeding" with wrong per-channel semantics."""
    from madmom.audio.signal import Signal

    chain = build_spec_processor()
    sig = Signal(wav_paths["stereo_44100"])
    try:
        chain(sig)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, we want the type+message
        return {"error_type": type(exc).__name__, "error_message": str(exc)}
    raise AssertionError(
        "expected feeding a raw stereo Signal through the standard spectrogram "
        "chain to raise (STFT requires mono) -- madmom's behavior may have "
        "changed; re-check docs/DESIGN.md and this script's assumptions."
    )


# ==========================================================================
# Phase 7a: toy hand-built HMM (TransitionModel.from_dense + DiscreteObservationModel)
# ==========================================================================

def generate_hmm_toy_fixtures() -> dict:
    from madmom.ml.hmm import DiscreteObservationModel, HiddenMarkovModel, TransitionModel

    rng = np.random.default_rng(HMM_TOY_SEED)
    num_states = 10
    num_obs_types = 5

    # dense transition matrix (row = prev_state, sums to 1), sparsified so the
    # resulting CSR structure has a realistic ragged degree per state.
    dense = rng.dirichlet(np.ones(num_states) * 0.5, size=num_states)
    dense[dense < 0.05] = 0
    dense = dense / dense.sum(axis=1, keepdims=True)

    prev_states, states, probs = [], [], []
    for i in range(num_states):
        for j in range(num_states):
            if dense[i, j] > 0:
                prev_states.append(i)
                states.append(j)
                probs.append(dense[i, j])

    tm = TransitionModel.from_dense(
        np.array(states), np.array(prev_states), np.array(probs)
    )

    obs_probs = rng.dirichlet(np.ones(num_obs_types) * 0.7, size=num_states)
    om = DiscreteObservationModel(obs_probs)

    hmm = HiddenMarkovModel(tm, om)
    obs_seq = rng.integers(0, num_obs_types, size=20)

    path, log_prob = hmm.viterbi(obs_seq)
    forward_out = hmm.forward(obs_seq)

    return {
        "dense_transition_matrix": dense,
        "tm_states": tm.states,
        "tm_pointers": tm.pointers,
        "tm_probabilities": tm.probabilities,
        "observation_probabilities": obs_probs,
        "observation_sequence": obs_seq,
        "viterbi_path": path,
        "viterbi_log_prob": np.array(log_prob),
        "forward_output": forward_out,
    }


# ==========================================================================
# Phase 7b: real BarStateSpace/BarTransitionModel/RNNDownBeatTrackingObservationModel
# metadata + CSR ground truth, for beats_per_bar in {3, 4} separately.
# ==========================================================================

def generate_beats_hmm_fixtures() -> dict:
    from madmom.features.beats_hmm import (
        BarStateSpace,
        BarTransitionModel,
        RNNDownBeatTrackingObservationModel,
    )

    fps = DBN_PARAMS["fps"]
    min_bpm = DBN_PARAMS["min_bpm"]
    max_bpm = DBN_PARAMS["max_bpm"]
    num_tempi = DBN_PARAMS["num_tempi"]
    transition_lambda = DBN_PARAMS["transition_lambda"]
    observation_lambda = DBN_PARAMS["observation_lambda"]

    min_interval = 60.0 * fps / max_bpm
    max_interval = 60.0 * fps / min_bpm

    out = {}
    for beats_per_bar in (3, 4):
        st = BarStateSpace(beats_per_bar, min_interval, max_interval, num_tempi)
        tm = BarTransitionModel(st, transition_lambda)
        om = RNNDownBeatTrackingObservationModel(st, observation_lambda)

        prefix = f"bpb{beats_per_bar}"
        out[f"{prefix}_num_states"] = np.array(st.num_states)
        out[f"{prefix}_num_beats"] = np.array(st.num_beats)
        # first_states/last_states are lists of `num_beats` arrays, each of
        # length num_tempi (constant across beats for a fixed BarStateSpace)
        # -- safe to stack into one 2D array.
        out[f"{prefix}_first_states"] = np.stack(st.first_states)
        out[f"{prefix}_last_states"] = np.stack(st.last_states)

        out[f"{prefix}_tm_num_states"] = np.array(tm.num_states)
        out[f"{prefix}_tm_num_transitions"] = np.array(tm.num_transitions)
        out[f"{prefix}_tm_states"] = tm.states
        out[f"{prefix}_tm_pointers"] = tm.pointers
        out[f"{prefix}_tm_probabilities"] = tm.probabilities

        out[f"{prefix}_om_pointers"] = om.pointers

    return out


# ==========================================================================
# Phase 7c: full DBNDownBeatTrackingProcessor decode on a synthetic ~30s
# seeded beat-activation array.
# ==========================================================================

def generate_dbn_fixtures() -> dict:
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    fps = DBN_PARAMS["fps"]
    rng = np.random.default_rng(DBN_ACTIVATION_SEED)
    n_frames = 30 * fps  # ~30s

    # Synthesize a plausible (beat, downbeat) activation array: periodic
    # gaussian-ish pulses at ~128bpm, downbeat pulses every 4th beat, plus
    # seeded noise. Kept strictly below sum()==1 per frame (renormalized
    # where needed) -- RNNDownBeatTrackingObservationModel.log_densities()
    # takes log(1 - sum(observations)), which is -inf/nan at the boundary;
    # avoiding that here keeps the fixture a "normal" decode, not an edge case.
    act = np.full((n_frames, 2), 0.02, dtype=np.float64)
    beat_period = int(fps * 60 / 128)
    width = 3
    for beat_idx, i in enumerate(range(0, n_frames, beat_period)):
        for w in range(-width, width + 1):
            idx = i + w
            if 0 <= idx < n_frames:
                act[idx, 0] = max(act[idx, 0], 0.02 + 0.6 * np.exp(-0.5 * (w / 1.2) ** 2))
        if beat_idx % 4 == 0:
            for w in range(-width, width + 1):
                idx = i + w
                if 0 <= idx < n_frames:
                    act[idx, 1] = max(
                        act[idx, 1], 0.02 + 0.3 * np.exp(-0.5 * (w / 1.2) ** 2)
                    )
    act += rng.normal(0, 0.005, size=act.shape)
    act = np.clip(act, 0.001, None)
    sums = act.sum(axis=1)
    over_budget = sums >= 0.98
    act[over_budget] *= (0.9 / sums[over_budget])[:, None]
    act = act.astype(np.float32)

    proc = DBNDownBeatTrackingProcessor(**DBN_PARAMS)
    beat_times = proc(act)

    return {
        "activations": act,
        "beat_times": np.asarray(beat_times),
    }


# ==========================================================================
# Manifest + orchestration
# ==========================================================================

def build_manifest(error_case: dict) -> dict:
    madmom_dist = importlib.metadata.distribution("madmom")
    madmom_commit = None
    try:
        direct_url = json.loads(
            (Path(madmom_dist._path) / "direct_url.json").read_text()
        )
        madmom_commit = direct_url.get("vcs_info", {}).get("commit_id")
    except Exception:
        pass

    return {
        "madmom_version": madmom_dist.version,
        "madmom_commit": madmom_commit,
        "madmom_source": "https://github.com/CPJKU/madmom",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "generation_command": f"{sys.executable} tools/generate_fixtures.py",
        "seeds": {
            "wav_seed": WAV_SEED,
            "hmm_toy_seed": HMM_TOY_SEED,
            "dbn_activation_seed": DBN_ACTIVATION_SEED,
        },
        "spectrogram_chain_params": {
            "frame_size": FRAME_SIZE,
            "fps": FPS,
            "num_bands": NUM_BANDS,
            "fmin": FMIN,
            "fmax": FMAX,
            "norm_filters": NORM_FILTERS,
            "log_mul": LOG_MUL,
            "log_add": LOG_ADD,
        },
        "dbn_params": DBN_PARAMS,
        "wav_duration_s": WAV_DURATION_S,
        "known_error_cases": {
            "stereo_full_chain": error_case,
        },
        "fixture_files": {
            "signal.npz": "Signal(path)/Signal(path,num_channels=1)/Signal(ndarray) outputs, per test wav",
            "framing.npz": "FramedSignalProcessor(frame_size=2048,fps=100) frame0/frame1/last frame + all-frames hash, per raw test wav (incl. multi-channel)",
            "stft.npz": "ShortTimeFourierTransformProcessor() complex STFT of mono-compatible signals + window-caching-gotcha demonstration",
            "filterbank.npz": "FilteredSpectrogramProcessor filterbank matrix (per sample rate) + filtered spectrogram outputs",
            "logspec.npz": "LogarithmicSpectrogramProcessor(mul=1,add=1) outputs",
            "full_chain.npz": "SequentialProcessor([frames,stft,filt,spec]) end-to-end output per mono-compatible test wav",
            "hmm_toy.npz": "hand-built 10-state TransitionModel/DiscreteObservationModel viterbi()/forward() outputs",
            "beats_hmm.npz": "BarStateSpace/BarTransitionModel/RNNDownBeatTrackingObservationModel metadata + CSR ground truth, beats_per_bar in {3,4}",
            "dbn_downbeat.npz": "DBNDownBeatTrackingProcessor(beats_per_bar=[3,4],fps=100) decode of a synthetic ~30s beat-activation array",
        },
    }


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not madmom_infer).\n"
            "Run it with the madmom-reference venv's interpreter -- see this file's "
            "module docstring for the exact command.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"Using madmom {madmom.__version__} from {madmom.__file__}")
    print(f"numpy {np.__version__}, scipy {scipy.__version__}, python {platform.python_version()}")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 0: synthesizing test wavs ...")
    wav_paths = make_test_wavs()

    print("Phase 1: Signal fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "signal.npz", **generate_signal_fixtures(wav_paths))

    print("Phase 2: FramedSignalProcessor fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "framing.npz", **generate_framing_fixtures(wav_paths))

    print("Phase 3: STFT fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "stft.npz", **generate_stft_fixtures(wav_paths))

    print("Phase 4: filterbank fixtures ...")
    np.savez_compressed(
        FIXTURES_DIR / "filterbank.npz", **generate_filterbank_fixtures(wav_paths)
    )

    print("Phase 5: log-spectrogram fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "logspec.npz", **generate_logspec_fixtures(wav_paths))

    print("Phase 6: full-chain fixtures ...")
    np.savez_compressed(
        FIXTURES_DIR / "full_chain.npz", **generate_full_chain_fixtures(wav_paths)
    )
    error_case = check_stereo_full_chain_error(wav_paths)
    print(f"  confirmed stereo full-chain error case: {error_case}")

    print("Phase 7a: toy HMM fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "hmm_toy.npz", **generate_hmm_toy_fixtures())

    print("Phase 7b: beats_hmm state-space/transition-model fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "beats_hmm.npz", **generate_beats_hmm_fixtures())

    print("Phase 7c: DBNDownBeatTrackingProcessor fixtures ...")
    np.savez_compressed(FIXTURES_DIR / "dbn_downbeat.npz", **generate_dbn_fixtures())

    print("Writing manifest.json ...")
    manifest = build_manifest(error_case)
    (FIXTURES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total_bytes = sum(
        f.stat().st_size for f in FIXTURES_DIR.rglob("*") if f.is_file()
    )
    print(f"Done. Total tests/fixtures/ size: {total_bytes / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
