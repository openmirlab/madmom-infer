"""User-facing contract tests for the task-level clean API."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import madmom_infer as mm
from madmom_infer.api import MadmomAnalyzer, _audio_signal

WAV = Path(__file__).parent / "fixtures" / "wavs" / "mono_44100.wav"


def test_top_level_import_stays_torch_free():
    assert "torch" not in sys.modules


def test_array_requires_sample_rate():
    with pytest.raises(ValueError, match="sample_rate is required"):
        mm.mfcc(np.zeros(4410, dtype=np.float32))


def test_path_mfcc_and_hpss():
    coeffs = mm.mfcc(WAV, sample_rate=44100)
    harmonic, percussive = mm.hpss(WAV, sample_rate=44100)
    assert coeffs.ndim == 2 and coeffs.shape[1] == 30
    assert harmonic.shape == percussive.shape


def test_resamples_array_to_model_rate():
    rate, data = wavfile.read(WAV)
    signal = _audio_signal(data[::2], sample_rate=rate // 2)
    assert signal.sample_rate == 44100
    assert signal.dtype == np.float32
    assert abs(len(signal) - len(data)) <= 2


def test_audio_boundary_has_stable_dtype_and_scale():
    rate, data = wavfile.read(WAV)
    native = _audio_signal(WAV, sample_rate=rate)
    resampled = _audio_signal(data[::2], sample_rate=rate // 2)
    assert native.dtype == resampled.dtype == np.float32
    assert np.max(np.abs(native)) <= 1.0
    assert np.max(np.abs(resampled)) <= 1.1


def test_reused_analyzer_feeds_cached_processor_a_stable_dtype(monkeypatch):
    rate, data = wavfile.read(WAV)
    seen_dtypes = []

    def processor(signal):
        seen_dtypes.append(signal.dtype)
        return np.zeros((1, 12), dtype=np.float32)

    analyzer = MadmomAnalyzer(tasks=("chroma",))
    monkeypatch.setattr(analyzer, "_build_processor", lambda task: processor)
    analyzer(WAV, sample_rate=rate)
    analyzer(data[::2], sample_rate=rate // 2)

    assert seen_dtypes == [np.dtype("float32"), np.dtype("float32")]
    assert len(analyzer._processors) == 1


@pytest.mark.network
@pytest.mark.parametrize("task", ["beats", "onsets"])
def test_analyzer_reuse_across_resampling_matches_fresh(task):
    rate, data = wavfile.read(WAV)
    half_rate_audio = data[::2]
    shared = MadmomAnalyzer(tasks=(task,))

    shared(WAV, sample_rate=rate)
    reused = shared(half_rate_audio, sample_rate=rate // 2)[task]
    fresh = MadmomAnalyzer(tasks=(task,))(
        half_rate_audio, sample_rate=rate // 2)[task]
    np.testing.assert_array_equal(reused, fresh)

    shared = MadmomAnalyzer(tasks=(task,))
    shared(half_rate_audio, sample_rate=rate // 2)
    reused = shared(WAV, sample_rate=rate)[task]
    fresh = MadmomAnalyzer(tasks=(task,))(WAV, sample_rate=rate)[task]
    np.testing.assert_array_equal(reused, fresh)


def test_analyzer_rejects_unknown_task():
    with pytest.raises(ValueError, match="unknown analysis task"):
        MadmomAnalyzer(tasks=("genre",))


def test_tempo_from_downbeat_activations_needs_both_tasks():
    with pytest.raises(ValueError, match="needs both 'tempo' and 'downbeats'"):
        MadmomAnalyzer(tasks=("tempo",), tempo_from_downbeat_activations=True)
    with pytest.raises(ValueError, match="needs both 'tempo' and 'downbeats'"):
        MadmomAnalyzer(tasks=("downbeats",), tempo_from_downbeat_activations=True)


def test_tempo_from_downbeat_activations_skips_the_beat_ensemble(monkeypatch):
    """The whole point of the flag: `RNNBeatProcessor` is never built or run."""
    analyzer = MadmomAnalyzer(tasks=("downbeats", "tempo"),
                              tempo_from_downbeat_activations=True)
    assert analyzer._build_processor("tempo") is None

    calls = {"downbeat_frontend": 0}

    def downbeat_frontend(signal):
        calls["downbeat_frontend"] += 1
        # (num_frames, 2): column 0 is the beat activation, column 1 the downbeat one
        activations = np.zeros((300, 2), dtype=np.float32)
        activations[::40, 0] = 1.0
        activations[::160, 1] = 1.0
        return activations

    monkeypatch.setattr(analyzer, "_build_processor",
                        lambda task: (downbeat_frontend, lambda act: act)
                        if task == "downbeats" else None)
    result = analyzer(np.zeros(44100, dtype=np.float32), sample_rate=44100)

    # one ensemble run, feeding both tasks
    assert calls["downbeat_frontend"] == 1
    assert set(result.values) == {"downbeats", "tempo"}
    assert result["tempo"].shape[1] == 2


def test_downbeat_activations_are_not_shared_when_the_flag_is_off(monkeypatch):
    """Default stays bug-for-bug: `tempo` runs its own ensemble."""
    analyzer = MadmomAnalyzer(tasks=("downbeats", "tempo"))
    calls = {"downbeats": 0, "tempo": 0}

    def downbeat_frontend(signal):
        calls["downbeats"] += 1
        return np.zeros((300, 2), dtype=np.float32)

    def beat_ensemble(signal):
        calls["tempo"] += 1
        return np.zeros(300, dtype=np.float32)

    monkeypatch.setattr(analyzer, "_build_processor",
                        lambda task: (downbeat_frontend, lambda act: act)
                        if task == "downbeats" else beat_ensemble)
    analyzer(np.zeros(44100, dtype=np.float32), sample_rate=44100)

    assert calls == {"downbeats": 1, "tempo": 1}


@pytest.mark.network
def test_tempo_from_downbeat_activations_leaves_downbeats_bit_identical():
    """`downbeats` never rides on the flag; only `tempo`'s input changes.

    Deliberately no assertion on the tempo VALUES: the two activations are not
    interchangeable (see `MadmomAnalyzer`'s docstring), and this fixture is 1.5 s
    of audio -- far too short for tempo estimation to mean anything, so its two
    leading candidates disagree (240.0 vs 230.8 BPM) purely as noise. What the
    flag guarantees is the shape of the contract, not agreement.
    """
    rate, _ = wavfile.read(WAV)
    shared = MadmomAnalyzer(tasks=("downbeats", "tempo"),
                            tempo_from_downbeat_activations=True)(WAV, sample_rate=rate)
    separate = MadmomAnalyzer(tasks=("downbeats", "tempo"))(WAV, sample_rate=rate)

    np.testing.assert_array_equal(shared["downbeats"], separate["downbeats"])
    assert shared["tempo"].ndim == 2 and shared["tempo"].shape[1] == 2
    assert np.all(shared["tempo"][:, 0] > 0)
