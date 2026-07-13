"""User-facing contract tests for the task-level clean API."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import madmom_infer as mm
from madmom_infer.api import Analyzer, _audio_signal

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

    analyzer = Analyzer(tasks=("chroma",))
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
    shared = Analyzer(tasks=(task,))

    shared(WAV, sample_rate=rate)
    reused = shared(half_rate_audio, sample_rate=rate // 2)[task]
    fresh = Analyzer(tasks=(task,))(
        half_rate_audio, sample_rate=rate // 2)[task]
    np.testing.assert_array_equal(reused, fresh)

    shared = Analyzer(tasks=(task,))
    shared(half_rate_audio, sample_rate=rate // 2)
    reused = shared(WAV, sample_rate=rate)[task]
    fresh = Analyzer(tasks=(task,))(WAV, sample_rate=rate)[task]
    np.testing.assert_array_equal(reused, fresh)


def test_analyzer_rejects_unknown_task():
    with pytest.raises(ValueError, match="unknown analysis task"):
        Analyzer(tasks=("genre",))
