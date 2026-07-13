"""Task-level MIR API hiding processor composition and audio normalization.

The existing ``audio`` and ``features`` modules remain the advanced,
madmom-compatible surface. This module owns the small user-facing vocabulary:
give it audio, choose a musical task, and receive the final semantic result.
"""

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from threading import RLock

import numpy as np
from scipy.signal import resample_poly

from .audio.signal import Signal
from .audio.spectrogram import Spectrogram

MODEL_SAMPLE_RATE = 44100
TASKS = frozenset({
    "onsets", "beats", "downbeats", "tempo", "key", "chords", "notes",
    "chroma", "mfcc", "hpss",
})


def _sample_rate_of(data):
    if isinstance(data, Signal):
        return data.sample_rate
    if isinstance(data, Spectrogram):
        return data.stft.frames.signal.sample_rate
    return None


def _audio_signal(audio, sample_rate=None, target_rate=MODEL_SAMPLE_RATE):
    """Return normalized mono float32 audio at ``target_rate``.

    The stable dtype is part of the clean-API boundary: advanced processors
    cache dtype-scaled FFT windows, so allowing dtype to vary across calls
    would make a reusable ``Analyzer`` depend on its input history.
    """
    if isinstance(audio, Spectrogram):
        raise TypeError("this task expects audio, not a Spectrogram")
    known_rate = _sample_rate_of(audio)
    if known_rate is not None and sample_rate is not None:
        if int(known_rate) != int(sample_rate):
            raise ValueError("sample_rate conflicts with the input metadata")
    if isinstance(audio, np.ndarray) and sample_rate is None:
        raise ValueError("sample_rate is required for NumPy audio arrays")
    if isinstance(audio, (str, Path)):
        signal = Signal(audio, sample_rate=None, num_channels=1)
    elif isinstance(audio, Signal):
        signal = Signal(audio, sample_rate=known_rate, num_channels=1)
    else:
        signal = Signal(np.asarray(audio), sample_rate=sample_rate,
                        num_channels=1)
    source_rate = int(signal.sample_rate)
    data = np.asarray(signal)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32, copy=False)
    if source_rate == target_rate:
        return Signal(data, sample_rate=target_rate, num_channels=1)
    factor = gcd(source_rate, target_rate)
    data = resample_poly(data, target_rate // factor,
                         source_rate // factor, axis=0)
    return Signal(data.astype(np.float32, copy=False),
                  sample_rate=target_rate, num_channels=1)


def _spectrogram(audio, sample_rate=None):
    if isinstance(audio, Spectrogram):
        known_rate = _sample_rate_of(audio)
        if sample_rate is not None and int(sample_rate) != int(known_rate):
            raise ValueError("sample_rate conflicts with the input metadata")
        return audio
    return Spectrogram(_audio_signal(audio, sample_rate))


@dataclass(frozen=True)
class AnalysisResult:
    """Named results returned by a multi-task :class:`Analyzer` call."""

    values: dict

    def __getitem__(self, task):
        return self.values[task]

    def __getattr__(self, task):
        try:
            return self.values[task]
        except KeyError as exc:
            raise AttributeError(task) from exc


class Analyzer:
    """Lazily build and reuse canonical pipelines for selected MIR tasks."""

    def __init__(self, tasks=TASKS, beats_per_bar=(3, 4)):
        self.tasks = tuple(tasks)
        unknown = set(self.tasks) - TASKS
        if unknown:
            raise ValueError("unknown analysis task(s): %s" % sorted(unknown))
        self.beats_per_bar = beats_per_bar
        self._processors = {}
        self._call_lock = RLock()

    def _processor(self, task):
        if task not in self._processors:
            self._processors[task] = self._build_processor(task)
        return self._processors[task]

    def _build_processor(self, task):
        if task == "onsets":
            from .features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
            return CNNOnsetProcessor(), OnsetPeakPickingProcessor(fps=100)
        if task in ("beats", "tempo"):
            from .features.beats import RNNBeatProcessor
            return RNNBeatProcessor()
        if task == "downbeats":
            from .features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
            return (RNNDownBeatProcessor(), DBNDownBeatTrackingProcessor(
                beats_per_bar=self.beats_per_bar, fps=100))
        if task == "key":
            from .features.key import CNNKeyRecognitionProcessor
            return CNNKeyRecognitionProcessor()
        if task in ("chords", "chroma"):
            from .audio.chroma import DeepChromaProcessor
            return DeepChromaProcessor()
        if task == "notes":
            from .features.notes import CNNPianoNoteProcessor, ADSRNoteTrackingProcessor
            return CNNPianoNoteProcessor(), ADSRNoteTrackingProcessor()
        return None

    def __call__(self, audio, *, sample_rate=None):
        with self._call_lock:
            return self._analyze(audio, sample_rate=sample_rate)

    def _analyze(self, audio, *, sample_rate=None):
        signal = _audio_signal(audio, sample_rate)
        values = {}
        shared = {}
        for task in self.tasks:
            if task == "onsets":
                frontend, decode = self._processor(task)
                values[task] = decode(frontend(signal))
            elif task in ("beats", "tempo"):
                if "beat_activations" not in shared:
                    shared["beat_activations"] = self._processor(task)(signal)
                if task == "beats":
                    from .features.beats import DBNBeatTrackingProcessor
                    values[task] = DBNBeatTrackingProcessor(fps=100)(shared["beat_activations"])
                else:
                    from .features.tempo import TempoEstimationProcessor
                    values[task] = TempoEstimationProcessor(fps=100)(shared["beat_activations"])
            elif task == "downbeats":
                frontend, decode = self._processor(task)
                values[task] = decode(frontend(signal))
            elif task == "key":
                from .features.key import key_prediction_to_label
                values[task] = key_prediction_to_label(self._processor(task)(signal))
            elif task in ("chroma", "chords"):
                if "chroma" not in shared:
                    shared["chroma"] = self._processor(task)(signal)
                if task == "chroma":
                    values[task] = shared["chroma"]
                else:
                    from .features.chords import DeepChromaChordRecognitionProcessor
                    values[task] = DeepChromaChordRecognitionProcessor()(shared["chroma"])
            elif task == "notes":
                frontend, decode = self._processor(task)
                values[task] = decode(frontend(signal))
            elif task == "mfcc":
                from .audio.cepstrogram import MFCC
                values[task] = np.asarray(MFCC(Spectrogram(signal)))
            elif task == "hpss":
                from .audio.hpss import HPSS
                values[task] = HPSS()(Spectrogram(signal))
        return AnalysisResult(values)


def analyze(audio, *, tasks=TASKS, sample_rate=None, beats_per_bar=(3, 4)):
    return Analyzer(tasks=tasks, beats_per_bar=beats_per_bar)(
        audio, sample_rate=sample_rate)


def _one(task, audio, sample_rate=None, **kwargs):
    return Analyzer(tasks=(task,), **kwargs)(audio, sample_rate=sample_rate)[task]


def detect_onsets(audio, *, sample_rate=None): return _one("onsets", audio, sample_rate)
def detect_beats(audio, *, sample_rate=None): return _one("beats", audio, sample_rate)
def detect_downbeats(audio, *, sample_rate=None, beats_per_bar=(3, 4)):
    return _one("downbeats", audio, sample_rate, beats_per_bar=beats_per_bar)
def estimate_tempo(audio, *, sample_rate=None): return _one("tempo", audio, sample_rate)
def detect_key(audio, *, sample_rate=None): return _one("key", audio, sample_rate)
def recognize_chords(audio, *, sample_rate=None): return _one("chords", audio, sample_rate)
def transcribe_notes(audio, *, sample_rate=None): return _one("notes", audio, sample_rate)
def chroma(audio, *, sample_rate=None): return _one("chroma", audio, sample_rate)


def mfcc(audio, *, sample_rate=None, **options):
    from .audio.cepstrogram import MFCC
    return np.asarray(MFCC(_spectrogram(audio, sample_rate), **options))


def hpss(audio, *, sample_rate=None, **options):
    from .audio.hpss import HPSS
    return HPSS(**options)(_spectrogram(audio, sample_rate))
