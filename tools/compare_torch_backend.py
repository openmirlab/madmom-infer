#!/usr/bin/env python3
"""Measure the torch backend (`madmom_infer/torch/features/`) against the
numpy reference on real (non-fixture) audio: for every one of the 9
pipelines, run the numpy `SequentialProcessor` and the torch
`TorchPipelineProcessor` adapter on the same audio file, report the
activation-level numeric agreement, decode both activations with the
matching numpy decoder, and report whether the decoded results agree
(exact equality, or a quantified difference if not) plus wall-clock timing
for numpy / torch-cpu / torch-cuda.

This is a manual measurement tool, not a pytest test (no assertions,
network access required for weight downloads, and it is meant to be run
against arbitrary real-world audio the repo does not -- and must not --
ship, per the CC-BY-NC-SA weights policy and general copyright hygiene).

Usage:
    uv run python tools/compare_torch_backend.py AUDIO.wav [AUDIO2.wav ...] \\
        --device cuda --dtype float32

`--allow-tf32` (default off): on Ampere+ GPUs, torch's own default
(`torch.backends.cudnn.allow_tf32=True`,
`torch.backends.cuda.matmul.allow_tf32=True`) makes the CNN-heavy
pipelines (onsets_cnn, key, notes_cnn) drift up to ~1e-3 from the numpy
reference on CUDA -- passing this flag re-enables that default; leaving it
off (this tool's default) forces both flags `False` for a tighter ~1e-6
match. See `madmom_infer/torch/features/adapter.py`'s module header for
the full measurement.

Reads: madmom_infer.torch.features (build_pipeline, TorchPipelineProcessor),
the 9 numpy feature-family modules (imported lazily), and their matching
decoders (DBNDownBeatTrackingProcessor,
DBNBeatTrackingProcessor, OnsetPeakPickingProcessor,
key_prediction_to_label, DeepChromaChordRecognitionProcessor,
CRFChordRecognitionProcessor, NoteOnsetPeakPickingProcessor,
ADSRNoteTrackingProcessor).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from madmom_infer.torch.features import build_pipeline
from madmom_infer.torch.features.adapter import TorchPipelineProcessor


def _numpy_processor(name):
    if name == "downbeats":
        from madmom_infer.features.downbeats import RNNDownBeatProcessor
        return RNNDownBeatProcessor()
    if name == "beats":
        from madmom_infer.features.beats import RNNBeatProcessor
        return RNNBeatProcessor()
    if name == "onsets_rnn":
        from madmom_infer.features.onsets import RNNOnsetProcessor
        return RNNOnsetProcessor()
    if name == "onsets_cnn":
        from madmom_infer.features.onsets import CNNOnsetProcessor
        return CNNOnsetProcessor()
    if name == "key":
        from madmom_infer.features.key import CNNKeyRecognitionProcessor
        return CNNKeyRecognitionProcessor()
    if name == "chroma":
        from madmom_infer.audio.chroma import DeepChromaProcessor
        return DeepChromaProcessor()
    if name == "chords_feat":
        from madmom_infer.features.chords import CNNChordFeatureProcessor
        return CNNChordFeatureProcessor()
    if name == "notes_rnn":
        from madmom_infer.features.notes import RNNPianoNoteProcessor
        return RNNPianoNoteProcessor()
    if name == "notes_cnn":
        from madmom_infer.features.notes import CNNPianoNoteProcessor
        return CNNPianoNoteProcessor()
    raise ValueError(name)


def _decode(name, activations):
    """Decode `activations` with the matching numpy decoder. Returns
    `(decoded, describe_fn)` where `describe_fn(decoded)` renders a short
    string for comparison/printing."""
    if name == "downbeats":
        from madmom_infer.features.downbeats import DBNDownBeatTrackingProcessor
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        return dbn(activations)
    if name == "beats":
        from madmom_infer.features.beats import DBNBeatTrackingProcessor
        dbn = DBNBeatTrackingProcessor(fps=100)
        return dbn(activations)
    if name in ("onsets_rnn", "onsets_cnn"):
        from madmom_infer.features.onsets import OnsetPeakPickingProcessor
        pp = OnsetPeakPickingProcessor(fps=100)
        return pp(activations)
    if name == "key":
        from madmom_infer.features.key import key_prediction_to_label
        return key_prediction_to_label(activations)
    if name == "chroma":
        from madmom_infer.features.chords import DeepChromaChordRecognitionProcessor
        decode = DeepChromaChordRecognitionProcessor()
        return decode(activations)
    if name == "chords_feat":
        from madmom_infer.features.chords import CRFChordRecognitionProcessor
        decode = CRFChordRecognitionProcessor()
        return decode(activations)
    if name == "notes_rnn":
        from madmom_infer.features.notes import NoteOnsetPeakPickingProcessor
        pp = NoteOnsetPeakPickingProcessor(fps=100, pitch_offset=21)
        return pp(activations)
    if name == "notes_cnn":
        from madmom_infer.features.notes import ADSRNoteTrackingProcessor
        adsr = ADSRNoteTrackingProcessor(fps=100)
        return adsr(activations)
    raise ValueError(name)


def _compare_decoded(name, decoded_np, decoded_torch):
    if name == "key":
        return decoded_np == decoded_torch, f"np={decoded_np!r} torch={decoded_torch!r}"
    a = np.asarray(decoded_np)
    b = np.asarray(decoded_torch)
    if a.shape != b.shape:
        return False, f"shape mismatch: np={a.shape} torch={b.shape}"
    if a.dtype.names:  # structured array (chord segments)
        starts_eq = np.allclose(a["start"], b["start"], atol=1e-6)
        ends_eq = np.allclose(a["end"], b["end"], atol=1e-6)
        labels_eq = bool(np.all(a["label"] == b["label"]))
        ok = starts_eq and ends_eq and labels_eq
        max_time_off = float(np.max(np.abs(a["start"] - b["start"]))) if a.size else 0.0
        return ok, f"labels_eq={labels_eq} max_time_offset={max_time_off:.6f}"
    exact = np.array_equal(a, b)
    if exact:
        return True, "exact"
    if a.size == 0 and b.size == 0:
        return True, "both empty"
    if a.size != b.size:
        return False, f"count mismatch: np={a.size} torch={b.size}"
    max_off = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return False, f"max offset={max_off:.6f}"


def _activation_diff(np_act, torch_act):
    a = np.asarray(np_act).astype(np.float64)
    b = np.asarray(torch_act).astype(np.float64)
    if a.shape != b.shape:
        return dict(shape_mismatch=(a.shape, b.shape))
    diff = np.abs(a - b)
    value_range = max(float(a.max() - a.min()), 1e-12)
    return dict(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        max_rel_to_range_diff=float(diff.max() / value_range),
    )


PIPELINE_NAMES = [
    "downbeats", "beats", "onsets_rnn", "onsets_cnn", "key", "chroma",
    "chords_feat", "notes_rnn", "notes_cnn",
]


def run(audio_path, device, dtype):
    print(f"\n=== {audio_path} (device={device}, dtype={dtype}) ===")
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]

    for name in PIPELINE_NAMES:
        print(f"\n--- {name} ---")
        numpy_proc = _numpy_processor(name)

        t0 = time.perf_counter()
        np_act = np.asarray(numpy_proc(audio_path))
        t_numpy = time.perf_counter() - t0

        pipeline = build_pipeline(name, dtype=torch_dtype).to(device).eval()
        adapter = TorchPipelineProcessor(pipeline, device=device, dtype=torch_dtype)

        t0 = time.perf_counter()
        torch_act = adapter(audio_path)
        if device == "cuda":
            torch.cuda.synchronize()
        t_torch = time.perf_counter() - t0

        diff = _activation_diff(np_act, torch_act)
        print(f"  activation diff: {diff}")
        print(f"  timing: numpy={t_numpy:.3f}s torch[{device}]={t_torch:.3f}s")

        try:
            decoded_np = _decode(name, np_act)
            decoded_torch = _decode(name, torch_act)
            ok, detail = _compare_decoded(name, decoded_np, decoded_torch)
            print(f"  decoded agreement: {'IDENTICAL' if ok else 'DIFFERS'} ({detail})")
        except Exception as exc:  # pragma: no cover - measurement tool
            print(f"  decode step failed: {exc!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", help="path(s) to audio file(s)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--allow-tf32", action="store_true",
        help="allow TF32 matmul/cudnn on CUDA (default: off, forced False "
             "for tighter numpy parity -- see this tool's module header)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is available")

    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    for audio_path in args.audio:
        run(audio_path, args.device, args.dtype)


if __name__ == "__main__":
    main()
