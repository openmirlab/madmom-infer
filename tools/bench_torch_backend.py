#!/usr/bin/env python3
"""Timing harness for `madmom_infer.torch.features` pipelines (the
`SpectrogramFrontend` + `to_torch`-converted NN forward pass) against the
numpy reference processor, on real audio.

Written to establish a baseline BEFORE optimizing the torch recurrent
path (`madmom_infer/torch/ml/nn/stacked.py`) and to re-measure after each
change -- see that module's header and the implementation report for the
numbers this tool produced. Not a pytest test: it's a manual measurement
tool (network access needed for weight downloads, wall-clock timing is
meaningless as a pass/fail assertion).

For each requested pipeline, measures:

- the numpy `SequentialProcessor`'s own `.process(audio_path)` wall time;
- the torch pipeline's forward pass on each requested device, under
  `torch.no_grad()`, with warmup iterations, `torch.cuda.synchronize()`
  around every timed CUDA call, and the MEDIAN of `--repeats` timed runs
  (median, not mean, to resist one slow outlier -- e.g. a CUDA context
  hiccup -- skewing the number);
- the same torch pipeline with `requires_grad=True` timing a forward +
  `.sum().backward()` pass (training-like usage) on `cuda` if available;
- a batched run (`--batch B` independent copies of the same waveform,
  stacked on a leading batch axis) on each device, no_grad only.

`torch.backends.cudnn.allow_tf32`/`torch.backends.cuda.matmul.allow_tf32`
are forced off (this port's numerics are float32-exact-verified against
real madmom, TF32's reduced mantissa would silently reintroduce drift on
Ampere+ GPUs).

Usage:
    uv run python tools/bench_torch_backend.py AUDIO.wav [AUDIO2.wav ...] \\
        --devices cpu cuda --pipelines downbeats beats --repeats 5 --batch 4

Reads: torch, madmom_infer.torch.features (build_pipeline),
madmom_infer.features/audio.chroma (the numpy reference processors,
imported lazily per pipeline name); read by: nothing (a standalone CLI
tool, not imported elsewhere).
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from madmom_infer.torch.features import build_pipeline  # noqa: E402

_ALL_PIPELINES = [
    "downbeats", "beats", "onsets_rnn", "onsets_cnn", "key", "chroma",
    "chords_feat", "notes_rnn", "notes_cnn",
]


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


def _median_time(fn, repeats, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _load_waveform(path):
    from madmom_infer.audio.signal import Signal
    sig = Signal(path, sample_rate=44100, num_channels=1)
    return np.asarray(sig, dtype=np.float32) / 32768.0 if np.issubdtype(
        sig.dtype, np.integer
    ) else np.asarray(sig, dtype=np.float32)


def _bench_numpy(name, audio_path, repeats):
    proc = _numpy_processor(name)

    def run():
        proc(audio_path)

    return _median_time(run, repeats, warmup=1)


def _bench_torch_forward(pipeline, waveform, device, repeats, batch=None):
    pipeline = pipeline.to(device)
    x = torch.as_tensor(waveform, dtype=torch.float32, device=device)
    if batch is not None:
        x = x.unsqueeze(0).expand(batch, -1).contiguous()

    def run():
        with torch.no_grad():
            out = pipeline(x)
            if device == "cuda":
                torch.cuda.synchronize()
        return out

    return _median_time(run, repeats, warmup=2)


def _bench_torch_backward(pipeline, waveform, device, repeats):
    pipeline = pipeline.to(device)
    x0 = torch.as_tensor(waveform, dtype=torch.float32, device=device)

    def run():
        x = x0.clone().requires_grad_(True)
        out = pipeline(x)
        loss = out.float().sum()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()

    return _median_time(run, repeats, warmup=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", help="audio file(s) to benchmark")
    parser.add_argument("--devices", nargs="+", default=["cpu"],
                         choices=["cpu", "cuda"])
    parser.add_argument("--pipelines", nargs="+", default=_ALL_PIPELINES,
                         choices=_ALL_PIPELINES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--skip-backward", action="store_true",
        help="skip the fwd+bwd (training-like) timing -- useful for the "
             "--no-stack baseline, where the unfused per-timestep autograd "
             "graph makes backward extremely slow.",
    )
    parser.add_argument(
        "--no-stack", action="store_true",
        help="disable the ensemble/gate-stacked performance path "
             "(madmom_infer.torch.ml.nn.stacked) and force the original "
             "per-network EnsembleModule/NeuralNetworkModule loop, for a "
             "baseline/before-vs-after comparison.",
    )
    args = parser.parse_args()

    if args.no_stack:
        import madmom_infer.torch.ml.nn.convert as _convert
        _convert._try_stack_networks = lambda networks, trainable: None

    devices = [d for d in args.devices if d != "cuda" or torch.cuda.is_available()]
    if "cuda" in args.devices and "cuda" not in devices:
        print("cuda requested but not available -- skipping")

    for audio_path in args.audio:
        waveform = _load_waveform(audio_path)
        print(f"\n=== {audio_path} ({len(waveform) / 44100:.1f}s) ===")
        for name in args.pipelines:
            print(f"\n-- {name} --")
            try:
                t_np = _bench_numpy(name, audio_path, args.repeats)
                print(f"  numpy                 : {t_np * 1000:8.1f} ms")
            except Exception as exc:  # noqa: BLE001
                print(f"  numpy                 : FAILED ({exc})")

            pipeline = build_pipeline(name).eval()

            for device in devices:
                try:
                    t_fwd = _bench_torch_forward(pipeline, waveform, device,
                                                  args.repeats)
                    print(f"  torch {device:4s} no_grad B=1  : {t_fwd * 1000:8.1f} ms")
                except Exception as exc:  # noqa: BLE001
                    print(f"  torch {device:4s} no_grad B=1  : FAILED ({exc})")

                try:
                    t_batch = _bench_torch_forward(
                        pipeline, waveform, device, args.repeats, batch=args.batch
                    )
                    print(f"  torch {device:4s} no_grad B={args.batch:<2d} : "
                          f"{t_batch * 1000:8.1f} ms")
                except Exception as exc:  # noqa: BLE001
                    print(f"  torch {device:4s} no_grad B={args.batch:<2d} : "
                          f"FAILED ({exc})")

            if "cuda" in devices and not args.skip_backward:
                try:
                    t_bwd = _bench_torch_backward(pipeline, waveform, "cuda",
                                                   args.repeats)
                    print(f"  torch cuda fwd+bwd B=1: {t_bwd * 1000:8.1f} ms")
                except Exception as exc:  # noqa: BLE001
                    print(f"  torch cuda fwd+bwd B=1: FAILED ({exc})")


if __name__ == "__main__":
    main()
