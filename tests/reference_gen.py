"""Standalone golden-reference generator for tests/test_signal.py.

Computes real madmom's `Signal`/`FramedSignal`/`remix` outputs for a fixed
set of deterministic test cases and dumps them to an .npz file. Must be run
under a *working* madmom install (this repo has none by design -- see
docs/DESIGN.md); `test_signal.py` invokes this script as a subprocess of
`/home/worzpro/Desktop/dev/openmirlab/all-in-one-fix/.venv/bin/python`
(madmom 0.17.dev0, numpy 1.23.5, scipy 1.15.3).

Deliberately standalone: no import of `madmom_infer` anywhere in this file,
so it stays runnable from any environment that has real `madmom` installed,
independent of this package (per the task brief: don't depend on another
workstream's fixture harness -- this script is this workstream's own,
self-sufficient one).

Usage: python reference_gen.py <wav_dir> <output_npz_path>

`<wav_dir>` must contain, written by the caller (test_signal.py) before this
script runs:
  - mono16_44100.wav    (int16 PCM, mono, 44100 Hz)
  - stereo16_44100.wav  (int16 PCM, stereo, 44100 Hz, channel sums crafted
                         to be odd so the mono-downmix mean is always x.5 --
                         the truncate-vs-round trap)
  - mono16_48000.wav    (int16 PCM, mono, 48000 Hz)
  - mono_float32.wav    (float32 PCM, mono, 44100 Hz)
"""

import sys

import numpy as np
from madmom.audio.signal import FramedSignal, FramedSignalProcessor, Signal, remix


def _dump_frames(results, prefix, frames):
    results[prefix + "__num_frames"] = np.array(frames.num_frames)
    results[prefix + "__hop_size"] = np.array(frames.hop_size)
    results[prefix + "__frame0"] = np.asarray(frames[0])
    results[prefix + "__frame1"] = np.asarray(frames[1])
    results[prefix + "__frame_last"] = np.asarray(frames[frames.num_frames - 1])
    results[prefix + "__dtype"] = np.array(str(frames[0].dtype))


def main():
    wav_dir, out_path = sys.argv[1], sys.argv[2]
    results = {}

    # -- A. array-based cases (no file I/O; exercises Signal-from-ndarray +
    #       FramedSignal framing/origin/num_frames math in isolation) -----
    sig_a = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=10)
    frames_center = FramedSignal(sig_a, frame_size=6, hop_size=2, origin=0)
    _dump_frames(results, "arr_center", frames_center)

    frames_left = FramedSignal(sig_a, frame_size=6, hop_size=2, origin="left")
    results["arr_left__origin"] = np.array(frames_left.origin)
    results["arr_left__frame0"] = np.asarray(frames_left[0])

    frames_right = FramedSignal(sig_a, frame_size=6, hop_size=2, origin="right")
    results["arr_right__origin"] = np.array(frames_right.origin)
    results["arr_right__frame0"] = np.asarray(frames_right[0])

    # num_frames formula divergence: 'normal' (ceil) vs 'extend' (floor+1)
    sig_b = Signal(np.arange(1, 10, dtype=np.int16), sample_rate=10)  # len 9
    fn = FramedSignal(sig_b, frame_size=4, hop_size=3, end="normal")
    fe = FramedSignal(sig_b, frame_size=4, hop_size=3, end="extend")
    results["numframes_normal"] = np.array(fn.num_frames)
    results["numframes_extend"] = np.array(fe.num_frames)

    # 48kHz hop_size derived from fps
    sig_c = Signal(np.arange(1, 21, dtype=np.int16), sample_rate=48000)
    frames_48k = FramedSignalProcessor(frame_size=2048, fps=100)(sig_c)
    results["fps48k__hop_size"] = np.array(frames_48k.hop_size)

    # remix mono-downmix truncation, direct (positive and negative x.5 mean)
    stereo_small = np.array(
        [[3, 4], [1, 2], [-3, -4], [-1, -2], [5, 6]], dtype=np.int16
    )
    results["remix_x5__mono"] = remix(stereo_small, 1)

    # -- B. wav-file-based cases (exercises the scipy.io.wavfile loading
    #       path: dtype flow-through, mmap read, real container parsing) --
    sig_mono16 = Signal("%s/mono16_44100.wav" % wav_dir)
    results["mono16_44100__dtype"] = np.array(str(sig_mono16.dtype))
    results["mono16_44100__sample_rate"] = np.array(sig_mono16.sample_rate)
    results["mono16_44100__data"] = np.asarray(sig_mono16)
    frames_mono16 = FramedSignalProcessor(frame_size=2048, fps=100)(sig_mono16)
    _dump_frames(results, "mono16_44100", frames_mono16)

    sig_stereo_mono = Signal("%s/stereo16_44100.wav" % wav_dir, num_channels=1)
    results["stereo16_mono__dtype"] = np.array(str(sig_stereo_mono.dtype))
    results["stereo16_mono__data"] = np.asarray(sig_stereo_mono)

    sig_stereo_raw = Signal("%s/stereo16_44100.wav" % wav_dir)
    results["stereo16_raw__data"] = np.asarray(sig_stereo_raw)
    results["stereo16_raw__num_channels"] = np.array(sig_stereo_raw.num_channels)

    sig_48k = Signal("%s/mono16_48000.wav" % wav_dir)
    results["mono16_48000__sample_rate"] = np.array(sig_48k.sample_rate)
    frames_48000 = FramedSignalProcessor(frame_size=2048, fps=100)(sig_48k)
    _dump_frames(results, "mono16_48000", frames_48000)

    sig_float32 = Signal("%s/mono_float32.wav" % wav_dir)
    results["mono_float32__dtype"] = np.array(str(sig_float32.dtype))
    results["mono_float32__data"] = np.asarray(sig_float32)

    np.savez(out_path, **results)


if __name__ == "__main__":
    main()
