"""Wave-4g golden-fixture generator: `audio/cepstrogram.py` (`Cepstrogram`/
`MFCC`), `audio/hpss.py` (`HarmonicPercussiveSourceSeparation`), and the six
`audio/signal.py` leftover functions (`attenuate`, `rescale`, `trim`,
`energy`, `root_mean_square`, `sound_pressure_level`) the 4b audit-table
TO-VERIFY flag left open -- the 4g sibling of `tools/generate_onset_
fixtures.py`, same conventions (own file, own fixture files, independently
regenerable without touching prior waves' already-committed fixtures).

**All fixtures here are OUTPUT-only, same economy as `generate_onset_
fixtures.py`.** Every target function is deterministic given (a) the shared
test wav and (b) this project's own already-golden-fixture-proven Phase-1
DSP chain (`SignalProcessor` -> `FramedSignalProcessor` -> `ShortTimeFourier
TransformProcessor` -> `SpectrogramProcessor`) -- so this generator records
only each function's OUTPUT array (real madmom, real pipeline, same wav);
`tests/test_cepstrogram.py`/`test_hpss.py`/`test_signal.py` reconstruct the
identical input via THIS PORT's OWN already-proven pipeline and feed it
through this port's own new function -- no intermediate spectrogram object
needs to survive serialization, and these tests run fully OFFLINE.

HOW TO RUN -- same real-madmom reference venv as every prior wave
(`madmom-reference/.venv`, Python 3.10.18, numpy 1.23.5, scipy 1.15.3):

    /home/worzpro/Desktop/dev/openmirlab/madmom-reference/.venv/bin/python \\
        tools/generate_leftovers_fixtures.py

Only ONE representative wav (`mono_44100.wav`) is needed -- none of this
wave's targets are model-dependent (no NN weights, no per-dtype/per-channel
activation drift to characterize), matching the economy `generate_key_
fixtures.py`/`generate_onset_fixtures.py` already established for their own
pure-DSP-function fixtures.

Reads: real `madmom` (audio.signal, audio.cepstrogram, audio.hpss,
audio.spectrogram, audio.stft), numpy. Writes: tests/fixtures/
cepstrogram.npz, tests/fixtures/hpss.npz, tests/fixtures/
signal_leftovers.npz. Read by: tests/test_cepstrogram.py,
tests/test_hpss.py, tests/test_signal.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
WAVS_DIR = FIXTURES_DIR / "wavs"

DSP_CASE_WAV = "mono_44100.wav"
FRAME_SIZE = 2048
FPS = 100


def _spectrogram(wav_path):
    from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
    from madmom.audio.spectrogram import SpectrogramProcessor
    from madmom.audio.stft import ShortTimeFourierTransformProcessor

    sig = SignalProcessor(num_channels=1, sample_rate=44100)
    frames = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)
    stft = ShortTimeFourierTransformProcessor()
    spec = SpectrogramProcessor()
    return spec(stft(frames(sig(str(wav_path)))))


def generate_cepstrogram_fixtures() -> dict:
    """`Cepstrogram` (default DCT transform, works fine from a plain
    `Spectrogram`) and `MFCC` outputs.

    **Real, confirmed upstream bug**: `MFCC.__new__` (`cepstrogram.py:
    197-200`) unconditionally evaluates `spectrogram.filterbank is not
    None or ...` -- but the base `Spectrogram` class never sets a
    `.filterbank` attribute at all (only `FilteredSpectrogram` does, as a
    plain attribute; `LogarithmicSpectrogram.filterbank` is a PROPERTY that
    forwards to `self.spectrogram.filterbank`, which raises the same way
    for an underlying plain `Spectrogram`). So `MFCC(plain_spectrogram)`
    -- the seemingly-primary, most obvious use case, and what a raw wav
    path/array argument builds internally too -- unconditionally raises
    `AttributeError: 'Spectrogram' object has no attribute 'filterbank'`,
    confirmed directly against the reference venv. The ONLY input that
    doesn't crash is an ALREADY-`FilteredSpectrogram` instance: its real
    `.filterbank` attribute is not None, which trips the "Spectrogram was
    filtered or scaled already, redo calculation!" warn-and-recompute
    branch -- which discards that filter, rebuilds a fresh plain
    `Spectrogram` from `.stft`, and proceeds normally (no more attribute
    checks after that point). So this generator only records the ONE
    actually-working construction path."""
    from madmom.audio.cepstrogram import MFCC, Cepstrogram
    from madmom.audio.filters import LogarithmicFilterbank
    from madmom.audio.spectrogram import FilteredSpectrogramProcessor

    spec = _spectrogram(WAVS_DIR / DSP_CASE_WAV)

    out = {"spectrogram_input": np.asarray(spec)}
    out["cepstrogram_default"] = np.asarray(Cepstrogram(spec))

    filt_spec = FilteredSpectrogramProcessor(
        filterbank=LogarithmicFilterbank, num_bands=12, fmin=30, fmax=17000,
        norm_filters=True,
    )(spec)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["mfcc_default"] = np.asarray(MFCC(filt_spec))
        out["mfcc_custom"] = np.asarray(
            MFCC(filt_spec, num_bands=13, fmin=20.0, fmax=8000.0,
                 norm_filters=False)
        )
    return out


def generate_hpss_fixtures() -> dict:
    """`HarmonicPercussiveSourceSeparation.slices()`/`.masks()` (the two
    actually-working helper methods, see `madmom_infer/audio/hpss.py`'s
    module header for why `.process()` itself is a faithfully-reproduced
    upstream bug, not fixture-tested here)."""
    from madmom.audio.hpss import HarmonicPercussiveSourceSeparation

    spec = _spectrogram(WAVS_DIR / DSP_CASE_WAV)
    spec_arr = np.asarray(spec)

    out = {"spectrogram_input": spec_arr}

    binary = HarmonicPercussiveSourceSeparation(masking="binary")
    h_slice, p_slice = binary.slices(spec_arr)
    out["harmonic_slice"] = h_slice
    out["percussive_slice"] = p_slice
    h_mask_bin, p_mask_bin = binary.masks(h_slice, p_slice)
    out["harmonic_mask_binary"] = h_mask_bin
    out["percussive_mask_binary"] = p_mask_bin

    soft = HarmonicPercussiveSourceSeparation(masking=2.0)
    h_mask_soft, p_mask_soft = soft.masks(h_slice, p_slice)
    out["harmonic_mask_soft"] = h_mask_soft
    out["percussive_mask_soft"] = p_mask_soft

    # non-default filter sizes, to exercise slices() beyond the class
    # defaults
    custom = HarmonicPercussiveSourceSeparation(
        harmonic_filter=(9, 1), percussive_filter=(1, 9)
    )
    h_slice_c, p_slice_c = custom.slices(spec_arr)
    out["harmonic_slice_custom"] = h_slice_c
    out["percussive_slice_custom"] = p_slice_c

    return out


def generate_signal_leftovers_fixtures() -> dict:
    """`attenuate`/`rescale`/`trim`/`energy`/`root_mean_square`/
    `sound_pressure_level` outputs -- resolving the 4b audit-table
    TO-VERIFY flag (see `madmom_infer/audio/signal.py`'s module header)."""
    from madmom.audio.signal import (
        FramedSignalProcessor, Signal, attenuate, energy, rescale,
        root_mean_square, sound_pressure_level, trim,
    )

    sig = Signal(str(WAVS_DIR / DSP_CASE_WAV), num_channels=1)
    sig_data = np.asarray(sig)

    out = {"signal_input": sig_data}

    out["attenuate_6db"] = np.asarray(attenuate(sig, 6.0))
    out["attenuate_0db"] = np.asarray(attenuate(sig, 0.0))
    out["rescale_float32"] = np.asarray(rescale(sig, dtype=np.float32))
    out["rescale_float64"] = np.asarray(rescale(sig, dtype=np.float64))

    # trim(): synthetic zero-padding around a real chunk of signal, so the
    # trimmed-length outcome is unambiguous (a real .wav has no guaranteed
    # exact-zero runs at either end).
    padded = np.concatenate([
        np.zeros(10, dtype=sig.dtype), sig_data[:200],
        np.zeros(15, dtype=sig.dtype),
    ])
    out["trim_input"] = padded
    out["trim_fb"] = np.asarray(trim(padded, where="fb"))
    out["trim_f"] = np.asarray(trim(padded, where="f"))
    out["trim_b"] = np.asarray(trim(padded, where="b"))

    out["energy_1d"] = np.asarray(energy(sig_data))
    out["root_mean_square_1d"] = np.asarray(root_mean_square(sig_data))
    out["sound_pressure_level_1d"] = np.asarray(
        sound_pressure_level(sig_data))

    frames = FramedSignalProcessor(frame_size=FRAME_SIZE, fps=FPS)(sig)
    out["energy_framed"] = np.asarray(energy(frames))
    out["root_mean_square_framed"] = np.asarray(root_mean_square(frames))
    out["sound_pressure_level_framed"] = np.asarray(
        sound_pressure_level(frames))

    # float signal exercises the p_ref=1.0 (not integer-dtype-max) default
    float_sig_data = sig_data.astype(np.float32) / 32768.0
    out["energy_float"] = np.asarray(energy(float_sig_data))
    out["sound_pressure_level_float"] = np.asarray(
        sound_pressure_level(float_sig_data))

    return out


def main() -> None:
    try:
        import madmom  # noqa: F401
    except ImportError as exc:
        print(
            "ERROR: this script needs the real `madmom` package (not "
            "madmom_infer). Run it with the madmom-reference venv's "
            "interpreter -- see this file's module docstring for the exact "
            "command.",
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

    print("4g-1: Cepstrogram/MFCC fixtures ...")
    cep_fixtures = generate_cepstrogram_fixtures()
    np.savez_compressed(FIXTURES_DIR / "cepstrogram.npz", **cep_fixtures)

    print("4g-2: HPSS slices()/masks() fixtures ...")
    hpss_fixtures = generate_hpss_fixtures()
    np.savez_compressed(FIXTURES_DIR / "hpss.npz", **hpss_fixtures)

    print("4g-3: audio/signal.py leftover-function fixtures ...")
    signal_fixtures = generate_signal_leftovers_fixtures()
    np.savez_compressed(FIXTURES_DIR / "signal_leftovers.npz",
                         **signal_fixtures)

    print("Done.")


if __name__ == "__main__":
    main()
