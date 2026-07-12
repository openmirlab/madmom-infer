"""Short-Time Fourier Transform -- composition port of madmom.audio.stft's
`ShortTimeFourierTransformProcessor`/`ShortTimeFourierTransform`, the stage
that windows each frame from `FramedSignalProcessor` and takes its (real)
FFT. Composition, not `np.ndarray` subclassing (docs/DESIGN.md C.2, same
rationale as `audio/signal.py`): `ShortTimeFourierTransform` wraps a plain
`complex64` array in `.data` plus the metadata (`frames`, `window`,
`fft_window`, ...) real madmom attaches via `__array_finalize__`.

Phase-1 scope: only `ShortTimeFourierTransformProcessor`/
`ShortTimeFourierTransform` plus the free functions they need
(`stft`, `fft_frequencies`) are ported -- these are the only surface
all-in-one-infer's `build_spec_processor()` exercises
(`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`). Still NOT ported:
`pyfftw` acceleration (`rfft_builder`, `fftw=` passthrough -- `pyfftw` is not
a project dependency and was never present in the reference madmom install
this port's fixtures were generated against, so the `fftw` code paths in
upstream are dead code for this port's purposes anyway).

Wave 4b addition: `phase()`, `Phase`, `local_group_delay()`/`lgd` (alias),
`LocalGroupDelay`/`LGD` (alias) -- the phase-vocoder-style analysis chain
`features/onsets.py`'s phase-deviation onset family
(`phase_deviation`/`weighted_phase_deviation`/`normalized_weighted_phase_
deviation`/`complex_domain`/`rectified_complex_domain`/`complex_flux`) reads
off `spectrogram.stft.phase()` and `...phase().lgd()`. Composition port, same
`.data`-wrapping pattern as `ShortTimeFourierTransform` itself (not an
`np.ndarray` subclass -- docs/DESIGN.md C.2).

**Bug, replicated on purpose (golden-fixture mandate, bugs included -- see
CLAUDE.md).** Upstream `LocalGroupDelay.__new__` (`stft.py:682-686`) reads:

    def __new__(cls, phase, **kwargs):
        if not isinstance(stft, Phase):
            phase = Phase(phase, circular_shift=True, **kwargs)
        ...

`stft` here is NOT the `phase` parameter -- it's an undefined local name
that Python resolves to the MODULE-LEVEL `stft` function (this same file
defines `def stft(frames, window, ...)` above). `isinstance(<function>,
Phase)` is always `False`, so `not isinstance(...)` is always `True`: this
branch is unconditionally taken, regardless of whether the `phase` argument
passed in was already a `Phase` instance. The practical effect: constructing
`LocalGroupDelay(existing_phase_instance)` does NOT reuse that instance (as
the docstring implies) -- it always rebuilds a fresh `Phase` from
`existing_phase_instance.stft` (unwrapped one level, since `Phase.__new__`
itself DOES correctly check `isinstance(stft, Phase)`) with `circular_shift`
forced to `True`. Confirmed via direct inspection of the upstream source
(not merely suspected); this port's `LocalGroupDelay.__init__` below always
takes the "reconstruct" branch to match, and is documented there too.

Two bit-identity traps replicated HERE ON PURPOSE, not fixed (see
tests/fixtures/README.md "Surprises" and CLAUDE.md's golden-fixture
philosophy -- bit-parity with real madmom, bugs included, is the mandate):

1. **int16 (integer-dtype) window scaling.** `ShortTimeFourierTransform`
   divides the FFT window by `np.iinfo(frames.signal.dtype).max` when the
   underlying `Signal` has an integer dtype, INSTEAD of rescaling the signal
   itself (`madmom-upstream/madmom/audio/stft.py:337-349`). This keeps int16
   PCM data un-rescaled all the way through framing (memory-mapping stays
   possible) while still producing a float-range STFT. For a float-dtype
   signal, `np.iinfo(dtype)` raises `ValueError` and the window is used
   completely unscaled.

2. **Window-caching gotcha, LOUDLY FLAGGED.** `ShortTimeFourierTransform`
   only (re)computes/rescales `fft_window` when it is given as `None`
   (`stft.py:331`: `if fft_window is None:`). `ShortTimeFourierTransformProcessor`
   caches whatever `fft_window` its `ShortTimeFourierTransform` instance ended
   up using, and passes that CACHED value into every subsequent call
   (`stft.py:505-510`). Consequence: **a single reused
   `ShortTimeFourierTransformProcessor` instance silently keeps the FIRST
   call's dtype-scaled window on every later call, even if a later call's
   signal has a completely different dtype** -- no error, no warning, just
   wrong numbers. This is a real bug in upstream madmom, reproduced exactly
   here (not "fixed") because Phase-1's mandate is bit-identical output,
   and it happens to be harmless for all-in-one-infer's own usage (which
   always reuses one `stft` instance across mono int16 stems -- same dtype
   every call, see `all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`).
   Pinned by `tests/test_stft.py`'s window-caching test against
   `tests/fixtures/stft.npz`'s `window_caching_reused_output` /
   `window_caching_fresh_output` / `window_caching_max_abs_diff`.

STFT also requires a strictly 2D (`num_frames`, `frame_size`) input -- a raw
multi-channel `FramedSignal` (3D) raises `ValueError` here, matching
upstream's "no multi-channel support" `TODO`
(`madmom-upstream/madmom/audio/stft.py:80-84`, confirmed empirically in
`tests/fixtures/manifest.json`'s `known_error_cases.stereo_full_chain`) --
this is why all-in-one-infer always downmixes to mono before framing.

Reads: madmom_infer/audio/signal.py (FramedSignal), madmom_infer/processors.py
(Processor), numpy, scipy.fft; read by: madmom_infer/audio/spectrogram.py
(planned, via Spectrogram's `isinstance(..., ShortTimeFourierTransform)`
check and `.bin_frequencies`).
"""

import numpy as np
import scipy.fft

from ..processors import Processor
from .signal import FramedSignal

STFT_DTYPE = np.complex64


def fft_frequencies(num_fft_bins, sample_rate):
    """Frequencies of the FFT bins [Hz].

    Port of `madmom.audio.stft.fft_frequencies` (`stft.py:29-46`).
    """
    return np.fft.fftfreq(num_fft_bins * 2, 1.0 / sample_rate)[:num_fft_bins]


def stft(frames, window, fft_size=None, circular_shift=False,
         include_nyquist=False):
    """Complex Short-Time Fourier Transform of a framed signal.

    Verbatim algorithmic port of `madmom.audio.stft.stft`
    (`stft.py:49-133`), minus the `fftw`/pyfftw acceleration parameter (see
    module header). `frames` is anything exposing `.ndim`/`.shape` and
    yielding one frame per iteration (a `FramedSignal` in practice, which
    this module's `ShortTimeFourierTransform` always passes in) -- `window`
    (`fft_window`, already dtype-scaled by the caller) is applied per frame
    before the real FFT.
    """
    if frames.ndim != 2:
        # TODO: add multi-channel support (matches upstream's own TODO)
        raise ValueError(
            "frames must be a 2D array or iterable, got %s with shape %s."
            % (type(frames), frames.shape)
        )

    num_frames, frame_size = frames.shape

    if fft_size is None:
        fft_size = frame_size
    num_fft_bins = fft_size >> 1
    if include_nyquist:
        num_fft_bins += 1

    if circular_shift:
        fft_shift = frame_size >> 1

    data = np.empty((num_frames, num_fft_bins), STFT_DTYPE)

    for f, frame in enumerate(frames):
        if circular_shift:
            if window is not None:
                signal = np.multiply(frame, window)
            else:
                signal = frame
            fft_signal = np.zeros(fft_size)
            fft_signal[:fft_shift] = signal[fft_shift:]
            fft_signal[-fft_shift:] = signal[:fft_shift]
        else:
            if window is not None:
                fft_signal = np.multiply(frame, window)
            else:
                fft_signal = frame
        data[f] = scipy.fft.fft(fft_signal, n=fft_size, axis=0)[:num_fft_bins]

    return data


class ShortTimeFourierTransform:
    """The complex STFT of a `FramedSignal`.

    Composition port of `madmom.audio.stft.ShortTimeFourierTransform`
    (`stft.py:202-429`): `.data` holds the `complex64` STFT matrix
    (shape `(num_frames, num_bins)`), plus the `frames`/`window`/
    `fft_window`/`fft_size`/`circular_shift`/`include_nyquist` metadata real
    madmom stores as ndarray-subclass attributes. See the module header for
    the int16-window-scaling and window-caching traps this class's
    `fft_window is None` check (below) exists to reproduce exactly.
    """

    def __init__(self, frames, window=np.hanning, fft_size=None,
                 circular_shift=False, include_nyquist=False,
                 fft_window=None, **kwargs):
        if isinstance(frames, ShortTimeFourierTransform):
            # already a STFT, use the frames thereof
            frames = frames.frames
        # instantiate a FramedSignal if needed
        if not isinstance(frames, FramedSignal):
            frames = FramedSignal(frames, **kwargs)

        # size of the frames
        frame_size = frames.shape[1]

        if fft_window is None:
            # CRITICAL: this is the only place the dtype-scaled window is
            # (re)computed -- a caller (ShortTimeFourierTransformProcessor)
            # passing in a non-None, previously-cached `fft_window` skips
            # all of this, even if `frames`' dtype has since changed. See
            # module header, trap 2.
            if hasattr(window, "__call__"):
                window = window(frame_size)
            try:
                # if the signal is not scaled, scale the window accordingly
                max_range = float(np.iinfo(frames.signal.dtype).max)
                try:
                    fft_window = window / max_range
                except TypeError:
                    # window is None -- can't scale it, so build a uniform
                    # window and scale that instead
                    fft_window = np.ones(frame_size) / max_range
            except ValueError:
                # non-integer dtype (e.g. float32/float64): no scaling needed
                fft_window = window

        data = stft(frames, fft_window, fft_size=fft_size,
                    circular_shift=circular_shift,
                    include_nyquist=include_nyquist)

        self.data = data
        self.frames = frames
        self.window = window
        self.fft_window = fft_window
        self.fft_size = fft_size if fft_size else frame_size
        self.circular_shift = circular_shift
        self.include_nyquist = include_nyquist

    # -- numpy interop, mirroring audio/signal.py's Signal -------------
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # intentionally returns a plain ndarray, not a wrapped instance --
        # see Signal.__getitem__'s docstring for the same composition-vs-
        # subclass rationale.
        return self.data[index]

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self.data)

    @property
    def num_bins(self):
        """Number of FFT bins."""
        return int(self.data.shape[1])

    @property
    def bin_frequencies(self):
        """Bin frequencies [Hz]."""
        return fft_frequencies(self.num_bins, self.frames.signal.sample_rate)

    def phase(self, **kwargs):
        """Return the `Phase` of this STFT. Port of
        `ShortTimeFourierTransform.phase` (`stft.py:427-438`), Wave 4b."""
        return Phase(self, **kwargs)


STFT = ShortTimeFourierTransform


class ShortTimeFourierTransformProcessor(Processor):
    """Processor wrapper: compute the STFT of a `FramedSignal`.

    Composition port of `madmom.audio.stft.ShortTimeFourierTransformProcessor`
    (`stft.py:432-552`). Caches the `fft_window` its `ShortTimeFourierTransform`
    ended up using, across calls -- see module header, trap 2, for the exact
    replicated (not fixed) consequence of reusing one instance across
    differing-dtype inputs.
    """

    def __init__(self, window=np.hanning, fft_size=None, circular_shift=False,
                 include_nyquist=False, **kwargs):
        self.window = window
        self.fft_size = fft_size
        self.circular_shift = circular_shift
        self.include_nyquist = include_nyquist
        # caching only, not intended for general use (matches upstream's own
        # comment, stft.py:479-481) -- this is exactly the state that makes
        # trap 2 (module header) possible.
        self.fft_window = None

    def process(self, data, **kwargs):
        """Perform FFT on a framed signal and return the STFT.

        Returns a `ShortTimeFourierTransform` instance, matching
        `ShortTimeFourierTransformProcessor.process` (`stft.py:483-511`).
        """
        data = ShortTimeFourierTransform(
            data, window=self.window, fft_size=self.fft_size,
            circular_shift=self.circular_shift,
            include_nyquist=self.include_nyquist,
            fft_window=self.fft_window, **kwargs,
        )
        # cache the window used for FFT (see class docstring)
        self.fft_window = data.fft_window
        return data


STFTProcessor = ShortTimeFourierTransformProcessor


# ---------------------------------------------------------------------------
# Phase / LocalGroupDelay -- Wave 4b addition (onset phase-deviation family)
# ---------------------------------------------------------------------------
def phase(stft_data):
    """Phase of a complex STFT array. Verbatim port of
    `madmom.audio.stft.phase` (`stft.py:136-152`): `np.angle(stft_data)`."""
    return np.angle(stft_data)


def local_group_delay(phase_data):
    """Local group delay (derivative of unwrapped phase over frequency) of
    a `(num_frames, num_bins)` phase array.

    Verbatim port of `madmom.audio.stft.local_group_delay`
    (`stft.py:155-184`).
    """
    if phase_data.ndim != 2:
        raise ValueError("phase must be a 2D array")
    unwrapped_phase = np.unwrap(phase_data)
    unwrapped_phase[:, :-1] -= unwrapped_phase[:, 1:]
    unwrapped_phase[:, -1] = 0
    return unwrapped_phase


# alias
lgd = local_group_delay


class Phase:
    """The phase of a `ShortTimeFourierTransform`.

    Composition port of `madmom.audio.stft.Phase` (`stft.py:554-644`).
    `.data` is `np.angle(stft)`, `float32` (matches upstream's implicit
    downcast: `np.angle` of a `complex64` array returns `float32`).
    `circular_shift` defaults to `True` when a fresh `STFT` must be built
    from a non-STFT input (matches upstream's `kwargs.pop('circular_shift',
    True)`) -- phase information is only meaningful with a circular-shifted
    STFT (`stft.py:110-123`'s `fft_shift` swap), and a non-circular-shift
    STFT emits a `RuntimeWarning` here, matching upstream, rather than
    silently producing wrong-but-plausible-looking phase values.
    """

    def __init__(self, stft, **kwargs):
        if isinstance(stft, Phase):
            # already a Phase, use its STFT
            stft = stft.stft
        if not isinstance(stft, ShortTimeFourierTransform):
            circular_shift = kwargs.pop("circular_shift", True)
            stft = ShortTimeFourierTransform(
                stft, circular_shift=circular_shift, **kwargs
            )
        if not stft.circular_shift:
            import warnings

            warnings.warn(
                "`circular_shift` of the STFT must be set to 'True' for "
                "correct phase",
                RuntimeWarning,
            )
        self.data = phase(np.asarray(stft)).astype(np.float32)
        self.stft = stft

    # -- numpy interop, mirroring audio/signal.py's Signal -------------
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self.data)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.data.shape[1])

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies

    def local_group_delay(self, **kwargs):
        """Return the `LocalGroupDelay` of this phase. Port of
        `Phase.local_group_delay` (`stft.py:614-628`)."""
        return LocalGroupDelay(self, **kwargs)

    lgd = local_group_delay


class LocalGroupDelay:
    """The local group delay of a `Phase`.

    Composition port of `madmom.audio.stft.LocalGroupDelay`
    (`stft.py:646-712`). **Reproduces a real upstream bug on purpose** -- see
    this module's header for the full explanation: `LocalGroupDelay.__new__`
    checks `isinstance(stft, Phase)` where `stft` is an undefined name that
    Python resolves to the module-level `stft()` FUNCTION, not the `phase`
    argument, so the condition is always `True` and a fresh `Phase` is always
    rebuilt (with `circular_shift` forced to `True`) regardless of whether
    the input was already a `Phase` instance. This class's `__init__` below
    always takes that same "always rebuild" branch, matching upstream's
    actual (buggy) behavior, not its docstring's stated intent.
    """

    def __init__(self, phase_or_stft, **kwargs):
        # bug-for-bug port (see class + module docstrings): upstream never
        # actually reuses an already-built Phase here, it always rebuilds.
        phase_obj = Phase(phase_or_stft, circular_shift=True, **kwargs)
        if not phase_obj.stft.circular_shift:
            import warnings

            warnings.warn(
                "`circular_shift` of the STFT must be set to 'True' for "
                "correct local group delay"
            )
        self.data = local_group_delay(np.asarray(phase_obj))
        self.phase = phase_obj
        self.stft = phase_obj.stft

    # -- numpy interop, mirroring audio/signal.py's Signal -------------
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self.data)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.data.shape[1])

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies


# alias
LGD = LocalGroupDelay
