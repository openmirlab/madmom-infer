"""Signal/FramedSignal layer -- composition-based port of madmom.audio.signal.

`Signal` wraps a plain numpy array (`.data`) plus `sample_rate`/metadata
instead of subclassing `np.ndarray` (madmom's own approach,
`madmom-upstream/madmom/audio/signal.py:506`). Per docs/DESIGN.md C.2 this is
a deliberate, documented deviation: ndarray subclassing needs
`__array_finalize__`/pickle-`__reduce__` boilerplate just to keep metadata
attached (`signal.py:634-656`) and has no structurally-parallel torch
equivalent, whereas composition lets a future `madmom_infer.torch.audio.
signal.Signal` share the same shape. `Signal` implements `__array__` so
`np.asarray(sig)` / `np.dot(sig, x)` etc. keep working transparently, and
forwards the handful of accesses the real phase-1 call sites need (`.dtype`,
`.sample_rate`, `.num_channels`, indexing) -- see `all-in-one-fix/src/
allin1_infer/spectrogram.py:19-22,90-104`.

`FramedSignalProcessor`/`FramedSignal` reproduce madmom's exact frame
indexing arithmetic (`signal.py:860-962,974-1393`) -- see `signal_frame()`
below for the origin/padding trap this stage exists to get exactly right.

Deliberately NOT ported (out of phase-1 scope, no ffmpeg/PyAudio dependency
in this project): `resample()`'s ffmpeg-based resampling
(`signal.py:226-263`), the `dtype`-driven implicit rescale that
`load_wave_file` falls back to ffmpeg for on mismatch (`madmom-upstream/
madmom/io/audio.py:622-625`), and the online/live-audio `Stream` class plus
`FramedSignalProcessor`'s `'stream'`-origin PyAudio integration
(`signal.py:1396-1504`, `processors.py:836-906`). Where madmom would silently
fall back to ffmpeg, this port raises a clear `NotImplementedError` instead
of guessing.

Wave 4b addition: `smooth()` (Hamming-window or custom-kernel 1D/2D
convolution smoothing) -- needed by `features/onsets.py`'s `peak_picking`,
which imports it as `smooth_signal`. Not a Phase-1 module, but lives here
because it's `madmom.audio.signal.smooth` upstream.

Reads: numpy, scipy.io.wavfile, scipy.signal (smooth's 2D path),
madmom_infer.processors.Processor; read by: madmom_infer/audio/stft.py
(via `FramedSignal.signal.dtype` for the int16 window-scaling convention),
madmom_infer/features/onsets.py (`smooth`, as `smooth_signal`).
"""

import numpy as np

from ..processors import Processor


# ---------------------------------------------------------------------------
# module-level defaults, mirroring madmom-upstream/madmom/audio/signal.py:
# 496-503, 965-970
# ---------------------------------------------------------------------------
SAMPLE_RATE = None
NUM_CHANNELS = None
CHANNEL = None
START = None
STOP = None
NORM = False
GAIN = 0.0
DTYPE = None

FRAME_SIZE = 2048
HOP_SIZE = 441.0
FPS = None
ORIGIN = 0
END_OF_SIGNAL = "normal"
NUM_FRAMES = None


# ---------------------------------------------------------------------------
# signal functions -- ported verbatim from madmom-upstream/madmom/audio/
# signal.py, the subset Signal's constructor needs
# ---------------------------------------------------------------------------
def remix(signal, num_channels, channel=None):
    """Remix `signal` to have `num_channels` channels.

    Verbatim port of `madmom.audio.signal.remix`
    (`madmom-upstream/madmom/audio/signal.py:170-223`). Only mono<->arbitrary
    conversions are supported (matching the original).

    CRITICAL bit-identity detail (the mono-downmix truncation trap): when
    down-mixing an integer-dtype signal to mono, `np.mean(signal, axis=-1)`
    upcasts to float64 first (numpy's default accumulator for integer
    input), and `.astype(signal.dtype)` then TRUNCATES toward zero -- it does
    not round. E.g. two int16 channels with values 3 and 4 average to 3.5,
    which truncates to 3, not 4. Verified empirically against the real
    madmom 0.17.dev0 install (`all-in-one-fix/.venv`): `remix(np.array([[3,
    4], [-3, -4]], dtype=np.int16), 1)` returns `[3, -3]` (not `[4, -4]` and
    not `[4, -3]`/`[3, -4]` from banker's-rounding-style behavior).
    """
    if num_channels == signal.ndim or num_channels is None:
        # return as many channels as there are
        return signal
    elif num_channels == 1 and signal.ndim > 1:
        if channel is None:
            # down-mix to mono; converted to float internally (np.mean's
            # default float64 accumulator for integer dtypes) then TRUNCATED
            # back to the original dtype via .astype() -- not rounded
            return np.mean(signal, axis=-1).astype(signal.dtype)
        else:
            # use the requested channel verbatim
            return signal[:, channel]
    elif num_channels > 1 and signal.ndim == 1:
        # up-mix a mono signal simply by copying channels
        return np.tile(signal[:, np.newaxis], num_channels)
    else:
        raise NotImplementedError(
            "Requested %d channels, but got %d channels and channel "
            "conversion is not implemented." % (num_channels, signal.shape[1])
        )


def normalize(signal):
    """Normalize `signal` to have maximum amplitude.

    Port of `madmom.audio.signal.normalize` (`signal.py:134-167`).
    """
    scaling = float(np.max(np.abs(signal)))
    if np.issubdtype(signal.dtype, np.integer):
        if signal.dtype in (np.int16, np.int32):
            scaling /= np.iinfo(signal.dtype).max
        else:
            raise ValueError(
                "only float and np.int16/32 dtypes supported, not %s."
                % signal.dtype
            )
    return np.asarray(signal / scaling, dtype=signal.dtype)


def adjust_gain(signal, gain):
    """Adjust the gain of `signal` [dB].

    Port of `madmom.audio.signal.adjust_gain` (`signal.py:71-103`).
    """
    gain = np.power(np.sqrt(10.0), 0.1 * gain)
    if gain > 1 and np.issubdtype(signal.dtype, np.integer):
        raise ValueError(
            "positive gain adjustments are only supported for float dtypes."
        )
    return np.asarray(signal * gain, dtype=signal.dtype)


def smooth(signal, kernel):
    """Smooth `signal` along its first axis with `kernel`.

    Verbatim port of `madmom.audio.signal.smooth` (`signal.py:20-68`) -- Wave
    4b addition, not Phase 1 (the earlier 4.0 audit table mis-listed this as
    already PORTED; it was not actually present in this module until
    `features/onsets.py`'s `peak_picking` needed it, see CLAUDE.md's audit
    table correction). If `kernel` is a plain (non-numpy) integer, a Hamming
    window of that length is used; if it's already a numpy array, it's used
    as-is directly as the convolution kernel.
    """
    if kernel is None:
        return signal
    elif isinstance(kernel, (int, np.integer)):
        if kernel == 0:
            return signal
        elif kernel > 1:
            kernel = np.hamming(kernel)
        else:
            raise ValueError(
                "can't create a smoothing kernel of size %d" % kernel
            )
    elif isinstance(kernel, np.ndarray):
        kernel = kernel
    else:
        raise ValueError("can't smooth signal with %s" % kernel)
    if signal.ndim == 1:
        return np.convolve(signal, kernel, "same")
    elif signal.ndim == 2:
        from scipy.signal import convolve2d

        return convolve2d(signal, kernel[:, np.newaxis], "same")
    else:
        raise ValueError("signal must be either 1D or 2D")


def _load_wave_file(filename, sample_rate=None, num_channels=None,
                     channel=None, start=None, stop=None, dtype=None):
    """Load a .wav file's raw samples, madmom's `load_wave_file` semantics.

    Port of `madmom.io.audio.load_wave_file`
    (`madmom-upstream/madmom/io/audio.py:594-668`), reading via
    `scipy.io.wavfile.read(filename, mmap=True)` exactly like the original --
    this is what keeps PCM `int16` data `int16` (no float rescale): scipy
    returns the file's native dtype verbatim, and madmom's `Signal` defaults
    to `dtype=None` (`signal.py:503`, "keep whatever dtype the source has"),
    so the dtype flows straight through to `FramedSignal` frames.

    Deliberately NOT ported: on a `sample_rate`/`dtype` mismatch, real madmom
    raises `ValueError` here, which `load_audio_file` catches and retries via
    its ffmpeg loader (`io/audio.py:753-765`). This project has no ffmpeg
    dependency, so both mismatches raise `NotImplementedError` directly
    instead of silently mis-loading.
    """
    from scipy.io import wavfile

    file_sample_rate, signal = wavfile.read(str(filename), mmap=True)
    if sample_rate is not None and sample_rate != file_sample_rate:
        raise NotImplementedError(
            "requested sample_rate=%r differs from the file's native rate "
            "%r Hz; resampling requires ffmpeg and is out of scope for "
            "madmom-infer phase 1 (madmom itself falls back to an ffmpeg "
            "loader in this case, io/audio.py:753-765)."
            % (sample_rate, file_sample_rate)
        )
    if dtype is not None and signal.dtype != dtype:
        raise NotImplementedError(
            "requested dtype=%r differs from the file's native dtype %r; "
            "dtype-converting rescale during load requires ffmpeg and is "
            "out of scope for madmom-infer phase 1." % (dtype, signal.dtype)
        )
    # `start`/`stop` positions are rounded to the closest sample (seconds);
    # this must happen BEFORE remixing, matching load_wave_file's order
    if start is not None:
        start = int(start * file_sample_rate)
    if stop is not None:
        stop = min(len(signal), int(stop * file_sample_rate))
    if start is not None or stop is not None:
        signal = signal[start:stop]
    if channel is not None and num_channels is None:
        num_channels = 1
    if num_channels is not None:
        signal = remix(signal, num_channels, channel)
    return signal, file_sample_rate


# ---------------------------------------------------------------------------
# Signal: composition, not ndarray subclass (docs/DESIGN.md C.2)
# ---------------------------------------------------------------------------
class Signal:
    """A signal: a plain numpy array plus `sample_rate` and related metadata.

    Composition port of `madmom.audio.signal.Signal`
    (`madmom-upstream/madmom/audio/signal.py:506-711`). Construction mirrors
    `Signal.__new__` (`signal.py:600-632`):

    1. If `data` is not already an array (a path or path-like), load it as a
       .wav file (`_load_wave_file`, above).
    2. Otherwise wrap the given array as-is -- **`np.asarray(data)` does not
       copy** when `data` is already an ndarray of a compatible dtype, so
       constructing `Signal(int16_array, sample_rate=...)` is a zero-copy
       view over the caller's array, exactly like madmom's own
       `np.asarray(data).view(cls)` (`signal.py:612`). Verified empirically:
       `np.shares_memory(Signal(arr, ...), arr)` is `True` against the real
       madmom install.
    3. Remix to `num_channels` if given (this runs even for the from-file
       path, where `_load_wave_file` may have already remixed once --
       harmless, since `remix()` is a no-op once the channel count already
       matches, matching madmom's own double-remix at `signal.py:614-616`).
    4. Normalize / adjust gain if requested.

    One subtle, faithfully-reproduced madmom quirk: the `dtype` constructor
    argument is used ONLY for the from-file loading path (forwarded to
    `_load_wave_file`/ffmpeg in real madmom) -- for an already-ndarray `data`
    argument, `dtype` is not applied at all (`signal.py:600-632` never
    references `dtype` outside the `load_audio_file` call). This looks like
    it could be a bug in upstream madmom, but this port replicates it
    exactly rather than silently "fixing" divergent behavior.

    Resampling (`sample_rate` different from an existing `Signal`'s own
    rate) requires madmom's ffmpeg-backed `resample()`
    (`signal.py:226-263`), which is out of scope here; that case raises
    `NotImplementedError`.
    """

    def __init__(self, data, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                 channel=CHANNEL, start=START, stop=STOP, norm=NORM,
                 gain=GAIN, dtype=DTYPE):
        prior_sample_rate = None
        if isinstance(data, Signal):
            # re-wrapping an existing Signal: keep its rate for the
            # resample-mismatch check below, matching signal.py:624-625
            prior_sample_rate = data.sample_rate
            data = data.data
        if not isinstance(data, np.ndarray):
            # not an array -- try to load it as a .wav file
            data, sample_rate = _load_wave_file(
                data, sample_rate=sample_rate, num_channels=num_channels,
                channel=channel, start=start, stop=stop, dtype=dtype,
            )
        else:
            # from-array path: no copy if `data` is already ndarray-shaped
            # correctly (dtype is NOT applied here, see docstring above)
            data = np.asarray(data)
            if prior_sample_rate is not None and sample_rate is None:
                sample_rate = prior_sample_rate

        # remix to the desired number of channels (see remix()'s docstring
        # for the mono-downmix truncation trap)
        if num_channels:
            data = remix(data, num_channels, channel)
        # normalize signal if needed
        if norm:
            data = normalize(data)
        # adjust the gain if needed
        if gain is not None and gain != 0:
            data = adjust_gain(data, gain)
        # resampling an already-rated Signal to a different rate needs
        # madmom's ffmpeg-backed resample(); out of scope here
        if (prior_sample_rate is not None and sample_rate is not None
                and sample_rate != prior_sample_rate):
            raise NotImplementedError(
                "resampling (%r Hz -> %r Hz) requires ffmpeg and is out of "
                "scope for madmom-infer phase 1." % (prior_sample_rate, sample_rate)
            )

        self.data = data
        self.sample_rate = sample_rate
        # start/stop bookkeeping only (madmom does not actually trim an
        # already-ndarray `data` by start/stop -- only the file-loading path
        # does, inside _load_wave_file -- signal.py:626-630)
        self.start = start
        self.stop = stop
        if start is not None and sample_rate is not None:
            self.stop = start + float(len(data)) / sample_rate

    # -- numpy interop -------------------------------------------------
    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # NOTE: intentionally returns a plain ndarray, not a Signal -- one
        # documented composition-vs-subclass deviation from madmom (whose
        # ndarray-subclass slicing auto-preserves the Signal type via
        # __array_finalize__, signal.py:634-640). Every real phase-1 call
        # site (FramedSignal's frame slicing, stft.py) only needs the raw
        # samples + dtype, not per-slice Signal metadata.
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
    def num_samples(self):
        """Number of samples."""
        return len(self.data)

    @property
    def num_channels(self):
        """Number of channels."""
        if self.data.ndim == 1:
            return 1
        return self.data.shape[1]

    @property
    def length(self):
        """Length of signal in seconds."""
        if self.sample_rate is None:
            return None
        return float(self.num_samples) / self.sample_rate

    def __repr__(self):
        return "Signal(%r, sample_rate=%r)" % (self.data, self.sample_rate)


class SignalProcessor(Processor):
    """Processor wrapper: load/convert a file or array into a `Signal`.

    Port of `madmom.audio.signal.SignalProcessor`
    (`madmom-upstream/madmom/audio/signal.py:714-796`). Phase 2 needs this
    because `RNNDownBeatProcessor` (`madmom_infer/features/downbeats.py`)
    starts its pipeline with `SignalProcessor(num_channels=1,
    sample_rate=44100)` -- Phase 1 never needed a `Processor` wrapper around
    `Signal` since its own golden-fixture tests constructed `Signal` directly.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
                 start=START, stop=STOP, norm=NORM, gain=GAIN, dtype=DTYPE,
                 **kwargs):
        # pylint: disable=unused-argument
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.start = start
        self.stop = stop
        self.norm = norm
        self.gain = gain
        self.dtype = dtype

    def process(self, data, **kwargs):
        """Process the given audio file/array into a `Signal`.

        Matches `madmom.audio.signal.SignalProcessor.process`
        (`signal.py:771-796`).
        """
        args = dict(sample_rate=self.sample_rate,
                    num_channels=self.num_channels, start=self.start,
                    stop=self.stop, norm=self.norm, gain=self.gain,
                    dtype=self.dtype)
        args.update(kwargs)
        return Signal(data, **args)


# ---------------------------------------------------------------------------
# frame splitting
# ---------------------------------------------------------------------------
def signal_frame(signal, index, frame_size, hop_size, origin=0, pad=0):
    """Return the frame at `index` of `signal`.

    Verbatim algorithmic port of `madmom.audio.signal.signal_frame`
    (`madmom-upstream/madmom/audio/signal.py:860-962`) -- this is the #1
    off-by-one trap in the whole module, spelled out here for the next
    stage's implementer:

    - `ref_sample = int(index * hop_size)`: the reference sample of frame
      `index` (frame 0's reference sample is the signal's first sample).
    - `start = ref_sample - frame_size // 2 - int(origin)`,
      `stop = start + frame_size`: the window is always centered on
      `ref_sample`; `origin` shifts it. `origin=0` (the default, madmom's
      'center'/'offline') centers the frame ON the reference sample, so
      frame 0 straddles the signal boundary and is LEFT-padded with
      `frame_size // 2` samples (verified: for `frame_size=2048`, frame 0's
      first 1024 samples are the pad value, the remaining 1024 are
      `signal[0:1024]`). `origin=(frame_size - 1) // 2`-ish ('left'/'past'/
      'online') shifts the window fully to the left of the reference sample
      (frame 0 is almost entirely padding, only the last sample is real).
      `origin=-(frame_size // 2)`-ish ('right'/'future'/'stream') shifts it
      fully to the right (frame 0 == `signal[0:frame_size]`, no left
      padding at all). See `FramedSignal.__init__` below for the exact
      literal->integer origin translation, which uses ordinary (non-floor)
      division then truncates via `int()`.
    - If the frame fits entirely inside `[0, num_samples)`, this is a
      **plain slice** (a view, not a copy, for the in-bounds case).
    - Otherwise a `frame_size`-length buffer is allocated (via
      `np.repeat(signal[:1], frame_size, axis=0)`, which preserves dtype and
      any extra dimensions e.g. multi-channel) and back-filled: out-of-range
      samples get the literal `pad` value (default `0`) unless
      `pad == 'repeat'`, in which case the first/last real sample is
      repeated instead of zero-padding.

    `signal` here is a plain ndarray (e.g. `FramedSignal.signal.data`), not a
    `Signal` -- this function has no dependency on the `Signal` wrapper.
    """
    frame_size = int(frame_size)
    num_samples = len(signal)
    ref_sample = int(index * hop_size)
    start = ref_sample - frame_size // 2 - int(origin)
    stop = start + frame_size

    if start >= 0 and stop <= num_samples:
        # normal read operation, return appropriate section (a view)
        return signal[start:stop]

    # part of the frame falls outside the signal, padding needed
    frame = np.repeat(signal[:1], frame_size, axis=0)

    left, right = 0, 0
    if start < 0:
        left = min(stop, 0) - start
        frame[:left] = np.repeat(signal[:1], left, axis=0)
        if pad != "repeat":
            frame[:left] = pad
        start = 0
    if stop > num_samples:
        right = stop - max(start, num_samples)
        frame[-right:] = np.repeat(signal[-1:], right, axis=0)
        if pad != "repeat":
            frame[-right:] = pad
        stop = num_samples

    frame[left:frame_size - right] = signal[min(start, num_samples):
                                             max(stop, 0)]
    return frame


def _translate_origin(origin, frame_size):
    """Translate a literal `origin` value to its numeric equivalent.

    Port of the translation block in `madmom.audio.signal.FramedSignal.
    __init__` (`madmom-upstream/madmom/audio/signal.py:1125-1142`). Integer
    `origin` values pass through unchanged; the literal strings are
    equivalent to:

    - `'center'`/`'offline'` -> `0` (window centered on the reference
      sample);
    - `'left'`/`'past'`/`'online'` -> `(frame_size - 1) / 2` (window to the
      left of -- i.e. only using past information relative to -- the
      reference sample; used to simulate online/causal processing);
    - `'right'`/`'future'`/`'stream'` -> `-(frame_size / 2)` (window to the
      right of the reference sample; used for live-stream single-frame
      retrieval).

    Note this uses ordinary (float, non-floor) division -- the caller
    truncates the result via `int()` afterwards (matching madmom exactly,
    including the sign: for even `frame_size` the 'left'/'right' origins are
    NOT simple negatives of each other after truncation, e.g. frame_size=6
    gives left origin `int(2.5) == 2` but right origin `int(-3.0) == -3`).
    """
    if origin in ("center", "offline"):
        return 0
    elif origin in ("left", "past", "online"):
        return (frame_size - 1) / 2
    elif origin in ("right", "future", "stream"):
        return -(frame_size / 2)
    return origin


class FramedSignal:
    """Splits a `Signal` into frames; iterable and indexable.

    Composition port of `madmom.audio.signal.FramedSignal`
    (`madmom-upstream/madmom/audio/signal.py:974-1251`). Frame `i`'s
    reference sample is `i * hop_size` (rounded via `int()`); `hop_size` can
    be a float (e.g. derived from `fps`), in which case successive frames'
    reference samples are NOT exactly `hop_size` apart in general (ordinary
    float->int truncation per frame, not accumulated rounding error
    correction) -- this matches madmom's own behavior exactly since both
    compute `int(index * hop_size)` fresh per frame.

    `num_frames`, if not given explicitly, is derived from `end`:

    - `'normal'` (default): `ceil(len(signal) / hop_size)` -- stop as soon
      as the whole signal is covered by at least one frame (pads at most one
      frame's worth past the signal end).
    - `'extend'`: `floor(len(signal) / hop_size + 1)` -- keep returning
      frames as long as any part of the frame overlaps the signal.

    `.signal` (a `Signal` instance) is exposed so a later STFT stage can
    read `.signal.dtype` -- this is exactly the hook madmom's own
    `ShortTimeFourierTransform.__new__` uses to divide the FFT window by
    `np.iinfo(frames.signal.dtype).max` for integer-dtype signals
    (`madmom-upstream/madmom/audio/stft.py:339-349`), which keeps the
    signal itself un-rescaled (int16 stays int16 through framing) while
    still producing the right-magnitude STFT.
    """

    def __init__(self, signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                 fps=FPS, origin=ORIGIN, end=END_OF_SIGNAL,
                 num_frames=NUM_FRAMES, **kwargs):
        if not isinstance(signal, Signal):
            signal = Signal(signal, **kwargs)
        self.signal = signal

        if frame_size:
            self.frame_size = int(frame_size)
        if hop_size:
            self.hop_size = float(hop_size)
        if fps:
            # fps overwrites hop_size, derived from the signal's sample rate
            self.hop_size = self.signal.sample_rate / float(fps)

        self.origin = int(_translate_origin(origin, self.frame_size))

        if num_frames is None:
            if end == "extend":
                num_frames = np.floor(
                    len(self.signal) / float(self.hop_size) + 1
                )
            elif end == "normal":
                num_frames = np.ceil(len(self.signal) / float(self.hop_size))
            else:
                raise ValueError("end of signal handling %r unknown" % end)
        self.num_frames = int(num_frames)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            if index < 0:
                index += self.num_frames
            if index < self.num_frames:
                return signal_frame(
                    self.signal.data, index, frame_size=self.frame_size,
                    hop_size=self.hop_size, origin=self.origin,
                )
            raise IndexError("end of signal reached")
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.num_frames)
            if step != 1:
                raise ValueError("only slices with a step size of 1 supported")
            num_frames = stop - start
            origin = self.origin - self.hop_size * start
            return FramedSignal(
                self.signal, frame_size=self.frame_size,
                hop_size=self.hop_size, origin=origin, num_frames=num_frames,
            )
        else:
            raise TypeError("frame indices must be slices or integers")

    def __len__(self):
        return self.num_frames

    @property
    def frame_rate(self):
        """Frame rate (same as fps)."""
        if self.signal.sample_rate is None:
            return None
        return float(self.signal.sample_rate) / self.hop_size

    @property
    def fps(self):
        """Frames per second."""
        return self.frame_rate

    @property
    def overlap_factor(self):
        """Overlapping factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.frame_size

    @property
    def shape(self):
        """Shape of the FramedSignal: (num_frames, frame_size[, num_channels])."""
        shape = (self.num_frames, self.frame_size)
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels,)
        return shape

    @property
    def ndim(self):
        return len(self.shape)


class FramedSignalProcessor(Processor):
    """Processor wrapper: slice a `Signal` into frames.

    Composition port of `madmom.audio.signal.FramedSignalProcessor`
    (`madmom-upstream/madmom/audio/signal.py:1254-1393`). A
    `madmom_infer.processors.Processor` subclass, so it composes into a
    `SequentialProcessor` exactly like `all-in-one-infer`'s
    `FramedSignalProcessor(frame_size=2048, fps=100)`
    (`all-in-one-fix/src/allin1_infer/spectrogram.py:28-31`).
    """

    def __init__(self, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=FPS,
                 origin=ORIGIN, end=END_OF_SIGNAL, num_frames=NUM_FRAMES,
                 **kwargs):
        # pylint: disable=unused-argument
        # Wave 4b: `**kwargs` catch-all added to match upstream's own
        # `FramedSignalProcessor.__init__` signature exactly (`signal.py:
        # 1298-1300`, which this port had dropped) -- needed so
        # `features/onsets.py`'s `SpectralOnsetProcessor` can blindly
        # kwargs-forward one shared dict across its whole pre-processing
        # chain (`SignalProcessor`/`FramedSignalProcessor`/
        # `ShortTimeFourierTransformProcessor`/... all silently absorb keys
        # meant for a different stage), exactly like upstream's own
        # equivalently-loose processors do.
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fps = fps  # not converted here, forwarded to FramedSignal
        self.origin = origin
        self.end = end
        self.num_frames = num_frames

    def process(self, data, **kwargs):
        """Slice `data` (a `Signal`, path, or array) into overlapping frames.

        Returns a `FramedSignal` instance, matching
        `madmom.audio.signal.FramedSignalProcessor.process`
        (`signal.py:1309-1336`).
        """
        args = dict(frame_size=self.frame_size, hop_size=self.hop_size,
                    fps=self.fps, origin=self.origin, end=self.end,
                    num_frames=self.num_frames)
        args.update(kwargs)
        if self.origin == "stream":
            # always use the last `frame_size` samples in live-stream mode
            data = data[-self.frame_size:]
        return FramedSignal(data, **args)
