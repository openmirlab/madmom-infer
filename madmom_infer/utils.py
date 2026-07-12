"""Narrow carve-out of madmom.utils -- ONLY the two free functions this wave's
onset code actually needs, not a general port of `utils/__init__.py` (which
stays EXCLUDE per CLAUDE.md's 4.0 audit: I/O and annotation-file helpers, out
of this project's `features/`, `audio/`, `ml/` scope). Both were found to be
real, non-speculative dependencies while porting `features/onsets.py` (Wave
4b): `StrideLayer` (`madmom_infer/ml/nn/layers.py`, needed by
`onsets_cnn.pkl`) calls `segment_axis`, and `OnsetPeakPickingProcessor`
(`madmom_infer/features/onsets.py`) calls `combine_events`. See CLAUDE.md's
4.0 audit table, `utils/*` row, for the correction this addition makes.

`segment_axis` here is a DELIBERATELY NARROWER port than
`madmom.utils.segment_axis` (`madmom-upstream/madmom/utils/__init__.py:
542-664`, which supports arbitrary `axis`, `hop_size`, and 3 `end` modes):
only `axis=0` and `end='cut'` are implemented, because that is the ONLY
combination `StrideLayer.activate()` (`madmom-upstream/madmom/ml/nn/
layers.py:1001-1019`, `segment_axis(data, self.block_size, 1, axis=0,
end='cut')`) ever calls it with. Implemented via
`numpy.lib.stride_tricks.sliding_window_view` rather than upstream's manual
`np.ndarray.__new__(strides=..., buffer=...)` construction -- both produce
numerically identical windows (`window[i, t, ...] == data[i + t, ...]` for
hop_size=1), `sliding_window_view` is the modern, well-tested numpy-stdlib
way to express the same zero-copy sliding-window view, just with the new
window axis appended at the end instead of inserted after `axis` -- fixed up
with one `np.moveaxis` call to match upstream's axis ordering (and therefore
`StrideLayer`'s subsequent `.reshape(len(data), -1)` flattening order)
exactly.

`combine_events` is a verbatim, full port (all 3 `combine` modes) of
`madmom.utils.combine_events` (`madmom-upstream/madmom/utils/__init__.py:
275-328`) -- cheap enough to port completely rather than narrowing, even
though `OnsetPeakPickingProcessor` only ever calls it with `combine='left'`.

Reads: numpy; read by: madmom_infer/ml/nn/layers.py (StrideLayer),
madmom_infer/features/onsets.py (OnsetPeakPickingProcessor).
"""

import numpy as np


def segment_axis(signal, frame_size, hop_size, axis=0, end="cut"):
    """Chop `signal` into overlapping frames along `axis=0` with hop
    `hop_size`, discarding any trailing remainder (`end='cut'`).

    Narrow port of `madmom.utils.segment_axis` -- see module header for
    exactly which subset of upstream's generality this implements (only what
    `StrideLayer` needs) and why `axis`/`end` are still accepted as
    parameters (signature parity) but rejected outside their one supported
    value.
    """
    if axis != 0:
        raise NotImplementedError(
            "only `axis=0` implemented (see madmom_infer/utils.py's module "
            "header -- this is a narrow carve-out of madmom.utils."
            "segment_axis, not a full port)."
        )
    if end != "cut":
        raise NotImplementedError(
            "only `end='cut'` implemented (see madmom_infer/utils.py's "
            "module header)."
        )
    frame_size = int(frame_size)
    hop_size = int(hop_size)
    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")
    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")
    length = signal.shape[0]
    if length < frame_size:
        raise ValueError(
            "Not enough data points to segment array in 'cut' mode; try "
            "end='pad' or end='wrap' (not implemented here, see module "
            "header)."
        )
    # sliding_window_view appends the new window axis at the end
    # (..., frame_size); upstream's own strided construction inserts it
    # right after `axis` instead -- move it there to match exactly (this is
    # what StrideLayer's later `.reshape(len(data), -1)` call relies on).
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, window_shape=frame_size, axis=0
    )
    windows = np.moveaxis(windows, -1, 1)
    if hop_size != 1:
        windows = windows[::hop_size]
    return windows


def combine_events(events, delta, combine="mean"):
    """Combine all `events` within `delta` of each other.

    Verbatim port of `madmom.utils.combine_events`
    (`madmom-upstream/madmom/utils/__init__.py:275-328`).
    """
    # add a small value to delta, otherwise we end up in floating point hell
    delta += 1e-12
    # return immediately if possible
    if len(events) <= 1:
        return events
    # convert to numpy array or create a copy if needed
    events = np.array(events, dtype=float)
    # can handle only 1D events
    if events.ndim > 1:
        raise ValueError("only 1-dimensional events supported.")
    # set start position
    idx = 0
    # get first event
    left = events[idx]
    # iterate over all remaining events
    for right in events[1:]:
        if right - left <= delta:
            # combine the two events
            if combine == "mean":
                left = events[idx] = 0.5 * (right + left)
            elif combine == "left":
                left = events[idx] = left
            elif combine == "right":
                left = events[idx] = right
            else:
                raise ValueError(
                    "don't know how to combine two events with %s" % combine
                )
        else:
            # move forward
            idx += 1
            left = events[idx] = right
    # return the combined events
    return events[: idx + 1]
