"""Neural-network layers -- port of the subset of madmom.ml.nn.layers that
the Phase-2 target ensemble (`DOWNBEATS_BLSTM`, 8x `downbeats_blstm_N.pkl`),
the 4a target model (`KEY_CNN`, `key/2018/key_cnn.pkl`), and the 4b onset
models (`ONSETS_RNN`/`ONSETS_BRNN`/`ONSETS_CNN`) actually need, enumerated
by inspecting the pickled files themselves with `pickletools` (not guessed
from reading source): every one of the 8 `downbeats_blstm_*` files
references exactly `NeuralNetwork`, `BidirectionalLayer`, `LSTMLayer`,
`Gate`, `Cell`, `FeedForwardLayer` (plus `activations.sigmoid`/`tanh`/
`softmax`); `key_cnn.pkl` references exactly `NeuralNetwork`,
`ConvolutionalLayer`, `MaxPoolLayer`, `BatchNormLayer`, `PadLayer`,
`AverageLayer` (plus `activations.elu`/`linear`, both already ported in
Phase 2 for other reasons); `onsets_rnn_*.pkl`/`onsets_brnn_*.pkl` reference
only already-ported classes (`FeedForwardLayer`/`RecurrentLayer`, plus
`BidirectionalLayer` for the BRNN family -- no new classes needed);
`onsets_cnn.pkl` references `BatchNormLayer`, `ConvolutionalLayer`,
`FeedForwardLayer`, `MaxPoolLayer` (all already ported by 4a) PLUS
`StrideLayer` (new in 4b) -- see madmom_infer/ml/nn/unpickle.py's module
header for the full class-path mapping table this finding produced.

Deliberately NOT ported (no reference in any Phase-2/4a/4b target pickle;
left for a future wave that targets a model actually using them):
`SequentialLayer`, `ParallelLayer`, `MultiTaskLayer` (composition-only
wrappers, no pickled model targeted so far routes through them),
`RecurrentLayer` is kept (it's `Gate`/`Cell`'s and `LSTMLayer`'s own base
class), `GRUCell`/`GRULayer` (gated-recurrent-unit variant --
`DOWNBEATS_BGRU` models, tentatively 4c, see CLAUDE.md), `TransposeLayer`/
`ReshapeLayer` (needed by `notes_cnn.pkl`, not `onsets_cnn.pkl` -- confirmed
by 4b's own pickletools walk of `onsets_cnn.pkl`, which references
`StrideLayer` but neither of these; stays TO-PORT for 4e), `TCNBlock`/
`TCNLayer` (temporal-conv-net -- e.g. `BEATS_TCN`, permanently EXCLUDED, no
shipped model references them).

`ConvolutionalLayer`'s `convolve()` helper: upstream tries `cv2.filter2D`
first (faster for some kernel sizes) and falls back to
`scipy.ndimage.convolve` if opencv isn't importable. This project has no
`opencv-python` dependency and never will (it's not in `pyproject.toml`,
and the reference venv used to record `key_cnn`'s golden fixtures also has
no `cv2` installed -- confirmed empirically, see
`tools/generate_key_fixtures.py`), so this port only implements the
`scipy.ndimage.convolve` path (`_convolve_scipy`/`_kernel_margins`) --
exactly what real madmom actually executes in the environment this port's
own fixtures were recorded from, not a speculative subset.

Faithful-port note: `Layer`/`RecurrentLayer`/`Gate`/`LSTMLayer` intentionally
do NOT define `__init__`-time construction paths that pickled models rely
on -- pickle restores instances via `NEWOBJ` (bypassing `__init__` entirely)
plus a direct `__dict__`/`__setstate__` restore. This is why the classes
below only need matching ATTRIBUTE NAMES (`weights`, `bias`,
`recurrent_weights`, `peephole_weights`, `activation_fn`, `input_gate`,
`forget_gate`, `cell`, `output_gate`, `fwd_layer`, `bwd_layer`, `stride`,
`pad`, `beta`, `gamma`, `mean`, `inv_std`, `size`, `axis`, `width`, `axes`,
`value`, `dtype`, `keepdims`), not constructor-argument-perfect `__init__`s,
to be valid unpickle targets -- verified via `pickletools.dis()` against all
8 `downbeats_blstm_*.pkl` files (see unpickle.py header) and, for the CNN
layers below, against `key_cnn.pkl`.

Reads: numpy, scipy.ndimage (convolve, maximum_filter),
madmom_infer.ml.nn.activations (sigmoid, tanh); read by:
madmom_infer/ml/nn/unpickle.py (as unpickle targets),
madmom_infer/ml/nn/__init__.py (NeuralNetwork.process's per-layer loop).
"""

import numpy as np
from scipy.ndimage import convolve as _scipy_convolve, maximum_filter

from .activations import sigmoid, tanh

NN_DTYPE = np.float32


class Layer:
    """Generic callable network layer. Port of `layers.Layer`
    (`madmom-upstream/madmom/ml/nn/layers.py:22-54`)."""

    def __call__(self, *args, **kwargs):
        return self.activate(*args, **kwargs)

    def activate(self, data):
        """Activate the layer. Must be implemented by subclasses."""
        raise NotImplementedError("must be implemented by subclass.")

    def reset(self):
        """Reset the layer to its initial state (no-op by default)."""
        return None


class FeedForwardLayer(Layer):
    """Feed-forward network layer: `activation_fn(data @ weights + bias)`.

    Port of `layers.FeedForwardLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:190-229`). `weights` has shape
    `(num_inputs, num_hiddens)`; `bias` is flattened to 1D at construction.
    """

    def __init__(self, weights, bias, activation_fn=None):
        self.weights = weights
        self.bias = bias.flatten()
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        """Activate FeedForwardLayer: `data` shape `(num_frames,
        num_inputs)` -> `(num_frames, num_hiddens)`."""
        out = np.dot(data, self.weights) + self.bias
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)
        return out


class RecurrentLayer(FeedForwardLayer):
    """Recurrent network layer: adds a `recurrent_weights @ prev_output`
    term to each frame's feed-forward output, applied one frame at a time.

    Port of `layers.RecurrentLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:232-323`). `_prev` (the
    previous time step's output) is deliberately excluded from
    `__getstate__`/pickling (it's transient runtime state, not a model
    parameter) and reinitialized to `init` on `__setstate__`/`reset()`.
    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh,
                 init=None):
        super().__init__(weights, bias, activation_fn)
        self.recurrent_weights = recurrent_weights
        if init is None:
            init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_prev", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "init"):
            self.init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self._prev = self.init

    def reset(self, init=None):
        """Reset RecurrentLayer to its initial state."""
        self._prev = init if init is not None else self.init

    def activate(self, data, reset=True):
        """Activate RecurrentLayer: `data` shape `(num_frames, num_inputs)`
        -> `(num_frames, num_hiddens)`, one frame at a time, feeding each
        frame's output into the next frame's recurrent term."""
        if reset:
            self.reset()
        out = np.dot(data, self.weights) + self.bias
        for i in range(len(data)):
            out[i] += np.dot(self._prev, self.recurrent_weights)
            if self.activation_fn is not None:
                out[i] = self.activation_fn(out[i])
            self._prev = out[i]
        return out


class BidirectionalLayer(Layer):
    """Bidirectional network layer: runs `fwd_layer` forward and
    `bwd_layer` on the time-reversed input, then stacks (`hstack`) the two
    activations.

    Port of `layers.BidirectionalLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:326-367`).
    """

    def __init__(self, fwd_layer, bwd_layer):
        self.fwd_layer = fwd_layer
        self.bwd_layer = bwd_layer

    def activate(self, data, **kwargs):
        """Activate BidirectionalLayer: `data` shape `(num_frames,
        num_inputs)` -> `(num_frames, 2 * num_hiddens)`."""
        fwd = self.fwd_layer(data, **kwargs)
        bwd = self.bwd_layer(data[::-1], **kwargs)
        return np.hstack((fwd, bwd[::-1]))


# -- LSTM stuff --------------------------------------------------------
class Gate(RecurrentLayer):
    """A gate as used inside `LSTMLayer` (input/forget/output gates).

    Port of `layers.Gate` (`madmom-upstream/madmom/ml/nn/layers.py:371-430`).
    Unlike a bare `RecurrentLayer`, a `Gate.activate()` call takes the
    CURRENT frame's data plus the previous frame's output/state directly
    (no internal per-frame loop of its own -- `LSTMLayer.activate()` drives
    the frame loop and calls each gate/cell once per frame). Should not be
    used standalone, only inside an `LSTMLayer`.
    """

    def __init__(self, weights, bias, recurrent_weights, peephole_weights=None,
                 activation_fn=sigmoid):
        super().__init__(weights, bias, recurrent_weights,
                          activation_fn=activation_fn)
        if peephole_weights is not None:
            peephole_weights = peephole_weights.flatten()
        self.peephole_weights = peephole_weights

    def activate(self, data, prev, state=None):
        """Activate the gate with the current frame's `data`, the previous
        frame's output `prev`, and (if peephole connections are used) the
        current/previous cell `state`."""
        out = np.dot(data, self.weights) + self.bias
        if self.peephole_weights is not None:
            out += state * self.peephole_weights
        out += np.dot(prev, self.recurrent_weights)
        return self.activation_fn(out)


class Cell(Gate):
    """A cell as used inside `LSTMLayer`: a `Gate` without peephole
    connections, `tanh`-activated by default.

    Port of `layers.Cell` (`madmom-upstream/madmom/ml/nn/layers.py:433-458`).
    Should not be used standalone, only inside an `LSTMLayer`.
    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn=tanh):
        super().__init__(weights, bias, recurrent_weights,
                          activation_fn=activation_fn)


class LSTMLayer(RecurrentLayer):
    """Recurrent layer with Long Short-Term Memory units: four `Gate`/`Cell`
    sub-layers (input gate, forget gate, cell, output gate), driven one
    frame at a time.

    Port of `layers.LSTMLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:461-587`). `_prev`/`_state`
    (previous output/cell state) are transient runtime attributes, excluded
    from pickling exactly like `RecurrentLayer._prev` (see that class's
    docstring).
    """

    def __init__(self, input_gate, forget_gate, cell, output_gate,
                 activation_fn=tanh, init=None, cell_init=None):
        # pylint: disable=super-init-not-called
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell = cell
        self.output_gate = output_gate
        self.activation_fn = activation_fn
        if init is None:
            init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init
        if cell_init is None:
            cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self.cell_init = cell_init
        self._state = self.cell_init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_prev", None)
        state.pop("_state", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "init"):
            self.init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        if not hasattr(self, "cell_init"):
            self.cell_init = np.zeros(self.cell.bias.size, dtype=NN_DTYPE)
        self._prev = self.init
        self._state = self.cell_init

    def reset(self, init=None, cell_init=None):
        """Reset LSTMLayer to its initial state (hidden output + cell)."""
        self._prev = init if init is not None else self.init
        self._state = cell_init if cell_init is not None else self.cell_init

    def activate(self, data, reset=True):
        """Activate LSTMLayer: `data` shape `(num_frames, num_inputs)` ->
        `(num_frames, num_hiddens)`, one frame at a time."""
        if reset:
            self.reset()
        size = len(data)
        out = np.zeros((size, self.cell.bias.size), dtype=NN_DTYPE)
        for i in range(size):
            data_ = data[i]
            ig = self.input_gate.activate(data_, self._prev, self._state)
            fg = self.forget_gate.activate(data_, self._prev, self._state)
            cell = self.cell.activate(data_, self._prev)
            self._state = cell * ig + self._state * fg
            og = self.output_gate.activate(data_, self._prev, self._state)
            out[i] = self.activation_fn(self._state) * og
            self._prev = out[i]
        return out


# -- CNN stuff (4a: key_cnn.pkl's layer family) -------------------------
def _kernel_margins(kernel_shape, margin_shift, pad="valid"):
    """Determine the margin to cut off a 'full'-mode convolution/correlation
    result to get the 'valid' or 'same' output size.

    Verbatim port of `layers._kernel_margins`
    (`madmom-upstream/madmom/ml/nn/layers.py:785-831`). `margin_shift`
    (True for the scipy convolution path, False for the opencv path this
    project doesn't implement -- see module header) shifts the borders by
    one pixel for even-sized kernels, matching `scipy.ndimage.convolve`'s
    slightly different even-kernel centering versus opencv's.
    """
    if pad == "same":
        return None, None, None, None
    elif pad != "valid":
        raise NotImplementedError("only `pad` == \"valid\" implemented.")

    start_x = int(np.floor(kernel_shape[0] / 2.0))
    start_y = int(np.floor(kernel_shape[1] / 2.0))

    margin_shift = -1 if margin_shift else 0
    if kernel_shape[0] % 2 == 0:
        end_x = start_x - 1
        start_x += margin_shift
        end_x -= margin_shift
    else:
        end_x = start_x
    start_x = start_x if start_x > 0 else None
    end_x = -end_x if end_x > 0 else None

    if kernel_shape[1] % 2 == 0:
        end_y = start_y - 1
        start_y += margin_shift
        end_y -= margin_shift
    else:
        end_y = start_y
    start_y = start_y if start_y > 0 else None
    end_y = -end_y if end_y > 0 else None

    return start_x, end_x, start_y, end_y


def _convolve_scipy(x, k, pad):
    """`scipy.ndimage.convolve`-backed 2D convolution, cropped to `pad`'s
    output size. Verbatim port of `layers._convolve_scipy`
    (`madmom-upstream/madmom/ml/nn/layers.py:855-859`) -- the only backend
    this port implements, see module header re: no `cv2` dependency."""
    sx, ex, sy, ey = _kernel_margins(k.shape, margin_shift=True, pad=pad)
    return _scipy_convolve(x, k, mode="constant")[sx:ex, sy:ey]


def convolve(data, kernel, pad="valid"):
    """Convolve `data` with `kernel`.

    Port of `layers.convolve` (`madmom-upstream/madmom/ml/nn/layers.py:
    862-895`), minus the opencv-backed fast path (`_convolve_opencv`) --
    this project has no `cv2` dependency (see module header), so this
    always takes upstream's own scipy fallback branch, which is also the
    branch the reference venv used to record `key_cnn`'s golden fixtures
    actually executes (confirmed: no `cv2` installed there either).
    """
    return _convolve_scipy(data, kernel, pad)


class ConvolutionalLayer(FeedForwardLayer):
    """Convolutional network layer: 2D convolution over `(num_frames,
    num_bins[, num_channels])` data, one kernel per output feature map, per
    input channel, summed across channels, plus bias and an activation
    function.

    Port of `layers.ConvolutionalLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:898-984`). `weights` has shape
    `(num_channels, num_features, kernel_time, kernel_freq)`; a missing
    channel axis on the input (`data.ndim == 2`) is treated as a single
    channel. Only `pad='valid'` is exercised by `key_cnn.pkl` (every
    `ConvolutionalLayer` there is preceded by an explicit `PadLayer`, so the
    convolution itself always crops rather than zero-pads), but `'same'` is
    ported too since it's a two-line branch in the original.
    """

    def __init__(self, weights, bias, stride=None, pad="valid",
                 activation_fn=None):
        super().__init__(weights, bias, activation_fn)
        self.stride = stride
        self.pad = pad

    def activate(self, data, **kwargs):
        """Activate ConvolutionalLayer: `data` shape `(num_frames, num_bins[,
        num_channels])` -> `(num_frames', num_bins', num_features)`."""
        # if no channel dimension given, assume 1 channel
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))

        # determine output shape and allocate memory
        num_frames, num_bins, num_channels = data.shape
        num_channels_w, num_features, size_time, size_freq = self.weights.shape
        if num_channels_w != num_channels:
            raise ValueError(
                "Number of channels in weight vector different from "
                "number of channels of input data!"
            )
        # adjust the output number of frames and bins depending on `pad`
        if self.pad == "valid":
            num_frames -= (size_time - 1)
            num_bins -= (size_freq - 1)
        elif self.pad != "same":
            raise NotImplementedError("`pad` is neither \"valid\" nor \"same\"")

        # init the output array with Fortran ordering (column major)
        out = np.zeros((num_frames, num_bins, num_features),
                        dtype=NN_DTYPE, order="F")
        # iterate over all channels
        for c in range(num_channels):
            channel = data[:, :, c]
            # convolve each channel separately with each filter
            for w, weights in enumerate(self.weights[c]):
                conv = convolve(channel, weights, self.pad)
                out[:, :, w] += conv
        # add bias to each feature map and apply activation function
        out += self.bias
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)

        # use only selected parts of the output
        if self.stride not in (None, 1, (1, 1)):
            out = out[::self.stride[0], ::self.stride[1]]

        return out


class MaxPoolLayer(Layer):
    """2D max-pooling network layer.

    Port of `layers.MaxPoolLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:1022-1092`), via
    `scipy.ndimage.maximum_filter`. `axis`, if set, ignores `size`/`stride`
    and just takes `np.max` along that axis (not used by `key_cnn.pkl`,
    which always pools spatially with `axis=None`, but ported for
    completeness/faithfulness).
    """

    def __init__(self, size, stride=None, axis=None):
        self.size = size
        if stride is None:
            stride = size
        self.stride = stride
        self.axis = axis

    def __setstate__(self, state):
        # restore pickled instance attributes
        self.__dict__.update(state)
        # old models do not have `axis`, thus create it -- matches upstream
        # (layers.py:1048-1051), kept even though key_cnn.pkl's own
        # MaxPoolLayer instances already carry `axis` (confirmed by
        # pickletools inspection), for faithfulness with older models.
        if not hasattr(self, "axis"):
            self.axis = None

    def activate(self, data, **kwargs):
        """Activate MaxPoolLayer: `data` shape `(num_frames, num_bins[,
        num_channels])` -> max-pooled data of the same rank."""
        if self.axis is not None:
            if self.stride is not None:
                raise NotImplementedError("`axis` with `stride` not supported")
            return np.max(data, axis=self.axis)
        # define which part of the maximum filtered data to return
        slice_dim_1 = slice(self.size[0] // 2,
                             data.shape[0] - (self.size[0] - 1) // 2,
                             self.stride[0])
        slice_dim_2 = slice(self.size[1] // 2,
                             data.shape[1] - (self.size[1] - 1) // 2,
                             self.stride[1])

        if len(data.shape) == 2:
            # filter the data as is
            return maximum_filter(data, self.size,
                                   mode="constant")[slice_dim_1, slice_dim_2]
        elif len(data.shape) == 3:
            # filter each channel separately
            data = [maximum_filter(data[:, :, c], self.size, mode="constant")
                    [slice_dim_1, slice_dim_2] for c in range(data.shape[2])]
            # join channels and return as array
            return np.dstack(data)
        else:
            raise ValueError("`data` must be either 2 or 3-dimensional")


class BatchNormLayer(Layer):
    """Batch normalization layer with activation function. The previous
    layer is usually linear with no bias -- this layer's `beta` parameter
    replaces it.

    Port of `layers.BatchNormLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:1095-1152`). `beta`/`gamma`/
    `mean`/`inv_std` must broadcast against the incoming data (in
    `key_cnn.pkl` they broadcast against the last, feature-map axis).

    References
    ----------
    .. [1] "Batch Normalization: Accelerating Deep Network Training by
           Reducing Internal Covariate Shift", Sergey Ioffe and Christian
           Szegedy, http://arxiv.org/abs/1502.03167, 2015.
    """

    def __init__(self, beta, gamma, mean, inv_std, activation_fn=None):
        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.inv_std = inv_std
        self.activation_fn = activation_fn

    def activate(self, data, **kwargs):
        """Activate BatchNormLayer: normalize then apply `activation_fn`."""
        out = (data - self.mean) * (self.gamma * self.inv_std) + self.beta
        if self.activation_fn is not None:
            self.activation_fn(out, out=out)
        return out


class AverageLayer(Layer):
    """Average layer: `np.mean` over the given axis/axes.

    Port of `layers.AverageLayer`
    (`madmom-upstream/madmom/ml/nn/layers.py:1226-1266`). `key_cnn.pkl`'s
    final layer is one of these, with `axis=(0, 1)` -- averaging the CNN's
    per-frame, per-bin class-map output down to one 24-class vector per
    clip (a global-average-pooling head).
    """

    def __init__(self, axis=None, dtype=None, keepdims=False):
        self.axis = axis
        self.dtype = dtype
        self.keepdims = keepdims

    def activate(self, data, **kwargs):
        """Activate AverageLayer: `np.mean(data, axis=self.axis, ...)`."""
        return np.mean(data, axis=self.axis, dtype=self.dtype,
                        keepdims=self.keepdims)


class PadLayer(Layer):
    """Padding layer that pads the input with a constant value along the
    given axes.

    Port of `layers.PadLayer` (`madmom-upstream/madmom/ml/nn/layers.py:
    1269-1311`). `key_cnn.pkl` uses this before every `ConvolutionalLayer`
    (`width` in `{1, 2}`, `axes=(0, 1)`, i.e. pad the time and frequency
    axes) to implement 'same'-style convolutions via an explicit pad +
    'valid' convolution, rather than `ConvolutionalLayer(pad='same')`
    directly.
    """

    def __init__(self, width, axes, value=0.0):
        self.width = width
        self.axes = axes
        self.value = value

    def activate(self, data, **kwargs):
        """Activate PadLayer: pad `self.axes` by `self.width` on both
        sides with `self.value`."""
        shape = list(data.shape)
        data_idxs = [slice(None) for _ in range(len(shape))]
        for a in self.axes:
            shape[a] += self.width * 2
            data_idxs[a] = slice(self.width, -self.width)
        data_padded = np.full(tuple(shape), self.value)
        data_padded[tuple(data_idxs)] = data
        return data_padded


class StrideLayer(Layer):
    """Stride layer: re-arrange the data into overlapping blocks of
    `block_size` consecutive frames along axis 0, flattened, ready for a
    following dense (`FeedForwardLayer`) layer.

    Port of `layers.StrideLayer` (`madmom-upstream/madmom/ml/nn/layers.py:
    987-1019`) -- new in Wave 4b, `onsets_cnn.pkl`'s own layer stack
    (confirmed by `pickletools`, see module header). `segment_axis` is the
    narrow (`axis=0`, `end='cut'`, `hop_size=1`) carve-out
    `madmom_infer/utils.py` ports -- exactly what this layer calls it with.
    """

    def __init__(self, block_size):
        self.block_size = block_size

    def activate(self, data, **kwargs):
        """Activate StrideLayer: `data` shape `(num_frames, ...)` ->
        `(num_frames - block_size + 1, block_size * prod(...))`."""
        from ...utils import segment_axis

        data = segment_axis(data, self.block_size, 1, axis=0, end="cut")
        return data.reshape(len(data), -1)
