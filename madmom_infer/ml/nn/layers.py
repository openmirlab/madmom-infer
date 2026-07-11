"""Neural-network layers -- port of the subset of madmom.ml.nn.layers that
the Phase-2 target ensemble (`DOWNBEATS_BLSTM`, 8x `downbeats_blstm_N.pkl`)
actually needs, enumerated by inspecting the pickled files themselves with
`pickletools` (not guessed from reading source): every one of the 8 files
references exactly `NeuralNetwork`, `BidirectionalLayer`, `LSTMLayer`,
`Gate`, `Cell`, `FeedForwardLayer` (plus `activations.sigmoid`/`tanh`/
`softmax`) -- see madmom_infer/ml/nn/unpickle.py's module header for the
full class-path mapping table this finding produced.

Deliberately NOT ported (no reference in any Phase-2 target pickle; left for
a future phase that targets a model actually using them): `SequentialLayer`,
`ParallelLayer`, `MultiTaskLayer` (composition-only wrappers, no pickled
model in this phase happens to route through them), `RecurrentLayer` is kept
(it's `Gate`/`Cell`'s and `LSTMLayer`'s own base class), `GRUCell`/`GRULayer`
(gated-recurrent-unit variant -- `DOWNBEATS_BGRU` models, out of Phase-2
scope, see README), `ConvolutionalLayer`/`StrideLayer`/`MaxPoolLayer`/
`BatchNormLayer`/`TransposeLayer`/`ReshapeLayer`/`AverageLayer`/`PadLayer`
(CNN-era layers -- e.g. `NOTES_CNN`, `CHORDS_CNN_FEAT`, out of Phase-2
scope), `TCNBlock`/`TCNLayer` (temporal-conv-net -- e.g. `BEATS_TCN`, out of
Phase-2 scope), and the free `convolve()`/`_kernel_margins()`/opencv-vs-scipy
convolution dispatch (only used by `ConvolutionalLayer`).

Faithful-port note: `Layer`/`RecurrentLayer`/`Gate`/`LSTMLayer` intentionally
do NOT define `__init__`-time construction paths that pickled models rely
on -- pickle restores instances via `NEWOBJ` (bypassing `__init__` entirely)
plus a direct `__dict__`/`__setstate__` restore. This is why the classes
below only need matching ATTRIBUTE NAMES (`weights`, `bias`,
`recurrent_weights`, `peephole_weights`, `activation_fn`, `input_gate`,
`forget_gate`, `cell`, `output_gate`, `fwd_layer`, `bwd_layer`), not
constructor-argument-perfect `__init__`s, to be valid unpickle targets --
verified via `pickletools.dis()` against all 8 target `.pkl` files (see
unpickle.py header).

Reads: numpy, madmom_infer.ml.nn.activations (sigmoid, tanh); read by:
madmom_infer/ml/nn/unpickle.py (as unpickle targets),
madmom_infer/ml/nn/__init__.py (NeuralNetwork.process's per-layer loop).
"""

import numpy as np

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
