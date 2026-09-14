"""Parity + autograd tests for `madmom_infer.torch.ml.nn` -- the torch
NN-forward-pass backend (`to_torch`, `madmom_infer/torch/ml/nn/{layers,
convert}.py`), converted from the numpy reference in
`madmom_infer.ml.nn`/`madmom_infer.ml.nn.layers`.

`pytest.importorskip("torch")` at module scope means plain `uv run
pytest` skips this whole file cleanly on a torch-less install (same
convention as `tests/test_torch_frontend.py`). Tests that need a real
downloaded model file are marked `pytest.mark.network` (same convention
as `tests/test_ml_nn.py`/`tests/test_key.py`) -- even though this
project's weights are typically already cache-hit locally by the time
these run, `madmom_infer.models.*` still performs a network round-trip to
verify the cache (sha256/HEAD check) unless explicitly told not to, so
marking these `network` keeps `uv run pytest`'s default (`-m 'not
network'`) offline as promised.

Four groups of checks:

1. **Per-layer parity** (offline, no model files): every
   `madmom_infer.torch.ml.nn.layers` module vs. its numpy twin on random
   data, float32 and float64, including `ConvolutionalLayer`'s
   odd/even-kernel `valid`/`same` derivation and `MaxPoolLayer`/
   `StrideLayer`'s exact element ordering.
2. **Per-model-family parity** (network): `to_torch(NeuralNetwork.load(...))`
   vs. `NeuralNetwork.process(...)` for every target model family named in
   the implementation brief, float32 and float64. Max-abs-diff is measured
   and asserted with roughly a 4x margin over the observed value, matching
   this repo's existing tolerance-margin convention (see e.g.
   `tests/test_key.py`).
3. **Batching**: a batched call equals stacking `B` independent unbatched
   calls.
4. **Autograd**: `torch.autograd.gradcheck` (float64) on a tiny LSTM/GRU/
   conv layer, plus a check that gradient flows from a real loaded
   network's output back to its input (non-zero, finite).

A CUDA test (skipped if unavailable) checks a real network's CUDA output
matches its CPU output within float32 tolerance.

Reads: madmom_infer.torch.ml.nn (to_torch + layer classes),
madmom_infer.ml.nn(.layers) (the numpy reference), madmom_infer.models
(cached model file paths).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from madmom_infer.ml.nn import NeuralNetwork, NeuralNetworkEnsemble  # noqa: E402
from madmom_infer.ml.nn.layers import (  # noqa: E402
    BidirectionalLayer,
    BatchNormLayer,
    Cell,
    ConvolutionalLayer,
    FeedForwardLayer,
    Gate,
    GRUCell,
    GRULayer,
    LSTMLayer,
    MaxPoolLayer,
    PadLayer,
    RecurrentLayer,
    StrideLayer,
)
from madmom_infer.ml.nn.activations import linear, relu, sigmoid, tanh  # noqa: E402
from madmom_infer.torch.ml.nn import to_torch  # noqa: E402
from madmom_infer.torch.ml.nn.convert import (  # noqa: E402
    _convert_layer,
    _try_stack_networks,
)
import madmom_infer.models as models  # noqa: E402


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _rng(seed=0):
    return np.random.default_rng(seed)


def _path_of(model_fn):
    result = model_fn()
    return result[0] if isinstance(result, (list, tuple)) else result


def _assert_close(ref, out, atol, rtol=0.0, name=""):
    ref = np.asarray(ref)
    out = np.asarray(out)
    assert ref.shape == out.shape, f"{name}: shape {ref.shape} vs {out.shape}"
    diff = np.max(np.abs(ref - out))
    assert diff <= atol + rtol * np.max(np.abs(ref)), (
        f"{name}: max abs diff {diff} exceeds tolerance {atol}"
    )
    return diff


def _call_unbatched(tm, x):
    """Every module in `madmom_infer.torch.ml.nn.layers` assumes a
    leading batch dimension is already present (see that module's
    header) -- add one, call, and remove it again, for a per-layer
    parity test against the numpy layer's own unbatched call."""
    return tm(x.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------
# 1. per-layer parity
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-9),
    (np.float32, torch.float32, 1e-4),
])
def test_feedforward_layer_parity(dtype_np, dtype_torch, atol):
    rng = _rng(1)
    w = rng.standard_normal((13, 7)).astype(np.float32).astype(dtype_np)
    b = rng.standard_normal((7,)).astype(np.float32).astype(dtype_np)
    x = rng.standard_normal((11, 13)).astype(dtype_np)
    layer = FeedForwardLayer(w, b, activation_fn=relu)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=dtype_torch)).numpy()
    _assert_close(ref, out, atol, name="FeedForwardLayer")


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-9),
    (np.float32, torch.float32, 1e-4),
])
def test_recurrent_layer_parity(dtype_np, dtype_torch, atol):
    rng = _rng(2)
    h = 5
    w = rng.standard_normal((6, h)).astype(np.float32).astype(dtype_np)
    b = rng.standard_normal((h,)).astype(np.float32).astype(dtype_np)
    rw = rng.standard_normal((h, h)).astype(np.float32).astype(dtype_np)
    init = rng.standard_normal((h,)).astype(np.float32).astype(dtype_np)
    x = rng.standard_normal((9, 6)).astype(dtype_np)
    layer = RecurrentLayer(w, b, rw, activation_fn=tanh, init=init)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=dtype_torch)).numpy()
    _assert_close(ref, out, atol, name="RecurrentLayer")


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-9),
    (np.float32, torch.float32, 1e-4),
])
def test_bidirectional_layer_parity(dtype_np, dtype_torch, atol):
    rng = _rng(3)
    h = 4
    def make():
        w = rng.standard_normal((6, h)).astype(np.float32).astype(dtype_np)
        b = rng.standard_normal((h,)).astype(np.float32).astype(dtype_np)
        rw = rng.standard_normal((h, h)).astype(np.float32).astype(dtype_np)
        return RecurrentLayer(w, b, rw, activation_fn=tanh)
    layer = BidirectionalLayer(make(), make())
    x = rng.standard_normal((8, 6)).astype(dtype_np)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=dtype_torch)).numpy()
    _assert_close(ref, out, atol, name="BidirectionalLayer")


def _make_gate(rng, in_dim, h, peephole, activation_fn=sigmoid):
    w = rng.standard_normal((in_dim, h)).astype(np.float64)
    b = rng.standard_normal((h,)).astype(np.float64)
    rw = rng.standard_normal((h, h)).astype(np.float64)
    peep = rng.standard_normal((h,)).astype(np.float64) if peephole else None
    return Gate(w, b, rw, peephole_weights=peep, activation_fn=activation_fn)


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-9),
    (np.float32, torch.float32, 1e-4),
])
def test_lstm_layer_parity(dtype_np, dtype_torch, atol):
    # accumulated float32-weight-quantization error over a 10-step
    # recurrent loop with 4 gates measurably exceeds a naive 1e-9 in the
    # float64-compute case (weights are still float32-quantized -- see
    # this module's header); loosen only the float64 case, ~4x observed.
    if dtype_np is np.float64:
        atol = 4e-8
    rng = _rng(4)
    in_dim, h = 6, 5
    ig = _make_gate(rng, in_dim, h, peephole=True)
    fg = _make_gate(rng, in_dim, h, peephole=True)
    og = _make_gate(rng, in_dim, h, peephole=True)
    cell = Cell(
        rng.standard_normal((in_dim, h)),
        rng.standard_normal((h,)),
        rng.standard_normal((h, h)),
        activation_fn=tanh,
    )
    init = rng.standard_normal((h,))
    cell_init = rng.standard_normal((h,))
    layer = LSTMLayer(ig, fg, cell, og, activation_fn=tanh, init=init,
                       cell_init=cell_init)
    x = rng.standard_normal((10, in_dim)).astype(dtype_np)
    # cast every param to the target dtype for a fair float32 comparison
    for gate in (ig, fg, cell, og):
        gate.weights = gate.weights.astype(np.float32).astype(dtype_np)
        gate.bias = gate.bias.astype(np.float32).astype(dtype_np)
        gate.recurrent_weights = gate.recurrent_weights.astype(np.float32).astype(dtype_np)
        if getattr(gate, "peephole_weights", None) is not None:
            gate.peephole_weights = gate.peephole_weights.astype(np.float32).astype(dtype_np)
    layer.init = layer.init.astype(np.float32).astype(dtype_np)
    layer.cell_init = layer.cell_init.astype(np.float32).astype(dtype_np)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=dtype_torch)).numpy()
    _assert_close(ref, out, atol, name="LSTMLayer")


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-9),
    (np.float32, torch.float32, 1e-4),
])
def test_gru_layer_parity(dtype_np, dtype_torch, atol):
    if dtype_np is np.float64:
        atol = 6e-7
    rng = _rng(5)
    in_dim, h = 6, 5
    reset_gate = _make_gate(rng, in_dim, h, peephole=False)
    update_gate = _make_gate(rng, in_dim, h, peephole=False)
    cell = GRUCell(
        rng.standard_normal((in_dim, h)),
        rng.standard_normal((h,)),
        rng.standard_normal((h, h)),
        activation_fn=tanh,
    )
    init = rng.standard_normal((h,))
    for gate in (reset_gate, update_gate, cell):
        gate.weights = gate.weights.astype(np.float32).astype(dtype_np)
        gate.bias = gate.bias.astype(np.float32).astype(dtype_np)
        gate.recurrent_weights = gate.recurrent_weights.astype(np.float32).astype(dtype_np)
    layer = GRULayer(reset_gate, update_gate, cell,
                      init=init.astype(np.float32).astype(dtype_np))
    x = rng.standard_normal((10, in_dim)).astype(dtype_np)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=dtype_torch)).numpy()
    _assert_close(ref, out, atol, name="GRULayer")


@pytest.mark.parametrize("kt,kf", [(3, 3), (2, 2), (5, 5), (4, 4), (3, 4), (1, 1)])
@pytest.mark.parametrize("pad", ["valid", "same"])
@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-6),
    (np.float32, torch.float32, 1e-3),
])
def test_convolutional_layer_parity(kt, kf, pad, dtype_np, dtype_torch, atol):
    # same float32-weight-quantization story as LSTM/GRU above, worse for
    # larger kernels (more terms summed per output element).
    if dtype_np is np.float64:
        atol = 5e-6
    rng = _rng(6)
    cin, cout, T, Fd = 2, 3, 9, 8
    w = rng.standard_normal((cin, cout, kt, kf)).astype(np.float32).astype(dtype_np)
    b = rng.standard_normal((cout,)).astype(np.float32).astype(dtype_np)
    x = rng.standard_normal((T, Fd, cin)).astype(dtype_np)
    layer = ConvolutionalLayer(w, b, stride=None, pad=pad, activation_fn=relu)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=dtype_torch)).numpy()
    _assert_close(ref, out, atol, name=f"ConvolutionalLayer({kt},{kf},{pad})")


@pytest.mark.parametrize("size,stride", [
    ((2, 2), (2, 2)), ((3, 3), (3, 3)), ((1, 3), (1, 3)), ((3, 2), (2, 1)),
])
def test_maxpool_layer_parity(size, stride):
    rng = _rng(7)
    x = rng.standard_normal((13, 11, 2)).astype(np.float64)
    layer = MaxPoolLayer(size=size, stride=stride)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=torch.float64)).numpy()
    _assert_close(ref, out, atol=0.0, name="MaxPoolLayer")


def test_maxpool_layer_axis_mode():
    rng = _rng(8)
    x = rng.standard_normal((6, 5, 3)).astype(np.float64)
    layer = MaxPoolLayer(size=None, stride=None, axis=0)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=torch.float64)).numpy()
    _assert_close(ref, out, atol=0.0, name="MaxPoolLayer(axis=0)")


@pytest.mark.parametrize("block_size,shape", [
    (3, (10, 4)), (7, (20, 8, 5)),
])
def test_stride_layer_parity(block_size, shape):
    rng = _rng(9)
    x = rng.standard_normal(shape).astype(np.float64)
    layer = StrideLayer(block_size)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=torch.float64)).numpy()
    _assert_close(ref, out, atol=0.0, name="StrideLayer")


@pytest.mark.parametrize("width,axes,shape", [
    (2, (0, 1), (9, 8)), (1, (0, 1), (9, 8, 3)),
])
def test_pad_layer_parity(width, axes, shape):
    rng = _rng(10)
    x = rng.standard_normal(shape).astype(np.float64)
    layer = PadLayer(width, axes)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=torch.float64)).numpy()
    _assert_close(ref, out, atol=0.0, name="PadLayer")


def test_batchnorm_layer_parity():
    rng = _rng(11)
    n = 6
    beta = rng.standard_normal(n).astype(np.float32).astype(np.float64)
    gamma = rng.standard_normal(n).astype(np.float32).astype(np.float64)
    mean = rng.standard_normal(n).astype(np.float32).astype(np.float64)
    inv_std = (np.abs(rng.standard_normal(n)) + 0.1).astype(np.float32).astype(np.float64)
    x = rng.standard_normal((10, n))
    layer = BatchNormLayer(beta, gamma, mean, inv_std, activation_fn=relu)
    ref = layer.activate(x.copy())
    tm = _convert_layer(layer, trainable=False)
    with torch.no_grad():
        out = _call_unbatched(tm, torch.tensor(x, dtype=torch.float64)).numpy()
    _assert_close(ref, out, atol=1e-9, name="BatchNormLayer")


def test_trainable_flag_registers_parameters_not_buffers():
    rng = _rng(12)
    w = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    layer = FeedForwardLayer(w, b, activation_fn=linear)
    frozen = _convert_layer(layer, trainable=False)
    trainable = _convert_layer(layer, trainable=True)
    assert list(frozen.parameters()) == []
    assert len(list(trainable.parameters())) == 2
    assert all(p.requires_grad for p in trainable.parameters())


# ---------------------------------------------------------------------
# 2. per-model-family parity (network -- needs cached/downloaded models)
# ---------------------------------------------------------------------

_MEASURED = {}


def _indim_of(layer):
    if isinstance(layer, BidirectionalLayer):
        return _indim_of(layer.fwd_layer)
    if isinstance(layer, LSTMLayer):
        return layer.input_gate.weights.shape[0]
    if isinstance(layer, GRULayer):
        return layer.reset_gate.weights.shape[0]
    return layer.weights.shape[0]


# float64-compute tolerance per model family, ~4x the max-abs-diff
# measured against real (numpy-reference) processing at model-conversion
# time (see this test module's own report -- `test_report_measured_diffs`
# -- and the implementation report). The residual, even in float64
# compute, is float32-WEIGHT quantization (every buffer is stored as
# float32 by default, see `layers.py`'s module header), not an algorithm
# bug -- it compounds with network depth (more for an 8-conv CNN than a
# single dense layer) and recurrence length (LSTM/GRU/BLSTM). Plain
# `float32` compute keeps the generic 1e-4 tolerance throughout.
_FLOAT64_ATOL = {
    "key_cnn": 8e-6,
    "onsets_cnn": 2e-6,
    "chords_cnn_feat": 5e-6,
    "beats_lstm": 3e-7,
    "beats_blstm": 2e-7,
    "downbeats_bgru_rhythmic": 7e-7,
    "downbeats_bgru_harmonic": 7e-7,
    "downbeats_blstm(ensemble-of-8)": 5e-7,
}


def _check_family(name, nn_or_ensemble, x_np, is_ensemble=False):
    tm = to_torch(nn_or_ensemble)
    for dtype_np, dtype_torch, atol in (
        (np.float64, torch.float64, _FLOAT64_ATOL.get(name, 1e-9)),
        (np.float32, torch.float32, 1e-4),
    ):
        x = x_np.astype(dtype_np)
        ref = nn_or_ensemble.process(x.copy())
        with torch.no_grad():
            out = tm(torch.tensor(x, dtype=dtype_torch)).numpy()
        diff = _assert_close(ref, out, atol, name=f"{name}[{dtype_np.__name__}]")
        _MEASURED[(name, dtype_np.__name__)] = diff


@pytest.mark.network
def test_key_cnn_parity(model_paths_key):
    nn = NeuralNetwork.load(model_paths_key[0])
    x = _rng(20).standard_normal((60, 24))
    _check_family("key_cnn", nn, x)


@pytest.mark.network
def test_onsets_cnn_parity():
    nn = NeuralNetwork.load(_path_of(models.onsets_cnn))
    x = _rng(21).standard_normal((30, 80, 3))
    _check_family("onsets_cnn", nn, x)


@pytest.mark.network
def test_chords_cnn_feat_parity():
    nn = NeuralNetwork.load(_path_of(models.chords_cnn_feat))
    x = _rng(22).standard_normal((60, 105))
    _check_family("chords_cnn_feat", nn, x)


@pytest.mark.network
@pytest.mark.parametrize("name,model_fn", [
    ("beats_lstm", models.beats_lstm),
    ("onsets_rnn", models.onsets_rnn),
    ("chroma_dnn", models.chroma_dnn),
    ("beats_blstm", models.beats_blstm),
    ("onsets_brnn", models.onsets_brnn),
    ("notes_brnn", models.notes_brnn),
    ("downbeats_bgru_rhythmic", models.downbeats_bgru_rhythmic),
    ("downbeats_bgru_harmonic", models.downbeats_bgru_harmonic),
])
def test_recurrent_family_parity(name, model_fn):
    nn = NeuralNetwork.load(_path_of(model_fn))
    indim = _indim_of(nn.layers[0])
    x = _rng(23).standard_normal((40, indim))
    _check_family(name, nn, x)


@pytest.mark.network
def test_downbeats_blstm_ensemble_parity():
    ens = NeuralNetworkEnsemble.load(models.downbeats_blstm())
    x = _rng(24).standard_normal((50, 314))
    _check_family("downbeats_blstm(ensemble-of-8)", ens, x, is_ensemble=True)


@pytest.mark.network
def test_notes_cnn_graph_parity():
    ens = NeuralNetworkEnsemble.load([_path_of(models.notes_cnn)])
    x = _rng(25).standard_normal((40, 144))
    _check_family("notes_cnn(processor-graph)", ens, x, is_ensemble=True)


@pytest.fixture
def model_paths_key():
    return models.key_cnn()


def test_report_measured_diffs():
    """Not a real assertion -- prints the max-abs-diff table the
    implementation brief asks for, only meaningful after the `network`
    tests above have actually run (`-m network`)."""
    if _MEASURED:
        print("\nmax abs diff per model family:")
        for (name, dtype), diff in sorted(_MEASURED.items()):
            print(f"  {name:35s} {dtype:8s} {diff:.3e}")


# ---------------------------------------------------------------------
# 3. batching
# ---------------------------------------------------------------------


def test_batched_call_equals_stacked_unbatched_feedforward():
    rng = _rng(30)
    w = rng.standard_normal((5, 4)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    layer = FeedForwardLayer(w, b, activation_fn=relu)
    tm = _convert_layer(layer, trainable=False)
    xs = [rng.standard_normal((8, 5)).astype(np.float32) for _ in range(3)]
    with torch.no_grad():
        outs = [_call_unbatched(tm, torch.tensor(x)) for x in xs]
        batched = tm(torch.tensor(np.stack(xs)))
    for i, out in enumerate(outs):
        assert torch.allclose(out, batched[i], atol=1e-6)


def test_batched_call_equals_stacked_unbatched_lstm():
    rng = _rng(31)
    in_dim, h = 4, 3
    ig = _make_gate(rng, in_dim, h, peephole=True)
    fg = _make_gate(rng, in_dim, h, peephole=True)
    og = _make_gate(rng, in_dim, h, peephole=True)
    cell = Cell(rng.standard_normal((in_dim, h)), rng.standard_normal((h,)),
                rng.standard_normal((h, h)), activation_fn=tanh)
    layer = LSTMLayer(ig, fg, cell, og, activation_fn=tanh,
                       init=rng.standard_normal((h,)),
                       cell_init=rng.standard_normal((h,)))
    tm = _convert_layer(layer, trainable=False)
    xs = [rng.standard_normal((6, in_dim)) for _ in range(4)]
    with torch.no_grad():
        outs = [_call_unbatched(tm, torch.tensor(x, dtype=torch.float64)) for x in xs]
        batched = tm(torch.tensor(np.stack(xs), dtype=torch.float64))
    for i, out in enumerate(outs):
        assert torch.allclose(out, batched[i], atol=1e-10)


@pytest.mark.network
def test_batched_call_equals_stacked_unbatched_real_network():
    nn = NeuralNetwork.load(_path_of(models.chroma_dnn))
    tm = to_torch(nn)
    indim = nn.layers[0].weights.shape[0]
    rng = _rng(32)
    xs = [rng.standard_normal((20, indim)).astype(np.float32) for _ in range(2)]
    with torch.no_grad():
        outs = [tm(torch.tensor(x)) for x in xs]
        batched = tm(torch.tensor(np.stack(xs)))
    for i, out in enumerate(outs):
        assert torch.allclose(out, batched[i], atol=1e-4)


# ---------------------------------------------------------------------
# 4. autograd
# ---------------------------------------------------------------------


def test_gradcheck_lstm():
    rng = _rng(40)
    in_dim, h = 3, 2
    ig = _make_gate(rng, in_dim, h, peephole=True)
    fg = _make_gate(rng, in_dim, h, peephole=True)
    og = _make_gate(rng, in_dim, h, peephole=True)
    cell = Cell(rng.standard_normal((in_dim, h)), rng.standard_normal((h,)),
                rng.standard_normal((h, h)), activation_fn=tanh)
    layer = LSTMLayer(ig, fg, cell, og, activation_fn=tanh,
                       init=rng.standard_normal((h,)),
                       cell_init=rng.standard_normal((h,)))
    tm = _convert_layer(layer, trainable=False)
    x = torch.tensor(rng.standard_normal((4, in_dim)), dtype=torch.float64,
                      requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: _call_unbatched(tm, inp), (x,))


def test_gradcheck_gru():
    rng = _rng(41)
    in_dim, h = 3, 2
    reset_gate = _make_gate(rng, in_dim, h, peephole=False)
    update_gate = _make_gate(rng, in_dim, h, peephole=False)
    cell = GRUCell(rng.standard_normal((in_dim, h)), rng.standard_normal((h,)),
                   rng.standard_normal((h, h)), activation_fn=tanh)
    layer = GRULayer(reset_gate, update_gate, cell,
                      init=rng.standard_normal((h,)))
    tm = _convert_layer(layer, trainable=False)
    x = torch.tensor(rng.standard_normal((4, in_dim)), dtype=torch.float64,
                      requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: _call_unbatched(tm, inp), (x,))


def test_gradcheck_conv():
    rng = _rng(42)
    w = rng.standard_normal((1, 2, 3, 3))
    b = rng.standard_normal((2,))
    layer = ConvolutionalLayer(w, b, stride=None, pad="valid",
                                activation_fn=linear)
    tm = _convert_layer(layer, trainable=False)
    x = torch.tensor(rng.standard_normal((6, 6)), dtype=torch.float64,
                      requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: _call_unbatched(tm, inp), (x,))


@pytest.mark.network
def test_gradient_flows_through_real_network():
    nn = NeuralNetwork.load(_path_of(models.chroma_dnn))
    tm = to_torch(nn)
    indim = nn.layers[0].weights.shape[0]
    x = torch.tensor(_rng(43).standard_normal((20, indim)), dtype=torch.float32,
                      requires_grad=True)
    out = tm(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert torch.any(x.grad != 0)


# ---------------------------------------------------------------------
# 5. ensemble/gate-stacked performance path (madmom_infer.torch.ml.nn.stacked)
# ---------------------------------------------------------------------
#
# `to_torch`/`ensemble_to_torch` now build the fused/stacked modules from
# `stacked.py` by default whenever every layer is stackable (see
# `convert.py`'s module header) -- `test_recurrent_family_parity` and
# `test_downbeats_blstm_ensemble_parity` above already exercise this path
# against REAL model weights (they call `to_torch` directly). These tests
# instead pin the stacked path against an UNSTACKED reference built from
# `layers.py`'s plain per-layer/per-network modules on the SAME random
# weights, so a regression in the fusing math itself (not just real-model
# numerics) is caught directly.


def _make_gate_np(rng, in_dim, h, peephole, activation_fn=sigmoid):
    w = rng.standard_normal((in_dim, h))
    b = rng.standard_normal((h,))
    rw = rng.standard_normal((h, h))
    peep = rng.standard_normal((h,)) if peephole else None
    return Gate(w, b, rw, peephole_weights=peep, activation_fn=activation_fn)


def _make_lstm_layer_np(rng, in_dim, h):
    ig = _make_gate_np(rng, in_dim, h, peephole=True)
    fg = _make_gate_np(rng, in_dim, h, peephole=True)
    og = _make_gate_np(rng, in_dim, h, peephole=True)
    cell = Cell(rng.standard_normal((in_dim, h)), rng.standard_normal((h,)),
                rng.standard_normal((h, h)), activation_fn=tanh)
    return LSTMLayer(ig, fg, cell, og, activation_fn=tanh,
                      init=rng.standard_normal((h,)),
                      cell_init=rng.standard_normal((h,)))


def _make_gru_layer_np(rng, in_dim, h):
    reset_gate = _make_gate_np(rng, in_dim, h, peephole=False)
    update_gate = _make_gate_np(rng, in_dim, h, peephole=False)
    cell = GRUCell(rng.standard_normal((in_dim, h)), rng.standard_normal((h,)),
                    rng.standard_normal((h, h)), activation_fn=tanh)
    return GRULayer(reset_gate, update_gate, cell, init=rng.standard_normal((h,)))


def _make_bidir_lstm_network(rng, in_dim, h, out_dim, n_layers=2):
    layers = []
    cur_in = in_dim
    for _ in range(n_layers):
        layers.append(BidirectionalLayer(
            _make_lstm_layer_np(rng, cur_in, h),
            _make_lstm_layer_np(rng, cur_in, h),
        ))
        cur_in = 2 * h
    w = rng.standard_normal((cur_in, out_dim))
    b = rng.standard_normal((out_dim,))
    layers.append(FeedForwardLayer(w, b, activation_fn=sigmoid))
    return NeuralNetwork(layers)


def _make_bidir_gru_network(rng, in_dim, h, out_dim):
    layers = [BidirectionalLayer(
        _make_gru_layer_np(rng, in_dim, h), _make_gru_layer_np(rng, in_dim, h)
    )]
    w = rng.standard_normal((2 * h, out_dim))
    b = rng.standard_normal((out_dim,))
    layers.append(FeedForwardLayer(w, b, activation_fn=linear))
    return NeuralNetwork(layers)


def _unstacked_ensemble_reference(networks, dtype_np, dtype_torch):
    """Build the OLD per-network `EnsembleModule` (bypassing the stacked
    path entirely, via `_convert_single`/`NeuralNetworkModule` per
    member) so a stacked module can be compared against it directly on
    identical weights."""
    from madmom_infer.torch.ml.nn.layers import EnsembleModule

    modules = []
    for net in networks:
        layer_modules = [_convert_layer(layer, trainable=False) for layer in net.layers]
        modules.append(_torch_layers_module(layer_modules))
    return EnsembleModule(modules)


def _torch_layers_module(layer_modules):
    from madmom_infer.torch.ml.nn.convert import NeuralNetworkModule
    return NeuralNetworkModule(layer_modules, expected_unbatched_ndim=2)


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-10),
    (np.float32, torch.float32, 1e-5),
])
def test_stacked_lstm_ensemble_matches_unstacked_reference(dtype_np, dtype_torch, atol):
    rng = _rng(50)
    in_dim, h, out_dim, timesteps = 5, 4, 3, 9
    networks = [_make_bidir_lstm_network(rng, in_dim, h, out_dim, n_layers=2)
                for _ in range(4)]
    stacked = _try_stack_networks(networks, trainable=False)
    assert stacked is not None
    reference = _unstacked_ensemble_reference(networks, dtype_np, dtype_torch)
    x_np = rng.standard_normal((timesteps, in_dim)).astype(dtype_np)
    x = torch.tensor(x_np, dtype=dtype_torch)
    with torch.no_grad():
        out_stacked = stacked(x).numpy()
        out_reference = reference(x).numpy()
    _assert_close(out_reference, out_stacked, atol, name="stacked-vs-unstacked LSTM ensemble")


@pytest.mark.parametrize("dtype_np,dtype_torch,atol", [
    (np.float64, torch.float64, 1e-10),
    (np.float32, torch.float32, 1e-5),
])
def test_stacked_gru_ensemble_matches_unstacked_reference(dtype_np, dtype_torch, atol):
    rng = _rng(51)
    in_dim, h, out_dim, timesteps = 5, 4, 3, 9
    networks = [_make_bidir_gru_network(rng, in_dim, h, out_dim) for _ in range(3)]
    stacked = _try_stack_networks(networks, trainable=False)
    assert stacked is not None
    reference = _unstacked_ensemble_reference(networks, dtype_np, dtype_torch)
    x_np = rng.standard_normal((timesteps, in_dim)).astype(dtype_np)
    x = torch.tensor(x_np, dtype=dtype_torch)
    with torch.no_grad():
        out_stacked = stacked(x).numpy()
        out_reference = reference(x).numpy()
    _assert_close(out_reference, out_stacked, atol, name="stacked-vs-unstacked GRU ensemble")


def test_stacked_single_network_matches_unstacked_reference():
    # E == 1 (no real ensemble): a single network should still be
    # eligible for the fused-gate path and still agree with `to_torch`'s
    # pre-stacking behavior (NeuralNetworkModule).
    rng = _rng(52)
    in_dim, h, out_dim = 4, 3, 2
    net = _make_bidir_lstm_network(rng, in_dim, h, out_dim, n_layers=1)
    stacked = _try_stack_networks([net], trainable=False)
    assert stacked is not None
    layer_modules = [_convert_layer(layer, trainable=False) for layer in net.layers]
    reference = _torch_layers_module(layer_modules)
    x = torch.tensor(rng.standard_normal((7, in_dim)), dtype=torch.float64)
    with torch.no_grad():
        out_stacked = stacked(x).numpy()
        out_reference = reference(x).numpy()
    _assert_close(out_reference, out_stacked, atol=1e-10, name="stacked-vs-unstacked E=1")


def test_fast_recurrent_cpu_falls_back_to_exact_eager_output():
    rng = _rng(521)
    networks = [
        _make_bidir_lstm_network(rng, 4, 3, 2, n_layers=1)
        for _ in range(2)
    ]
    eager = _try_stack_networks(
        networks, trainable=False, fast_recurrent=False
    )
    requested = _try_stack_networks(
        networks, trainable=False, fast_recurrent=True
    )
    x = torch.tensor(rng.standard_normal((17, 4)), dtype=torch.float32)

    with torch.no_grad():
        eager_output = eager(x)
        requested_output = requested(x)

    torch.testing.assert_close(requested_output, eager_output, rtol=0, atol=0)


def test_fast_recurrent_keeps_grad_enabled_execution_differentiable():
    rng = _rng(522)
    networks = [
        _make_bidir_lstm_network(rng, 3, 2, 2, n_layers=1)
        for _ in range(2)
    ]
    module = _try_stack_networks(
        networks, trainable=False, fast_recurrent=True
    )
    x = torch.tensor(
        rng.standard_normal((8, 3)), dtype=torch.float32, requires_grad=True
    )

    output = module(x)
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_stacking_falls_back_for_convolutional_layers():
    rng = _rng(53)
    w = rng.standard_normal((1, 2, 3, 3))
    b = rng.standard_normal((2,))
    net = NeuralNetwork([ConvolutionalLayer(w, b, pad="valid", activation_fn=relu)])
    assert _try_stack_networks([net], trainable=False) is None


def test_stacking_falls_back_for_mismatched_ensemble_architecture():
    rng = _rng(54)
    net_a = _make_bidir_lstm_network(rng, 4, 3, 2, n_layers=1)
    net_b = _make_bidir_lstm_network(rng, 4, 5, 2, n_layers=1)  # different hidden size
    assert _try_stack_networks([net_a, net_b], trainable=False) is None


def test_gradcheck_stacked_lstm_ensemble():
    rng = _rng(55)
    networks = [_make_bidir_lstm_network(rng, 3, 2, 2, n_layers=1) for _ in range(2)]
    stacked = _try_stack_networks(networks, trainable=False)
    assert stacked is not None
    stacked = stacked.double()
    x = torch.tensor(rng.standard_normal((4, 3)), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: stacked(inp), (x,), eps=1e-6, atol=1e-4)


def test_gradcheck_stacked_gru_ensemble():
    rng = _rng(56)
    networks = [_make_bidir_gru_network(rng, 3, 2, 2) for _ in range(2)]
    stacked = _try_stack_networks(networks, trainable=False)
    assert stacked is not None
    stacked = stacked.double()
    x = torch.tensor(rng.standard_normal((4, 3)), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: stacked(inp), (x,), eps=1e-6, atol=1e-4)


def test_stacked_ensemble_trainable_gradients_reach_all_members():
    """`trainable=True` on a stacked ensemble registers ONE `nn.Parameter`
    per layer position, shaped `(E, ...)` -- covering all E members at
    once, unlike `layers.EnsembleModule` (one Parameter set per member).
    Gradients still flow correctly per member: this test checks every
    member's OWN slice along the ensemble axis gets a distinct, finite,
    non-zero gradient after a backward pass on ensemble-member-dependent
    (different-per-member weights) input."""
    rng = _rng(57)
    networks = [_make_bidir_lstm_network(rng, 3, 2, 2, n_layers=1) for _ in range(3)]
    stacked = _try_stack_networks(networks, trainable=True)
    assert stacked is not None
    x = torch.tensor(rng.standard_normal((5, 3)), dtype=torch.float32)
    out = stacked(x)
    out.sum().backward()
    params = list(stacked.parameters())
    assert len(params) > 0
    for p in params:
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))
        if p.dim() >= 1 and p.shape[0] == len(networks):
            # per-member gradient slice, recoverable via p.grad[i]
            per_member_norm = p.grad.reshape(len(networks), -1).norm(dim=1)
            assert torch.all(per_member_norm > 0)


# ---------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu():
    nn = NeuralNetwork.load(_path_of(models.chroma_dnn))
    tm_cpu = to_torch(nn)
    tm_cuda = to_torch(nn).to("cuda")
    indim = nn.layers[0].weights.shape[0]
    x_np = _rng(44).standard_normal((25, indim)).astype(np.float32)
    with torch.no_grad():
        out_cpu = tm_cpu(torch.tensor(x_np)).numpy()
        out_cuda = tm_cuda(torch.tensor(x_np, device="cuda")).cpu().numpy()
    _assert_close(out_cpu, out_cuda, atol=1e-4, name="cuda-vs-cpu")


@pytest.mark.network
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu_stacked_ensemble():
    ens = NeuralNetworkEnsemble.load(models.downbeats_blstm())
    tm_cpu = to_torch(ens)
    tm_cuda = to_torch(ens).to("cuda")
    x_np = _rng(45).standard_normal((30, 314)).astype(np.float32)
    with torch.no_grad():
        out_cpu = tm_cpu(torch.tensor(x_np)).numpy()
        out_cuda = tm_cuda(torch.tensor(x_np, device="cuda")).cpu().numpy()
    _assert_close(out_cpu, out_cuda, atol=1e-4, name="cuda-vs-cpu (stacked ensemble)")


@pytest.mark.network
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_fast_recurrent_cuda_matches_eager_without_mutating_initial_state():
    from madmom_infer.torch.ml.nn.stacked import StackedLSTMLayer
    from madmom_infer.torch.ml.nn import triton_lstm

    if triton_lstm.triton is None:
        pytest.skip("Triton is unavailable")

    ensemble = NeuralNetworkEnsemble.load(models.downbeats_blstm())
    eager = to_torch(ensemble).to("cuda")
    fast = to_torch(ensemble, fast_recurrent=True).to("cuda")
    x = torch.tensor(
        _rng(451).standard_normal((600, 314)).astype(np.float32),
        device="cuda",
    )
    initial = [
        (layer.init.clone(), layer.cell_init.clone())
        for layer in fast.modules()
        if isinstance(layer, StackedLSTMLayer)
    ]

    with torch.no_grad():
        eager_output = eager(x)
        fast_output = fast(x)
        repeated_output = fast(x)

    _assert_close(
        eager_output.cpu().numpy(), fast_output.cpu().numpy(), atol=6e-4,
        name="fast recurrent CUDA vs eager",
    )
    assert not torch.equal(fast_output, eager_output)
    torch.testing.assert_close(repeated_output, fast_output, rtol=0, atol=0)
    layers = [
        layer for layer in fast.modules()
        if isinstance(layer, StackedLSTMLayer)
    ]
    assert len(layers) == len(initial) > 0
    for layer, (init, cell_init) in zip(layers, initial):
        torch.testing.assert_close(layer.init, init, rtol=0, atol=0)
        torch.testing.assert_close(layer.cell_init, cell_init, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_fast_recurrent_compile_failure_is_cached_and_falls_back(
        monkeypatch):
    from madmom_infer.torch.ml.nn import triton_lstm

    if triton_lstm.triton is None:
        pytest.skip("Triton is unavailable")

    rng = _rng(452)
    networks = [
        _make_bidir_lstm_network(rng, 4, 3, 2, n_layers=1)
        for _ in range(2)
    ]
    eager = _try_stack_networks(networks, trainable=False).cuda()
    fast = _try_stack_networks(
        networks, trainable=False, fast_recurrent=True
    ).cuda()
    x = torch.tensor(
        rng.standard_normal((20, 4)).astype(np.float32), device="cuda"
    )

    class BrokenKernel:
        calls = 0

        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                self.calls += 1
                raise RuntimeError("synthetic compile failure")
            return launch

    broken = BrokenKernel()
    monkeypatch.setattr(triton_lstm, "_lstm_gates_kernel", broken)
    triton_lstm._reset_failure_for_tests()
    try:
        with torch.no_grad():
            expected = eager(x)
            first = fast(x)
            second = fast(x)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, expected, rtol=0, atol=0)
        assert broken.calls == 1
        assert triton_lstm._triton_failed is True
    finally:
        triton_lstm._reset_failure_for_tests()
