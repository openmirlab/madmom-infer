"""`TorchPipelineProcessor`: a `madmom_infer.processors.Processor` that
wraps one of `pipelines.py`'s torch `nn.Module`s so it can be dropped in
wherever a numpy processor is expected (a path, an already-loaded
`madmom_infer.audio.signal.Signal`, or a raw ndarray in, a numpy array
out) -- this is the numpy-facing boundary of the torch backend; wiring
`backend="torch"` into the numpy processors themselves is a later step,
not this one.

**Int-PCM scaling convention (load-bearing, read before editing).** The
numpy backend never rescales an integer-PCM `Signal` -- it keeps int16
samples unrescaled all the way through framing and instead divides the
*FFT window* by `np.iinfo(dtype).max` (`madmom_infer/audio/stft.py`'s
module header, "trap 1"). The torch frontend takes the opposite, but
mathematically equivalent, path: it assumes an already-float waveform and
applies no such special case (`torch/audio/frontend.py`'s module
docstring). Multiplication is associative/commutative here
(`frame * (window / max) == (frame / max) * window`), so this adapter
reproduces the numpy result exactly by doing the rescaling on the
*signal* instead, once, right here: an integer-dtype `Signal` is divided
by `np.iinfo(dtype).max` before conversion to a torch tensor; a
float-dtype `Signal` is used as-is (matching the numpy "non-integer
dtype: no scaling needed" branch).

**TF32 finding (measured, not a bug in this module).** On Ampere+ GPUs,
torch's default `torch.backends.cudnn.allow_tf32=True` /
`torch.backends.cuda.matmul.allow_tf32=True` makes the CNN-heavy pipelines
(`OnsetCNNPipeline`, `KeyPipeline`, `NoteCNNPipeline`) drift up to ~1e-3
from the numpy reference on CUDA -- that drops to ~1e-6 with both flags set
`False`. This module deliberately does NOT mutate those global torch flags
itself (a library importing a side-effecting global setting on every
process that imports it would be its own surprise); decoded results
(beats/onsets/notes/key labels) were still identical either way in every
case measured. Set the flags yourself before running on CUDA if you need
the tighter tolerance (`tools/compare_torch_backend.py --allow-tf32`
controls this for that tool specifically, default off).

Reads: torch, numpy, madmom_infer.backends (validate_torch_device),
madmom_infer.processors (Processor),
madmom_infer.audio.signal (SignalProcessor, Signal); read by:
madmom_infer/torch/features/__init__.py, madmom_infer/backends.py,
tests/test_torch_pipelines.py, tests/test_torch_backend.py,
tools/compare_torch_backend.py.
"""

from __future__ import annotations

import numpy as np
import torch

from madmom_infer.audio.signal import Signal, SignalProcessor
from madmom_infer.backends import validate_torch_device
from madmom_infer.processors import Processor


def _module_device(module):
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def waveform_from_signal(signal, dtype=np.float64):
    """Convert a `Signal` (or anything `Signal(...)` accepts) into a mono
    float waveform array, applying the int-PCM scaling convention
    documented in this module's header."""
    if not isinstance(signal, Signal):
        signal = Signal(signal, num_channels=1, sample_rate=44100)
    array = np.asarray(signal.data)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(dtype) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(dtype)
    return array


class TorchPipelineProcessor(Processor):
    """Adapt a `pipelines.py` torch `nn.Module` into a numpy-in/numpy-out
    `Processor`. `module` should already be constructed (e.g. via
    `madmom_infer.torch.features.build_pipeline`); this class only handles
    I/O conversion and device placement, never model construction.

    `device=None` (default) runs on whatever device `module`'s own
    parameters/buffers already live on (call `module.to(device)` first to
    choose); `dtype` (default `torch.float32`) is the tensor precision fed
    into the module -- must match the dtype `module`'s own buffers were
    built with.
    """

    def __init__(self, module, device=None, dtype=torch.float32):
        validate_torch_device(device)
        self.module = module.eval()
        self.device = torch.device(device) if device is not None else _module_device(module)
        self.dtype = dtype
        self._signal_processor = SignalProcessor(num_channels=1, sample_rate=44100)

    def process(self, data, **kwargs):
        """Run `data` (a path, `Signal`, or ndarray) through the wrapped
        torch pipeline and return a numpy array, matching the dtype
        (`float32`) the numpy processor it mirrors returns."""
        signal = self._signal_processor(data)
        waveform = waveform_from_signal(signal, dtype=np.float64)
        tensor = torch.as_tensor(waveform, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.module(tensor)
        return output.detach().to("cpu").numpy().astype(np.float32)
