"""Processor composition core -- port of madmom.processors' inference-time
call semantics: `Processor` (a one-method abstract callable),
`SequentialProcessor` (a mutable-sequence-of-Processors whose own `.process()`
is a plain left-to-right fold), `ParallelProcessor` (fan the same input out to
every processor, return a list -- Phase 2's `NeuralNetworkEnsemble` is a
`ParallelProcessor` of per-model `NeuralNetwork`s), and `BufferProcessor` (a
tiny stateful ring-buffer used by `SpectrogramDifferenceProcessor` to buffer
context frames). Every `audio/*` stage in this package is designed to be an
instance of `Processor` so pipelines compose exactly the way
`all-in-one-infer`'s `SequentialProcessor([frames, stft, filt, spec])` already
expects (`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`).

Deliberately NOT ported (see docs/DESIGN.md A.2 and CLAUDE.md): argparse
plumbing (`Processor.add_arguments`/`io_arguments`), pickle-based
`Processor.load`/`.dump` (madmom uses this to ship pickled trained models
directly via `pickle.load` with no restriction -- Phase 2 instead ships its
own restricted, class-allowlisted unpickler, see `madmom_infer/ml/nn/
unpickle.py`), `ParallelProcessor`'s `multiprocessing.Pool`-based
`num_threads` dispatch (`madmom-upstream/madmom/processors.py:442-455` --
ported as a plain sequential `map()`, since this project's goal is correct
forward-pass results, not multiprocessing throughput; CLAUDE.md's "don't
oversell perf" stance applies here too), `OnlineProcessor`/`OutputProcessor`/
`IOProcessor`/`Stream`-related streaming-mode machinery (this project targets
offline, whole-clip inference only), and the `process_single`/`process_batch`/
`process_online` CLI-batch-file helpers.

Reads: nothing (pure Python, stdlib `collections.abc.MutableSequence` only);
read by: madmom_infer/audio/*.py, madmom_infer/ml/nn/*.py to compose the DSP
and NN-ensemble pipelines.
"""

from collections.abc import MutableSequence

import numpy as np


class Processor:
    """Abstract base class for processing data.

    Port of `madmom.processors.Processor` (`madmom-upstream/madmom/
    processors.py:30-114`), minus the pickle-based `load`/`dump` classmethods
    (`processors.py:36-88`) -- those exist in madmom to un/pickle trained
    models from disk, which has no equivalent in an inference-only,
    no-bundled-weights project.
    """

    def process(self, data, **kwargs):
        """Process the data.

        Must be implemented by the derived class; it should process the
        given data and return the processed output.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def __call__(self, *args, **kwargs):
        # this magic method makes a Processor callable
        # (madmom-upstream/madmom/processors.py:112-114)
        return self.process(*args, **kwargs)


def _process(process_tuple):
    """Apply a single (processor, data, kwargs) tuple.

    Port of `madmom.processors._process` (`processors.py:247-285`), minus the
    multiprocessing-pickling concern the original docstring calls out (that
    concern only matters for `ParallelProcessor`'s pool dispatch, which this
    package does not implement). Kept as a free function -- rather than
    inlined into `SequentialProcessor.process` -- because it also has to
    handle two non-`Processor` cases madmom's own composition relies on:
    a `None` entry (pass the data through unchanged) and a plain callable
    accepting a single `data` argument (no `**kwargs` forwarding for those,
    matching the original).
    """
    processor, data, kwargs = process_tuple
    # do not process the data if the processor is None
    if processor is None:
        return data
    # call the Processor with data and kwargs
    elif isinstance(processor, Processor):
        return processor(data, **kwargs)
    # just call whatever we got here (e.g. a function) without kwargs
    return processor(data)


class SequentialProcessor(MutableSequence, Processor):
    """Processor for sequential processing of data.

    Port of `madmom.processors.SequentialProcessor`
    (`madmom-upstream/madmom/processors.py:288-419`). `processors` is a list
    of `Processor` instances (or plain callables, or nested lists/tuples,
    which get wrapped as a nested `SequentialProcessor`) to be applied in
    order: `SequentialProcessor([a, b, c])(data)` computes `c(b(a(data)))`,
    exactly like `all-in-one-infer`'s `build_spec_processor()`
    (`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`).

    Implements the `MutableSequence` protocol (`__getitem__`, `__setitem__`,
    `__delitem__`, `__len__`, `insert`, plus `append`/`extend`) so a pipeline
    can be edited like a list, matching the original.
    """

    def __init__(self, processors):
        self.processors = []
        # iterate over all given processors and save them
        for processor in processors:
            # wrap lists and tuples as a SequentialProcessor
            if isinstance(processor, (list, tuple)):
                processor = SequentialProcessor(processor)
            self.processors.append(processor)

    def __getitem__(self, index):
        return self.processors[index]

    def __setitem__(self, index, processor):
        self.processors[index] = processor

    def __delitem__(self, index):
        del self.processors[index]

    def __len__(self):
        return len(self.processors)

    def insert(self, index, processor):
        self.processors.insert(index, processor)

    def append(self, other):
        self.processors.append(other)

    def extend(self, other):
        self.processors.extend(other)

    def process(self, data, **kwargs):
        """Process the data sequentially with the defined processing chain.

        `kwargs` are forwarded to every stage's `process()`/`__call__`, same
        as `madmom.processors.SequentialProcessor.process`
        (`processors.py:399-419`).
        """
        for processor in self.processors:
            data = _process((processor, data, kwargs))
        return data


class ParallelProcessor(SequentialProcessor):
    """Processor for parallel processing of data.

    Port of `madmom.processors.ParallelProcessor`
    (`madmom-upstream/madmom/processors.py:423-479`): every processor in
    `processors` is applied to the *same* input `data`, and the results are
    returned as a list (in processor order). `NeuralNetworkEnsemble`
    (`madmom_infer/ml/nn/__init__.py`) is exactly a `ParallelProcessor` over
    per-model `NeuralNetwork`s, and `RNNDownBeatProcessor`
    (`madmom_infer/features/downbeats.py`) uses one to run its three
    frame-size branches (1024/2048/4096) over the same input signal.

    `num_threads` is accepted for constructor-signature parity but always
    runs sequentially (plain `map()`) -- see this module's header for why the
    original's `multiprocessing.Pool` dispatch is out of scope.
    """

    def __init__(self, processors, num_threads=None):
        # pylint: disable=unused-argument
        super().__init__(processors)

    def process(self, data, **kwargs):
        """Process the data with every processor, using the same input.

        Returns a list of each processor's output, same as
        `madmom.processors.ParallelProcessor.process`
        (`processors.py:457-479`), minus the `multiprocessing.Pool.map`
        dispatch (see module header).
        """
        return [_process((p, data, kwargs)) for p in self.processors]


class BufferProcessor(Processor):
    """Buffer for processors which need context to do their processing.

    Port of `madmom.processors.BufferProcessor`
    (`madmom-upstream/madmom/processors.py:717-835`). Used by
    `SpectrogramDifferenceProcessor` (`madmom_infer/audio/spectrogram.py`) to
    hold the trailing `diff_frames` context needed to compute a temporal
    difference across streaming calls. Every Phase-2 call site in this
    project processes a whole clip in one shot with `reset=True` (the
    default), so the *stateful* continuation path (calling `process()`
    multiple times with `reset=False` to stream) is ported faithfully but not
    exercised by this project's own tests -- only the single-call,
    freshly-initialized-buffer path is golden-fixture verified.
    """

    def __init__(self, buffer_size=None, init=None, init_value=0):
        # if init is given, infer buffer_size from it
        if buffer_size is None and init is not None:
            buffer_size = init.shape
        elif isinstance(buffer_size, int):
            buffer_size = (buffer_size,)
        # init buffer if needed
        if buffer_size is not None and init is None:
            init = np.ones(buffer_size) * init_value
        self.buffer_size = buffer_size
        self.init = init
        self.data = init

    @property
    def buffer_length(self):
        """Length of the buffer (time steps)."""
        return self.buffer_size[0]

    def reset(self, init=None):
        """Reset BufferProcessor to its initial state."""
        self.data = init if init is not None else self.init

    def process(self, data, **kwargs):
        """Buffer the data.

        Shifts the buffer by the length of `data` and appends `data` at the
        end, returning the buffer's full (shifted) contents -- port of
        `madmom.processors.BufferProcessor.process` (`processors.py:775-812`).
        If `data`'s length is >= the buffer's length, the buffer is replaced
        outright by `data`'s last `buffer_length` items.
        """
        # expected minimum number of dimensions
        ndmin = len(self.buffer_size)
        if data.ndim < ndmin:
            data = np.array(data, ndmin=ndmin)
        data_length = len(data)
        if data_length >= self.buffer_length:
            self.data = data[-self.buffer_length:]
        else:
            self.data = np.roll(self.data, -data_length, axis=0)
            self.data[-data_length:] = data
        return self.data

    # alias for easier / more intuitive calling
    buffer = process

    def __getitem__(self, index):
        """Direct access to the buffer data (any numpy indexing method)."""
        return self.data[index]
