"""Processor composition core -- port of madmom.processors' inference-time
call semantics: `Processor` (a one-method abstract callable) and
`SequentialProcessor` (a mutable-sequence-of-Processors whose own `.process()`
is a plain left-to-right fold). Every `audio/*` stage in this package (and,
eventually, `ml`/`features`) is designed to be an instance of `Processor` so
pipelines compose exactly the way `all-in-one-infer`'s
`SequentialProcessor([frames, stft, filt, spec])` already expects
(`all-in-one-fix/src/allin1_infer/spectrogram.py:27-40`).

Deliberately NOT ported (phase-1 scope, see docs/DESIGN.md A.2 and CLAUDE.md):
argparse plumbing (`Processor.add_arguments`/`io_arguments`), pickle-based
`Processor.load`/`.dump` (madmom uses this to ship pickled trained models --
out of scope, this project never bundles pretrained weights), the
multiprocessing `ParallelProcessor`/`_process`'s pool-dispatch branch,
`OnlineProcessor`/`OutputProcessor`/`IOProcessor`/`BufferProcessor`/`Stream`
-related online-mode machinery, and the `process_single`/`process_batch`/
`process_online` CLI-batch-file helpers. None of these are exercised by the
in-memory, offline `Signal -> FramedSignal -> STFT -> filterbank -> log`
inference call path this project targets.

Reads: nothing (pure Python, stdlib `collections.abc.MutableSequence` only);
read by: madmom_infer/audio/*.py (planned) to compose the DSP pipeline.
"""

from collections.abc import MutableSequence


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
