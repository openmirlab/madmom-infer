"""Reimplementation of madmom.processors -- the `Processor` base class (a
callable with a uniform `process(data, **kwargs)` interface).

Minimal implementation added out of a hard dependency: workstream C
(ml/hmm.py, features/beats_hmm.py, features/downbeats.py) needs
`DBNDownBeatTrackingProcessor(Processor)` to be importable and callable, and
this file was still a Phase-1 stub. Only `Processor` (the abstract callable
base, `processors.py:30-114` in the original) is ported here -- `load()`/
`dump()` (pickling helpers) and `SequentialProcessor`/`ParallelProcessor`
(pipeline composition for the audio/* DSP chain, `processors.py:288-410`) are
task #2 of docs/DESIGN.md's phase-1 breakdown and are intentionally left for
that workstream to complete; this file should be reconciled, not silently
overwritten, if that workstream also touches it.

Reads: nothing beyond stdlib; used by madmom_infer/features/downbeats.py
(DBNDownBeatTrackingProcessor) and, eventually, madmom_infer/audio/*.py to
compose the DSP pipeline
"""


class Processor(object):
    """
    Abstract base class for processing data.

    """

    def process(self, data, **kwargs):
        """
        Process the data.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def __call__(self, *args, **kwargs):
        # this magic method makes a Processor callable
        return self.process(*args, **kwargs)
