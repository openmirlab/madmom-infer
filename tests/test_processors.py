"""Unit tests for madmom_infer.processors -- Processor/SequentialProcessor
composition semantics. These are pure-Python behavioral tests (no madmom
comparison needed: this module has no numerical output, just call-chaining
control flow), covering: `__call__` forwarding to `process()`, sequential
folding of a chain, nested list-wrapping, `None`-entry pass-through, plain
callables (non-Processor) in a chain, and the MutableSequence protocol
(`len`, indexing, `insert`, `append`, `extend`, `del`).

Reads: madmom_infer/processors.py
"""

import pytest

from madmom_infer.processors import Processor, SequentialProcessor, _process


class AddOne(Processor):
    def process(self, data, **kwargs):
        return data + 1


class MulTwo(Processor):
    def process(self, data, **kwargs):
        return data * 2


class RecordsKwargs(Processor):
    def process(self, data, **kwargs):
        return data, kwargs


def test_processor_process_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        Processor().process(1)


def test_processor_call_forwards_to_process():
    assert AddOne()(5) == 6


def test_sequential_processor_chains_in_order():
    # (5 + 1) * 2 == 12, order matters: MulTwo(AddOne(x)) != AddOne(MulTwo(x))
    chain = SequentialProcessor([AddOne(), MulTwo()])
    assert chain(5) == 12
    reverse = SequentialProcessor([MulTwo(), AddOne()])
    assert reverse(5) == 11


def test_sequential_processor_empty_chain_is_identity():
    assert SequentialProcessor([])(42) == 42


def test_sequential_processor_wraps_nested_lists_as_subchain():
    chain = SequentialProcessor([[AddOne(), AddOne()], MulTwo()])
    assert isinstance(chain.processors[0], SequentialProcessor)
    # (5 + 1 + 1) * 2 == 14
    assert chain(5) == 14


def test_sequential_processor_none_entry_passes_data_through():
    chain = SequentialProcessor([AddOne(), None, MulTwo()])
    assert chain(5) == 12


def test_sequential_processor_accepts_plain_callable():
    chain = SequentialProcessor([lambda x: x + 10, AddOne()])
    assert chain(5) == 16


def test_sequential_processor_forwards_kwargs_to_processor_stages():
    chain = SequentialProcessor([RecordsKwargs()])
    data, kwargs = chain(5, foo="bar")
    assert data == 5
    assert kwargs == {"foo": "bar"}


def test_sequential_processor_mutable_sequence_protocol():
    a, b, c = AddOne(), MulTwo(), AddOne()
    chain = SequentialProcessor([a, b])
    assert len(chain) == 2
    assert chain[0] is a
    assert chain[1] is b

    chain.append(c)
    assert len(chain) == 3
    assert chain[2] is c

    chain.insert(0, MulTwo())
    assert len(chain) == 4

    del chain[0]
    assert len(chain) == 3
    assert chain[0] is a

    chain[0] = c
    assert chain[0] is c

    chain.extend([AddOne(), AddOne()])
    assert len(chain) == 5


def test_process_helper_none_processor_passes_through():
    assert _process((None, 7, {})) == 7


def test_process_helper_plain_function_ignores_kwargs():
    assert _process((lambda x: x * 3, 4, {"unused": 1})) == 12


def test_process_helper_processor_forwards_kwargs():
    assert _process((RecordsKwargs(), 4, {"a": 1})) == (4, {"a": 1})


def test_sequential_processor_is_a_processor_and_callable_uniformly():
    chain = SequentialProcessor([AddOne()])
    assert isinstance(chain, Processor)
    # both call conventions must be equivalent
    assert chain.process(1) == chain(1)
