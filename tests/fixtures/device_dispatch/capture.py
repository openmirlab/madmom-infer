"""Capture untouched madmom-infer CPU dispatch outputs, not upstream goldens."""

import contextlib
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.request

REPO = Path('/home/worzpro/Desktop/dev/openmirlab/madmom-infer')
OUT = Path(__file__).resolve().parent
SOURCE_SHA = '269e948ab2475031d8c9bfb10eb1ec1ac5a6961f'
INPUT = Path('/home/worzpro/Desktop/dev/openmirlab/references/madmom-upstream/tests/data/audio/sample.wav')
INPUT_SHA = 'd06b3c39384c4a0cf59f0bd326eda2443edf668fbb90647e37c801e59602866c'
ENV = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
for name, value in ENV.items():
    assert os.environ.get(name) == value, (name, os.environ.get(name))
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def source_state():
    return {'head': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
            'production_python_sha256': {
                str(p.relative_to(REPO)): sha(p)
                for p in sorted((REPO / 'madmom_infer').rglob('*.py'))},
            'checkpoint_config_sha256': sha(REPO / 'madmom_infer/config/checkpoints.toml')}


before = source_state()
assert before['head'] == SOURCE_SHA and not before['status'], before
assert sha(INPUT) == INPUT_SHA
download_attempts = []


def no_download(*args, **kwargs):
    download_attempts.append(repr(args))
    raise AssertionError('Network acquisition forbidden during baseline capture')


urllib.request.urlopen = no_download
urllib.request.urlretrieve = no_download

import numpy as np
import scipy
from scipy.io import wavfile
import torch
from madmom_infer.audio.signal import Signal
from madmom_infer.features.beats import RNNBeatProcessor
from madmom_infer.features.downbeats import RNNBarProcessor
from madmom_infer.torch.features import TorchPipelineProcessor, build_pipeline
from madmom_infer import models

torch.set_num_threads(4)
assert not torch.cuda.is_available()
assert models._cache_root() == Path.home() / '.cache/madmom_infer/models'
config_path = REPO / 'madmom_infer/config/checkpoints.toml'
config = tomllib.loads(config_path.read_text())
checkpoints = []
for model_name, model in config['models'].items():
    for item in model['files']:
        path = models._cache_root() / item['path']
        actual = sha(path)
        assert actual == item['sha256'], path
        checkpoints.append({'model': model_name, 'path': str(path),
                            'sha256': actual, 'expected_sha256': item['sha256'],
                            'size_bytes': path.stat().st_size})
assert len(checkpoints) == 62, len(checkpoints)

sample_rate, pcm = wavfile.read(INPUT)
assert sample_rate == 44100 and pcm.shape == (123481,) and pcm.dtype == np.int16
music = pcm.astype(np.float32) / np.float32(32768)
silence = np.zeros_like(music)
assert np.count_nonzero(silence) == 0
beats = np.arange(.25, len(music) / sample_rate, .5)


def stats(array):
    finite = array[np.isfinite(array)].astype(np.float64)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256_raw_c_order': hashlib.sha256(array.tobytes(order='C')).hexdigest(),
            'nan_count': int(np.isnan(array).sum()),
            'inf_count': int(np.isinf(array).sum()),
            'rms': float(np.sqrt(np.mean(finite ** 2))),
            'peak': float(np.max(np.abs(finite))),
            'min': float(np.min(finite)), 'max': float(np.max(finite)),
            'std': float(np.std(finite))}


def capture():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    processors = {
        'common_beats': RNNBeatProcessor(backend='torch', device='cpu'),
        'bar': RNNBarProcessor(backend='torch', device='cpu'),
        'direct_key': TorchPipelineProcessor(build_pipeline('key').to('cpu'), device='cpu'),
    }
    arrays, summaries = {}, {}
    for path, processor in processors.items():
        for input_name, audio in [('music', music), ('silence', silence)]:
            start = time.monotonic()
            signal = Signal(audio.copy(), sample_rate=sample_rate)
            result = np.asarray(processor((signal, beats.copy())) if path == 'bar' else processor(signal)).copy()
            elapsed = time.monotonic() - start
            if path == 'bar':
                assert result.shape == (len(beats), 2)
                assert np.array_equal(result[:, 0], beats)
                assert np.isnan(result).sum() == 1 and np.isnan(result[-1, 1])
                activation = result[:-1, 1]
                assert np.isfinite(activation).all()
            else:
                assert result.size and np.isfinite(result).all()
                activation = result
            summary = stats(result)
            summary['activation_stats'] = stats(activation)
            summary['elapsed_seconds'] = elapsed
            if path == 'bar':
                summary['rms'] = summary['activation_stats']['rms']
                summary['peak'] = summary['activation_stats']['peak']
                summary['rms_peak_scope'] = 'result[:-1, 1]; timestamps and final sentinel NaN excluded'
            else:
                summary['rms_peak_scope'] = 'complete result'
            if input_name == 'music':
                assert summary['activation_stats']['rms'] > 1e-4, summary
                assert summary['activation_stats']['std'] > 1e-5, summary
            name = path + '__' + input_name
            arrays[name] = result
            summaries[name] = summary
            print(name, json.dumps(summary), flush=True)
    return arrays, summaries


first, first_summaries = capture()
second, second_summaries = capture()
for name in first:
    assert np.array_equal(first[name], second[name], equal_nan=True), name
    assert first[name].dtype == second[name].dtype
assert not download_attempts
after = source_state()
assert before == after, 'Repository changed during baseline capture'
for checkpoint in checkpoints:
    assert sha(checkpoint['path']) == checkpoint['sha256']
assert sha(INPUT) == INPUT_SHA
np.savez_compressed(OUT / 'outputs.npz', **first)
np.savez_compressed(OUT / 'repeat_outputs.npz', **second)
np.savez_compressed(OUT / 'inputs.npz', music=music, silence=silence, beats=beats)
numpy_config = io.StringIO()
with contextlib.redirect_stdout(numpy_config):
    np.show_config()
command = ' '.join([*(k + '=' + shlex.quote(v) for k, v in ENV.items()),
                    shlex.quote(sys.executable), shlex.quote(str(Path(__file__).resolve()))])
metadata = {
    'kind': 'Current-port CPU device-dispatch baseline; NOT an upstream golden fixture',
    'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command': command, 'cwd': str(Path.cwd()),
    'source_before': before, 'source_after': after,
    'script_sha256': sha(__file__), 'input_wav': str(INPUT), 'input_wav_sha256': INPUT_SHA,
    'input_normalization': 'scipy.io.wavfile int16 -> float32 / 32768; silence zeros_like(music)',
    'sample_rate': sample_rate, 'inputs': {k: stats(v) for k, v in {'music': music, 'silence': silence, 'beats': beats}.items()},
    'checkpoints': checkpoints, 'checkpoint_count': len(checkpoints),
    'runtime': {'python': sys.version, 'executable': sys.executable,
                'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'madmom_infer': importlib.metadata.version('madmom-infer'),
                'torch_config': torch.__config__.show(), 'numpy_config': numpy_config.getvalue(),
                'torch_threads': torch.get_num_threads(),
                'torch_interop_threads': torch.get_num_interop_threads(),
                'cpu': subprocess.check_output(['lscpu'], text=True),
                'platform': platform.platform(), 'environment': ENV,
                'cuda_available': torch.cuda.is_available(), 'seed': 1234},
    'network': {'urllib_urlopen_and_urlretrieve_blocked': True, 'download_attempts': download_attempts},
    'first': first_summaries, 'second': second_summaries,
    'repeat_array_equal_equal_nan': {name: True for name in first},
    'artifact_sha256': {name: sha(OUT / name) for name in ('outputs.npz', 'repeat_outputs.npz', 'inputs.npz')},
}
(OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('SUCCESS: all six outputs repeat exactly; source and all 62 checkpoints unchanged', flush=True)
