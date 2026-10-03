# Original CPU device-dispatch baseline

Captured **before device repairs**, from untouched madmom-infer commit
`269e948ab2475031d8c9bfb10eb1ec1ac5a6961f`. These are current-port regression
outputs, **not upstream madmom goldens**. No weights are included.

`inputs.npz` contains the existing upstream test sample as normalized float32
audio, equal-length exact silence, and supplied beat timestamps. `outputs.npz`
contains six outputs: common beat dispatch, RNNBar, and direct key adapter,
each for music and silence. RNNBar intentionally ends with one NaN activation;
its RMS/peak measurements exclude that sentinel and the timestamp column.
The nonzero silence predictions are the original model's actual predictions.

`capture.py` is the exact original recording script, retained for provenance;
its original absolute paths and clean-source assertions are intentional.
`metadata.json` records command, input/checkpoint/output hashes, 50 production
Python hashes before/after, CPU/build/thread details, original script hash,
and two captures that matched with `array_equal(equal_nan=True)`.

Portable replay from the repository root, using the recording environment
(Python 3.11, Torch 2.13.0+cu130, NumPy 2.4.6, SciPy 1.17.1, i5-13600K):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python tools/verify_device_baseline.py --device cpu
```

The replay verifies cached checkpoint checksums and blocks downloads. The
offline pytest regression explicitly skips if the recording build/CPU or
checkpoints are unavailable. Use the existing Torch-vs-NumPy tolerance tests
on other environments. Do not regenerate these outputs from repaired code.
