# Probe — does decode dtype change the detected beat grid?

> Feeds back into the pending decision: how to fix `_load_wave_file`'s hard
> rejection of 24-bit PCM WAV. Verdict below is measured, not argued.

## The fork

`madmom_infer/audio/signal.py:461` calls `scipy.io.wavfile.read(filename, mmap=True)`.
scipy raises `ValueError: mmap=True not compatible with 3-byte container size` for any
24-bit PCM WAV, so 24-bit input — an ordinary studio master export — cannot be read at all.

Candidates: (a) `mmap=False`, (b) switch to soundfile, (c) restore madmom's ffmpeg fallback.

## The load-bearing assumption

`_load_wave_file`'s docstring justifies `mmap=True` as deliberate: it "keeps PCM `int16`
data `int16` (no float rescale)... so the dtype flows straight through to `FramedSignal`
frames." The decision rests entirely on the implied consequence: **that decoded sample
dtype/scale materially changes the detected beat grid.** If true, any decoder change risks
silently shifting beat times. If false, the justification does not bind.

## Method

One decode of a real 24-bit master (`04_奉獻_71bpm_Fmaj V.wav`, 48kHz stereo, first 60s)
via `scipy.io.wavfile.read(mmap=False)` → int32. Conditions derived from that single array,
so audio content is bit-identical and only dtype/scale varies:

| Condition | dtype | Represents |
|---|---|---|
| A | int32 (native) | what `mmap=False` yields |
| C | float64 (`/2^31`) | what soundfile yields |
| D | float32 (`/2^31`) | a float32 decoder |

Each fed to `MadmomAnalyzer(tasks=('beats',))` at the file's native rate. CPU, deterministic.

**Verdict rule, pre-registered before the first trial:** if A, C and D agree (same BPM, all
beat times within 10 ms — one madmom frame at 100 fps), dtype is not load-bearing and
`mmap=False` suffices. If they diverge, the docstring binds and the decoder must preserve
native dtype.

## Data

    A_int32 vs C_float64_scaled : identical=True  max_delta=0.000000 ms
    A_int32 vs D_float32_scaled : identical=True  max_delta=0.000000 ms

Bit-identical, all 72 beats, both comparisons. madmom normalises internally; the dtype the
signal arrives in does not reach the result.

## Verdict

**The load-bearing assumption is false.** The docstring's description is accurate — the dtype
does flow through — but the consequence it implies does not exist. `mmap=False` is safe and
sufficient. Switching decoders is unnecessary: the contract declares `audio/wav` only, so
soundfile's broader format support buys nothing here, and it is now measured to buy no
correctness either.

## Incidental finding — the current workaround is the lossy step

A fourth condition (`>>16` → int16) was included to model 16-bit input. It is **not** a valid
dtype test — requantisation changes audio content, so two axes move — and it is reported here
only as context. It nevertheless measured something worth keeping:

    A_int32 vs B_int16_requantised: 4/72 beats moved, max=50.000 ms
       beat[5]:  24-bit=4.2400s   16-bit=4.2500s   delta=10.0 ms
       beat[6]:  24-bit=5.0600s   16-bit=5.0700s   delta=10.0 ms
       beat[12]: 24-bit=10.1300s  16-bit=10.1800s  delta=50.0 ms
       beat[48]: 24-bit=40.5700s  16-bit=40.5800s  delta=10.0 ms

Converting to 16-bit — today the only way to get any result from a 24-bit master — moves 4 of
72 beats by up to 50 ms. At this track's 71 BPM (845 ms per beat) that is ~6% of a beat, well
past a rounding artefact for downstream beat-aligned work. So `mmap=False` removes both the
false rejection and a silent precision cost nobody was told they were paying.

## Design defect in this probe, recorded

The first run's condition set compared A against the `>>16` int16 variant and treated the
result as a dtype signal. That was wrong: it varied dtype *and* content simultaneously, and
its 50 ms delta was misattributable. The follow-up run isolated the pure dtype axis (A vs C
vs D) and is what the verdict rests on. The mis-specified condition was retained as context,
not quietly dropped.

## Hands back

- **madmom-infer:** change `mmap=True` → `mmap=False` at `signal.py:461`; the docstring's
  stated rationale should be corrected, not preserved.
- **Phonon (separate, not fixed by the above):**
  `providers/madmom-infer/src/phonon_provider_madmom/app.py:382` catches bare `ValueError`
  and reports it as `invalid_audio` — "The audio could not produce a valid result for this
  instrument." That statement was false here: the audio was valid and one parameter away from
  a result. At least ten internal `raise ValueError` sites (e.g. "onset output must not be
  empty") funnel into the same false claim about the caller's data.
