"""Pretrained-weights acquisition layer -- runtime download of madmom's OWN
`.pkl` model files from the official upstream `CPJKU/madmom_models` GitHub
repository (the `madmom/models` git submodule upstream, never vendored into
this project's own git history), with an XDG-respecting local cache and
sha256 verification.

***********************************************************************
* LICENSE NOTICE, READ BEFORE CALLING ANY FUNCTION IN THIS MODULE:     *
* The `.pkl` files this module downloads are NOT covered by            *
* madmom-infer's BSD-2-Clause license. They are the original madmom    *
* authors' own trained model weights, separately licensed              *
* **CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)** --        *
* see https://creativecommons.org/licenses/by-nc-sa/4.0/ and the       *
* upstream `madmom/models/LICENSE` file. In particular: **NON-         *
* COMMERCIAL USE ONLY** for the weights themselves. madmom-infer's own *
* source code (this file included) remains BSD-2-Clause -- only the    *
* downloaded `.pkl` byte content carries the CC BY-NC-SA restriction.  *
* This project NEVER bundles/vendors these files (see README.md,       *
* NOTICE, CLAUDE.md) -- they are fetched at runtime, on demand, only   *
* when a caller explicitly asks for a specific model (e.g. via         *
* `RNNDownBeatProcessor()`), and cached locally for reuse.             *
***********************************************************************

**Where the file list/hashes came from**: `madmom.models.__init__`
(`madmom-upstream/madmom/models/__init__.py`, itself living inside the
`madmom/models` submodule -- CPJKU/madmom_models.git, branch `master`)
builds `DOWNBEATS_BLSTM` by globbing
`downbeats/2016/downbeats_blstm_[1-8].pkl` under the submodule's local
checkout. Since this project never checks out that submodule, the exact
relative paths were read directly from that file, and each file's identity
was pinned by downloading it from
`https://raw.githubusercontent.com/CPJKU/madmom_models/master/<relpath>`
and recording its sha256 -- cross-checked byte-for-byte (`sha256sum`) against
the copy already vendored inside a real, pip-installed madmom 0.17.dev0
wheel (`all-in-one-fix/.venv`), confirming the raw-GitHub copy and the
PyPI-wheel-vendored copy are identical.

4a addition: `KEY_CNN` (`key/2018/key_cnn.pkl`, `CNNKeyRecognitionProcessor`'s
single-network model, `madmom_infer/features/key.py`). Its sha256 was
computed directly from the file already present locally at
`../madmom-upstream/madmom/models/key/2018/key_cnn.pkl` (populated by Wave
4.0's `madmom/models` submodule checkout) and cross-checked byte-for-byte
against a fresh download from
`https://raw.githubusercontent.com/CPJKU/madmom_models/master/key/2018/key_cnn.pkl`
(both hash to `c58ba553be1048877662a663a2670c0051b3c2c66d109b6042ba722ed0bfc7a6`
-- confirmed 2026-07-12, network was available). `../madmom-upstream/madmom/
models/__init__.py`'s `KEY_CNN = models('key/2018/key_cnn.pkl')` resolves to
a single-element list (no glob wildcard -- unlike `DOWNBEATS_BLSTM`'s
8-file ensemble, `key/2017/*` exists in the submodule checkout but is not
`package_data`-shipped by a real madmom install and has no effect here, see
CLAUDE.md's 4.0 audit corrections), so `key_cnn()` below returns a
single-path list, matching `NeuralNetworkEnsemble.load()`'s expected input
shape for an ensemble of size 1.

4b addition: `ONSETS_RNN` (`onsets/2013/onsets_rnn_[1-8].pkl`, 8-network
unidirectional-RNN ensemble, `RNNOnsetProcessor(online=True)`),
`ONSETS_BRNN` (`onsets/2013/onsets_brnn_[1-8].pkl`, 8-network bidirectional
ensemble, `RNNOnsetProcessor()`'s default), and `ONSETS_CNN`
(`onsets/2013/onsets_cnn.pkl`, single-network, `CNNOnsetProcessor`) -- all
three model families `madmom_infer/features/onsets.py`'s processors load.
(`ONSETS_BRNN_PP`, `onsets/2014/onsets_brnn_pp_[1-8].pkl`, is a real
`package_data`-shipped model family too, but is only ever loaded by
upstream's `bin/SuperFluxNN` CLI script -- out of scope, `bin/` is a
Permanent exclusion, see CLAUDE.md -- no processor this project ports
references it, so it has no registry entry here.) All 17 sha256s were
computed directly from the files already present locally at
`../madmom-upstream/madmom/models/onsets/{2013,2014}/*.pkl` (Wave 4.0's
submodule checkout) and cross-checked byte-for-byte against fresh downloads
from `https://raw.githubusercontent.com/CPJKU/madmom_models/master/onsets/
...` for all 17 files -- identical, confirmed 2026-07-12, network was
available (slow/flaky in this sandbox, needed `curl --retry`, but all 17
downloads eventually succeeded and matched).

4c addition: `BEATS_LSTM` (`beats/2016/beats_lstm_[1-8].pkl`, 8-network
unidirectional-LSTM ensemble, `RNNBeatProcessor(online=True)`), `BEATS_BLSTM`
(`beats/2015/beats_blstm_[1-8].pkl`, 8-network bidirectional ensemble,
`RNNBeatProcessor()`'s default), and `DOWNBEATS_BGRU` (a 2-element list of
6-file lists -- `[rhythmic, harmonic]`, matching upstream's own
`models('downbeats/2016/downbeats_bgru_rhythmic_*.pkl')`/`..._harmonic_*.pkl`
order exactly -- `madmom_infer/features/downbeats.py`'s `RNNBarProcessor`).
All 28 sha256s (8+8+12) were computed directly from the files already
present locally at `../madmom-upstream/madmom/models/{beats,downbeats}/...`
(Wave 4.0's submodule checkout) and cross-checked byte-for-byte against
fresh downloads from `https://raw.githubusercontent.com/CPJKU/madmom_models/
master/...` for all 28 files -- identical, confirmed 2026-07-12, network was
available (parallel `curl --retry`, same technique 4b established).

4d addition: `CHROMA_DNN` (`chroma/2016/chroma_dnn.pkl`, single-network,
`madmom_infer/audio/chroma.py`'s `DeepChromaProcessor`), `CHORDS_DCCRF`
(`chords/2016/chords_dccrf.pkl`, single-network CRF,
`DeepChromaChordRecognitionProcessor`), `CHORDS_CNN_FEAT`
(`chords/2016/chords_cnnfeat.pkl`, single-network CNN,
`CNNChordFeatureProcessor`), and `CHORDS_CFCRF` (`chords/2016/
chords_cnncrf.pkl`, single-network CRF, `CRFChordRecognitionProcessor`) --
all `madmom_infer/features/chords.py`'s end-to-end targets. All 4 sha256s
were computed directly from the files already present locally at
`../madmom-upstream/madmom/models/{chroma,chords}/2016/*.pkl` (Wave 4.0's
submodule checkout) and cross-checked byte-for-byte against fresh downloads
from `https://raw.githubusercontent.com/CPJKU/madmom_models/master/...` for
all 4 files -- identical, confirmed 2026-07-13, network was available.
Note the upstream naming oddity, preserved here rather than "fixed": madmom's
own `models/__init__.py` names the `chords_cnncrf.pkl`-backed constant
`CHORDS_CFCRF` (not `CHORDS_CNNCRF`) -- `CF` stands for "CNN Feature" (the
CRF that decodes `CNNChordFeatureProcessor`'s output), not a typo for the
filename.

Every other model family madmom ships (`NOTES_CNN`, ...) would follow the
exact same `_ModelFile`/`download()` pattern -- adding one is a matter of
listing its relative paths + sha256s, not new machinery -- but is out of
scope until a processor that needs it is ported (see CLAUDE.md's wave plan).

Reads: urllib.request (stdlib, HTTPS GET), hashlib (stdlib, sha256), os/
pathlib (stdlib, XDG cache resolution); read by:
madmom_infer/features/downbeats.py (RNNDownBeatProcessor.__init__,
RNNBarProcessor.__init__), madmom_infer/features/key.py
(CNNKeyRecognitionProcessor.__init__), madmom_infer/features/onsets.py
(RNNOnsetProcessor.__init__, CNNOnsetProcessor.__init__),
madmom_infer/features/beats.py (RNNBeatProcessor.__init__),
madmom_infer/audio/chroma.py (DeepChromaProcessor.__init__),
madmom_infer/features/chords.py (DeepChromaChordRecognitionProcessor.
__init__, CNNChordFeatureProcessor.__init__, CRFChordRecognitionProcessor.
__init__).
"""

import hashlib
import os
import urllib.request
from pathlib import Path

MODELS_REPO_RAW_BASE = "https://raw.githubusercontent.com/CPJKU/madmom_models/master"


def _cache_root() -> Path:
    """Local cache directory root, respecting `XDG_CACHE_HOME`.

    `$XDG_CACHE_HOME/madmom_infer/models/` if set, else
    `~/.cache/madmom_infer/models/` -- never inside this project's own
    source tree (see `.gitignore`: `madmom_infer_models_cache/`, the default
    dev-time override some tests use, is also git-ignored as a belt-and-
    braces measure even though it is never the real default location).
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return base / "madmom_infer" / "models"


class _ModelFile:
    """One remote model file: its repo-relative path and known-good sha256."""

    __slots__ = ("relpath", "sha256")

    def __init__(self, relpath: str, sha256: str):
        self.relpath = relpath
        self.sha256 = sha256


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(model_file: "_ModelFile", cache_root: Path = None,
             force: bool = False) -> Path:
    """Download (if not already cached) one model file, verify its sha256,
    and return the local cache path.

    Parameters
    ----------
    model_file : _ModelFile
        Repo-relative path + known-good sha256 to fetch/verify.
    cache_root : Path, optional
        Override the cache root (default: `_cache_root()`, XDG-respecting).
    force : bool, optional
        Re-download even if a correctly-hashing cached copy already exists.

    Raises
    ------
    RuntimeError
        If the downloaded (or a pre-existing cached) file's sha256 does not
        match the recorded known-good value -- this is treated as loudly
        as a corrupted/tampered download, never silently accepted.
    """
    cache_root = cache_root or _cache_root()
    local_path = cache_root / model_file.relpath

    if local_path.exists() and not force:
        digest = _sha256_of(local_path)
        if digest == model_file.sha256:
            return local_path
        # stale/corrupt cache entry -- fall through and re-download
        local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MODELS_REPO_RAW_BASE}/{model_file.relpath}"
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=30) as resp, open(tmp_path, "wb") as fh:
        fh.write(resp.read())

    digest = _sha256_of(tmp_path)
    if digest != model_file.sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "downloaded model file %r sha256 mismatch: expected %s, got %s "
            "-- refusing to use it (corrupted download, or upstream "
            "CPJKU/madmom_models changed the file at this path/ref)."
            % (model_file.relpath, model_file.sha256, digest)
        )
    tmp_path.replace(local_path)
    return local_path


# ---------------------------------------------------------------------------
# DOWNBEATS_BLSTM: madmom's 8-network joint beat/downbeat BLSTM ensemble,
# used by RNNDownBeatProcessor -- the Phase-2 end-to-end target.
# sha256s verified against BOTH a fresh raw-GitHub download AND the copy
# vendored inside a real pip-installed madmom 0.17.dev0 wheel (identical).
# ---------------------------------------------------------------------------
_DOWNBEATS_BLSTM_FILES = [
    _ModelFile("downbeats/2016/downbeats_blstm_1.pkl",
               "b02c4b8aa963069449422fd5cc55bd4077105fe268b0d83036b7d68b5520f38d"),
    _ModelFile("downbeats/2016/downbeats_blstm_2.pkl",
               "6fa9d138ed84bed14ab5ef6f81f3a7e1a125ff8c571a255f6ce91ccf93390eee"),
    _ModelFile("downbeats/2016/downbeats_blstm_3.pkl",
               "adf6a3e3bf345a9172e8252a511dae28622afe26ee653354c0d4c9e37de8f89a"),
    _ModelFile("downbeats/2016/downbeats_blstm_4.pkl",
               "a4f9a09b1588ca74b0157251364650d3fe81e383019b5986941415e08b37fcde"),
    _ModelFile("downbeats/2016/downbeats_blstm_5.pkl",
               "e40aea9b8221babaf14a9211a48dccc3ebcecdf8bcd6d62d60f0949445de9bb6"),
    _ModelFile("downbeats/2016/downbeats_blstm_6.pkl",
               "89b6854c5d6b2381687731aedc9bce6c84894021d1c2eeb60d161cefe997d68c"),
    _ModelFile("downbeats/2016/downbeats_blstm_7.pkl",
               "371a319a66d32bee6ab8d4a15f6ae9e4ee22ff97b4276736bab93d9a994c06c1"),
    _ModelFile("downbeats/2016/downbeats_blstm_8.pkl",
               "a0d20e1922872a3e5b74df451d8b5701de6ca88d9f5739f313fb2991d7a84c77"),
]


def downbeats_blstm(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 8
    `downbeats_blstm_[1-8].pkl` files -- madmom's `DOWNBEATS_BLSTM` model
    list (`madmom-upstream/madmom/models/__init__.py`'s
    `models('downbeats/2016/downbeats_blstm_[1-8].pkl')`).

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _DOWNBEATS_BLSTM_FILES]


# ---------------------------------------------------------------------------
# KEY_CNN: madmom's single-network CNN key-recognition model, used by
# CNNKeyRecognitionProcessor -- the 4a end-to-end target.
# sha256 verified against BOTH a fresh raw-GitHub download AND the copy
# already checked out locally under ../madmom-upstream/madmom/models
# (identical, see this module's header).
# ---------------------------------------------------------------------------
_KEY_CNN_FILES = [
    _ModelFile("key/2018/key_cnn.pkl",
               "c58ba553be1048877662a663a2670c0051b3c2c66d109b6042ba722ed0bfc7a6"),
]


def key_cnn(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return the local path to `key_cnn.pkl` (as a
    single-element list) -- madmom's `KEY_CNN` model list
    (`madmom-upstream/madmom/models/__init__.py`'s
    `models('key/2018/key_cnn.pkl')`), the model
    `madmom_infer.features.key.CNNKeyRecognitionProcessor` loads by default.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _KEY_CNN_FILES]


# ---------------------------------------------------------------------------
# ONSETS_RNN / ONSETS_BRNN / ONSETS_CNN: madmom's onset-detection model
# families, used by RNNOnsetProcessor (RNN/BRNN) and CNNOnsetProcessor (CNN)
# -- the 4b end-to-end targets. sha256s verified against BOTH a fresh
# raw-GitHub download AND the copy already checked out locally under
# ../madmom-upstream/madmom/models (identical, see this module's header).
# ---------------------------------------------------------------------------
_ONSETS_RNN_FILES = [
    _ModelFile("onsets/2013/onsets_rnn_1.pkl",
               "5374f4a5fd12c9419de6195d77f20de7553b8b858d1408942b35bf04dc674901"),
    _ModelFile("onsets/2013/onsets_rnn_2.pkl",
               "2169734fc38a2366c49b98e2950f0743bcf7c659097db7798fbb0bc570bdfb22"),
    _ModelFile("onsets/2013/onsets_rnn_3.pkl",
               "bfb88d92ca0f7dcb11749edba9ff1c954536e0d84ba0660f6e201af9cc43050d"),
    _ModelFile("onsets/2013/onsets_rnn_4.pkl",
               "77af66643507873a0a1a6835c8dfccdb35ea0baace5e44ce0653211827a9db97"),
    _ModelFile("onsets/2013/onsets_rnn_5.pkl",
               "2c5b97872b568dab9e9aa67899c379026df5793f129595c070d4e58f0caa512d"),
    _ModelFile("onsets/2013/onsets_rnn_6.pkl",
               "3395a9be0e4c2b1fb37b58616a3ed499c6fd994bdd29e644144e8a06664180d5"),
    _ModelFile("onsets/2013/onsets_rnn_7.pkl",
               "d36d4f14b91aad277e6d5f8dad542297ada35f075a95cbb62f8fde34766c8e26"),
    _ModelFile("onsets/2013/onsets_rnn_8.pkl",
               "fafc8535a5654b2225739eff6803897f36f5945a411cbeaf36a2228ed00deaea"),
]

_ONSETS_BRNN_FILES = [
    _ModelFile("onsets/2013/onsets_brnn_1.pkl",
               "cd3552b4476f6cafb73c3937e07d954056b4ebd972114b67641ae5726c3233fd"),
    _ModelFile("onsets/2013/onsets_brnn_2.pkl",
               "041d72ce5fb5d466aeb65b1555f3cbb83db6ed659f30299c15cb5024f22cee21"),
    _ModelFile("onsets/2013/onsets_brnn_3.pkl",
               "0b2d2f805f5f9eca7a08174bfb3fdef7ed4cd04a75845dea925061262c634819"),
    _ModelFile("onsets/2013/onsets_brnn_4.pkl",
               "5394e8a6ef0d6b316a7dea0150ebf8d32b0a91267ba2cbb3108b9b8aabf11d17"),
    _ModelFile("onsets/2013/onsets_brnn_5.pkl",
               "cbe6967220a327e8691a45ab40adf6a98089c77b4969a6e9c961d308e3e29eb5"),
    _ModelFile("onsets/2013/onsets_brnn_6.pkl",
               "686cde0a80a83b3efb0986f8a4c8c170353727ebd03920fac525959e4d7f3f2f"),
    _ModelFile("onsets/2013/onsets_brnn_7.pkl",
               "14788a5834c73b09bebed233cb870cfd0af41915e6ee2639fbca1e43bdeae410"),
    _ModelFile("onsets/2013/onsets_brnn_8.pkl",
               "0b24555e609460b5a134645fb439e7088fb518007a165e6b9d469b7cf7d65de4"),
]

_ONSETS_CNN_FILES = [
    _ModelFile("onsets/2013/onsets_cnn.pkl",
               "ac7aad3f99c45ecea846406c867dec83fbdab114399b00bcb12eaaeffd5a990e"),
]


def onsets_rnn(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 8
    `onsets_rnn_[1-8].pkl` files -- madmom's `ONSETS_RNN` model list, used by
    `RNNOnsetProcessor(online=True)`.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _ONSETS_RNN_FILES]


def onsets_brnn(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 8
    `onsets_brnn_[1-8].pkl` files -- madmom's `ONSETS_BRNN` model list, used
    by `RNNOnsetProcessor()`'s default (`online=False`).

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _ONSETS_BRNN_FILES]


def onsets_cnn(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return the local path to `onsets_cnn.pkl`
    (as a single-element list) -- madmom's `ONSETS_CNN` model list, used by
    `CNNOnsetProcessor`.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _ONSETS_CNN_FILES]


# ---------------------------------------------------------------------------
# BEATS_LSTM / BEATS_BLSTM: madmom's beat-only RNN ensembles, used by
# RNNBeatProcessor -- the 4c end-to-end target. sha256s verified against
# BOTH a fresh raw-GitHub download AND the copy already checked out locally
# under ../madmom-upstream/madmom/models (identical, see this module's
# header).
# ---------------------------------------------------------------------------
_BEATS_LSTM_FILES = [
    _ModelFile("beats/2016/beats_lstm_1.pkl",
               "baab78a188d220d6972fdb91040f375ae711d96489bdb471ff7268124cc47725"),
    _ModelFile("beats/2016/beats_lstm_2.pkl",
               "af222e0cb79f49b680a87952e13dd755b71f5b81128b3318d4e99bdd458dba26"),
    _ModelFile("beats/2016/beats_lstm_3.pkl",
               "fc54ea5248fca54385312e17974cac07ca4a1a97432dc1dd1c02c364b4cd717a"),
    _ModelFile("beats/2016/beats_lstm_4.pkl",
               "7a355cf7167607f3543e73b8b2ba06ffa481d1cd7443376d4e3378ef6d28f1ae"),
    _ModelFile("beats/2016/beats_lstm_5.pkl",
               "849d563129e1b77df72c9b5e0d13a02f5a3eeffe3189b91a3edb73f864dc19c9"),
    _ModelFile("beats/2016/beats_lstm_6.pkl",
               "cae69664e3058ad3fed74dcd53414284096913cae8c9b2726375a555701202c8"),
    _ModelFile("beats/2016/beats_lstm_7.pkl",
               "bbc47b60c3c7e3f2026e234847aa91b41d3faf8cc61f81b8f523c4567a7b6c46"),
    _ModelFile("beats/2016/beats_lstm_8.pkl",
               "78459e9ef8af334ffa88eb8a36ababf04e51d7abdad2daaabd56dd99ccad2f3e"),
]

_BEATS_BLSTM_FILES = [
    _ModelFile("beats/2015/beats_blstm_1.pkl",
               "bd6545fe694264491ee95e03508d318d164b7f2abac1f48edd0121389506bf5f"),
    _ModelFile("beats/2015/beats_blstm_2.pkl",
               "89ab1defb1b34712bdd0c9f84a756178e809612720733ed7b3a5cc47420689d7"),
    _ModelFile("beats/2015/beats_blstm_3.pkl",
               "4a3f63f1fe361a354cb5f646349522b87482f29baadacf2d20441d5c52c5282a"),
    _ModelFile("beats/2015/beats_blstm_4.pkl",
               "4a4469fc00dc4c90b028866a9d268fd19429c6a3f298c70838571e90db248f18"),
    _ModelFile("beats/2015/beats_blstm_5.pkl",
               "017fd2e6808f43b9cddb61ac115c8d899f6fcdd3801338c6f76b1b30c3c3dbfd"),
    _ModelFile("beats/2015/beats_blstm_6.pkl",
               "055e8c8715ce4e71e9855e348ff1816fb7630a4d10117d8e4c11684770d64240"),
    _ModelFile("beats/2015/beats_blstm_7.pkl",
               "35ba5622e6c8f6c24b7efee1dbb31a78468b3106a7e4a8e3bfa2761f326b51dc"),
    _ModelFile("beats/2015/beats_blstm_8.pkl",
               "fc25b6890baaeecd2304e18078ee1ce6a0014a69fdba95590a4ebee6cd847f64"),
]


def beats_lstm(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 8
    `beats_lstm_[1-8].pkl` files -- madmom's `BEATS_LSTM` model list, used
    by `RNNBeatProcessor(online=True)`.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _BEATS_LSTM_FILES]


def beats_blstm(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 8
    `beats_blstm_[1-8].pkl` files -- madmom's `BEATS_BLSTM` model list, used
    by `RNNBeatProcessor()`'s default (`online=False`).

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _BEATS_BLSTM_FILES]


# ---------------------------------------------------------------------------
# DOWNBEATS_BGRU: madmom's beat-synchronous GRU ensembles (rhythmic +
# harmonic), used by RNNBarProcessor -- the 4c GRU end-to-end target.
# sha256s verified against BOTH a fresh raw-GitHub download AND the copy
# already checked out locally under ../madmom-upstream/madmom/models
# (identical, see this module's header).
# ---------------------------------------------------------------------------
_DOWNBEATS_BGRU_RHYTHMIC_FILES = [
    _ModelFile("downbeats/2016/downbeats_bgru_rhythmic_0.pkl",
               "50e7bcb8b6d5be58c35463caca79b624552469b23f94f353a97e319b38a3ee33"),
    _ModelFile("downbeats/2016/downbeats_bgru_rhythmic_1.pkl",
               "78f82313bec5f129bd9eed03a13718d6e81b5b6587c4bbdd7e5b4c4616e18436"),
    _ModelFile("downbeats/2016/downbeats_bgru_rhythmic_2.pkl",
               "d959a9478dd08ba3543c45df1ff89a80e8319d4c353f4e591ea93d52d317ff09"),
    _ModelFile("downbeats/2016/downbeats_bgru_rhythmic_3.pkl",
               "81089b0d9cfed8b21ffdb6584cac5f055ed97372612d6a0d4eb503494842763f"),
    _ModelFile("downbeats/2016/downbeats_bgru_rhythmic_4.pkl",
               "c3ea8025bbfc3f7660aea1c92b901c7f35b71c64d5c57a8485fd9011409f6a39"),
    _ModelFile("downbeats/2016/downbeats_bgru_rhythmic_5.pkl",
               "452708ef3c3a38523c259eaacc94dca84fb444e7a2553ee4a93fce20ef9397bb"),
]

_DOWNBEATS_BGRU_HARMONIC_FILES = [
    _ModelFile("downbeats/2016/downbeats_bgru_harmonic_0.pkl",
               "8d677435269025d71f1beb85c11e9c130eacb1fdbeca45d054969299fd18eea9"),
    _ModelFile("downbeats/2016/downbeats_bgru_harmonic_1.pkl",
               "f8797e7d8ce9d064714f3b9b4ccdc03413e5c91b25d6a275024efea72d8aef01"),
    _ModelFile("downbeats/2016/downbeats_bgru_harmonic_2.pkl",
               "a29d2acbffe75485e6476c63df461f90640bb9de6fd534b61a6bd8f938db46ee"),
    _ModelFile("downbeats/2016/downbeats_bgru_harmonic_3.pkl",
               "649ccc0a015020b74c5b437789926082bbbea7f9fe1950125b5e5d094b3d6d8f"),
    _ModelFile("downbeats/2016/downbeats_bgru_harmonic_4.pkl",
               "28ec3216a4368071eae0c2311bab138fca0b0dbca6791441b7ba290f34c29217"),
    _ModelFile("downbeats/2016/downbeats_bgru_harmonic_5.pkl",
               "82f71f78943399af650fccee59a9118ba64784e22fac4446ae820c5ec96d6933"),
]


def downbeats_bgru_rhythmic(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 6
    `downbeats_bgru_rhythmic_[0-5].pkl` files -- the first (`[0]`) element
    of madmom's `DOWNBEATS_BGRU` model list, used by `RNNBarProcessor`'s
    percussive-feature NN ensemble.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _DOWNBEATS_BGRU_RHYTHMIC_FILES]


def downbeats_bgru_harmonic(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return local paths to all 6
    `downbeats_bgru_harmonic_[0-5].pkl` files -- the second (`[1]`) element
    of madmom's `DOWNBEATS_BGRU` model list, used by `RNNBarProcessor`'s
    harmonic-feature NN ensemble.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _DOWNBEATS_BGRU_HARMONIC_FILES]


def downbeats_bgru(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return `[rhythmic_paths, harmonic_paths]` --
    madmom's own `DOWNBEATS_BGRU` list-of-lists shape exactly
    (`madmom-upstream/madmom/models/__init__.py`'s `DOWNBEATS_BGRU = [
    models('downbeats/2016/downbeats_bgru_rhythmic_*.pkl'), models(
    'downbeats/2016/downbeats_bgru_harmonic_*.pkl')]` -- `[0]` is rhythmic/
    percussive, `[1]` is harmonic, in that order).

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [downbeats_bgru_rhythmic(cache_root=cache_root, force=force),
            downbeats_bgru_harmonic(cache_root=cache_root, force=force)]


# ---------------------------------------------------------------------------
# CHROMA_DNN / CHORDS_DCCRF / CHORDS_CNN_FEAT / CHORDS_CFCRF: madmom's
# chroma/chord model families -- the 4d end-to-end targets. sha256s verified
# against BOTH a fresh raw-GitHub download AND the copy already checked out
# locally under ../madmom-upstream/madmom/models (identical, see this
# module's header).
# ---------------------------------------------------------------------------
_CHROMA_DNN_FILES = [
    _ModelFile("chroma/2016/chroma_dnn.pkl",
               "d91aff59113ec6b85de9c9b1aafb065ed0c756072284cebc61564237e901ca94"),
]

_CHORDS_DCCRF_FILES = [
    _ModelFile("chords/2016/chords_dccrf.pkl",
               "64e72027c989d6db8ecaa2f097592d3124b94f0362bde225d0c686a8a68d8d03"),
]

_CHORDS_CNN_FEAT_FILES = [
    _ModelFile("chords/2016/chords_cnnfeat.pkl",
               "59ac731a514880d7ec22ed9ee33935fc1683d2a03d8a7b4b81d26a3da944ce7d"),
]

_CHORDS_CFCRF_FILES = [
    _ModelFile("chords/2016/chords_cnncrf.pkl",
               "4b8a63b4bb5bea076cf1babc76d5cb0a3618e6125d9a3a998f441d61308676af"),
]


def chroma_dnn(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return the local path to `chroma_dnn.pkl`
    (as a single-element list) -- madmom's `CHROMA_DNN` model list, used by
    `madmom_infer.audio.chroma.DeepChromaProcessor`.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _CHROMA_DNN_FILES]


def chords_dccrf(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return the local path to `chords_dccrf.pkl`
    (as a single-element list) -- madmom's `CHORDS_DCCRF` model list, used by
    `DeepChromaChordRecognitionProcessor`.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _CHORDS_DCCRF_FILES]


def chords_cnn_feat(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return the local path to `chords_cnnfeat.pkl`
    (as a single-element list) -- madmom's `CHORDS_CNN_FEAT` model list, used
    by `CNNChordFeatureProcessor`.

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _CHORDS_CNN_FEAT_FILES]


def chords_cfcrf(cache_root: Path = None, force: bool = False):
    """Download (if needed) and return the local path to `chords_cnncrf.pkl`
    (as a single-element list) -- madmom's `CHORDS_CFCRF` model list, used by
    `CRFChordRecognitionProcessor` (the CRF that decodes
    `CNNChordFeatureProcessor`'s output -- see this module's header re: the
    `CFCRF`/`cnncrf.pkl` naming oddity).

    NON-COMMERCIAL USE ONLY for the downloaded weights (CC BY-NC-SA 4.0) --
    see this module's header.
    """
    return [download(f, cache_root=cache_root, force=force)
            for f in _CHORDS_CFCRF_FILES]
