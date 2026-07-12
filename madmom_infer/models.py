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

Every other model family madmom ships (`BEATS_LSTM`, `CHORDS_DCCRF`,
`NOTES_CNN`, ...) would follow the exact same `_ModelFile`/`download()`
pattern -- adding one is a matter of listing its relative paths + sha256s,
not new machinery -- but is out of scope until a processor that needs it is
ported (see CLAUDE.md's wave plan).

Reads: urllib.request (stdlib, HTTPS GET), hashlib (stdlib, sha256), os/
pathlib (stdlib, XDG cache resolution); read by:
madmom_infer/features/downbeats.py (RNNDownBeatProcessor.__init__),
madmom_infer/features/key.py (CNNKeyRecognitionProcessor.__init__),
madmom_infer/features/onsets.py (RNNOnsetProcessor.__init__,
CNNOnsetProcessor.__init__).
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
