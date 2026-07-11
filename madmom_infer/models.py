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

Only `DOWNBEATS_BLSTM` (the Phase-2 end-to-end target, `RNNDownBeatProcessor`)
has a populated, sha256-pinned file list today. Every other model family
madmom ships (`BEATS_LSTM`, `ONSETS_RNN`, `CHORDS_DCCRF`, `NOTES_CNN`, ...)
would follow the exact same `_ModelFile`/`download()` pattern -- adding one
is a matter of listing its relative paths + sha256s, not new machinery --
but is out of Phase-2 scope (see README's roadmap) until a processor that
needs it is ported.

Reads: urllib.request (stdlib, HTTPS GET), hashlib (stdlib, sha256), os/
pathlib (stdlib, XDG cache resolution); read by:
madmom_infer/features/downbeats.py (RNNDownBeatProcessor.__init__).
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
