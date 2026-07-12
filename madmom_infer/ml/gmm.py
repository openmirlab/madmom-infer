"""Numpy reimplementation of madmom.ml.gmm -- a minimal Gaussian Mixture
Model class with only the forward-inference path (`score`/`score_samples`)
ported, no `fit()` (this project never trains models, see CLAUDE.md's
golden-fixture philosophy: numpy backend is a reference-matching INFERENCE
port, not a training library). Wave 4f of the complete-port campaign; see
CLAUDE.md's audit table, `ml/gmm.py` row -- backs
`features/beats_hmm.py`'s `GMMPatternTrackingObservationModel`, in turn
`features/downbeats.py`'s `PatternTrackingProcessor`.

Upstream's own module docstring notes this is itself a vendored-and-trimmed
copy of an old `sklearn.mixture.GMM` (BSD-licensed, pre-`GaussianMixture`
sklearn API) -- this port is a further trim of THAT: `fit()` (needs
`sklearn.mixture.GMM` itself, an EM-training routine, permanently out of
scope) is dropped, everything `score()`/`score_samples()` actually reaches
is kept verbatim.

**Pickle-format finding, confirmed by `pickletools`-walking the actual
target `.pkl` files (`madmom-upstream/madmom/models/patterns/2013/
ballroom_pattern_{3,4}_4.pkl`), not guessed**: these GMM instances are
OLD-FORMAT pickles -- their pickled `__dict__` keys are `weights_`/`means_`/
`covars_` (trailing underscore, the old sklearn-`GMM`-style attribute
names), not this class's own `weights`/`means`/`covars`. `GMM.__setstate__`
below (verbatim port of upstream's own legacy-rename branch,
`madmom-upstream/madmom/ml/gmm.py:258-272`) is what makes unpickling work at
all -- standard `pickle.Unpickler`'s `BUILD` opcode calls `obj.__setstate__
(state)` when a class defines one (rather than a bare `obj.__dict__.update
(state)`), so this fires automatically through `madmom_infer.ml.nn.unpickle
.SafeUnpickler` (a `pickle.Unpickler` subclass -- only `find_class` is
overridden, `load_build` is untouched, same "restrict WHICH classes can be
named, not HOW an already-resolved class restores its own state" split as
every other target `.pkl` in this project). Confirmed empirically: loading
either target file with REAL madmom emits the exact "Please update your GMM
models..." `UserWarning` this method raises, matching the "OLD-FORMAT
pickle" finding wave 4c made for `downbeats_bgru_*.pkl`/`GRULayer
.__setstate__` -- same shape of gap, different model family.

Both target pattern files' GMMs use `covariance_type='full'` (confirmed by
loading with real madmom and inspecting `gmm.covariance_type` directly) --
`score_samples`/`score` are exercised end-to-end only through
`_log_multivariate_normal_density_full` in this project's own tests, but all
4 covariance-type branches are ported (cheap, pure numpy/scipy, matches
upstream's own `log_multivariate_normal_density` dispatch dict exactly) for
API completeness, matching this project's "port the whole forward-inference
surface, not just what one target model happens to exercise" convention
elsewhere (e.g. `ml/crf.py`, `ml/nn/layers.py`'s unused-by-any-target-but-
ported-for-completeness classes).

Reads: numpy, scipy.linalg; read by: madmom_infer/features/beats_hmm.py
(GMMPatternTrackingObservationModel.log_densities calls `gmm.score(...)`),
madmom_infer/ml/nn/unpickle.py (ALLOWED_GLOBALS entry, GMM as an unpickle
target).
"""

import numpy as np
from scipy import linalg


def logsumexp(arr, axis=0):
    """Compute `log(sum(exp(arr)))` along `axis`, minimizing over/underflow.

    Verbatim port of `madmom.ml.gmm.logsumexp` (`gmm.py:28-55`, itself
    copied from `sklearn.utils.extmath.logsumexp`).
    """
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out


def pinvh(a, cond=None, rcond=None, lower=True):
    """Moore-Penrose pseudo-inverse of a Hermitian/symmetric matrix, via its
    eigenvalue decomposition.

    Verbatim port of `madmom.ml.gmm.pinvh` (`gmm.py:58-106`, itself copied
    from `sklearn.utils.extmath.pinvh`). Not reached by `GMM.score_samples`
    itself (no target `.pkl` needs it), ported for API completeness --
    upstream exports it from this module too.
    """
    a = np.asarray_chkfinite(a)
    s, u = linalg.eigh(a, lower=lower)

    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = u.dtype.char.lower()
        factor = {"f": 1e3, "d": 1e6}
        cond = factor[t] * np.finfo(t).eps

    above_cutoff = abs(s) > cond * np.max(abs(s))
    psigma_diag = np.zeros_like(s)
    psigma_diag[above_cutoff] = 1.0 / s[above_cutoff]

    return np.dot(u * psigma_diag, np.conjugate(u).T)


def _log_multivariate_normal_density_diag(x, means, covars):
    """Gaussian log-density at `x` for a diagonal-covariance model.

    Verbatim port of `madmom.ml.gmm._log_multivariate_normal_density_diag`
    (`gmm.py:149-156`).
    """
    _, n_dim = x.shape
    lpr = -0.5 * (
        n_dim * np.log(2 * np.pi)
        + np.sum(np.log(covars), 1)
        + np.sum((means**2) / covars, 1)
        - 2 * np.dot(x, (means / covars).T)
        + np.dot(x**2, (1.0 / covars).T)
    )
    return lpr


def _log_multivariate_normal_density_spherical(x, means, covars):
    """Gaussian log-density at `x` for a spherical-covariance model.

    Verbatim port of
    `madmom.ml.gmm._log_multivariate_normal_density_spherical`
    (`gmm.py:159-166`).
    """
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if covars.shape[1] == 1:
        cv = np.tile(cv, (1, x.shape[-1]))
    return _log_multivariate_normal_density_diag(x, means, cv)


def _log_multivariate_normal_density_tied(x, means, covars):
    """Gaussian log-density at `x` for a tied-covariance model.

    Verbatim port of `madmom.ml.gmm._log_multivariate_normal_density_tied`
    (`gmm.py:169-172`).
    """
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(x, means, cv)


def _log_multivariate_normal_density_full(x, means, covars, min_covar=1.0e-7):
    """Gaussian log-density at `x` for a full-covariance model (via a
    Cholesky decomposition of each component's covariance matrix).

    Verbatim port of `madmom.ml.gmm._log_multivariate_normal_density_full`
    (`gmm.py:175-198`) -- the ONLY branch the shipped `PATTERNS_BALLROOM`
    GMMs exercise (both target `.pkl` files use `covariance_type='full'`,
    see this module's header).
    """
    n_samples, n_dim = x.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # the model is most probably stuck in a component with too few
            # observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(
                    cv + min_covar * np.eye(n_dim), lower=True
                )
            except linalg.LinAlgError:
                raise ValueError(
                    "'covars' must be symmetric, positive-definite"
                )

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (x - mu).T, lower=True).T
        log_prob[:, c] = -0.5 * (
            np.sum(cv_sol**2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det
        )

    return log_prob


_LOG_MULTIVARIATE_NORMAL_DENSITY_DICT = {
    "spherical": _log_multivariate_normal_density_spherical,
    "tied": _log_multivariate_normal_density_tied,
    "diag": _log_multivariate_normal_density_diag,
    "full": _log_multivariate_normal_density_full,
}


def log_multivariate_normal_density(x, means, covars, covariance_type="diag"):
    """Log probability of `x` under a multivariate Gaussian distribution,
    dispatching on `covariance_type`.

    Verbatim port of `madmom.ml.gmm.log_multivariate_normal_density`
    (`gmm.py:109-146`).
    """
    return _LOG_MULTIVARIATE_NORMAL_DENSITY_DICT[covariance_type](
        x, means, covars
    )


class GMM:
    """Gaussian Mixture Model -- forward-inference only (`score_samples`/
    `score`), no `fit()` (see this module's header).

    Composition port of `madmom.ml.gmm.GMM` (`gmm.py:202-330`, minus
    `fit()`, `gmm.py:332-385`).

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.
    covariance_type : {'diag', 'spherical', 'tied', 'full'}
        Type of covariance parameters to use. Defaults to 'full' (matching
        upstream's own default -- note this differs from
        `log_multivariate_normal_density`'s own `'diag'` default).

    Attributes
    ----------
    weights : numpy array, shape (n_components,)
        Mixing weight for each mixture component.
    means : numpy array, shape (n_components, n_features)
        Mean parameters for each mixture component.
    covars : numpy array
        Covariance parameters for each mixture component (shape depends on
        `covariance_type`, see `log_multivariate_normal_density`).
    """

    def __init__(self, n_components=1, covariance_type="full"):
        if covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for covariance_type: %s" % covariance_type
            )
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = None
        self.covars = None

    def __setstate__(self, state):
        """Restore a pickled `GMM`, transparently renaming the legacy
        (pre-0.16) `weights_`/`means_`/`covars_` attribute names to this
        class's own `weights`/`means`/`covars` if present.

        Verbatim port of `madmom.ml.gmm.GMM.__setstate__` (`gmm.py:258-272`)
        -- called automatically by the stdlib `pickle.Unpickler`'s `BUILD`
        opcode handling for any class that defines `__setstate__` (both real
        madmom's own bare `pickle.load` and this project's
        `madmom_infer.ml.nn.unpickle.SafeUnpickler`, which only overrides
        `find_class`, not this protocol-level restore step -- see this
        module's header). Confirmed empirically that both shipped
        `PATTERNS_BALLROOM` `.pkl` files take this legacy-rename branch (old
        pickle format, matching wave 4c's `GRULayer`/`downbeats_bgru_*.pkl`
        finding).
        """
        import warnings

        try:
            warnings.warn(
                "Please update your GMM models by loading them and "
                "saving them again. Loading old models will not "
                "work from version 0.16 onwards."
            )
            state["weights"] = state.pop("weights_")
            state["means"] = state.pop("means_")
            state["covars"] = state.pop("covars_")
        except KeyError:
            pass
        self.__dict__.update(state)

    def score_samples(self, x):
        """Per-sample log-likelihood of `x` under the model, plus the
        posterior responsibility of each mixture component for each sample.

        Verbatim port of `madmom.ml.gmm.GMM.score_samples`
        (`gmm.py:274-311`).

        Parameters
        ----------
        x : array_like, shape (n_samples, n_features)
            Data points.

        Returns
        -------
        log_prob : numpy array, shape (n_samples,)
            Log probability of each data point.
        responsibilities : numpy array, shape (n_samples, n_components)
            Posterior probability of each mixture component for each sample.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if x.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if x.shape[1] != self.means.shape[1]:
            raise ValueError("The shape of x is not compatible with self")

        lpr = log_multivariate_normal_density(
            x, self.means, self.covars, self.covariance_type
        ) + np.log(self.weights)
        log_prob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - log_prob[:, np.newaxis])
        return log_prob, responsibilities

    def score(self, x):
        """Log probability of `x` under the model.

        Verbatim port of `madmom.ml.gmm.GMM.score` (`gmm.py:313-330`).

        Parameters
        ----------
        x : array_like, shape (n_samples, n_features)
            Data points.

        Returns
        -------
        log_prob : numpy array, shape (n_samples,)
            Log probability of each data point.
        """
        log_prob, _ = self.score_samples(x)
        return log_prob
