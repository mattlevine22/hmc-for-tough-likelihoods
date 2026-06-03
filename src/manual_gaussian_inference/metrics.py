from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance


Array = jax.Array


def gaussian_kl_to_standard_normal(
    samples: Array, *, diagonal: bool = True, jitter: float = 1e-6
) -> Array:
    """Moment-matched Gaussian KL divergence to the standard normal.

    Fits a Gaussian to ``samples`` (shape ``(n, dim)``) and returns the analytic
    ``KL(N(mu_hat, Sigma_hat) || N(0, I_dim))``.

    ``diagonal=True`` (default) fits a diagonal covariance,

        0.5 * sum_i (var_i + mu_i**2 - 1 - log var_i),

    which is the appropriate, well-conditioned choice for this study: the target is
    isotropic and the per-coordinate sinusoid induces no cross-correlations, so the
    full covariance's ~dim^2/2 off-diagonal entries only add finite-sample noise
    (a large positive bias floor) without signal. ``diagonal=False`` uses the full
    sample covariance with a Cholesky log-determinant (no explicit inverse) and
    requires ``n > dim``. A small ``jitter`` stabilizes the log-variance/Cholesky.
    """
    samples = jnp.asarray(samples, dtype=float)
    n, dim = samples.shape
    mu = jnp.mean(samples, axis=0)
    if diagonal:
        var = jnp.var(samples, axis=0) + jitter
        return 0.5 * jnp.sum(var + mu**2 - 1.0 - jnp.log(var))
    centered = samples - mu
    cov = (centered.T @ centered) / (n - 1) + jitter * jnp.eye(dim)
    chol = jnp.linalg.cholesky(cov)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    return 0.5 * (jnp.trace(cov) + jnp.dot(mu, mu) - dim - logdet)


def wasserstein1_to_truth(samples: Array, reference: Array) -> float:
    """1-Wasserstein (W1) distance between 1-D ``samples`` and a reference draw.

    Both arguments are flattened to 1-D; ``reference`` should be a large iid draw
    from the true ``N(0, 1)`` target. Used for the ``d=1`` experiment.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    return float(wasserstein_distance(samples, reference))
