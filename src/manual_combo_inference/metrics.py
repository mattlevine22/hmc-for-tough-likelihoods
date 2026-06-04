from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance

from manual_discrete_time_inference import kalman_filter_loglik

Array = jax.Array


def wasserstein1_to_truth(samples, reference) -> float:
    """1-Wasserstein (W1) distance between 1-D ``samples`` and a reference draw.

    Both arguments are flattened to 1-D; ``reference`` should be a large iid draw from
    the exact KF posterior (see :func:`kf_posterior_reference_samples`)."""
    samples = np.asarray(samples, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    return float(wasserstein_distance(samples, reference))


def kf_posterior_grid(
    obs_values,
    *,
    ctrl_values=None,
    system_kwargs: dict | None = None,
    alpha_lo: float = -0.7,
    alpha_hi: float = 0.7,
    grid_size: int = 2001,
):
    """Exact posterior density of ``alpha`` on a fine grid, from the Kalman filter.

    Returns ``(grid, pdf)`` where ``pdf`` integrates to 1 over the grid (uniform prior
    on ``[-0.7, 0.7]`` cancels into the normalizer). This is the ground-truth posterior
    the approximate combinations are compared against."""
    grid = jnp.linspace(alpha_lo, alpha_hi, grid_size)

    def loglik(alpha):
        ll, _ = kalman_filter_loglik(
            alpha, obs_values=obs_values, ctrl_values=ctrl_values,
            system_kwargs=system_kwargs,
        )
        return ll

    ll = np.asarray(jax.vmap(loglik)(grid), dtype=float)
    grid_np = np.asarray(grid, dtype=float)
    weights = np.exp(ll - np.max(ll))
    pdf = weights / np.trapezoid(weights, grid_np)
    return grid_np, pdf


def kf_posterior_reference_samples(
    obs_values,
    seed: int,
    n: int,
    *,
    ctrl_values=None,
    system_kwargs: dict | None = None,
    alpha_lo: float = -0.7,
    alpha_hi: float = 0.7,
    grid_size: int = 2001,
):
    """Draw ``n`` iid samples from the exact KF posterior via inverse-CDF on the fine
    grid -- the reference for the W1-to-truth metric."""
    grid_np, pdf = kf_posterior_grid(
        obs_values, ctrl_values=ctrl_values, system_kwargs=system_kwargs,
        alpha_lo=alpha_lo, alpha_hi=alpha_hi, grid_size=grid_size,
    )
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    rng = np.random.default_rng(int(seed))
    u = rng.random(n)
    return np.interp(u, cdf, grid_np)
