from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from manual_discrete_time_inference import (
    kalman_filter_loglik,
    particle_filter_loglik,
    run_blackjax_nuts_1d as run_nuts_1d,
)

Array = jax.Array

# The three marginal-likelihood sources, by fidelity.
SOURCES = ("kf", "lofi", "hifi")
SOURCE_LABELS = {"kf": "KF", "lofi": "lofi PF", "hifi": "hifi PF"}

# Uniform prior on alpha (matches the discrete-time package).
_ALPHA_LO = -0.7
_ALPHA_HI = 0.7


def _log_prior(alpha: Array) -> Array:
    return jnp.where(
        (alpha >= _ALPHA_LO) & (alpha <= _ALPHA_HI),
        -jnp.log(_ALPHA_HI - _ALPHA_LO),
        -jnp.inf,
    )


def make_source_loglik(
    source: str,
    obs_values: Array,
    *,
    ctrl_values: Array | None = None,
    n_lofi: int = 100,
    n_hifi: int = 1000,
    fixed_key: Array | None = None,
    system_kwargs: dict | None = None,
):
    """Return a scalar log-marginal-likelihood function ``alpha -> log p(y|alpha)``
    for one source in ``{"kf", "lofi", "hifi"}``.

    ``kf`` is the exact Kalman-filter marginal likelihood (smooth, autodiff-exact).
    ``lofi``/``hifi`` are particle-filter estimates at ``n_lofi``/``n_hifi`` particles,
    evaluated under the *fixed* CRN key so the surface is a single deterministic-but-
    rough sample path; their ``jax.grad`` is the Fisher-identity score (via the
    ``stop_gradient_resample`` trick), not the path derivative of the estimate.
    """
    obs_values = jnp.asarray(obs_values)
    ctrl_values = None if ctrl_values is None else jnp.asarray(ctrl_values)
    if fixed_key is None:
        fixed_key = jr.PRNGKey(0)

    if source == "kf":
        def loglik_fn(alpha):
            ll, _ = kalman_filter_loglik(
                alpha,
                obs_values=obs_values,
                ctrl_values=ctrl_values,
                system_kwargs=system_kwargs,
            )
            return ll

        return loglik_fn

    if source in ("lofi", "hifi"):
        n_particles = n_lofi if source == "lofi" else n_hifi

        def loglik_fn(alpha):
            return particle_filter_loglik(
                alpha,
                obs_values=obs_values,
                n_particles=n_particles,
                key=fixed_key,
                ctrl_values=ctrl_values,
                system_kwargs=system_kwargs,
            )

        return loglik_fn

    raise ValueError(f"Unknown source: {source!r}. Expected one of {SOURCES}.")


def make_combo_logdensity(
    obs_values: Array,
    *,
    primal_source: str,
    grad_source: str,
    ctrl_values: Array | None = None,
    n_lofi: int = 100,
    n_hifi: int = 1000,
    fixed_key: Array | None = None,
    system_kwargs: dict | None = None,
):
    """Custom-JVP scalar log-density whose *value* comes from ``primal_source`` and
    whose *gradient* comes from ``grad_source`` (each in ``{"kf","lofi","hifi"}``).

    The point of the package: decouple the fidelity of the likelihood *value* the
    Metropolis correction sees from the fidelity of the *score* the leapfrog follows.
    The diagonal ``primal == grad`` cases reproduce the ordinary autodiff log-density
    for that source (``kf/kf`` is the exact gold standard); the off-diagonals are the
    mismatched combinations.

    The custom JVP returns ``(value, score * dalpha)`` -- linear in the tangent, so it
    transposes cleanly and ``jax.value_and_grad`` (used by BlackJAX) recovers the
    primal-source value together with the grad-source score. The uniform prior is
    folded into the value (``-inf`` outside ``[-0.7, 0.7]``); its gradient is zero
    inside the box.
    """
    primal_loglik = make_source_loglik(
        primal_source, obs_values,
        ctrl_values=ctrl_values, n_lofi=n_lofi, n_hifi=n_hifi,
        fixed_key=fixed_key, system_kwargs=system_kwargs,
    )
    grad_loglik = make_source_loglik(
        grad_source, obs_values,
        ctrl_values=ctrl_values, n_lofi=n_lofi, n_hifi=n_hifi,
        fixed_key=fixed_key, system_kwargs=system_kwargs,
    )
    grad_score = jax.grad(grad_loglik)

    @jax.custom_jvp
    def logdensity_fn(alpha):
        alpha = jnp.asarray(alpha)
        return _log_prior(alpha) + primal_loglik(alpha)

    @logdensity_fn.defjvp
    def _logdensity_jvp(primals, tangents):
        (alpha,) = primals
        (dalpha,) = tangents
        alpha = jnp.asarray(alpha)
        value = _log_prior(alpha) + primal_loglik(alpha)
        score = grad_score(alpha)  # Fisher/exact score from grad_source
        return value, score * dalpha

    return logdensity_fn


def run_walnuts_1d(
    logdensity_fn,
    seed: int,
    *,
    init_position: float = 0.35,
    num_warmup: int = 200,
    num_samples: int = 500,
    initial_step_size: float = 0.1,
    energy_threshold: float = 0.7,
    target_no_refinement_rate: float = 0.8,
    max_num_doublings: int = 8,
    max_num_micro_doublings: int = 8,
):
    """WALNUTS on a scalar ``alpha`` target via the ``walnuts`` kernel in the
    ``DanWaxman/blackjax`` fork (runs under ``jax_enable_x64`` since the fork's x64
    fix). The scalar ``logdensity_fn`` is wrapped to a length-1 vector for the kernel;
    samples are returned as a scalar array. Mirrors :func:`run_nuts_1d`.
    """
    import blackjax

    warmup_key, sample_key = jr.split(jr.PRNGKey(seed))

    def logdensity_vec(theta):
        return logdensity_fn(theta[0])

    initial_position = jnp.asarray([init_position], dtype=float)

    warmup = blackjax.walnuts_adaptation(
        logdensity_vec,
        inverse_mass_matrix=jnp.ones(1),
        initial_step_size=initial_step_size,
        target_no_refinement_rate=target_no_refinement_rate,
        energy_threshold=energy_threshold,
        max_num_doublings=max_num_doublings,
        max_num_micro_doublings=max_num_micro_doublings,
    )
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_warmup)

    kernel = blackjax.walnuts(logdensity_vec, **parameters)

    @jax.jit
    def inference_loop(initial_state, rng_key):
        keys = jr.split(rng_key, num_samples)

        def one_step(current_state, current_key):
            next_state, info = kernel.step(current_key, current_state)
            return next_state, (next_state.position, info)

        _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)
        return positions, infos

    positions, infos = inference_loop(state, sample_key)
    return {
        "samples": positions[:, 0],
        "infos": infos,
        "state": state,
        "parameters": parameters,
    }
