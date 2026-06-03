from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr


Array = jax.Array


@dataclasses.dataclass(frozen=True)
class PerturbedGaussianSystem:
    """A ``d``-dimensional standard normal target with an additive sinusoidal
    perturbation of its log-density.

    The true target is ``N(0, I_d)`` with log-density ``L(theta) = -0.5||theta||^2``
    (up to a constant). The *approximate* (rough) log-density handed to a sampler
    adds a deterministic per-coordinate sinusoid::

        log pi(theta) = L(theta) + sum_i amplitude_i * sin(frequency_i * theta_i)

    whose gradient is the exact path derivative. ``amplitude`` and ``frequency``
    are length-``dim`` arrays (isotropic settings are broadcast at construction).
    """

    dim: int
    amplitude: Array
    frequency: Array


def make_perturbed_gaussian_system(
    dim: int,
    amplitude: Array | float = 0.0,
    frequency: Array | float = 0.0,
) -> PerturbedGaussianSystem:
    """Build a system, broadcasting scalar (isotropic) ``amplitude``/``frequency``
    to length-``dim`` arrays and validating anisotropic vectors."""

    def _broadcast(value, name):
        value = jnp.asarray(value, dtype=float)
        if value.ndim == 0:
            return jnp.full((dim,), value)
        if value.shape == (dim,):
            return value
        raise ValueError(
            f"{name} must be a scalar or a length-{dim} vector, got shape {value.shape}."
        )

    return PerturbedGaussianSystem(
        dim=int(dim),
        amplitude=_broadcast(amplitude, "amplitude"),
        frequency=_broadcast(frequency, "frequency"),
    )


def true_log_density(system: PerturbedGaussianSystem, theta: Array) -> Array:
    """Log-density of the true target ``N(0, I_d)`` (normalized)."""
    theta = jnp.atleast_1d(jnp.asarray(theta))
    return -0.5 * (system.dim * jnp.log(2.0 * jnp.pi) + jnp.dot(theta, theta))


def true_grad_log_density(system: PerturbedGaussianSystem, theta: Array) -> Array:
    """Gradient of the true log-density: ``-theta``."""
    theta = jnp.atleast_1d(jnp.asarray(theta))
    return -theta


def perturbation(system: PerturbedGaussianSystem, theta: Array) -> Array:
    """Additive per-coordinate sinusoid ``sum_i A_i sin(omega_i theta_i)``."""
    theta = jnp.atleast_1d(jnp.asarray(theta))
    return jnp.sum(system.amplitude * jnp.sin(system.frequency * theta))


def perturbed_log_density(system: PerturbedGaussianSystem, theta: Array) -> Array:
    """Rough log-density handed to the sampler: ``L(theta) + perturbation``."""
    return true_log_density(system, theta) + perturbation(system, theta)


def perturbed_grad_log_density(system: PerturbedGaussianSystem, theta: Array) -> Array:
    """Analytic path derivative ``-theta + A*omega*cos(omega*theta)`` (equals
    ``jax.grad(perturbed_log_density)``)."""
    theta = jnp.atleast_1d(jnp.asarray(theta))
    return -theta + system.amplitude * system.frequency * jnp.cos(system.frequency * theta)


def simulate_true_samples(key: Array, system: PerturbedGaussianSystem, n: int) -> Array:
    """Draw ``n`` iid samples from the true target ``N(0, I_d)`` -- the ground-truth
    "data" the perturbed samples are compared against. Shape ``(n, dim)``."""
    return jr.normal(key, shape=(n, system.dim))


def make_blackjax_logdensity(
    system: PerturbedGaussianSystem,
    *,
    mode: str = "perturbed",
):
    """Return a closure ``logdensity_fn(theta)``.

    ``mode="exact"`` returns the true ``N(0, I_d)`` log-density; ``mode="perturbed"``
    returns the rough sinusoidally-perturbed log-density. No prior term is added --
    the target is itself a (proper) density.
    """
    if mode not in ("exact", "perturbed"):
        raise ValueError(f"Unknown mode: {mode!r}")

    def logdensity_fn(theta):
        theta = jnp.atleast_1d(jnp.asarray(theta))
        if mode == "exact":
            return true_log_density(system, theta)
        return perturbed_log_density(system, theta)

    return logdensity_fn


# --------------------------------------------------------------------------- #
# Stochastic-gradient variant (custom JVP): exact value, noisy gradient.
#
# Models the CRN particle-filter regime where the Fisher-identity gradient is a
# *consistent estimator of the true gradient* that is decoupled from -- not the
# path derivative of -- the log-likelihood value. We realize this with a
# ``jax.custom_jvp``: the primal value is controlled by ``value_mode``, while the
# gradient handed to the sampler is always ``-theta + grad_noise_scale * N(0, I_d)``
# (Gaussian noise around the true, non-sinusoidal gradient ``-theta``).
# --------------------------------------------------------------------------- #


def _broadcast_to_dim(value: Array | float, dim: int, name: str) -> Array:
    """Broadcast a scalar (isotropic) parameter to a length-``dim`` array, or pass a
    length-``dim`` vector (anisotropic) through after validation."""
    value = jnp.asarray(value, dtype=float)
    if value.ndim == 0:
        return jnp.full((dim,), value)
    if value.shape == (dim,):
        return value
    raise ValueError(
        f"{name} must be a scalar or a length-{dim} vector, got shape {value.shape}."
    )


def _position_dependent_key(base_key: Array, theta: Array) -> Array:
    """Deterministic per-position PRNG key -- a CRN-faithful rough field.

    Mirrors ``manual_ct_inference._position_dependent_key`` but for a length-``dim``
    position: floor-quantizes each coordinate to ``uint32`` and folds it into
    ``base_key`` (a tiny ``lax.scan`` over coordinates; a single fold for ``dim==1``).
    Distinct ``theta`` map to distinct keys, so the noise is reproducible yet
    effectively fresh at each position a leapfrog step visits.
    """
    theta = jnp.atleast_1d(jnp.asarray(theta))
    folded = jnp.asarray(jnp.floor((theta + 10.0) * 1_000_000.0), dtype=jnp.uint32)

    def _fold(key, f):
        return jr.fold_in(key, f), None

    key, _ = jax.lax.scan(_fold, base_key, folded)
    return key


@dataclasses.dataclass(frozen=True)
class NoisyGradientGaussianSystem:
    """A ``d``-dimensional standard normal target whose *gradient* (not value) is
    corrupted by Gaussian noise around the true gradient ``-theta``.

    The log-density value handed to the sampler is selected by ``value_mode`` in
    :func:`make_noisy_gradient_logdensity` (exact ``-0.5||theta||^2`` by default; an
    additive sinusoid; or additive hashed-position noise). Regardless of the value,
    the gradient is ``-theta + grad_noise_scale * eps`` with ``eps ~ N(0, I_d)`` drawn
    from a position-hashed key -- a deterministic-but-rough realization of a
    stochastic gradient (CRN). ``amplitude``/``frequency`` parameterize the optional
    sinusoidal value corruption; ``value_noise_scale`` the optional noisy value
    corruption. All four are length-``dim`` arrays (isotropic settings broadcast at
    construction).
    """

    dim: int
    grad_noise_scale: Array
    amplitude: Array
    frequency: Array
    value_noise_scale: Array


def make_noisy_gradient_system(
    dim: int,
    grad_noise_scale: Array | float = 0.0,
    amplitude: Array | float = 0.0,
    frequency: Array | float = 0.0,
    value_noise_scale: Array | float = 0.0,
) -> NoisyGradientGaussianSystem:
    """Build a system, broadcasting scalar (isotropic) parameters to length-``dim``
    arrays and validating anisotropic vectors."""
    return NoisyGradientGaussianSystem(
        dim=int(dim),
        grad_noise_scale=_broadcast_to_dim(grad_noise_scale, dim, "grad_noise_scale"),
        amplitude=_broadcast_to_dim(amplitude, dim, "amplitude"),
        frequency=_broadcast_to_dim(frequency, dim, "frequency"),
        value_noise_scale=_broadcast_to_dim(value_noise_scale, dim, "value_noise_scale"),
    )


def noisy_grad_log_density(
    system: NoisyGradientGaussianSystem, theta: Array, key: Array
) -> Array:
    """Reference noisy gradient ``-theta + grad_noise_scale * N(0, I_d)`` (the value
    the sampler's leapfrog sees via :func:`make_noisy_gradient_logdensity`). ``key``
    seeds the Gaussian noise; pass a position-hashed key for a CRN-reproducible
    field (see :func:`_position_dependent_key`)."""
    theta = jnp.atleast_1d(jnp.asarray(theta))
    eps = jr.normal(key, theta.shape)
    return -theta + system.grad_noise_scale * eps


def make_noisy_gradient_logdensity(
    system: NoisyGradientGaussianSystem,
    base_key: Array,
    *,
    value_mode: str = "exact",
):
    """Return a ``jax.custom_jvp`` closure ``logdensity_fn(theta)`` whose *value* is
    selected by ``value_mode`` but whose *gradient* is always the stochastic
    ``-theta + grad_noise_scale * N(0, I_d)`` (noise around the true, non-sinusoidal
    gradient).

    ``value_mode``:
      - ``"exact"``      -- value is the true ``N(0, I_d)`` log-density;
      - ``"sinusoidal"`` -- value adds ``sum_i amplitude_i * sin(frequency_i * theta_i)``;
      - ``"noisy"``      -- value adds hashed-position ``sum_i value_noise_scale_i * N(0, 1)``.

    The gradient noise is a deterministic function of ``theta`` (a position-hashed
    key folded into ``base_key``), so the surface is CRN-reproducible yet effectively
    fresh at each distinct ``theta`` a leapfrog step visits. The custom JVP returns
    ``(value, <noisy_grad, dtheta>)``; being linear in the tangent it transposes
    cleanly, so ``jax.value_and_grad`` (used by BlackJAX leapfrog) recovers the exact
    value together with the noisy gradient.
    """
    if value_mode not in ("exact", "sinusoidal", "noisy"):
        raise ValueError(f"Unknown value_mode: {value_mode!r}")

    def _value(theta, value_key):
        base = true_log_density(system, theta)
        if value_mode == "sinusoidal":
            return base + jnp.sum(system.amplitude * jnp.sin(system.frequency * theta))
        if value_mode == "noisy":
            eps = jr.normal(value_key, theta.shape)
            return base + jnp.sum(system.value_noise_scale * eps)
        return base

    @jax.custom_jvp
    def logdensity_fn(theta):
        theta = jnp.atleast_1d(jnp.asarray(theta))
        value_key = jr.split(_position_dependent_key(base_key, theta))[0]
        return _value(theta, value_key)

    @logdensity_fn.defjvp
    def logdensity_jvp(primals, tangents):
        (theta,) = primals
        (dtheta,) = tangents
        theta = jnp.atleast_1d(jnp.asarray(theta))
        dtheta = jnp.atleast_1d(jnp.asarray(dtheta))
        value_key, grad_key = jr.split(_position_dependent_key(base_key, theta))
        value = _value(theta, value_key)
        eps = jr.normal(grad_key, theta.shape)
        noisy_grad = -theta + system.grad_noise_scale * eps
        return value, jnp.vdot(noisy_grad, dtheta)

    return logdensity_fn


def _broadcast_init_position(init_position: Array | float, dim: int) -> Array:
    init = jnp.asarray(init_position, dtype=float)
    if init.ndim == 0:
        return jnp.full((dim,), init)
    if init.shape == (dim,):
        return init
    raise ValueError(
        f"init_position must be a scalar or a length-{dim} vector, got shape {init.shape}."
    )


def run_blackjax_nuts(
    logdensity_fn,
    seed: int,
    *,
    dim: int,
    init_position: Array | float = 0.0,
    num_warmup: int = 500,
    num_samples: int = 1000,
):
    """BlackJAX NUTS with window adaptation on a ``dim``-dimensional target."""
    import blackjax

    warmup_key, sample_key = jr.split(jr.PRNGKey(seed))
    initial_position = _broadcast_init_position(init_position, dim)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        progress_bar=False,
    )
    (state, parameters), _ = warmup.run(
        warmup_key,
        initial_position,
        num_steps=num_warmup,
    )

    kernel = blackjax.nuts(logdensity_fn, **parameters)

    @jax.jit
    def inference_loop(initial_state, rng_key):
        keys = jr.split(rng_key, num_samples)

        def one_step(current_state, current_key):
            next_state, info = kernel.step(current_key, current_state)
            return next_state, (next_state.position, info)

        _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)
        return positions, infos

    samples, infos = inference_loop(state, sample_key)
    return {
        "samples": samples,
        "infos": infos,
        "state": state,
        "parameters": parameters,
    }


def run_numpyro_nuts(
    logdensity_fn,
    seed: int,
    *,
    dim: int,
    init_position: Array | float = 0.0,
    num_warmup: int = 500,
    num_samples: int = 1000,
):
    """NumPyro NUTS mirror (potential = -logdensity) on a ``dim``-dimensional target."""
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(1)

    def potential_fn(theta):
        return -logdensity_fn(theta)

    kernel = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    initial_position = _broadcast_init_position(init_position, dim)
    mcmc.run(jr.PRNGKey(seed), init_params=initial_position)
    samples = mcmc.get_samples(group_by_chain=False)
    extra_fields = mcmc.get_extra_fields()
    return {
        "samples": samples,
        "extra_fields": extra_fields,
        "mcmc": mcmc,
    }


def run_walnuts(
    logdensity_fn,
    seed: int,
    *,
    dim: int,
    init_position: Array | float = 0.0,
    num_warmup: int = 500,
    num_samples: int = 1000,
    initial_step_size: float = 0.6,
    energy_threshold: float = 0.7,
    target_no_refinement_rate: float = 0.8,
    max_num_doublings: int = 8,
    max_num_micro_doublings: int = 8,
):
    """WALNUTS on a ``dim``-dimensional target, via the ``walnuts`` kernel in the
    ``DanWaxman/blackjax`` fork.

    Mirrors :func:`run_blackjax_nuts`: ``walnuts_adaptation`` performs warmup
    step-size adaptation (parallel to NUTS' ``window_adaptation``), then the tuned
    kernel is run in a scanned inference loop. ``energy_threshold`` is the
    within-step (micro) energy-error budget; ``initial_step_size`` only seeds the
    warmup, which adapts it. The metric is the identity (ideal for ``N(0, I_d)``).

    Note: ``walnuts_adaptation`` is run under JAX's default x32 precision -- its
    dual-averaging carry mixes float32/float64 and errors under ``jax_enable_x64``.
    """
    import blackjax

    warmup_key, sample_key = jr.split(jr.PRNGKey(seed))
    initial_position = _broadcast_init_position(init_position, dim)

    warmup = blackjax.walnuts_adaptation(
        logdensity_fn,
        inverse_mass_matrix=jnp.ones(dim),
        initial_step_size=initial_step_size,
        target_no_refinement_rate=target_no_refinement_rate,
        energy_threshold=energy_threshold,
        max_num_doublings=max_num_doublings,
        max_num_micro_doublings=max_num_micro_doublings,
    )
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_warmup)

    kernel = blackjax.walnuts(logdensity_fn, **parameters)

    @jax.jit
    def inference_loop(initial_state, rng_key):
        keys = jr.split(rng_key, num_samples)

        def one_step(current_state, current_key):
            next_state, info = kernel.step(current_key, current_state)
            return next_state, (next_state.position, info)

        _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)
        return positions, infos

    samples, infos = inference_loop(state, sample_key)
    return {
        "samples": samples,
        "infos": infos,
        "state": state,
        "parameters": parameters,
    }
