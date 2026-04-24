from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp


Array = jax.Array


@dataclasses.dataclass(frozen=True)
class DiscreteLTISystem:
    A: Array
    Q: Array
    H: Array
    R: Array
    B: Array
    D: Array
    initial_mean: Array
    initial_cov: Array


def make_lti_discrete_system(alpha: Array | float) -> DiscreteLTISystem:
    alpha = jnp.asarray(alpha)
    return make_lti_discrete_system_with_overrides(alpha)


def make_lti_discrete_system_with_overrides(
    alpha: Array | float,
    *,
    Q: Array | None = None,
    R: Array | None = None,
    B: Array | None = None,
    D: Array | None = None,
    initial_mean: Array | None = None,
    initial_cov: Array | None = None,
) -> DiscreteLTISystem:
    alpha = jnp.asarray(alpha)
    return DiscreteLTISystem(
        A=jnp.array([[alpha, 0.0], [0.0, 0.8]]),
        Q=0.1 * jnp.eye(2) if Q is None else jnp.asarray(Q),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.5**2]]) if R is None else jnp.asarray(R),
        B=jnp.array([[0.1], [0.0]]) if B is None else jnp.asarray(B),
        D=jnp.array([[0.01]]) if D is None else jnp.asarray(D),
        initial_mean=jnp.zeros(2) if initial_mean is None else jnp.asarray(initial_mean),
        initial_cov=jnp.eye(2) if initial_cov is None else jnp.asarray(initial_cov),
    )


def _control_values_or_zeros(length: int, control_dim: int, ctrl_values: Array | None) -> Array:
    if ctrl_values is None:
        return jnp.zeros((length, control_dim))
    return jnp.asarray(ctrl_values)


def _gaussian_logpdf(value: Array, mean: Array, cov: Array) -> Array:
    value = jnp.atleast_1d(value)
    mean = jnp.atleast_1d(mean)
    diff = value - mean
    chol = jnp.linalg.cholesky(cov)
    solved = jsp.linalg.solve_triangular(chol, diff, lower=True)
    maha = solved @ solved
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = value.shape[0]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + maha)


def simulate_lti_discrete_gaussian(
    key: Array,
    alpha: Array | float,
    times: Array,
    ctrl_values: Array | None = None,
    system_kwargs: dict | None = None,
) -> tuple[Array, Array, Array]:
    system = make_lti_discrete_system_with_overrides(alpha, **(system_kwargs or {}))
    times = jnp.asarray(times)
    ctrl_values = _control_values_or_zeros(times.shape[0], system.B.shape[1], ctrl_values)

    init_key, process_key, obs_key = jr.split(key, 3)
    x0 = jr.multivariate_normal(init_key, system.initial_mean, system.initial_cov)

    process_noise = jr.multivariate_normal(
        process_key,
        mean=jnp.zeros(system.A.shape[0]),
        cov=system.Q,
        shape=(max(times.shape[0] - 1, 0),),
    )

    def step(x_prev, inputs):
        u_prev, noise = inputs
        x_next = system.A @ x_prev + system.B @ u_prev + noise
        return x_next, x_next

    _, xs_tail = jax.lax.scan(
        step,
        x0,
        (ctrl_values[:-1], process_noise),
    )
    states = jnp.vstack([x0[None, :], xs_tail])

    obs_means = jax.vmap(lambda x, u: system.H @ x + system.D @ u)(states, ctrl_values)
    obs_noise = jr.multivariate_normal(
        obs_key,
        mean=jnp.zeros(system.H.shape[0]),
        cov=system.R,
        shape=(times.shape[0],),
    )
    observations = obs_means + obs_noise
    return times, states, observations[:, 0]


def kalman_filter_loglik(
    alpha: Array | float,
    obs_values: Array,
    ctrl_values: Array | None = None,
    system_kwargs: dict | None = None,
) -> tuple[Array, dict[str, Array]]:
    system = make_lti_discrete_system_with_overrides(alpha, **(system_kwargs or {}))
    observations = jnp.asarray(obs_values).reshape(-1, 1)
    ctrl_values = _control_values_or_zeros(observations.shape[0], system.B.shape[1], ctrl_values)

    y0 = observations[0]
    u0 = ctrl_values[0]
    innov_mean0 = system.H @ system.initial_mean + system.D @ u0
    innov_cov0 = system.H @ system.initial_cov @ system.H.T + system.R
    gain0 = system.initial_cov @ system.H.T @ jnp.linalg.inv(innov_cov0)
    mean0 = system.initial_mean + gain0 @ (y0 - innov_mean0)
    cov0 = system.initial_cov - gain0 @ innov_cov0 @ gain0.T
    loglik0 = _gaussian_logpdf(y0, innov_mean0, innov_cov0)

    def kf_step(carry, inputs):
        mean_prev, cov_prev, loglik_prev = carry
        u_prev, u_curr, y_curr = inputs
        mean_pred = system.A @ mean_prev + system.B @ u_prev
        cov_pred = system.A @ cov_prev @ system.A.T + system.Q
        innov_mean = system.H @ mean_pred + system.D @ u_curr
        innov_cov = system.H @ cov_pred @ system.H.T + system.R
        gain = cov_pred @ system.H.T @ jnp.linalg.inv(innov_cov)
        mean_filt = mean_pred + gain @ (y_curr - innov_mean)
        cov_filt = cov_pred - gain @ innov_cov @ gain.T
        loglik = loglik_prev + _gaussian_logpdf(y_curr, innov_mean, innov_cov)
        return (mean_filt, cov_filt, loglik), (mean_filt, cov_filt)

    (final_mean, final_cov, marginal_loglik), (means_tail, covs_tail) = jax.lax.scan(
        kf_step,
        (mean0, cov0, loglik0),
        (ctrl_values[:-1], ctrl_values[1:], observations[1:]),
    )
    filtered_means = jnp.vstack([mean0[None, :], means_tail])
    filtered_covs = jnp.concatenate([cov0[None, :, :], covs_tail], axis=0)
    return marginal_loglik, {
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs,
        "final_mean": final_mean,
        "final_covariance": final_cov,
    }


def particle_filter_loglik(
    alpha: Array | float,
    obs_values: Array,
    n_particles: int,
    key: Array,
    ctrl_values: Array | None = None,
    ess_threshold_ratio: float = 0.7,
    system_kwargs: dict | None = None,
) -> Array:
    system = make_lti_discrete_system_with_overrides(alpha, **(system_kwargs or {}))
    observations = jnp.asarray(obs_values).reshape(-1, 1)
    ctrl_values = _control_values_or_zeros(observations.shape[0], system.B.shape[1], ctrl_values)

    init_key, step_key_root = jr.split(key)
    step_keys = jr.split(step_key_root, observations.shape[0])
    particles = jr.multivariate_normal(
        init_key,
        mean=system.initial_mean,
        cov=system.initial_cov,
        shape=(n_particles,),
    )
    log_weights = jnp.full((n_particles,), -jnp.log(n_particles))
    loglik = jnp.asarray(0.0)

    def normalize_log_weights(log_w):
        log_norm = jsp.special.logsumexp(log_w)
        return log_w - log_norm, log_norm

    def effective_sample_size(log_w):
        return jnp.exp(-jsp.special.logsumexp(2.0 * log_w))

    def stop_gradient_resample(resample_key, particles_in, log_weights_in):
        indices = jr.choice(
            resample_key,
            n_particles,
            shape=(n_particles,),
            p=jnp.exp(log_weights_in),
            replace=True,
        )
        resampled_particles = particles_in[indices]
        new_log_weights = jnp.full_like(log_weights_in, -jnp.log(n_particles))
        resampled_log_weights = log_weights_in[indices]
        log_weights_out = (
            new_log_weights
            + resampled_log_weights
            - jax.lax.stop_gradient(resampled_log_weights)
        )
        log_weights_out, _ = normalize_log_weights(log_weights_out)
        return resampled_particles, log_weights_out

    def pf_step(carry, inputs):
        particles_prev, log_weights_prev, loglik_prev = carry
        idx, y_curr, u_prev, u_curr, step_key = inputs
        noise_key, resample_key = jr.split(step_key)

        def do_predict(particles_in):
            noise = jr.multivariate_normal(
                noise_key,
                mean=jnp.zeros(system.A.shape[0]),
                cov=system.Q,
                shape=(n_particles,),
            )
            return (
                jax.vmap(lambda x: system.A @ x + system.B @ u_prev)(particles_in)
                + noise
            )

        particles_pred = jax.lax.cond(
            idx > 0,
            do_predict,
            lambda p: p,
            particles_prev,
        )
        emission_means = jax.vmap(lambda x: system.H @ x + system.D @ u_curr)(particles_pred)
        obs_loglik = jax.vmap(lambda mean: _gaussian_logpdf(y_curr, mean, system.R))(emission_means)
        log_weights_updated = log_weights_prev + obs_loglik
        log_weights_updated, log_norm = normalize_log_weights(log_weights_updated)
        loglik_new = loglik_prev + log_norm
        ess = effective_sample_size(log_weights_updated)

        particles_resampled, log_weights_resampled = jax.lax.cond(
            ess < (ess_threshold_ratio * n_particles),
            lambda args: stop_gradient_resample(*args),
            lambda args: (args[1], args[2]),
            (resample_key, particles_pred, log_weights_updated),
        )
        return (particles_resampled, log_weights_resampled, loglik_new), None

    (_, _, final_loglik), _ = jax.lax.scan(
        pf_step,
        (particles, log_weights, loglik),
        (
            jnp.arange(observations.shape[0]),
            observations,
            jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0),
            ctrl_values,
            step_keys,
        ),
    )
    return final_loglik


def make_blackjax_logdensity(
    obs_values: Array,
    *,
    filter_name: str,
    ctrl_values: Array | None = None,
    n_particles: int | None = None,
    fixed_key: Array | None = None,
    system_kwargs: dict | None = None,
):
    obs_values = jnp.asarray(obs_values)
    ctrl_values = None if ctrl_values is None else jnp.asarray(ctrl_values)
    if fixed_key is None:
        fixed_key = jr.PRNGKey(0)

    def logdensity_fn(alpha):
        alpha = jnp.asarray(alpha)
        log_prior = jnp.where(
            (alpha >= -0.7) & (alpha <= 0.7),
            -jnp.log(1.4),
            -jnp.inf,
        )

        if filter_name == "kf":
            loglik, _ = kalman_filter_loglik(
                alpha,
                obs_values=obs_values,
                ctrl_values=ctrl_values,
                system_kwargs=system_kwargs,
            )
        elif filter_name == "pf":
            if n_particles is None:
                raise ValueError("n_particles is required for PF log density.")
            loglik = particle_filter_loglik(
                alpha,
                obs_values=obs_values,
                n_particles=n_particles,
                key=fixed_key,
                ctrl_values=ctrl_values,
                system_kwargs=system_kwargs,
            )
        else:
            raise ValueError(f"Unknown filter_name: {filter_name!r}")

        return log_prior + loglik

    return logdensity_fn


def run_blackjax_nuts_1d(
    logdensity_fn,
    seed: int,
    *,
    init_position: float = 0.35,
    num_warmup: int = 50,
    num_samples: int = 50,
):
    import blackjax

    warmup_key, sample_key = jr.split(jr.PRNGKey(seed))
    initial_position = jnp.asarray(init_position)

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


def run_numpyro_nuts_1d(
    logdensity_fn,
    seed: int,
    *,
    init_position: float = 0.35,
    num_warmup: int = 50,
    num_samples: int = 50,
):
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(1)

    def potential_fn(alpha):
        return -logdensity_fn(alpha)

    kernel = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    initial_position = jnp.asarray(init_position)
    mcmc.run(jr.PRNGKey(seed), init_params=initial_position)
    samples = mcmc.get_samples(group_by_chain=False)
    extra_fields = mcmc.get_extra_fields()
    return {
        "samples": samples,
        "extra_fields": extra_fields,
        "mcmc": mcmc,
    }
