from __future__ import annotations

import dataclasses

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp


Array = jax.Array


@dataclasses.dataclass(frozen=True)
class LTISystem:
    A: Array
    L: Array
    H: Array
    R: Array
    B: Array
    initial_mean: Array
    initial_cov: Array


def make_lti_system(rho: Array | float) -> LTISystem:
    rho = jnp.asarray(rho)
    return LTISystem(
        A=jnp.array([[-1.0, 0.0], [rho, -1.0]]),
        L=jnp.eye(2),
        H=jnp.array([[0.0, 1.0]]),
        R=jnp.array([[1.0**2]]),
        B=jnp.array([[0.0], [10.0]]),
        initial_mean=jnp.zeros(2),
        initial_cov=jnp.eye(2),
    )


def _control_values_or_zeros(times: Array, control_dim: int, ctrl_values: Array | None) -> Array:
    if ctrl_values is None:
        return jnp.zeros((times.shape[0], control_dim))
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


def _position_dependent_key(base_key: Array, rho: Array) -> Array:
    folded = jnp.asarray(jnp.floor((rho + 10.0) * 1_000_000.0), dtype=jnp.uint32)
    return jr.fold_in(base_key, folded)


def discretize_lti_sde(system: LTISystem, dt: Array) -> tuple[Array, Array, Array]:
    state_dim = system.A.shape[0]
    ll_t = system.L @ system.L.T

    van_loan = jnp.block(
        [
            [system.A, ll_t],
            [jnp.zeros_like(system.A), -system.A.T],
        ]
    )
    exp_vl = jsp.linalg.expm(van_loan * dt)
    transition = exp_vl[:state_dim, :state_dim]
    cross = exp_vl[:state_dim, state_dim:]
    process_cov = cross @ transition.T
    process_cov = 0.5 * (process_cov + process_cov.T)

    control_block = jnp.block(
        [
            [system.A, system.B],
            [jnp.zeros((system.B.shape[1], state_dim + system.B.shape[1]))],
        ]
    )
    exp_control = jsp.linalg.expm(control_block * dt)
    control_gain = exp_control[:state_dim, state_dim:]
    return transition, process_cov, control_gain


def simulate_lti_gaussian_euler_maruyama(
    key: Array,
    rho: Array | float,
    predict_times: Array,
    ctrl_values: Array | None = None,
) -> tuple[Array, Array, Array]:
    system = make_lti_system(rho)
    times = jnp.asarray(predict_times)
    ctrl_values = _control_values_or_zeros(times, system.B.shape[1], ctrl_values)

    init_key, process_key, obs_key = jr.split(key, 3)
    initial_chol = jnp.linalg.cholesky(system.initial_cov)
    x0 = system.initial_mean + initial_chol @ jr.normal(init_key, shape=(system.A.shape[0],))

    dts = jnp.diff(times)
    process_eps = jr.normal(process_key, shape=(dts.shape[0], system.L.shape[1]))

    def em_step(x_prev, inputs):
        dt, noise, u_prev = inputs
        drift = system.A @ x_prev + system.B @ u_prev
        x_next = x_prev + drift * dt + system.L @ (jnp.sqrt(dt) * noise)
        return x_next, x_next

    _, xs_tail = jax.lax.scan(
        em_step,
        x0,
        (dts, process_eps, ctrl_values[:-1]),
    )
    states = jnp.vstack([x0[None, :], xs_tail])

    obs_noise = jr.normal(obs_key, shape=(times.shape[0], system.H.shape[0]))
    obs_chol = jnp.linalg.cholesky(system.R)
    means = jax.vmap(lambda x: system.H @ x)(states)
    observations = means + obs_noise @ obs_chol.T
    return times, states, observations[:, 0]


def kalman_filter_loglik(
    rho: Array | float,
    obs_times: Array,
    obs_values: Array,
    ctrl_values: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    system = make_lti_system(rho)
    times = jnp.asarray(obs_times)
    observations = jnp.asarray(obs_values).reshape(-1, 1)
    ctrl_values = _control_values_or_zeros(times, system.B.shape[1], ctrl_values)

    initial_obs_mean = system.H @ system.initial_mean
    initial_obs_cov = system.H @ system.initial_cov @ system.H.T + system.R
    initial_gain = system.initial_cov @ system.H.T @ jnp.linalg.inv(initial_obs_cov)
    initial_mean = system.initial_mean + initial_gain @ (observations[0] - initial_obs_mean)
    initial_cov = system.initial_cov - initial_gain @ initial_obs_cov @ initial_gain.T
    initial_loglik = _gaussian_logpdf(observations[0], initial_obs_mean, initial_obs_cov)

    dts = jnp.diff(times)

    def kf_step(carry, inputs):
        mean_prev, cov_prev, loglik_prev = carry
        dt, y_t, u_prev = inputs
        transition, process_cov, control_gain = discretize_lti_sde(system, dt)
        mean_pred = transition @ mean_prev + control_gain @ u_prev
        cov_pred = transition @ cov_prev @ transition.T + process_cov
        innov_mean = system.H @ mean_pred
        innov_cov = system.H @ cov_pred @ system.H.T + system.R
        gain = cov_pred @ system.H.T @ jnp.linalg.inv(innov_cov)
        mean_filt = mean_pred + gain @ (y_t - innov_mean)
        cov_filt = cov_pred - gain @ innov_cov @ gain.T
        loglik = loglik_prev + _gaussian_logpdf(y_t, innov_mean, innov_cov)
        return (mean_filt, cov_filt, loglik), (mean_filt, cov_filt)

    (final_mean, final_cov, marginal_loglik), (means_tail, covs_tail) = jax.lax.scan(
        kf_step,
        (initial_mean, initial_cov, initial_loglik),
        (dts, observations[1:], ctrl_values[:-1]),
    )
    filtered_means = jnp.vstack([initial_mean[None, :], means_tail])
    filtered_covs = jnp.concatenate([initial_cov[None, :, :], covs_tail], axis=0)
    return marginal_loglik, {
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covs,
        "final_mean": final_mean,
        "final_covariance": final_cov,
    }


def particle_filter_loglik(
    rho: Array | float,
    obs_times: Array,
    obs_values: Array,
    n_particles: int,
    key: Array,
    ctrl_values: Array | None = None,
    ess_threshold_ratio: float = 0.7,
) -> Array:
    system = make_lti_system(rho)
    times = jnp.asarray(obs_times)
    observations = jnp.asarray(obs_values).reshape(-1, 1)
    ctrl_values = _control_values_or_zeros(times, system.B.shape[1], ctrl_values)

    init_key, step_key_root = jr.split(key)
    step_keys = jr.split(step_key_root, times.shape[0])
    initial_chol = jnp.linalg.cholesky(system.initial_cov)
    particles = (
        system.initial_mean[None, :]
        + jr.normal(init_key, shape=(n_particles, system.A.shape[0])) @ initial_chol.T
    )
    log_weights = jnp.full((n_particles,), -jnp.log(n_particles))
    loglik = jnp.asarray(0.0)

    def propagate_particles_diffrax(particles_in, dt, u_prev, subkey):
        dt0 = jnp.asarray(0.01)
        tol_vbt = dt0 / 2.0

        def single_particle(y0, particle_key):
            drift = lambda t, y, args: system.A @ y + system.B @ u_prev
            diffusion_matrix = system.L
            terms = dfx.MultiTerm(
                dfx.ODETerm(drift),
                dfx.ControlTerm(
                    lambda t, y, args: diffusion_matrix,
                    dfx.VirtualBrownianTree(
                        t0=0.0,
                        t1=dt,
                        tol=tol_vbt,
                        shape=(system.L.shape[1],),
                        key=particle_key,
                    ),
                ),
            )
            sol = dfx.diffeqsolve(
                terms,
                solver=dfx.Heun(),
                stepsize_controller=dfx.ConstantStepSize(),
                adjoint=dfx.RecursiveCheckpointAdjoint(),
                t0=0.0,
                t1=dt,
                dt0=dt0,
                y0=y0,
                saveat=dfx.SaveAt(t1=True),
                max_steps=1_000,
            )
            return sol.ys[0]

        keys = jr.split(subkey, particles_in.shape[0])
        return jax.vmap(single_particle)(particles_in, keys)

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
        idx, y_t, t_curr, t_prev, u_prev, step_key = inputs
        resample_key, propagate_key = jr.split(step_key)

        def do_predict(particles_in):
            return propagate_particles_diffrax(
                particles_in,
                t_curr - t_prev,
                u_prev,
                propagate_key,
            )

        particles_pred = jax.lax.cond(
            idx > 0,
            do_predict,
            lambda p: p,
            particles_prev,
        )
        emission_means = jax.vmap(lambda x: system.H @ x)(particles_pred)
        obs_loglik = jax.vmap(
            lambda mean: _gaussian_logpdf(y_t, mean, system.R)
        )(emission_means)
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

    (final_particles, final_log_weights, final_loglik), _ = jax.lax.scan(
        pf_step,
        (particles, log_weights, loglik),
        (
            jnp.arange(times.shape[0]),
            observations,
            times,
            jnp.concatenate([times[:1], times[:-1]]),
            jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0),
            step_keys,
        ),
    )
    del final_particles, final_log_weights
    return final_loglik


def make_blackjax_logdensity(
    obs_times: Array,
    obs_values: Array,
    *,
    filter_name: str,
    n_particles: int | None = None,
    fixed_key: Array | None = None,
    key_mode: str = "deterministic",
    ctrl_values: Array | None = None,
) -> callable:
    obs_times = jnp.asarray(obs_times)
    obs_values = jnp.asarray(obs_values)
    if fixed_key is None:
        fixed_key = jr.PRNGKey(0)

    def logdensity_fn(rho):
        rho = jnp.asarray(rho)
        log_prior = jnp.where(
            (rho >= 0.0) & (rho <= 5.0),
            -jnp.log(5.0),
            -jnp.inf,
        )

        if filter_name == "kf":
            loglik, _ = kalman_filter_loglik(
                rho,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_values=ctrl_values,
            )
        elif filter_name == "pf":
            if n_particles is None:
                raise ValueError("n_particles is required for PF log density.")
            if key_mode == "fixed":
                key = fixed_key
            elif key_mode == "hashed_position":
                key = _position_dependent_key(fixed_key, rho)
            else:
                raise ValueError(f"Unknown key_mode: {key_mode!r}")
            loglik = particle_filter_loglik(
                rho,
                obs_times=obs_times,
                obs_values=obs_values,
                n_particles=n_particles,
                key=key,
                ctrl_values=ctrl_values,
            )
        else:
            raise ValueError(f"Unknown filter_name: {filter_name!r}")

        return log_prior + loglik

    return logdensity_fn


def run_blackjax_nuts_1d(
    logdensity_fn,
    seed: int,
    *,
    init_position: float = 2.5,
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
