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


def run_walnuts_1d(
    logdensity_fn,
    seed: int,
    *,
    init_position: float = 0.35,
    num_warmup: int = 50,
    num_samples: int = 50,
    step_size: float = 0.05,
    delta: float = 0.5,
    i_max: int = 8,
    max_ell: int = 8,
):
    raw_logdensity_fn = logdensity_fn
    logdensity_fn = jax.jit(lambda alpha: raw_logdensity_fn(alpha))
    grad_logdensity_fn = jax.jit(jax.grad(lambda alpha: raw_logdensity_fn(alpha)))
    max_levels = i_max + 1

    def leapfrog(theta: Array, rho: Array, h: Array, n_steps: int) -> tuple[Array, Array]:
        grad0 = grad_logdensity_fn(theta)

        def body_fn(_, carry):
            theta_curr, rho_curr, grad_curr = carry
            rho_half = rho_curr + 0.5 * h * grad_curr
            theta_new = theta_curr + h * rho_half
            grad_new = grad_logdensity_fn(theta_new)
            rho_new = rho_half + 0.5 * h * grad_new
            return theta_new, rho_new, grad_new

        theta_out, rho_out, _ = jax.lax.fori_loop(0, n_steps, body_fn, (theta, rho, grad0))
        return theta_out, rho_out

    def hamiltonian(theta: Array, rho: Array) -> Array:
        return -logdensity_fn(theta) + 0.5 * rho * rho

    def logsumexp2(a: Array, b: Array) -> Array:
        m = jnp.maximum(a, b)
        both_finite = jnp.isfinite(a) & jnp.isfinite(b)
        return jnp.where(
            both_finite,
            m + jnp.log(jnp.exp(a - m) + jnp.exp(b - m)),
            jnp.where(jnp.isfinite(a), a, b),
        )

    def is_u_turn(l_theta: Array, l_rho: Array, r_theta: Array, r_rho: Array) -> Array:
        dtheta = r_theta - l_theta
        return (l_rho * dtheta < 0.0) | (r_rho * dtheta < 0.0)

    def p_micro_pmf(j: Array, i: Array) -> Array:
        return jnp.where(
            j == i,
            2.0 / 3.0,
            jnp.where(j == i + 1, 1.0 / 3.0, 0.0),
        )

    def micro(theta_0: Array, rho_0: Array, h_macro: Array, local_delta: Array) -> Array:
        def cond_fn(carry):
            ell, result, done = carry
            return (ell <= max_ell) & (~done)

        def body_fn(carry):
            ell, result, done = carry
            n_substeps = 1 << ell
            h = h_macro / n_substeps
            energy0 = hamiltonian(theta_0, rho_0)

            def step_fn(_, state):
                theta_curr, rho_curr, energy_max, energy_min, bad = state

                def active_fn(active_state):
                    theta_a, rho_a, energy_max_a, energy_min_a, _bad_a = active_state
                    theta_new, rho_new = leapfrog(theta_a, rho_a, h, 1)
                    energy_new = hamiltonian(theta_new, rho_new)
                    energy_max_new = jnp.maximum(energy_max_a, energy_new)
                    energy_min_new = jnp.minimum(energy_min_a, energy_new)
                    bad_new = (energy_max_new - energy_min_new) > local_delta
                    return theta_new, rho_new, energy_max_new, energy_min_new, bad_new

                return jax.lax.cond(
                    bad,
                    lambda s: s,
                    active_fn,
                    state,
                )

            _, _, _, _, bad = jax.lax.fori_loop(
                0,
                n_substeps,
                step_fn,
                (theta_0, rho_0, energy0, energy0, jnp.array(False)),
            )
            success = ~bad
            return ell + 1, jnp.where(success, ell, result), done | success

        ell0 = jnp.array(0, dtype=jnp.int32)
        result0 = jnp.array(max_ell + 1, dtype=jnp.int32)
        done0 = jnp.array(False)
        _, result, _ = jax.lax.while_loop(cond_fn, body_fn, (ell0, result0, done0))
        return result

    def sample_bernoulli_choice(key: Array) -> Array:
        return jnp.where(jr.bernoulli(key, p=2.0 / 3.0), 0, 1)

    def build_leaf(
        key: Array,
        theta: Array,
        rho: Array,
        logw_start: Array,
        direction: Array,
    ) -> tuple[Array, Array, Array, Array]:
        energy_old = hamiltonian(theta, rho)
        sample_key, next_key = jr.split(key)

        def forward_fn(_):
            ell_f = micro(theta, rho, step_size, delta)
            ell = ell_f + sample_bernoulli_choice(sample_key)
            n_substeps = 1 << ell
            theta_1, rho_1 = leapfrog(theta, rho, step_size / n_substeps, n_substeps)
            ell_b = micro(theta_1, -rho_1, step_size, delta)
            return theta_1, rho_1, ell, ell_f, ell_b

        def backward_fn(_):
            rho_neg = -rho
            ell_b = micro(theta, rho_neg, step_size, delta)
            ell = ell_b + sample_bernoulli_choice(sample_key)
            n_substeps = 1 << ell
            theta_1, rho_1_neg = leapfrog(theta, rho_neg, step_size / n_substeps, n_substeps)
            rho_1 = -rho_1_neg
            ell_f = micro(theta_1, rho_1, step_size, delta)
            return theta_1, rho_1, ell, ell_b, ell_f

        theta_1, rho_1, ell, base_ell, reverse_ell = jax.lax.cond(
            direction == 1,
            forward_fn,
            backward_fn,
            operand=None,
        )
        p_forward = p_micro_pmf(ell, base_ell)
        p_target = p_micro_pmf(ell, reverse_ell)
        energy_new = hamiltonian(theta_1, rho_1)

        def finite_weight_fn(_):
            return logw_start + (energy_old - energy_new) + jnp.log(p_target) - jnp.log(p_forward)

        logw_new = jax.lax.cond(
            (p_target > 0.0) & (p_forward > 0.0),
            finite_weight_fn,
            lambda _: -jnp.inf,
            operand=None,
        )
        return theta_1, rho_1, logw_new, next_key

    def barker_pick_right(key: Array, logt_left: Array, logt_right: Array) -> Array:
        log_total = logsumexp2(logt_left, logt_right)
        prob_right = jnp.where(jnp.isfinite(log_total), jnp.exp(logt_right - log_total), 0.0)
        return jr.bernoulli(key, p=prob_right)

    def build_subtree(
        key: Array,
        theta_start: Array,
        rho_start: Array,
        logw_start: Array,
        direction: Array,
        depth: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        stk_l_th0 = jnp.zeros((max_levels,))
        stk_l_rh0 = jnp.zeros((max_levels,))
        stk_r_th0 = jnp.zeros((max_levels,))
        stk_r_rh0 = jnp.zeros((max_levels,))
        stk_se_th0 = jnp.zeros((max_levels,))
        stk_se_rh0 = jnp.zeros((max_levels,))
        stk_logt0 = jnp.full((max_levels,), -jnp.inf)
        stk_valid0 = jnp.zeros((max_levels,), dtype=bool)

        n_leaves = 1 << depth

        def leaf_body(_, carry):
            (
                theta_curr,
                rho_curr,
                logw_curr,
                stk_l_th,
                stk_l_rh,
                stk_r_th,
                stk_r_rh,
                stk_se_th,
                stk_se_rh,
                stk_logt,
                stk_valid,
                root_l_th,
                root_l_rh,
                root_r_th,
                root_r_rh,
                root_se_th,
                root_se_rh,
                root_logt,
                root_loge,
                subtree_uturn,
                loop_key,
            ) = carry

            def active_leaf_fn(active_carry):
                (
                    theta_curr_a,
                    rho_curr_a,
                    logw_curr_a,
                    stk_l_th_a,
                    stk_l_rh_a,
                    stk_r_th_a,
                    stk_r_rh_a,
                    stk_se_th_a,
                    stk_se_rh_a,
                    stk_logt_a,
                    stk_valid_a,
                    root_l_th_a,
                    root_l_rh_a,
                    root_r_th_a,
                    root_r_rh_a,
                    root_se_th_a,
                    root_se_rh_a,
                    root_logt_a,
                    root_loge_a,
                    subtree_uturn_a,
                    loop_key_a,
                ) = active_carry

                leaf_key, merge_seed_key, next_loop_key = jr.split(loop_key_a, 3)
                theta_1, rho_1, logw_new, _ = build_leaf(
                    leaf_key,
                    theta_curr_a,
                    rho_curr_a,
                    logw_curr_a,
                    direction,
                )

                merge_init = (
                    jnp.array(0, dtype=jnp.int32),
                    theta_1,
                    rho_1,
                    theta_1,
                    rho_1,
                    theta_1,
                    rho_1,
                    logw_new,
                    stk_l_th_a,
                    stk_l_rh_a,
                    stk_r_th_a,
                    stk_r_rh_a,
                    stk_se_th_a,
                    stk_se_rh_a,
                    stk_logt_a,
                    stk_valid_a,
                    jnp.array(False),
                    root_l_th_a,
                    root_l_rh_a,
                    root_r_th_a,
                    root_r_rh_a,
                    root_se_th_a,
                    root_se_rh_a,
                    root_logt_a,
                    logw_new,
                    merge_seed_key,
                )

                def merge_cond_fn(merge_carry):
                    j = merge_carry[0]
                    stk_valid_m = merge_carry[15]
                    merge_done = merge_carry[16]
                    return (j < depth) & stk_valid_m[j] & (~merge_done)

                def merge_body_fn(merge_carry):
                    (
                        j,
                        cur_l_th,
                        cur_l_rh,
                        cur_r_th,
                        cur_r_rh,
                        cur_se_th,
                        cur_se_rh,
                        cur_logt,
                        stk_l_th_m,
                        stk_l_rh_m,
                        stk_r_th_m,
                        stk_r_rh_m,
                        stk_se_th_m,
                        stk_se_rh_m,
                        stk_logt_m,
                        stk_valid_m,
                        merge_done,
                        root_l_th_m,
                        root_l_rh_m,
                        root_r_th_m,
                        root_r_rh_m,
                        root_se_th_m,
                        root_se_rh_m,
                        root_logt_m,
                        root_loge_m,
                        merge_key_m,
                    ) = merge_carry

                    merged_l_th = jnp.where(direction == 1, stk_l_th_m[j], cur_l_th)
                    merged_l_rh = jnp.where(direction == 1, stk_l_rh_m[j], cur_l_rh)
                    merged_r_th = jnp.where(direction == 1, cur_r_th, stk_r_th_m[j])
                    merged_r_rh = jnp.where(direction == 1, cur_r_rh, stk_r_rh_m[j])

                    uturn_here = is_u_turn(merged_l_th, merged_l_rh, merged_r_th, merged_r_rh)
                    pick_key, next_merge_key = jr.split(merge_key_m)
                    pick_right = barker_pick_right(pick_key, stk_logt_m[j], cur_logt)
                    new_se_th = jnp.where(pick_right, cur_se_th, stk_se_th_m[j])
                    new_se_rh = jnp.where(pick_right, cur_se_rh, stk_se_rh_m[j])
                    merged_logt = logsumexp2(stk_logt_m[j], cur_logt)
                    stk_valid_new = stk_valid_m.at[j].set(False)

                    return (
                        j + 1,
                        merged_l_th,
                        merged_l_rh,
                        merged_r_th,
                        merged_r_rh,
                        new_se_th,
                        new_se_rh,
                        merged_logt,
                        stk_l_th_m,
                        stk_l_rh_m,
                        stk_r_th_m,
                        stk_r_rh_m,
                        stk_se_th_m,
                        stk_se_rh_m,
                        stk_logt_m,
                        stk_valid_new,
                        uturn_here,
                        jnp.where(uturn_here, merged_l_th, root_l_th_m),
                        jnp.where(uturn_here, merged_l_rh, root_l_rh_m),
                        jnp.where(uturn_here, merged_r_th, root_r_th_m),
                        jnp.where(uturn_here, merged_r_rh, root_r_rh_m),
                        jnp.where(uturn_here, new_se_th, root_se_th_m),
                        jnp.where(uturn_here, new_se_rh, root_se_rh_m),
                        jnp.where(uturn_here, merged_logt, root_logt_m),
                        root_loge_m,
                        next_merge_key,
                    )

                merge_final = jax.lax.while_loop(merge_cond_fn, merge_body_fn, merge_init)
                (
                    j_final,
                    cur_l_th,
                    cur_l_rh,
                    cur_r_th,
                    cur_r_rh,
                    cur_se_th,
                    cur_se_rh,
                    cur_logt,
                    stk_l_th_a,
                    stk_l_rh_a,
                    stk_r_th_a,
                    stk_r_rh_a,
                    stk_se_th_a,
                    stk_se_rh_a,
                    stk_logt_a,
                    stk_valid_a,
                    merge_done,
                    root_l_th_a,
                    root_l_rh_a,
                    root_r_th_a,
                    root_r_rh_a,
                    root_se_th_a,
                    root_se_rh_a,
                    root_logt_a,
                    root_loge_a,
                    _,
                ) = merge_final

                def park_fn(_):
                    return (
                        stk_l_th_a.at[j_final].set(cur_l_th),
                        stk_l_rh_a.at[j_final].set(cur_l_rh),
                        stk_r_th_a.at[j_final].set(cur_r_th),
                        stk_r_rh_a.at[j_final].set(cur_r_rh),
                        stk_se_th_a.at[j_final].set(cur_se_th),
                        stk_se_rh_a.at[j_final].set(cur_se_rh),
                        stk_logt_a.at[j_final].set(cur_logt),
                        stk_valid_a.at[j_final].set(True),
                    )

                def no_park_fn(_):
                    return (
                        stk_l_th_a,
                        stk_l_rh_a,
                        stk_r_th_a,
                        stk_r_rh_a,
                        stk_se_th_a,
                        stk_se_rh_a,
                        stk_logt_a,
                        stk_valid_a,
                    )

                (
                    stk_l_th_a,
                    stk_l_rh_a,
                    stk_r_th_a,
                    stk_r_rh_a,
                    stk_se_th_a,
                    stk_se_rh_a,
                    stk_logt_a,
                    stk_valid_a,
                ) = jax.lax.cond(merge_done, no_park_fn, park_fn, operand=None)

                return (
                    theta_1,
                    rho_1,
                    logw_new,
                    stk_l_th_a,
                    stk_l_rh_a,
                    stk_r_th_a,
                    stk_r_rh_a,
                    stk_se_th_a,
                    stk_se_rh_a,
                    stk_logt_a,
                    stk_valid_a,
                    root_l_th_a,
                    root_l_rh_a,
                    root_r_th_a,
                    root_r_rh_a,
                    root_se_th_a,
                    root_se_rh_a,
                    root_logt_a,
                    root_loge_a,
                    merge_done,
                    next_loop_key,
                )

            return jax.lax.cond(subtree_uturn, lambda c: c, active_leaf_fn, carry)

        init_carry = (
            theta_start,
            rho_start,
            logw_start,
            stk_l_th0,
            stk_l_rh0,
            stk_r_th0,
            stk_r_rh0,
            stk_se_th0,
            stk_se_rh0,
            stk_logt0,
            stk_valid0,
            theta_start,
            rho_start,
            theta_start,
            rho_start,
            theta_start,
            rho_start,
            -jnp.inf,
            logw_start,
            jnp.array(False),
            key,
        )
        final_carry = jax.lax.fori_loop(0, n_leaves, leaf_body, init_carry)
        (
            _theta_curr,
            _rho_curr,
            logw_curr,
            stk_l_th,
            stk_l_rh,
            stk_r_th,
            stk_r_rh,
            stk_se_th,
            stk_se_rh,
            stk_logt,
            _stk_valid,
            root_l_th,
            root_l_rh,
            root_r_th,
            root_r_rh,
            root_se_th,
            root_se_rh,
            root_logt,
            root_loge,
            subtree_uturn,
            next_key,
        ) = final_carry

        return (
            jnp.where(subtree_uturn, root_l_th, stk_l_th[depth]),
            jnp.where(subtree_uturn, root_l_rh, stk_l_rh[depth]),
            jnp.where(subtree_uturn, root_r_th, stk_r_th[depth]),
            jnp.where(subtree_uturn, root_r_rh, stk_r_rh[depth]),
            jnp.where(subtree_uturn, root_se_th, stk_se_th[depth]),
            jnp.where(subtree_uturn, root_se_rh, stk_se_rh[depth]),
            jnp.where(subtree_uturn, root_logt, stk_logt[depth]),
            jnp.where(subtree_uturn, root_loge, logw_curr),
            subtree_uturn,
            next_key,
        )

    def walnuts_transition(theta: Array, key: Array) -> tuple[Array, Array, Array, Array]:
        rho_key, loop_key = jr.split(key)
        rho = jr.normal(rho_key)
        logw_0 = logdensity_fn(theta) - 0.5 * rho * rho

        init_carry = (
            theta,
            rho,
            theta,
            rho,
            theta,
            logw_0,
            logw_0,
            logw_0,
            jnp.array(1, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            loop_key,
        )

        def outer_body(i, carry):
            (
                glob_l_th,
                glob_l_rh,
                glob_r_th,
                glob_r_rh,
                theta_tilde,
                glob_logt,
                glob_loge_fwd,
                glob_loge_bwd,
                n_leaves,
                depth_reached,
                done,
                outer_key,
            ) = carry

            def active_outer_fn(active_carry):
                (
                    glob_l_th_a,
                    glob_l_rh_a,
                    glob_r_th_a,
                    glob_r_rh_a,
                    theta_tilde_a,
                    glob_logt_a,
                    glob_loge_fwd_a,
                    glob_loge_bwd_a,
                    n_leaves_a,
                    _depth_reached_a,
                    _done_a,
                    outer_key_a,
                ) = active_carry

                dir_key, subtree_key, accept_key, next_outer_key = jr.split(outer_key_a, 4)
                direction = jnp.where(jr.bernoulli(dir_key), 1, -1)

                subtree_args = (
                    jnp.where(direction == 1, glob_r_th_a, glob_l_th_a),
                    jnp.where(direction == 1, glob_r_rh_a, glob_l_rh_a),
                    jnp.where(direction == 1, glob_loge_fwd_a, glob_loge_bwd_a),
                )

                e_l_th, e_l_rh, e_r_th, e_r_rh, e_se_th, _e_se_rh, e_logt, e_loge, e_uturn, _ = build_subtree(
                    subtree_key,
                    subtree_args[0],
                    subtree_args[1],
                    subtree_args[2],
                    direction,
                    jnp.asarray(i, dtype=jnp.int32),
                )

                accept_swap = jnp.log(jr.uniform(accept_key)) <= (e_logt - glob_logt_a)
                theta_tilde_new = jnp.where(accept_swap, e_se_th, theta_tilde_a)
                glob_logt_new = logsumexp2(glob_logt_a, e_logt)
                glob_r_th_new = jnp.where(direction == 1, e_r_th, glob_r_th_a)
                glob_r_rh_new = jnp.where(direction == 1, e_r_rh, glob_r_rh_a)
                glob_l_th_new = jnp.where(direction == -1, e_l_th, glob_l_th_a)
                glob_l_rh_new = jnp.where(direction == -1, e_l_rh, glob_l_rh_a)
                glob_loge_fwd_new = jnp.where(direction == 1, e_loge, glob_loge_fwd_a)
                glob_loge_bwd_new = jnp.where(direction == -1, e_loge, glob_loge_bwd_a)
                global_uturn = is_u_turn(glob_l_th_new, glob_l_rh_new, glob_r_th_new, glob_r_rh_new)
                done_new = e_uturn | global_uturn

                return (
                    glob_l_th_new,
                    glob_l_rh_new,
                    glob_r_th_new,
                    glob_r_rh_new,
                    theta_tilde_new,
                    glob_logt_new,
                    glob_loge_fwd_new,
                    glob_loge_bwd_new,
                    n_leaves_a + (1 << i),
                    jnp.asarray(i + 1, dtype=jnp.int32),
                    done_new,
                    next_outer_key,
                )

            return jax.lax.cond(done, lambda c: c, active_outer_fn, carry)

        final_carry = jax.lax.fori_loop(0, i_max, outer_body, init_carry)
        (
            _glob_l_th,
            _glob_l_rh,
            _glob_r_th,
            _glob_r_rh,
            theta_tilde,
            _glob_logt,
            _glob_loge_fwd,
            _glob_loge_bwd,
            n_leaves,
            depth_reached,
            _done,
            next_key,
        ) = final_carry
        return theta_tilde, depth_reached, n_leaves, next_key

    @jax.jit
    def inference_loop(initial_theta: Array, rng_key: Array):
        n_total = num_warmup + num_samples
        keys = jr.split(rng_key, n_total)

        def one_step(theta_curr, key_curr):
            theta_next, depth, orbit_size, _ = walnuts_transition(theta_curr, key_curr)
            return theta_next, (theta_next, depth, orbit_size)

        _, (all_samples, depths, orbit_sizes) = jax.lax.scan(one_step, initial_theta, keys)
        return all_samples, depths, orbit_sizes

    initial_theta = jnp.asarray(init_position)
    all_samples, depths, orbit_sizes = inference_loop(initial_theta, jr.PRNGKey(seed))
    return {
        "samples": all_samples[num_warmup:],
        "warmup_samples": all_samples[:num_warmup],
        "depths": depths[num_warmup:],
        "orbit_sizes": orbit_sizes[num_warmup:],
        "step_size": step_size,
        "delta": delta,
        "i_max": i_max,
        "max_ell": max_ell,
    }
