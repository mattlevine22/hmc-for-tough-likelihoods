"""Process-isolated, kill-based timeout wrapper for the combo samplers.

Some (primal, gradient) combinations make inference collapse: the sampler builds
enormous trajectories so a single warmup/sampling step can run for minutes inside an
uninterruptible JAX call. A JAX ``lax.scan`` (or even one ``kernel.step``) cannot be
interrupted from Python, so we run each method in a **separate spawned process**, stream
each post-warmup sample into a ``multiprocessing.Manager`` list, and hard-kill the process
on timeout. Whatever samples were produced before the kill survive in the manager list.

Contract (matches the notebook's requirement):
  - timeout reached after warmup -> return the samples collected so far;
  - timeout reached during warmup (no post-warmup sample produced) -> return ``None``
    (the caller turns this into NaN).
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

# WALNUTS-adaptation defaults (mirror combo.run_walnuts_1d); overridden by sampler_kwargs.
_WALNUTS_DEFAULTS = dict(
    initial_step_size=0.1,
    target_no_refinement_rate=0.8,
    energy_threshold=0.7,
    max_num_doublings=8,
    max_num_micro_doublings=8,
)


def _worker(shared, flag, spec):
    """Child entry point. Rebuilds the combo log-density from picklable ``spec``, runs
    warmup, then appends one scalar ``alpha`` sample per step to ``shared``. Sets
    ``flag['warmup_done']`` once warmup returns and ``flag['done']`` on clean finish."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    import blackjax

    from manual_combo_inference.combo import make_combo_logdensity

    obs_values = jnp.asarray(spec["obs_values"])
    ctrl_values = None if spec["ctrl_values"] is None else jnp.asarray(spec["ctrl_values"])
    system_kwargs = (
        None if spec["system_kwargs"] is None
        else {k: jnp.asarray(v) for k, v in spec["system_kwargs"].items()}
    )
    fixed_key = jr.PRNGKey(0) if spec["fixed_key"] is None else jnp.asarray(spec["fixed_key"])

    logdensity_fn = make_combo_logdensity(
        obs_values,
        primal_source=spec["primal"],
        grad_source=spec["grad"],
        ctrl_values=ctrl_values,
        n_lofi=spec["n_lofi"],
        n_hifi=spec["n_hifi"],
        fixed_key=fixed_key,
        system_kwargs=system_kwargs,
    )

    warmup_key, sample_key = jr.split(jr.PRNGKey(spec["seed"]))
    num_warmup = spec["num_warmup"]
    num_samples = spec["num_samples"]
    init_position = spec["init_position"]

    if spec["sampler"] == "nuts":
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn, progress_bar=False)
        (state, parameters), _ = warmup.run(
            warmup_key, jnp.asarray(init_position), num_steps=num_warmup
        )
        flag["warmup_done"] = True
        kernel = blackjax.nuts(logdensity_fn, **parameters)
        step = jax.jit(kernel.step)
        keys = jr.split(sample_key, num_samples)
        for idx in range(num_samples):
            state, _ = step(keys[idx], state)
            shared.append(float(state.position))
    elif spec["sampler"] == "walnuts":
        def logdensity_vec(theta):
            return logdensity_fn(theta[0])

        wkw = dict(_WALNUTS_DEFAULTS)
        wkw.update(spec["sampler_kwargs"] or {})
        warmup = blackjax.walnuts_adaptation(
            logdensity_vec, inverse_mass_matrix=jnp.ones(1), **wkw
        )
        (state, parameters), _ = warmup.run(
            warmup_key, jnp.asarray([init_position], dtype=float), num_warmup
        )
        flag["warmup_done"] = True
        kernel = blackjax.walnuts(logdensity_vec, **parameters)
        step = jax.jit(kernel.step)
        keys = jr.split(sample_key, num_samples)
        for idx in range(num_samples):
            state, _ = step(keys[idx], state)
            shared.append(float(state.position[0]))
    else:
        raise ValueError(f"Unknown sampler: {spec['sampler']!r}")

    flag["done"] = True


def run_combo_with_timeout(
    sampler_name: str,
    primal: str,
    grad: str,
    *,
    obs_values,
    ctrl_values=None,
    system_kwargs: dict | None = None,
    n_lofi: int = 100,
    n_hifi: int = 1000,
    fixed_key=None,
    seed: int = 0,
    init_position: float = 0.35,
    num_warmup: int = 100,
    num_samples: int = 200,
    sampler_kwargs: dict | None = None,
    timeout_s: float = 180.0,
):
    """Run one (primal, gradient) combination under a hard wall-clock ``timeout_s``.

    Returns a dict with ``samples`` (1-D float ``np.ndarray`` of collected post-warmup
    alpha draws, or ``None`` if warmup did not finish in time), ``warmup_done``,
    ``done`` (clean finish), ``timed_out``, ``elapsed`` (seconds), and ``n_samples``.
    """
    spec = {
        "sampler": sampler_name,
        "primal": primal,
        "grad": grad,
        "obs_values": np.asarray(obs_values),
        "ctrl_values": None if ctrl_values is None else np.asarray(ctrl_values),
        "system_kwargs": (
            None if system_kwargs is None
            else {k: np.asarray(v) for k, v in system_kwargs.items()}
        ),
        "n_lofi": int(n_lofi),
        "n_hifi": int(n_hifi),
        "fixed_key": None if fixed_key is None else np.asarray(fixed_key),
        "seed": int(seed),
        "init_position": float(init_position),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "sampler_kwargs": dict(sampler_kwargs or {}),
    }

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    shared = manager.list()
    flag = manager.dict(warmup_done=False, done=False)
    proc = ctx.Process(target=_worker, args=(shared, flag, spec), daemon=True)

    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout_s)
    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join(10.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
    elapsed = time.perf_counter() - t0

    samples_list = list(shared)
    warmup_done = bool(flag.get("warmup_done", False))
    done = bool(flag.get("done", False))
    manager.shutdown()

    samples = (
        np.asarray(samples_list, dtype=float)
        if (warmup_done and len(samples_list) > 0)
        else None
    )
    return {
        "samples": samples,
        "warmup_done": warmup_done,
        "done": done,
        "timed_out": timed_out,
        "elapsed": elapsed,
        "n_samples": len(samples_list),
    }
