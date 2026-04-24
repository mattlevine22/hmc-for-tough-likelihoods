from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _as_numpy(array_like):
    return np.asarray(array_like, dtype=float)


def plot_lti_synthetic_data(times, states, observations):
    times = _as_numpy(times)
    states = _as_numpy(states)
    observations = _as_numpy(observations)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(times, states[:, 0], color="C0", linewidth=2.0)
    axes[0].set_ylabel("state[0]")
    axes[0].set_title("Synthetic latent states and observations")
    axes[0].grid(alpha=0.2)

    axes[1].plot(times, states[:, 1], color="C1", linewidth=2.0)
    axes[1].set_ylabel("state[1]")
    axes[1].grid(alpha=0.2)

    axes[2].plot(times, observations, color="C2", linewidth=1.8, label="observation")
    axes[2].scatter(times, observations, color="C2", s=12, alpha=0.7)
    axes[2].set_ylabel("y")
    axes[2].set_xlabel("time")
    axes[2].grid(alpha=0.2)

    return fig, axes


def plot_lti_filter_comparison(
    *,
    parameter_grid,
    profile_band_q05,
    profile_band_q50,
    profile_band_q95,
    grad_band_q05,
    grad_band_q50,
    grad_band_q95,
    grad_numeric_q50,
    comparison_curves: Sequence[dict],
    parameter_name: str = "alpha",
    base_profile_label: str = "PF profile band",
    base_grad_label: str = "PF autodiff band",
    base_numeric_grad_label: str = "PF numerical grad",
):
    parameter_grid = _as_numpy(parameter_grid)
    profile_band_q05 = _as_numpy(profile_band_q05)
    profile_band_q50 = _as_numpy(profile_band_q50)
    profile_band_q95 = _as_numpy(profile_band_q95)
    grad_band_q05 = _as_numpy(grad_band_q05)
    grad_band_q50 = _as_numpy(grad_band_q50)
    grad_band_q95 = _as_numpy(grad_band_q95)
    grad_numeric_q50 = _as_numpy(grad_numeric_q50)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    profile_ax, grad_ax = axes

    profile_ax.fill_between(
        parameter_grid,
        profile_band_q05,
        profile_band_q95,
        color="C1",
        alpha=0.18,
        label=base_profile_label,
    )
    profile_ax.plot(
        parameter_grid,
        profile_band_q50,
        color="C1",
        linewidth=2.0,
        linestyle="--",
        label=f"{base_profile_label} median",
    )

    grad_ax.fill_between(
        parameter_grid,
        grad_band_q05,
        grad_band_q95,
        color="C1",
        alpha=0.18,
        label=base_grad_label,
    )
    grad_ax.plot(
        parameter_grid,
        grad_band_q50,
        color="C1",
        linewidth=2.0,
        linestyle="--",
        label=f"{base_grad_label} median",
    )
    grad_ax.plot(
        parameter_grid,
        grad_numeric_q50,
        color="0.25",
        linewidth=1.8,
        linestyle=":",
        label=base_numeric_grad_label,
    )

    for curve in comparison_curves:
        color = curve.get("color", "C0")
        linestyle = curve.get("linestyle", "-")
        alpha = curve.get("alpha", 1.0)
        linewidth = curve.get("linewidth", 2.0)
        label = curve["label"]

        profile_ax.plot(
            parameter_grid,
            _as_numpy(curve["profile"]),
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )
        grad_ax.plot(
            parameter_grid,
            _as_numpy(curve["grad"]),
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )

    profile_ax.set_ylabel("log marginal likelihood")
    profile_ax.set_title("Profile likelihood comparison")
    profile_ax.grid(alpha=0.2)
    profile_ax.legend(loc="best")

    grad_ax.set_xlabel(parameter_name)
    grad_ax.set_ylabel(f"d/d{parameter_name} log likelihood")
    grad_ax.set_title("Gradient comparison")
    grad_ax.grid(alpha=0.2)
    grad_ax.legend(loc="best")

    return fig, axes
