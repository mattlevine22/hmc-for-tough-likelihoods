from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _as_numpy(array_like):
    return np.asarray(array_like, dtype=float)


def plot_likelihood_curves(theta_grid, true_curve, perturbed_curves, *, labels=None):
    """1-D true log-density vs several perturbed log-densities over a theta grid."""
    theta_grid = _as_numpy(theta_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(theta_grid, _as_numpy(true_curve), color="0.2", linewidth=3.0, label="true")

    for idx, curve in enumerate(perturbed_curves):
        label = None if labels is None else labels[idx]
        ax.plot(theta_grid, _as_numpy(curve), linewidth=2.0, alpha=0.9, label=label)

    ax.set_xlabel(r"$\theta$")
    if labels is not None:
        ax.legend(loc="best", frameon=False)
    sns.despine(ax=ax)
    return fig, ax


def plot_gradient_field(theta_grid, true_grad, noisy_grads, *, labels=None):
    """True gradient vs several noisy-gradient fields over a theta grid.

    ``true_grad`` is the thick dark reference line ``-theta``; each entry of
    ``noisy_grads`` (one per noise scale) is scattered as the rough force field the
    sampler's leapfrog actually integrates against."""
    theta_grid = _as_numpy(theta_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for idx, grad in enumerate(noisy_grads):
        label = None if labels is None else labels[idx]
        ax.scatter(theta_grid, _as_numpy(grad), s=10, alpha=0.6, label=label)
    ax.plot(
        theta_grid, _as_numpy(true_grad), color="0.2", linewidth=3.0,
        label="true", zorder=3,
    )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\nabla \log \pi$")
    if labels is not None:
        ax.legend(loc="best", frameon=False)
    sns.despine(ax=ax)
    return fig, ax


def plot_profile_1d(theta_grid, profile_values, *, coord_index=0):
    """1-D profile of the (perturbed) log-density along one coordinate, others at 0."""
    theta_grid = _as_numpy(theta_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(theta_grid, _as_numpy(profile_values), color="C0", linewidth=3.0)
    ax.set_xlabel(rf"$\theta_{{{coord_index}}}$ (others at 0)")
    sns.despine(ax=ax)
    return fig, ax


def plot_profile_2d(theta_grid_x, theta_grid_y, profile_grid, *, coord_indices=(0, 1)):
    """2-D profile of the (perturbed) log-density over two coordinates, others at 0.

    ``profile_grid`` is indexed ``[i_x, j_y]`` (x along the first axis)."""
    theta_grid_x = _as_numpy(theta_grid_x)
    theta_grid_y = _as_numpy(theta_grid_y)
    profile_grid = _as_numpy(profile_grid)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    extent = [theta_grid_x.min(), theta_grid_x.max(), theta_grid_y.min(), theta_grid_y.max()]
    im = ax.imshow(profile_grid.T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ci, cj = coord_indices
    ax.set_xlabel(rf"$\theta_{{{ci}}}$")
    ax.set_ylabel(rf"$\theta_{{{cj}}}$")
    fig.colorbar(im, ax=ax, label="log density")
    sns.despine(ax=ax)
    return fig, ax


def _fmt_tick(value):
    """Compact tick label: scientific for very small/large, else 3 sig figs."""
    v = float(value)
    if v == 0:
        return "0"
    if abs(v) < 1e-2 or abs(v) >= 1e4:
        return f"{v:.1e}"
    return f"{v:.3g}"


def _shared_vmin_vmax(*arrays, positive_only=False):
    """Shared color limits across several grids (finite values only)."""
    vals = np.concatenate([_as_numpy(a).ravel() for a in arrays])
    finite = vals[np.isfinite(vals)]
    if positive_only:
        finite = finite[finite > 0]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def plot_nuts_walnuts_heatmaps(
    a_grid,
    omega_grid,
    nuts_values,
    walnuts_values,
    *,
    quantity_label,
    xlabel="A",
    ylabel=r"$\omega$",
    log_color=False,
    cmap="viridis",
):
    """NUTS vs WALNUTS heatmaps of one quantity over a 2-D grid.

    Both panels share a single color scale (shared ``vmin``/``vmax``) and one
    colorbar so they are directly comparable. ``values`` arrays are indexed
    ``[i_a, j_omega]`` (``a_grid`` along the first axis / x, ``omega_grid`` along the
    second / y); pass ``xlabel``/``ylabel`` for other grids. Set ``log_color=True``
    for quantities spanning orders of magnitude (e.g. divergence).
    """
    a_grid = _as_numpy(a_grid)
    omega_grid = _as_numpy(omega_grid)
    nuts_values = _as_numpy(nuts_values)
    walnuts_values = _as_numpy(walnuts_values)

    vmin, vmax = _shared_vmin_vmax(nuts_values, walnuts_values, positive_only=log_color)
    if log_color and vmin is not None:
        norm_kw = {"norm": mcolors.LogNorm(vmin=vmin, vmax=vmax)}
    else:
        norm_kw = {"vmin": vmin, "vmax": vmax}

    n_a, n_w = len(a_grid), len(omega_grid)
    fig, (ax_n, ax_w) = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    im = None
    for ax, values, title in (
        (ax_n, nuts_values, "NUTS"),
        (ax_w, walnuts_values, "WALNUTS"),
    ):
        # Cells sit at integer indices and ticks carry the true grid values, so the
        # axis labels are correct for ANY spacing (e.g. a logarithmic A grid) --
        # unlike an extent-based axis, which would space the labels linearly.
        im = ax.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            **norm_kw,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(np.arange(n_a))
        ax.set_xticklabels([_fmt_tick(a) for a in a_grid], rotation=45, ha="right")
        sns.despine(ax=ax)

    ax_n.set_ylabel(ylabel)
    ax_n.set_yticks(np.arange(n_w))
    ax_n.set_yticklabels([_fmt_tick(w) for w in omega_grid])
    ax_w.tick_params(labelleft=False)
    fig.colorbar(im, ax=[ax_n, ax_w], label=quantity_label)
    return fig, (ax_n, ax_w)


def plot_sweep_nuts_walnuts(
    sigma_grid,
    nuts_values,
    walnuts_values,
    *,
    quantity_label,
    xlabel=r"$\sigma$ (gradient noise)",
    log_x=False,
    log_y=False,
):
    """NUTS vs WALNUTS line plot of one quantity over a 1-D sweep.

    The 1-D analog of :func:`plot_nuts_walnuts_heatmaps`: both series are drawn on a
    single panel, so they inherently share axes and are directly comparable.
    ``sigma_grid`` is the swept parameter (gradient-noise scale by default; pass
    ``xlabel`` for other sweeps); ``log_x``/``log_y`` enable log scaling for sweeps
    or quantities spanning orders of magnitude.
    """
    sigma_grid = _as_numpy(sigma_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(
        sigma_grid, _as_numpy(nuts_values), color="C0", linewidth=3.0,
        marker="o", label="NUTS",
    )
    ax.plot(
        sigma_grid, _as_numpy(walnuts_values), color="C1", linewidth=3.0,
        marker="s", label="WALNUTS",
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(quantity_label)
    ax.legend(loc="best", frameon=False)
    sns.despine(ax=ax)
    return fig, ax


def plot_histogram_vs_truth(samples, true_pdf_fn, *, n_bins=60):
    """1-D histogram of samples overlaid with the analytic true density."""
    samples = _as_numpy(samples).ravel()

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.hist(samples, bins=n_bins, density=True, color="C0", alpha=0.4)
    xs = np.linspace(samples.min(), samples.max(), 400)
    ax.plot(xs, _as_numpy(true_pdf_fn(xs)), color="0.2", linewidth=3.0)
    ax.set_xlabel(r"$\theta$")
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    return fig, ax


def plot_pairplot_vs_truth(samples, truth_samples, *, coords: Sequence[int] = (0, 1, 2, 3)):
    """Corner pairplot of sampled vs ground-truth draws over a few coordinates."""
    import pandas as pd

    samples = _as_numpy(samples)
    truth_samples = _as_numpy(truth_samples)
    coords = list(coords)
    columns = [rf"$\theta_{{{c}}}$" for c in coords]

    df_sampled = pd.DataFrame(samples[:, coords], columns=columns)
    df_sampled["source"] = "sampled"
    df_truth = pd.DataFrame(truth_samples[:, coords], columns=columns)
    df_truth["source"] = "truth"
    df = pd.concat([df_sampled, df_truth], ignore_index=True)

    grid = sns.pairplot(
        df,
        hue="source",
        corner=True,
        diag_kind="kde",
        plot_kws={"s": 8, "alpha": 0.3, "edgecolor": "none"},
    )
    return grid
