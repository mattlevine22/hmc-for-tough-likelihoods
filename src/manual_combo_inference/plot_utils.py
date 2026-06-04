from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _as_numpy(array_like):
    return np.asarray(array_like, dtype=float)


def plot_loglik_score_curves(
    alpha_grid, loglik_by_source, score_by_source, *, labels=None
):
    """Overlaid log-likelihood (L) and score (dL) profiles across sources.

    ``loglik_by_source``/``score_by_source`` are aligned sequences (one curve per
    source) over ``alpha_grid``. Since the value of any combination comes from its
    primal source and the gradient from its grad source, these two panels fully
    determine all 9 (primal, gradient) combinations."""
    alpha_grid = _as_numpy(alpha_grid)

    fig, (ax_l, ax_d) = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    for idx, (curve_l, curve_d) in enumerate(zip(loglik_by_source, score_by_source)):
        label = None if labels is None else labels[idx]
        ax_l.plot(alpha_grid, _as_numpy(curve_l), linewidth=2.4, alpha=0.95, label=label)
        ax_d.plot(alpha_grid, _as_numpy(curve_d), linewidth=2.4, alpha=0.95, label=label)

    ax_l.set_title(r"$L(\alpha)$")
    ax_d.set_title(r"$dL(\alpha)$")
    for ax in (ax_l, ax_d):
        ax.set_xlabel(r"$\alpha$")
        sns.despine(ax=ax)
    if labels is not None:
        ax_l.legend(loc="best", frameon=False)
    return fig, (ax_l, ax_d)


def _shared_vmin_vmax(*arrays, positive_only=False):
    vals = np.concatenate([_as_numpy(a).ravel() for a in arrays])
    finite = vals[np.isfinite(vals)]
    if positive_only:
        finite = finite[finite > 0]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def plot_combo_heatmaps(
    nuts_values,
    walnuts_values=None,
    *,
    primal_sources,
    grad_sources,
    quantity_label,
    log_color=False,
    cmap="viridis",
):
    """NUTS (and optionally WALNUTS) heatmaps of one quantity over the (primal,
    gradient) combo grid.

    ``*_values`` are indexed ``[i_primal, j_grad]`` -- rows are the primal (value)
    source, columns the gradient (score) source. When ``walnuts_values`` is ``None``
    (e.g. WALNUTS was disabled) a single NUTS panel is drawn; otherwise both panels
    share a single color scale and one colorbar. Set ``log_color=True`` for quantities
    spanning orders of magnitude (e.g. W1, ESS/s)."""
    primal_sources = list(primal_sources)
    grad_sources = list(grad_sources)

    panels = [("NUTS", _as_numpy(nuts_values))]
    if walnuts_values is not None:
        panels.append(("WALNUTS", _as_numpy(walnuts_values)))

    vmin, vmax = _shared_vmin_vmax(*[v for _, v in panels], positive_only=log_color)
    if log_color and vmin is not None:
        norm_kw = {"norm": mcolors.LogNorm(vmin=vmin, vmax=vmax)}
    else:
        norm_kw = {"vmin": vmin, "vmax": vmax}

    # NaN cells (e.g. a method that timed out in warmup) are masked and drawn grey.
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.85")

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4.5),
        sharex=True, sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    im = None
    for ax, (title, values) in zip(axes, panels):
        masked = np.ma.masked_invalid(values)
        im = ax.imshow(masked, origin="upper", aspect="auto", cmap=cmap_obj, **norm_kw)
        ax.set_title(title)
        ax.set_xlabel("gradient (score) source")
        ax.set_xticks(np.arange(len(grad_sources)))
        ax.set_xticklabels(grad_sources, rotation=45, ha="right")
        # Annotate each cell: its value, or "n/a" where it timed out.
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                            color="w", fontsize=8)
                else:
                    ax.text(j, i, "n/a", ha="center", va="center",
                            color="0.4", fontsize=8)
        sns.despine(ax=ax, left=True, bottom=True)

    axes[0].set_ylabel("primal (value) source")
    axes[0].set_yticks(np.arange(len(primal_sources)))
    axes[0].set_yticklabels(primal_sources)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=list(axes), label=quantity_label)
    return fig, tuple(axes)


def plot_combo_histograms(
    samples_by_combo,
    *,
    primal_sources,
    grad_sources,
    ref_grid,
    ref_pdf,
    true_alpha=None,
    bins=40,
    xlim=None,
):
    """3x3 grid of posterior histograms, one per (primal, gradient) combination, each
    overlaid with the exact KF posterior (thick line). Rows are the primal source,
    columns the gradient source. ``samples_by_combo`` is a dict keyed by
    ``(primal, grad)``."""
    primal_sources = list(primal_sources)
    grad_sources = list(grad_sources)
    ref_grid = _as_numpy(ref_grid)
    ref_pdf = _as_numpy(ref_pdf)
    n_p, n_g = len(primal_sources), len(grad_sources)

    fig, axes = plt.subplots(
        n_p, n_g, figsize=(3.2 * n_g, 2.4 * n_p),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for i, ps in enumerate(primal_sources):
        for j, gs in enumerate(grad_sources):
            ax = axes[i, j]
            raw = samples_by_combo.get((ps, gs))
            samples = None if raw is None else np.asarray(raw, dtype=float).ravel()
            ax.plot(ref_grid, ref_pdf, color="0.2", linewidth=2.0)
            if samples is None or samples.size == 0:
                # Method timed out before producing samples (warmup).
                ax.text(0.5, 0.5, "no samples\n(timed out)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8, color="0.4")
            else:
                ax.hist(samples, bins=bins, density=True, color="C0", alpha=0.4)
            if true_alpha is not None:
                ax.axvline(float(true_alpha), color="C3", linewidth=1.5, linestyle="--")
            if i == 0:
                ax.set_title(f"grad = {gs}")
            if j == 0:
                ax.set_ylabel(f"primal = {ps}")
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)

    if xlim is not None:
        axes[0, 0].set_xlim(*xlim)
    return fig, axes


def plot_combo_failures(
    failures_by_sampler,
    *,
    primal_sources,
    grad_sources,
    n_replicas,
    cmap="Reds",
):
    """Integer failure-count heatmap (one panel per sampler) over the (primal,
    gradient) combo grid. ``failures_by_sampler`` is a dict ``{sampler_name: array}``
    of how many of ``n_replicas`` replicas failed to produce a usable chain; rows are
    the primal source, columns the gradient source."""
    primal_sources = list(primal_sources)
    grad_sources = list(grad_sources)
    panels = list(failures_by_sampler.items())

    cmap_obj = plt.get_cmap(cmap)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4.5),
        sharex=True, sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    im = None
    for ax, (name, values) in zip(axes, panels):
        values = _as_numpy(values)
        im = ax.imshow(values, origin="upper", aspect="auto", cmap=cmap_obj,
                       vmin=0, vmax=n_replicas)
        ax.set_title(name.upper())
        ax.set_xlabel("gradient (score) source")
        ax.set_xticks(np.arange(len(grad_sources)))
        ax.set_xticklabels(grad_sources, rotation=45, ha="right")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                # White text on dark (high-count) cells, dark text on light ones.
                text_color = "w" if values[i, j] > 0.6 * n_replicas else "0.2"
                ax.text(j, i, str(int(round(values[i, j]))), ha="center",
                        va="center", color=text_color, fontsize=9)
        sns.despine(ax=ax, left=True, bottom=True)

    axes[0].set_ylabel("primal (value) source")
    axes[0].set_yticks(np.arange(len(primal_sources)))
    axes[0].set_yticklabels(primal_sources)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.colorbar(im, ax=list(axes), label=f"# failures (of {n_replicas})")
    return fig, tuple(axes)
