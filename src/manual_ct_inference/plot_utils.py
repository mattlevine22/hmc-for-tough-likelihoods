import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_lti_synthetic_data(times, states, observations):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(times, states[:, 0], color="C0", label="Latent state $x_0(t)$")
    axs[0].set_ylabel("State")
    axs[0].legend(loc="upper right")
    sns.despine(ax=axs[0])

    axs[1].plot(times, states[:, 1], color="C1", label="Latent state $x_1(t)$")
    axs[1].set_ylabel("State")
    axs[1].legend(loc="upper right")
    sns.despine(ax=axs[1])

    axs[2].plot(
        times,
        observations,
        color="C2",
        linewidth=1.5,
        linestyle="--",
        label="Observed output $y(t)$",
    )
    axs[2].set_ylabel("Observation")
    axs[2].set_xlabel("Time")
    axs[2].legend(loc="upper right")
    sns.despine(ax=axs[2])

    plt.tight_layout()
    plt.show()


def plot_lti_filter_comparison(
    parameter_grid,
    profile_band_q05,
    profile_band_q50,
    profile_band_q95,
    grad_band_q05,
    grad_band_q50,
    grad_band_q95,
    grad_numeric_q50,
    comparison_curves,
    parameter_name="rho",
    base_profile_label="DPF band",
    base_grad_label="DPF autodiff band",
    base_numeric_grad_label="DPF median numerical",
    profile_title="Profile likelihood comparisons across filters and particle counts",
    grad_title="Autodiff gradient comparisons across filters and particle counts",
):
    fig, (ax_profile, ax_grad) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    x = np.array(parameter_grid)

    ax_profile.fill_between(
        x,
        np.array(profile_band_q05),
        np.array(profile_band_q95),
        color="C0",
        alpha=0.12,
        label=base_profile_label,
    )
    ax_profile.plot(
        x,
        np.array(profile_band_q50),
        color="C0",
        linewidth=2.5,
        alpha=0.9,
        label=base_profile_label.replace("band", "median"),
    )

    for curve in comparison_curves:
        ax_profile.plot(
            x,
            np.array(curve["profile"]),
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=curve.get("linewidth", 2),
            alpha=curve.get("alpha", 0.95),
            label=curve["label"],
        )

    ax_profile.set_ylabel(r"$\log p(y_{1:T} \mid \theta)$")
    ax_profile.set_title(profile_title)
    ax_profile.set_yscale("symlog")
    ax_profile.legend(loc="best", ncols=2)
    sns.despine(ax=ax_profile)

    ax_grad.fill_between(
        x,
        np.array(grad_band_q05),
        np.array(grad_band_q95),
        color="C0",
        alpha=0.12,
        label=base_grad_label,
    )
    ax_grad.plot(
        x,
        np.array(grad_band_q50),
        color="C0",
        linewidth=2.5,
        alpha=0.9,
        label=base_grad_label.replace("band", "median"),
    )
    ax_grad.plot(
        x,
        np.array(grad_numeric_q50),
        color="0.35",
        linewidth=1.6,
        linestyle=":",
        label=base_numeric_grad_label,
    )

    for curve in comparison_curves:
        ax_grad.plot(
            x,
            np.array(curve["grad"]),
            color=curve["color"],
            linestyle=curve["linestyle"],
            linewidth=curve.get("linewidth", 2),
            alpha=curve.get("alpha", 0.95),
            label=f'{curve["label"]} gradient',
        )

    ax_grad.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    ax_grad.set_xlabel(parameter_name)
    ax_grad.set_ylabel(r"$\partial \log p(y_{1:T} \mid \theta) / \partial \theta$")
    ax_grad.set_title(grad_title)
    ax_grad.legend(loc="best", ncols=2)
    sns.despine(ax=ax_grad)

    plt.tight_layout()
    plt.show()
