# Tough Likelihoods

Parameter inference in dynamical systems and state-space models can be
surprisingly difficult, even when the latent dynamics are conceptually simple.
One reason is that the likelihood surface seen by an inference algorithm is
often not the "true" surface, but an approximation induced by filtering or
simulation methods used when the marginal likelihood is analytically
intractable. Those approximations can introduce roughness, bias, discontinuous
behaviour, and other pathologies that make posterior sampling much harder than
the underlying model structure might suggest.

This repository is a research sandbox for studying those failures directly. The
goal is to identify common failure modes in parameter inference for dynamical
systems, then develop sampling strategies that remain efficient and reliable in
the presence of irregular approximate likelihoods. In practice, that means
asking questions like: when standard NUTS breaks down, does a method such as
WALNUTS behave better? When do we need specialized adaptation, better
randomness control, or different estimators entirely?

## What Is In This Repo

Right now the repo contains a discrete-time linear time-invariant (LTI)
Gaussian example with a single unknown parameter. It is a useful first testbed
because we can compute the marginal likelihood exactly with a Kalman filter,
then compare that ground truth against particle-based approximations and study
how controlled degradations distort the surface seen by downstream samplers.

The current example includes:

- synthetic data generation for a 2D latent linear Gaussian model
- exact marginal likelihood evaluation with a Kalman filter
- approximate marginal likelihood evaluation with a bootstrap particle filter
- posterior inference over a scalar system parameter with BlackJAX NUTS
- notebook-based visualisation of trajectories, posterior samples, and filter
  comparisons

The main entry point is:

- Notebook: `notebooks/lti_discrete_time_low_level.ipynb`

The supporting Python package lives under:

- `src/manual_discrete_time_inference/`

## Repository Layout

```text
.
|-- notebooks/
|   `-- lti_discrete_time_low_level.ipynb
|-- src/
|   `-- manual_discrete_time_inference/
|       |-- __init__.py
|       |-- lti_discrete.py
|       `-- plot_utils.py
|-- example.toml
|-- pyproject.toml
`-- uv.lock
```

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) for environment management

The project dependencies are declared in [`pyproject.toml`](pyproject.toml).

## Quick Start

From a fresh clone:

```bash
uv sync
```

Launch Jupyter Lab:

```bash
uv run jupyter lab notebooks/lti_discrete_time_low_level.ipynb
```

Or execute the notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/lti_discrete_time_low_level.ipynb
```

## Using The Code

After installing dependencies, you can import the package directly:

```python
from manual_discrete_time_inference import (
    kalman_filter_loglik,
    make_lti_discrete_system,
    particle_filter_loglik,
    run_blackjax_nuts_1d,
    simulate_lti_discrete_gaussian,
)
```

The core implementation is in
[`src/manual_discrete_time_inference/lti_discrete.py`](src/manual_discrete_time_inference/lti_discrete.py).

## Why This Repo Exists

This project is useful as a compact research sandbox when you want to:

- study how approximate likelihoods alter posterior geometry
- compare exact and particle-based inference in a controlled setting
- identify failure modes for gradient-based samplers on rough likelihoods
- prototype more robust inference strategies for tough state-space problems
- keep the implementation small enough to read end-to-end

## Development Notes

- Dependencies are locked in `uv.lock`.
- Source code uses a `src/` layout.
- The notebook is intended to be the easiest place to start.

If you want to turn this folder into a standalone Git repository locally:

```bash
git init
git add .
git commit -m "Initial commit"
```

## Status

This repository currently documents and ships the discrete-time example above.
If additional continuous-time examples are added later, they should be
documented as separate notebook/package sections rather than assumed here.
