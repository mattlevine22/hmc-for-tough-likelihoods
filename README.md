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

The repo currently contains two low-level benchmark cases:

1. A discrete-time linear time-invariant (LTI) Gaussian model with one unknown
parameter.
2. A continuous-time LTI Gaussian SDE with one unknown parameter.

Both notebooks compare exact Kalman-filter marginal likelihoods against
particle-filter approximations and then study the consequences for NUTS-style
inference. The point is not only to have two examples, but to have two examples
with qualitatively different levels of difficulty.

### Case 1: Discrete-Time LTI

Notebook: `notebooks/lti_discrete_time_low_level.ipynb`

This is the mild case.

- fast to run
- exact marginal likelihood available from the Kalman filter
- particle-filter distortions are visible but relatively controlled
- useful as a first sandbox for profiling roughness, bias, and sampler behavior

### Case 2: Continuous-Time LTI

Notebook: `notebooks/lti_gaussian_low_level.ipynb`

This is the more catastrophic case.

- much slower to run because the particle propagation is continuous-time
- exact marginal likelihood still available as a baseline after discretization
- approximate likelihood geometry is substantially harsher
- intended as the stronger failure-mode benchmark when the discrete-time case is
  not difficult enough

The core Python implementation lives under:

- `src/manual_ct_inference/`
- `src/manual_discrete_time_inference/`
- `src/hmc_for_tough_likelihoods/` (clean public import surface for the
  discrete-time API)

## Repository Layout

```text
.
|-- notebooks/
|   |-- lti_discrete_time_low_level.ipynb
|   |-- lti_gaussian_low_level.ipynb
|   `-- lti_gaussian_low_level_copy.ipynb
|-- src/
|   |-- hmc_for_tough_likelihoods/
|   |   `-- __init__.py
|   |-- manual_ct_inference/
|   |   |-- __init__.py
|   |   |-- lti_gaussian.py
|   |   `-- plot_utils.py
|   `-- manual_discrete_time_inference/
|       |-- __init__.py
|       |-- lti_discrete.py
|       `-- plot_utils.py
|-- example.toml
|-- pyproject.toml
`-- uv.lock
```

## Benchmarks

The two current benchmarks play different roles:

- `lti_discrete_time_low_level.ipynb`: fast, mild, and easy to iterate on
- `lti_gaussian_low_level.ipynb`: slow, harsher, and intended to expose more
  serious failure modes

If you are new to the repo, start with the discrete-time notebook. If you want
the more difficult stress test, move to the continuous-time notebook.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) for environment management

The project dependencies are declared in [`pyproject.toml`](pyproject.toml).

## Quick Start

From a fresh clone:

```bash
uv sync
```

Launch the discrete-time notebook:

```bash
uv run jupyter lab notebooks/lti_discrete_time_low_level.ipynb
```

Launch the continuous-time notebook:

```bash
uv run jupyter lab notebooks/lti_gaussian_low_level.ipynb
```

Or execute either notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/lti_discrete_time_low_level.ipynb
```

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/lti_gaussian_low_level.ipynb
```

## Using The Code

After installing dependencies, you can import the discrete-time package through
the clean public API:

```python
from hmc_for_tough_likelihoods import (
    kalman_filter_loglik,
    make_lti_discrete_system,
    particle_filter_loglik,
    run_blackjax_nuts_1d,
    simulate_lti_discrete_gaussian,
)
```

The continuous-time notebook currently imports its backing code directly from:

```python
from manual_ct_inference import (
    kalman_filter_loglik,
    make_blackjax_logdensity,
    particle_filter_loglik,
    run_blackjax_nuts_1d,
    simulate_lti_gaussian_euler_maruyama,
)
```

The main implementations live in:

- [`src/manual_discrete_time_inference/lti_discrete.py`](src/manual_discrete_time_inference/lti_discrete.py)
- [`src/manual_ct_inference/lti_gaussian.py`](src/manual_ct_inference/lti_gaussian.py)

## Why This Repo Exists

This project is useful as a compact research sandbox when you want to:

- study how approximate likelihoods alter posterior geometry
- compare mild and catastrophic failure regimes in a controlled setting
- identify failure modes for gradient-based samplers on rough likelihoods
- prototype more robust HMC- and NUTS-like strategies for tough state-space
  problems
- keep the implementation small enough to read end-to-end

## Development Notes

- Dependencies are locked in `uv.lock`.
- Source code uses a `src/` layout.
- The discrete-time notebook is the easiest place to start.
- The continuous-time notebook is intentionally slower and more computationally
  demanding.

If you want to turn this folder into a standalone Git repository locally:

```bash
git init
git add .
git commit -m "Initial commit"
```

## Status

The discrete-time and continuous-time LTI examples are both now included.

- The discrete-time case is the fast, mild benchmark.
- The continuous-time case is the slow, more pathological benchmark.

Future additions can extend this ladder of difficulty, but these two cases are
the current baseline pair for comparing exact and approximate likelihood
surfaces.
