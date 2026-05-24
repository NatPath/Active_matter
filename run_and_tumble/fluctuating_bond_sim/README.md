# Fluctuating bond simulator

A small local project for simulating the **1D periodic lattice with a single localized fluctuating bond** using the **probability-field method**.

This code is organized for the workflow you were using in chat:

- simulate the one-body probability field on a ring
- sample thermalized states
- compute selected equal-time connected correlation slices \(C(x,y)\)
- optionally add the fixed-\(N\) correction
- plot the raw cuts and the collapse
- optionally compute the full covariance matrix

The implementation is designed to be easy to modify.

## Model

- lattice sites: `0, 1, ..., L-1`
- periodic boundary conditions
- special bond: `(m, m+1 mod L)`, by default centered with `m = L // 2`
- bulk directed rates: `1/2` left and `1/2` right
- special bond orientation `sigma = +1`:
  - `m -> m+1` rate `1`
  - `m+1 -> m` rate `0`
- special bond orientation `sigma = -1`: reversed
- sweep-wise approximation:
  - during one sweep the orientation is fixed
  - between sweeps it flips with probability `1 - exp(-flip_rate_per_sweep)`

This is the same approximation used in the notebook version.

## Directory layout

```text
fluctuating_bond_sim/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_collapse.py
│   ├── run_full_covariance.py
│   └── run_smoke_test.py
└── fluctuating_bond/
    ├── __init__.py
    ├── analysis.py
    ├── io_utils.py
    ├── model.py
    ├── plotting.py
    └── simulate.py
```

## Install

Create a local environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

### 1. Reproduce the \(L=1024\) collapse with the fixed-\(N\) correction

This matches the workflow from the chat:
- `L = 1024`
- `rho = 100`
- cuts `y = 20:10:200`
- add the correction `+N/L^2`

```bash
python scripts/run_collapse.py \
  --L 1024 \
  --rho 100 \
  --n-burn 6000 \
  --n-samples 12000 \
  --sample-stride 3 \
  --y-start 20 \
  --y-stop 200 \
  --y-step 10 \
  --use-fix-term \
  --seed 123 \
  --output-dir outputs/L1024_fixed
```

This will save:

- `config.json`
- `runtime_report.txt`
- `mean_density.npy`
- `slice_data.npz`
- `cuts.png`
- `collapse.png`
- `collapse_only.png`

### 2. Full covariance run

```bash
python scripts/run_full_covariance.py \
  --L 512 \
  --rho 100 \
  --n-burn 4000 \
  --n-samples 6000 \
  --sample-stride 3 \
  --seed 123 \
  --output-dir outputs/full_cov_L512
```

This saves the full covariance matrix and a heatmap.

### 3. Small smoke test

```bash
python scripts/run_smoke_test.py
```

### 4. Rebuild the collapse from saved outputs only

If `mean_density.npy`, `slice_data.npz`, and `config.json` already exist, you
can regenerate the corrected cuts and collapse without rerunning the simulation:

```bash
python scripts/run_collapse.py \
  --input-dir outputs/L1024_fixed \
  --output-dir outputs/L1024_fixed
```

If you omit `--output-dir`, it defaults to the same directory as `--input-dir`.

## Fixed-\(N\) correction

The collapse workflow reconstructs the connected fixed-\(N\) occupancy
correlations from the sampled mean fields \(m_t = N p_t\). Off diagonal there
is then a trivial background

\[
C_{ij}^{\mathrm{trivial}} = -\frac{N}{L^2}, \qquad i \neq j.
\]

The script option

```bash
--use-fix-term
```

adds

\[
\frac{N}{L^2}
\]

to the off-diagonal part of those connected slices before plotting the cuts and
the collapse. This is the quantity that is collapsed, so the corrected tails
approach zero near the boundary.

For the corrected cuts and collapse, the single diagonal point at `x=y` is then
replaced by the midpoint average of its neighboring corrected values. This keeps
the local self-correlation spike from obscuring the broader signal.

If you want the more precise nonuniform correction, use

```bash
--fix-term-mode mean_field
```

which adds

\[
N \bar p(x)\bar p(y)
\]

instead of the flat approximation \(N/L^2\).

Available modes are:

- `none`
- `flat`
- `mean_field`

`--use-fix-term` is shorthand for `--fix-term-mode flat`.

## Notes on performance

For collapse plots, the **selected-slices** workflow is the right default.
It is much cheaper than constructing the full \(L \times L\) covariance matrix.

### Main cost

The dominant cost is repeated application of

\[
p_{t+1} = e^{Q_{\sigma_t}} p_t
\]

through `scipy.sparse.linalg.expm_multiply`.

Runtime scales roughly linearly with

\[
n_{\text{burn}} + n_{\text{samples}} \times \text{sample_stride}.
\]

### First optimizations to try

1. Increase `sample_stride` if the sampled states are strongly correlated.
2. Use selected slices instead of full covariance.
3. Use fewer `x/y` points for exploratory runs.
4. Batch several seeds as separate local jobs if you want more independent statistics.

## Parameters you will probably edit most

Inside the command line:

- `--L`
- `--rho`
- `--n-burn`
- `--n-samples`
- `--sample-stride`
- `--flip-rate-per-sweep`
- `--seed`
- `--y-start`, `--y-stop`, `--y-step`
- `--x-window-multiple`
- `--fix-term-mode`
- `--collapse-power`

`--collapse-power` sets the exponent \(p\) in the collapse variable

\[
y^p \times \text{(corrected connected correlation)}.
\]

The default is `2`. This means you can test alternative power laws cheaply with
the reload-only path, for example

```bash
python scripts/run_collapse.py \
  --input-dir outputs/L1024_fixed \
  --output-dir outputs/L1024_power_1p5 \
  --fix-term-mode flat \
  --collapse-power 1.5
```

For `collapse-power != 2`, the script does not draw the current continuum-shape
fit, because that overlay is specific to the \(y^2\) collapse.

## Output conventions

Coordinates are always reported **relative to the source bond center**, with the source at `m = L//2`.

For each selected cut the saved `slice_data.npz` contains arrays:

- `y_values`
- `x_values_y_<y>`
- `cxy_y_<y>` for the raw connected fixed-\(N\) correlation
- `cxy_corrected_y_<y>` after the finite-size correction and diagonal-point interpolation

## Adapting to more special bonds later

This package is written so that the model construction is separated from analysis.
If you later want multiple independently fluctuating bonds, the main thing to replace is the generator builder in `fluctuating_bond/model.py`.

The current package is intentionally limited to the single-bond case so the code stays clean.
