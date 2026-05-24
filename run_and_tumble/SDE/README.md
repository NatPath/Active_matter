# Continuum SDE Models for Localized Fluctuating Forces

This directory contains prototype Python dashboards and a newer headless Julia
implementation for continuum stochastic differential equation (SDE) versions of
localized fluctuating-force simulations. The goal is to model the continuum
analog of a lattice drive whose sign or amplitude is white-noise-like in time
and localized in space.

The Python files are interactive exploratory dashboards. The Julia path is the
recommended path for production, replica averaging, and cluster use.

## Quick LLM Context

- Physical problem: passive bath particles diffuse in a periodic domain while a
  localized force profile is multiplied by a temporally white random signal.
- Continuum bath equation:

  ```text
  dx_i = mu_bath * sum_k phi(x_i - R_k) dW_k + sqrt(2 D_bath) dB_i
  ```

- `phi` is a localized spatial profile, currently Gaussian or compact bump.
- `dW_k` is a force-level noise shared by all bath particles, so it induces
  spatially correlated density fluctuations.
- `dB_i` is independent thermal bath noise for each particle.
- The Julia implementation stores online density-count variance accumulators,
  not full trajectories by default.
- Multiple force centers are already represented as a vector of centers.
- Optional mobile force centers are implemented as a model extension:

  ```text
  dR_k = -force_mobility * S_k dW_k + sqrt(2 force_diffusivity) dB^R_k
  S_k = sum_i phi(x_i - R_k)
  ```

- Production-scale runs should be submitted through the Condor DAG wrapper:

  ```text
  cluster_scripts/submit_fluctuating_force_sde_dag.sh
  ```

## Files

Prototype dashboards:

- `fluctuating_force_variance_dashboard.py`
  - 1D animated dashboard.
  - Single fixed localized fluctuating force at the origin.
  - Shows total density-count variance and an active tail diagnostic.

- `2D_fluctuating_force_variance_dashboard.py`
  - 2D animated dashboard.
  - Single fixed localized fluctuating force at the origin.
  - Shows active-variance heatmap, angular dependence, and radial tail.

Production Julia path:

- `../src/fluctuating_force_sde/modules_fluctuating_force_sde.jl`
  - Core SDE implementation and online statistics.

- `../run_fluctuating_force_sde.jl`
  - YAML-driven runner that writes JLD2 result files.

- `../utility_scripts/analyze_fluctuating_force_sde.jl`
  - Aggregates replica JLD2 files into CSV and aggregate JLD2 outputs.

- `../cluster_scripts/generate_fluctuating_force_sde_configs.sh`
  - Generates YAML configs.

- `../cluster_scripts/run_fluctuating_force_sde_from_config.sh`
  - Cluster/local wrapper for one config.

- `../cluster_scripts/submit_fluctuating_force_sde_dag.sh`
  - Condor DAG wrapper for replica production and final aggregation.

- `../cluster_scripts/analyze_fluctuating_force_sde.sh`
  - Shell wrapper around the aggregate utility.

## Mathematical Model

### Domain and State

The bath contains `N` passive particles in a periodic domain of side length `L`.
The implemented dimensions are `dims = 1` and `dims = 2`.

The periodic domain is stored in centered coordinates:

```text
Omega = [-L/2, L/2)^dims
```

Bath particle positions are

```text
x_i(t) in Omega,  i = 1, ..., N.
```

There are `K` fluctuating force centers:

```text
R_k(t) in Omega,  k = 1, ..., K.
```

For fixed-force runs, `R_k` is constant. For mobile-force runs, `R_k` evolves
stochastically.

### Localized Profile

The force profile is a scalar envelope evaluated with minimum-image periodic
distance:

```text
r_ik = x_i - R_k  using the minimum periodic image.
```

The default Gaussian profile is

```text
phi(r) = f0 * exp(-|r|^2 / (2 sigma_f^2)).
```

The compact bump option is

```text
u = |r| / sigma_f
phi(r) = f0 * exp(1 - 1/(1 - u^2))   for u < 1
phi(r) = 0                           for u >= 1.
```

### Bath SDE

For each force center `k`, the simulation draws a vector Brownian increment
`dW_k` with variance `dt` in each spatial component. This is a force-level noise:
all bath particles see the same `dW_k`, weighted by their local profile value.

The Euler-Maruyama update corresponds to

```text
dx_i =
    mu_bath * sum_{k=1}^K phi(x_i - R_k) dW_k
    + sqrt(2 D_bath) dB_i.
```

Here:

- `mu_bath` is the bath coupling to the fluctuating force.
- `D_bath` is the independent bath diffusivity.
- `dB_i` is independent Brownian noise for each particle.
- Different force centers have independent `dW_k` by default.

This is the continuum interpretation of a spatially localized forcing whose
temporal signal is white noise.

### Mobile Force Extension

Mobile force centers are optional and controlled by:

```yaml
mobile_forces: true
force_mobility: <value>
force_diffusivity: <value>
```

The implemented mobile-force rule is:

```text
S_k = sum_i phi(x_i - R_k)
dR_k = -force_mobility * S_k dW_k + sqrt(2 force_diffusivity) dB^R_k.
```

This mirrors the existing coupled-SDE active-object convention in the repository:
the object responds to the same fluctuating bath signal through the profile sum.
The sign convention is chosen so that positive `force_mobility` gives a response
opposite to the bath displacement induced by the same noise.

Important: this mobile-force rule is a modeling choice in the current code. If a
specific derivation from the fluctuation-induced diffusion-gradient paper should
be enforced exactly, verify the target equation against the paper and update this
section and the implementation together.

## Observables

### Density Counts

At sampling times, particles are binned into a regular grid.

For 1D:

```text
n_b(t) = number of particles in bin b.
```

For 2D:

```text
n_{b_x,b_y}(t) = number of particles in grid cell (b_x,b_y).
```

The simulator stores only online sums:

```text
sum_counts[b]  = sum_t n_b(t)
sum_counts2[b] = sum_t n_b(t)^2
```

The total count variance is reconstructed as

```text
Var[n_b] = <n_b^2> - <n_b>^2.
```

### Thermal Offset Subtraction

The dashboards and Julia aggregate separate a background variance from the
localized active contribution.

For 1D, the offset is the average variance in edge bins:

```text
thermal_offset = average of the left and right edge-bin variances.
```

For 2D, the offset is taken from the largest radial bin in the radial summary.

The active variance is stored as

```text
active_variance = max(total_variance - thermal_offset, variance_floor).
```

The floor avoids zeros on log-log plots. It should not be interpreted as a
measured signal.

### Scaling Diagnostics

The original dashboards use reference slopes:

- 1D: active variance tail compared with `1/x^2`.
- 2D: radial active variance tail compared with `1/r^4`.

These are diagnostics for the hypothesized far-field response of localized
white-in-time forcing. The code does not automatically fit or validate these
exponents yet; it writes binned data so fits can be performed downstream.

## Configuration Keys

A minimal 1D fixed-force YAML:

```yaml
description: "one_dimensional_fixed_origin"

dims: 1
L: 60.0
N: 8000

D_bath: 1.0
dt: 0.05
mu_bath: 5.0
f0: 1.0
sigma_f: 0.5
profile_type: "gaussian"

force_centers: [0.0]
mobile_forces: false
force_mobility: 0.0
force_diffusivity: 0.0

warmup_steps: 1000
n_steps: 10000
sample_interval: 1

n_bins: 80
edge_bins_for_offset: 5
variance_floor: 1e-6

history_interval: 1000
max_history_records: 20000
save_force_history: true

seed: 0
performance_mode: true
cluster_mode: false
save_dir: "saved_states/fluctuating_force_sde"
```

A minimal 2D fixed-force YAML:

```yaml
description: "two_dimensional_fixed_origin"

dims: 2
L: 40.0
N: 60000

D_bath: 1.0
dt: 0.05
mu_bath: 6.0
f0: 1.0
sigma_f: 0.5
profile_type: "gaussian"

force_centers: [[0.0, 0.0]]
mobile_forces: false

warmup_steps: 10000
n_steps: 100000
sample_interval: 200

n_bins: 60
n_radial_bins: 20
radial_min: 0.8
variance_floor: 1e-6

save_force_history: false
seed: 0
performance_mode: true
save_dir: "saved_states/fluctuating_force_sde"
```

Multiple 1D force centers:

```yaml
dims: 1
force_centers: [-8.0, 0.0, 8.0]
```

Multiple 2D force centers:

```yaml
dims: 2
force_centers:
  - [-8.0, 0.0]
  - [8.0, 0.0]
```

Mobile force centers:

```yaml
mobile_forces: true
force_mobility: 1e-4
force_diffusivity: 0.0
```

## Local Running Examples

Use local runs only for smoke tests or small exploratory checks. Production
work should use the submit wrapper below.

### Live Two-Force Distance Sweep

For the specific diagnostic "two fixed fluctuating forces separated by distance
`d`, measure the density-count variance on top of the force locations, and plot
that variance as a function of `d`", use:

```bash
JULIA_NUM_THREADS="$(nproc)" julia --startup-file=no \
  SDE/run_two_force_distance_sweep_live.jl \
  --distances 4,6,8,12,16,24,32,40 \
  --L 128 \
  --N 20000 \
  --dt 0.002 \
  --mu_bath 1.0 \
  --D_bath 1.0 \
  --f0 1.0 \
  --sigma_f 1.5 \
  --warmup_steps 20000 \
  --production_steps 120000 \
  --chunk_steps 2000 \
  --sample_interval 20 \
  --replicas 2 \
  --n_bins 256 \
  --plot_interval_seconds 8 \
  --output_dir analysis_outputs/fluctuating_force_sde/two_force_distance_sweep_live \
  --save_tag two_force_local
```

The script runs independent `(d, replica)` jobs with Julia threads. It updates:

```text
analysis_outputs/fluctuating_force_sde/two_force_distance_sweep_live/two_force_local_live.png
analysis_outputs/fluctuating_force_sde/two_force_distance_sweep_live/two_force_local_metrics.csv
```

The live plot panel contains:

- linear `variance on force bins` versus separation `d`;
- log-log `(variance - farthest-d offset)` versus `d`;
- a dashed reference line with slope `-2`;
- per-distance sample progress;
- representative density-variance profiles.

For a desktop Julia session, add `--display` if you want Plots.jl to also try to
show a live plot window. Without `--display`, the PNG is overwritten every
`plot_interval_seconds`, which is usually less disruptive to performance.

The measured quantity is:

```text
V_on(d) = average of Var[n_b] over the bins containing the two force centers
```

where the force centers are placed at:

```text
R_1 = -d/2,   R_2 = +d/2.
```

The log-log panel subtracts the largest-distance value:

```text
V_plot(d) = V_on(d) - V_on(d_max).
```

The farthest point is therefore the offset reference and is not itself visible
on the log-log plot when the subtraction makes it zero.

### Generate a 1D Config

```bash
CONFIG_ROOT=/tmp/ffsde_cfg \
CONFIG_NAME=smoke_1d \
DIMS=1 \
L=10 \
N=100 \
DT=0.01 \
WARMUP_STEPS=10 \
PRODUCTION_STEPS=100 \
N_BINS=20 \
SAVE_FORCE_HISTORY=false \
SAVE_DIR=/tmp/ffsde_states \
bash cluster_scripts/generate_fluctuating_force_sde_configs.sh
```

### Estimate Runtime Scale

```bash
julia --startup-file=no run_fluctuating_force_sde.jl \
  --config /tmp/ffsde_cfg/smoke_1d.yaml \
  --estimate_only
```

This prints the number of particle-force evaluations:

```text
total particle-force evaluations =
    (warmup_steps + n_steps) * N * number_of_forces.
```

### Run a Small Local Replica

```bash
julia --startup-file=no run_fluctuating_force_sde.jl \
  --config /tmp/ffsde_cfg/smoke_1d.yaml \
  --save_tag smoke_1d \
  --performance_mode
```

Expected output files:

```text
/tmp/ffsde_states/fluctuating_force_sde_1D_L10_N100_nf1_id-smoke_1d.jld2
/tmp/ffsde_states/fluctuating_force_sde_1D_L10_N100_nf1_id-smoke_1d_summary.txt
```

### Aggregate Local Replicas

```bash
julia --startup-file=no utility_scripts/analyze_fluctuating_force_sde.jl \
  --state_dir /tmp/ffsde_states \
  --output_dir /tmp/ffsde_analysis \
  --save_tag smoke_1d \
  --no_plot
```

Expected aggregate outputs:

```text
/tmp/ffsde_analysis/smoke_1d_bins.csv
/tmp/ffsde_analysis/smoke_1d_aggregate.jld2
/tmp/ffsde_analysis/smoke_1d_summary.txt
```

For 2D aggregates, an additional radial CSV is written:

```text
<save_tag>_radial.csv
```

## Cluster Running Examples

Heavy production runs should be submitted as cluster jobs. Do not run large
`N`, long `n_steps`, or heavy aggregation directly on a login node.

### Two-Force 600-CPU Distance Sweep

For the two fixed-force separation scan `d = 4,6,8,12,16,24,32,40`, `L=160`,
and approximately 600 single-core replica jobs, use the dedicated submit
wrapper. It creates one independent checkpoint-like JLD2 result per
`(distance, replica)` and a final aggregate node that writes the distance-sweep
CSV/JLD2/summary/PNG.

Dry-run first:

```bash
bash cluster_scripts/submit_fluctuating_force_two_force_sweep_10h.sh --run_id ffsde_two_force_L160_600cpu_v1 --target_hours 10 --total_cpus 600 --no_submit
```

Inspect the generated files:

```bash
cat runs/fluctuating_force_sde/two_force_distance_sweep/ffsde_two_force_L160_600cpu_v1/run_info.txt && sed -n '1,12p' runs/fluctuating_force_sde/two_force_distance_sweep/ffsde_two_force_L160_600cpu_v1/manifest.csv && sed -n '1,40p' runs/fluctuating_force_sde/two_force_distance_sweep/ffsde_two_force_L160_600cpu_v1/submit/fluctuating_force_two_force_sweep.dag
```

Submit after inspection:

```bash
bash cluster_scripts/submit_fluctuating_force_two_force_sweep_10h.sh --run_id ffsde_two_force_L160_600cpu_v1 --target_hours 10 --total_cpus 600
```

Fetch aggregate outputs after completion:

```bash
bash cluster_scripts/copy_data_from_cluster.sh --run_id ffsde_two_force_L160_600cpu_v1 --run_family fluctuating_force_sde --sync_scope aggregation
```

Default scientific/run parameters in this wrapper:

```text
L=160
N=50000
distances=4,6,8,12,16,24,32,40
num_replicas_per_distance=75
total_replica_jobs=600
dt=0.002
D_bath=1.0
mu_bath=1.0
f0=1.0
sigma_f=1.5
n_bins=256
sample_interval=20
warmup_fraction=0.12
target_utilization=0.85
particle_force_evals_per_second=4000000
```

The computed step counts are written to `run_info.txt`. With the defaults above,
the wrapper estimates the per-job production length from:

```text
total_steps ~= target_hours * 3600 * target_utilization * particle_force_evals_per_second / (N * 2)
```

The factor `2` is the number of force centers. If a short pilot on the cluster
shows a different throughput, rerun the dry-run with an override such as:

```bash
PARTICLE_FORCE_EVALS_PER_SECOND=2500000 bash cluster_scripts/submit_fluctuating_force_two_force_sweep_10h.sh --run_id ffsde_two_force_L160_600cpu_v1 --target_hours 10 --total_cpus 600 --no_submit
```

### Prepare a Production Config

Example 1D production-style config:

```bash
CONFIG_ROOT=configuration_files/fluctuating_force_sde/one_force_1d \
CONFIG_NAME=one_force_1d_L256_N25600 \
DIMS=1 \
L=256 \
N=25600 \
DT=0.001 \
D_BATH=1.0 \
MU_BATH=1.0 \
F0=1.0 \
SIGMA_F=1.5 \
WARMUP_STEPS=100000 \
PRODUCTION_STEPS=1000000 \
SAMPLE_INTERVAL=10 \
N_BINS=256 \
SAVE_FORCE_HISTORY=false \
SAVE_DIR=saved_states/fluctuating_force_sde \
bash cluster_scripts/generate_fluctuating_force_sde_configs.sh
```

Example 2D production-style config:

```bash
CONFIG_ROOT=configuration_files/fluctuating_force_sde/one_force_2d \
CONFIG_NAME=one_force_2d_L64_N200000 \
DIMS=2 \
L=64 \
N=200000 \
DT=0.001 \
D_BATH=1.0 \
MU_BATH=1.0 \
F0=1.0 \
SIGMA_F=1.5 \
WARMUP_STEPS=100000 \
PRODUCTION_STEPS=500000 \
SAMPLE_INTERVAL=20 \
N_BINS=96 \
N_RADIAL_BINS=32 \
RADIAL_MIN=1.0 \
SAVE_FORCE_HISTORY=false \
SAVE_DIR=saved_states/fluctuating_force_sde \
bash cluster_scripts/generate_fluctuating_force_sde_configs.sh
```

### Dry-Run DAG Creation

Use `--no_submit` first to inspect the generated run root:

```bash
bash cluster_scripts/submit_fluctuating_force_sde_dag.sh \
  --config configuration_files/fluctuating_force_sde/one_force_1d/one_force_1d_L256_N25600.yaml \
  --num_replicas 32 \
  --run_id ffsde_1d_L256_nf1_test \
  --request_cpus 1 \
  --request_memory "6 GB" \
  --aggregate_request_cpus 1 \
  --no_submit
```

This creates:

```text
runs/fluctuating_force_sde/<run_id>/
  configs/
  submit/
  logs/
  states/
  analysis/
  manifest.csv
  run_info.txt
```

Before submitting a real job, inspect:

```bash
cat runs/fluctuating_force_sde/ffsde_1d_L256_nf1_test/run_info.txt
sed -n '1,20p' runs/fluctuating_force_sde/ffsde_1d_L256_nf1_test/manifest.csv
sed -n '1,120p' runs/fluctuating_force_sde/ffsde_1d_L256_nf1_test/submit/fluctuating_force_sde.dag
```

### Submit Production DAG

After inspecting the dry-run files, remove `--no_submit`:

```bash
bash cluster_scripts/submit_fluctuating_force_sde_dag.sh \
  --config configuration_files/fluctuating_force_sde/one_force_1d/one_force_1d_L256_N25600.yaml \
  --num_replicas 32 \
  --run_id ffsde_1d_L256_nf1_prod \
  --request_cpus 1 \
  --request_memory "6 GB" \
  --aggregate_request_cpus 1
```

The final DAG node aggregates all replica JLD2 files under:

```text
runs/fluctuating_force_sde/<run_id>/analysis/
```

## Output Format

Each replica JLD2 contains:

```text
result
param
state
metadata
```

Important `result` keys:

```text
result["result_type"]
result["parameters"]
result["final_state"]
result["sample_count"]
result["bins"]
result["radial"]     # only meaningful for dims = 2
result["forces"]
result["history"]
result["stability"]
```

Important `result["bins"]` keys:

```text
edges
centers
grid_shape
sum_counts
sum_counts2
mean_counts
variance_total
thermal_offset
variance_active
```

Important `result["radial"]` keys for 2D:

```text
edges
centers
cell_counts
variance_total
thermal_offset
variance_active
```

Important `result["forces"]` keys:

```text
sum_profile_sums
sum_profile_sums2
mean_profile_sums
variance_profile_sums
```

Important `result["stability"]` keys:

```text
thermal_rms
thermal_rms_over_sigma_f
bath_active_single_profile_rms
bath_active_single_profile_rms_over_sigma_f
max_abs_bath_active_step
max_abs_bath_thermal_step
max_abs_force_center_step
```

The stability ratios are useful for checking whether `dt` is small compared with
the profile width `sigma_f`.

## Scientific Interpretation Notes

### What the SDE Is Designed to Test

The model tests whether a localized force with zero mean in time but nonzero
temporal variance induces a measurable density-variance profile in the bath.

The key mechanism is that the active noise is common to many particles near the
force center. Therefore the forcing does not average away like independent
particle noise. Instead it creates correlated density fluctuations whose spatial
structure is controlled by diffusion and the force profile.

### Relation to Lattice Fluctuating-Force Runs

The lattice fluctuating-bond model applies a localized current drive on a bond
whose direction changes in time. In the continuum SDE model, that localized drive
is replaced by a smooth profile `phi(r)` multiplied by a white-noise increment.

The intended correspondence is:

```text
lattice local fluctuating current source
    -> continuum localized stochastic drift

telegraph or Monte-Carlo time fluctuations
    -> white-noise temporal idealization

bond-localized spatial support
    -> Gaussian or compact profile with width sigma_f
```

This README does not claim that the continuum limit is already calibrated. The
amplitude mapping between lattice force parameters and SDE parameters should be
treated as a separate calibration problem.

### Interpretation of Active Variance

`variance_total` contains both ordinary finite-particle bin-count variance and
the extra variance induced by the fluctuating force.

`variance_active` is a diagnostic subtraction:

```text
variance_active = variance_total - estimated_far_field_offset.
```

This is useful for visualizing tails and comparing scaling forms, but it is not
an exact decomposition unless the far-field offset is known to be purely
thermal/background.

### Suggested Thesis Wording

The following wording is intentionally plain and can be adapted:

```text
To isolate the continuum response to a localized zero-mean fluctuating drive, we
simulated passive Brownian particles in a periodic domain subject to a stochastic
drift field with a fixed spatial envelope. At each time step, the drive amplitude
was sampled as a Brownian increment shared by all particles, while each particle
also received independent thermal Brownian noise. This implements a force that
is white in time but spatially localized. Density fluctuations were measured by
binning particles and accumulating the time variance of the bin occupancies.
```

For mobile force centers:

```text
In the mobile-center extension, each force center responds to the same stochastic
signal that it applies to the bath, with a mobility proportional to the local
profile sum over bath particles. This gives a minimal coupled SDE model for a
fluctuating object whose effective diffusion depends on the surrounding density
field.
```

## Current Limitations

- The Python dashboard visual references are not automatically generated by the
  Julia pipeline yet.
- The Julia analyzer writes CSV/JLD2 aggregates but does not yet fit scaling
  exponents.
- Multiple forces are independent by default; correlated force noises are not
  implemented yet.
- Mobile force centers share the same noise that drives the bath, following the
  repository's coupled-SDE active-object convention. If a different convention is
  required by a paper derivation, update the model explicitly.
- The thermal-offset subtraction is a diagnostic, not a proof of active/passive
  variance separation.

## Development Checks

Run focused tests:

```bash
julia --startup-file=no test_fluctuating_force_sde.jl
```

Check shell wrapper syntax:

```bash
bash -n cluster_scripts/generate_fluctuating_force_sde_configs.sh
bash -n cluster_scripts/run_fluctuating_force_sde_from_config.sh
bash -n cluster_scripts/submit_fluctuating_force_sde_dag.sh
bash -n cluster_scripts/analyze_fluctuating_force_sde.sh
```

For any production or aggregate workflow that is computationally heavy, use the
submit wrapper and inspect the generated configs, submit files, logs, manifest,
and `run_info.txt` before drawing conclusions from a failed run.
