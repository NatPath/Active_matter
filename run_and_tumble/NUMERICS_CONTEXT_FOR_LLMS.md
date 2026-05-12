# Numerics Context for LLM Discussions

This note is meant as a compact handoff for discussing the simulations with an
LLM. It is not a debugging report. The immediate question is why a nontrivial
reflection-symmetric component appears in the density-correlation signal for
localized fluctuating drives in a 1D diffusive system.

## Current Comparison Runs

The two main runs are fetched locally under `run_and_tumble/cluster_results`.
This note uses ASCII names such as `rho_avg`; the Julia code often uses the
corresponding Greek-symbol field names, e.g. `state.rho_avg` here means
the same field written with a Greek rho in the source.

### Fluctuating bond force

Run ID:

```text
single_origin_bond_L2048_rho100_f1_ffr1
```

Local metadata:

```text
run_and_tumble/cluster_results/runs/single_origin_bond/managed/single_origin_bond_L2048_rho100_f1_ffr1/run_spec.yaml
```

Important parameters:

```text
L = 2048
rho0 = 100
D = 1
T = 1
gamma = 0
potential_type = zero
fluctuation_type = no-fluctuation
force bond = [1024, 1025]
force magnitude f = 1
force flip rate ffr = 1
forcing_rate_scheme = symmetric_normalized
bond_pass_count_mode = all_forcing_bonds
target replicas = 600
```

Latest local aggregate at the time this note was written:

```text
run_and_tumble/cluster_results/runs/single_origin_bond/managed/single_origin_bond_L2048_rho100_f1_ffr1/aggregated/1D_pot-zero_fluc-no-fluctuation_L2048_rho1.0e+02_eps0.00_a0.00_g0.000_D1.0_V0.0_f1.0_ffr1.0000_ic-aggregated_t2657700000_tstats2657700000_id-aggregated_managed_20260510m113345.jld2
```

### Fluctuating local potential

Run ID:

```text
diffusive_1d_pmlr_L2048_rho100_gamma1_V16
```

Local metadata:

```text
run_and_tumble/cluster_results/runs/diffusive_1d_pmlr/managed/diffusive_1d_pmlr_L2048_rho100_gamma1_V16/run_spec.yaml
```

Important parameters:

```text
L = 2048
rho0 = 100
D = 1
T = 1
gamma = 1
potential_type = ratchet_PmLr
fluctuation_type = profile_switch
potential strength V = 16
forcing magnitude = 0
forcing_rate_scheme = symmetric_normalized
bond_pass_count_mode = none
target replicas = 600
```

The local managed spec records this as `potential_strength: 16`; production
configs and aggregate filenames also expose it as `potential_magnitude`/`V16`.

Latest local aggregate at the time this note was written:

```text
run_and_tumble/cluster_results/runs/diffusive_1d_pmlr/managed/diffusive_1d_pmlr_L2048_rho100_gamma1_V16/aggregated/1D_pot-ratchet_PmLr_fluc-profile_switch_L2048_rho1.0e+02_eps0.00_a0.00_g1.000_D1.0_V16.0_f0.0_ffr0.0000_ic-aggregated_t568400000_tstats568400000_id-aggregated_managed_20260430m102826.jld2
```

There are older aggregates in the same folder with `g0.000` in the filename.
For the current `gamma=1, V=16` question, the `g1.000` aggregate is the relevant
one.

## Core Model

The relevant implementation is in:

```text
run_and_tumble/src/diffusive/modules_diffusive_no_activity.jl
run_and_tumble/src/common/potentials.jl
run_and_tumble/src/common/plot_utils.jl
run_and_tumble/load_and_plot.jl
```

The system is a periodic lattice gas of passive diffusive particles. Occupancy is
not excluded: many particles may occupy the same site. For a 1D run,
`N = round(rho0 * L)`, so the two runs above have:

```text
N = 100 * 2048 = 204800 particles.
```

One call to `update!` advances the simulation by one sweep. Internally, one sweep
contains `N` microsteps. Each microstep chooses one of `N` particles or one
potential-update channel, together with a left/right action. The potential
channel is therefore selected about once per sweep, up to the negligible
`N/(N+1)` correction.

For particle moves in 1D, the code chooses a candidate nearest-neighbor hop and
accepts it with a probability based on the potential difference and any directed
bond force on that bond.

With the current `symmetric_normalized` rate scheme,

```text
p(hop i -> j) =
    (D + F_directed(i -> j)) / (D + max_bond_force)
    * min(1, exp(-(V[j] - V[i]) / T)).
```

Here `F_directed` is positive along the active direction of a forced bond,
negative opposite to it, and zero on unforced bonds. The normalization keeps all
probabilities <= 1.

For the force run with `D=1, f=1, V=0`:

```text
on unforced bonds: p = 1 / 2
on the active forced direction: p = 1
on the reverse forced direction: p = 0
```

The force direction flips independently after each microstep with probability
`ffr / N`, clamped to `[0,1]`. Thus `ffr=1` means about one force-direction flip
per sweep. The flip is a deterministic toggle of the `BondForce.direction_flag`.

For the potential run there is no force, so `max_bond_force=0`. With `D=1,T=1`,
the hop acceptance is simply Metropolis:

```text
p(hop i -> j) = min(1, exp(-(V[j] - V[i]))).
```

The potential update channel accepts with probability `gamma`. Thus `gamma=1`
means roughly one potential profile refresh per sweep.

## Fluctuating Force Scheme

The force object is a `BondForce` with:

```text
bond_indices = ([1024], [1025])
direction_flag = true or false
magnitude = 1
```

When `direction_flag=true`, the active direction is `1024 -> 1025`. When false,
the active direction is `1025 -> 1024`. The code toggles this flag when a force
fluctuation event occurs.

The mean force direction is zero over long times if the telegraph process is
balanced. However, the instantaneous dynamics is strongly anisotropic on the
origin bond. In the current `D=f=1` case, the active direction is fully accepted
and the reverse direction is fully blocked.

Because the force is implemented directly in the hopping rate, not as a scalar
potential, it is best thought of as a localized fluctuating current drive at the
origin bond.

## Fluctuating Potential Scheme

The potential run uses:

```text
potential_type = ratchet_PmLr
fluctuation_type = profile_switch
potential strength = 16
```

In `potentials.jl`, `ratchet_PmLr` is a profile-switch potential built from four
two-site profiles:

```text
smudge
left_smudge
minus_smudge
left_minus_smudge
```

For `L=2048`, the helper currently constructs these profiles around
`location = L/2 = 1024`, so the nonzero sites are the pair near `1023,1024`.
This is close to the origin convention but is not identical to the force-run
bond `[1024,1025]`; this one-site or half-site convention should be kept in mind
when comparing centered plots.

The four profiles are positive/negative and left/right variants. With default
profile probabilities, a potential update samples one of the profiles uniformly.
The update is a random resampling, not a deterministic toggle. It may resample
the same profile as before.

The average potential over the four profiles is zero, but the hopping rates are
nonlinear in `V[j]-V[i]`. Therefore a zero mean potential does not imply zero
mean effect on correlations.

## Stored Density Observables

The simulation stores running time averages:

```text
rho_avg[i] = <rho_i>
rho_matrix_avg_cuts[:full][i,j] = <rho_i rho_j>
```

For 1D, the full second-moment matrix is usually available in
`state.rho_matrix_avg_cuts[:full]`. Aggregated files may also contain an exact
connected aggregate under:

```text
:agg_connected_corr_full_exact
```

The connected density correlation used in plotting is generally:

```text
C_raw(i,j) = <rho_i rho_j> - <rho_i><rho_j>
```

For the noninteracting diffusive particle model at fixed total particle number,
the plotting/collapse code often adds the independent-particle finite-N baseline
correction:

```text
C(i,j) = C_raw(i,j) + N / L^2.
```

For the current `L=2048, rho0=100` runs,

```text
N / L^2 = rho0 / L = 100 / 2048 = 0.048828125.
```

This matters when comparing to theory. Check whether the plotted quantity is
`C_raw` or the baseline-corrected `C`. In `load_and_plot.jl`, the helper
`connected_corr_mat_1d` adds this correction. In `plot_utils.jl`,
`one_dimensional_data_collapse_curves` also adds it before decomposition.

The diagonal/contact entries are often smoothed for plotting by replacing the
self/contact point with an average of neighboring values. Do not interpret the
smoothed contact point as a direct measurement.

## Symmetric and Antisymmetric Decomposition

The current issue concerns the reflection-even part of the density-correlation
cut around the localized drive.

For site-centered 1D cuts, the code reflects about an origin site, usually
`L/2` unless selected-site metadata provides another origin:

```text
C_sym(x)  = 0.5 * (C(x) + C(reflected x))
C_anti(x) = 0.5 * (C(x) - C(reflected x))
```

For bond-centered 1D collapse, used when a single forcing bond and a full
correlation matrix are available, the reflection center is the origin bond
center. For the force run this is the bond `[1024,1025]`, i.e. coordinate
`1024.5`.

The data-collapse machinery then plots curves such as:

```text
C(x, y) * y^n versus x / y
```

with separate output directories for:

```text
full_data
antisymmetric
symmetric
```

The simple long-distance expectation for a localized current-like dipole in 1D
is mostly antisymmetric under reflection about the origin. The observed issue is
that the `symmetric` component is not negligible in the simulation data.

## Why a Symmetric Part Is Plausible

The following are not conclusions, but useful hypotheses for theory discussions.

1. The force has zero mean but nonzero variance.

   In the force run, the instantaneous bond is strongly directional. Even if the
   mean current drive averages to zero, the current noise injected at the origin
   is localized and reflection-even at second order. A density covariance is a
   second-order observable, so it can couple to the variance of the drive, not
   only to the mean drive.

2. The potential has zero mean but nonlinear rates.

   The four `ratchet_PmLr` profiles average to zero as potentials, but the
   dynamics depends on `min(1, exp(-Delta V))`. A zero-mean profile ensemble can
   still produce a nonzero even correction to density correlations through
   nonlinear response and localized changes in escape/arrival rates.

3. The two fluctuation processes are temporally different.

   The force process toggles direction at event times. The potential process
   resamples one of four profiles at event times. With the same nominal rate
   (`ffr=1` versus `gamma=1`), their autocorrelation structures are not identical.

4. The force normalization changes the background hop rate.

   In `symmetric_normalized`, the presence of a force with `f=1` makes unforced
   bond hops accept with probability `1/2`, while the no-force potential run has
   free hops accepted with probability `1` away from potential gradients. Thus
   the force run is not just a local perturbation of an otherwise identical
   background clock; the normalization changes the effective diffusion scale.

5. Centering conventions can mix apparent symmetry.

   The force run is naturally bond-centered at `[1024,1025]`. The `ratchet_PmLr`
   potential helper places its two-site profiles at `1023,1024` for `L=2048`.
   A half-site or one-site mismatch can mix symmetric and antisymmetric pieces
   when comparing force-centered and potential-centered plots.

6. The fixed-N baseline correction is a uniform offset.

   Adding `N/L^2` is correct for removing the independent fixed-number
   anticorrelation baseline, but it is reflection-even. If the symmetric
   component under discussion is small, confirm whether it persists in both
   `C_raw` and the baseline-corrected `C`.

## Suggested Questions for Another LLM

When asking for theoretical help, frame the problem like this:

```text
I simulate noninteracting diffusive particles on a 1D periodic lattice with
fixed total particle number. A localized drive at the origin fluctuates either
as a two-state bond force or as a four-state local potential profile. The mean
drive is zero, but the density covariance C(i,j) shows a non-negligible
reflection-symmetric component after decomposing correlation cuts about the
origin. I want to understand whether this symmetric part is expected from
second-order response/noise injection, from the stochastic protocol, or from
analysis conventions such as baseline correction and centering.
```

Concrete things to ask:

```text
What is the expected symmetry of the density covariance response to a zero-mean
localized fluctuating current source?

Can the variance of a zero-mean local drive generate a reflection-even density
covariance in 1D diffusion?

How should the answer differ between a fluctuating bond force and a fluctuating
scalar potential with zero mean but nonlinear Metropolis rates?

What correlation function should be compared to fluctuating hydrodynamics:
<rho rho> - <rho><rho>, or the fixed-N-baseline-corrected version?

How sensitive is the symmetric/antisymmetric split to choosing a site-centered
origin versus a bond-centered origin?
```

## Code Pointers

Main numerical update:

```text
run_and_tumble/src/diffusive/modules_diffusive_no_activity.jl
  FPDiffusive.update!
  calculate_jump_probability
  directed_bond_forcing_1d
  bond_rate_prefactor
  potential_update!
  bondforce_update!
  update_and_compute_correlations!
```

Potential definitions:

```text
run_and_tumble/src/common/potentials.jl
  potential_args("ratchet_PmLr", ...)
  choose_potential(... fluctuation_type="profile_switch")
  ProfileSwitchPotential
  IndependentFluctuatingPoints
  BondForce
```

Correlation and plotting:

```text
run_and_tumble/src/common/plot_utils.jl
  connected_correlation_fix_term
  connected_correlation_matrix_1d
  reflection_decomposed_cut_1d
  reflection_decomposed_cut_1d_about_bond_center
  one_dimensional_data_collapse_curves

run_and_tumble/load_and_plot.jl
  connected_corr_mat_1d
  bond_centered_cut_1d
  default_bond_centered_collapse_1d
```

Managed run specs:

```text
run_and_tumble/cluster_scripts/init_managed_diffusive.sh
run_and_tumble/cluster_scripts/managed_diffusive_common.sh
run_and_tumble/cluster_scripts/managed_diffusive_1d_pmlr_README.md
run_and_tumble/cluster_scripts/managed_diffusive_1d_center_bond_README.md
```
