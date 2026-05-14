# Numerical Schemes Context for LLM Discussions

This note is a scheme-level handoff for the `run_and_tumble` numerics. It is
meant to help another LLM understand what the simulations are doing before
discussing physics, scaling, or expected signals. It deliberately stays away
from plot-specific analysis conventions and focuses on the update rules,
fluctuating potentials, fluctuating forces, two-force runs, and active objects.

## Main Code Paths

Core passive diffusive bath:

```text
run_and_tumble/src/diffusive/modules_diffusive_no_activity.jl
run_and_tumble/run_diffusive_no_activity.jl
```

Shared potentials and force objects:

```text
run_and_tumble/src/common/potentials.jl
```

Active-object extension:

```text
run_and_tumble/run_active_objects.jl
run_and_tumble/src/active_objects/modules_active_objects.jl
```

Common plotting and analysis helpers:

```text
run_and_tumble/src/common/plot_utils.jl
run_and_tumble/load_and_plot.jl
```

Cluster workflow examples:

```text
run_and_tumble/cluster_scripts/generate_single_origin_bond_configs.sh
run_and_tumble/cluster_scripts/generate_two_force_d_sweep_configs.sh
run_and_tumble/cluster_scripts/managed_diffusive_1d_pmlr_README.md
run_and_tumble/cluster_scripts/submit_active_objects_two_objects_hard_refresh_hist.sh
```

## Shared Diffusive Bath

The passive bath is a periodic lattice gas of independent diffusive particles.
Occupancy is not excluded: many particles can occupy the same lattice site.

For a 1D run:

```text
N = round(rho0 * L)
```

For a 2D run:

```text
N = round(rho0 * Lx * Ly)
```

The state stores particle positions, the instantaneous density field `rho`, the
time-averaged density `rho_avg`, selected second-moment/correlation arrays, and
optional bond-passage statistics. The same bath update is used by the
single-force, two-force, fluctuating-potential, and active-object simulations.

One call to `FPDiffusive.update!` advances the system by one sweep. In 1D, a
sweep consists of `N` microsteps. At each microstep, the code randomly chooses
one of the `N` particles or one extra potential-update channel, and chooses a
left/right candidate direction. In 2D, the same idea is used with four candidate
directions.

For a particle candidate move from site `i` to nearest neighbor `j`, the
acceptance probability is

```text
p(i -> j) =
    force_prefactor(i -> j) * min(1, exp(-(V[j] - V[i]) / T)).
```

The scalar potential contribution is Metropolis-like. The bond-force
contribution enters through `force_prefactor`, described below.

If the potential channel is selected, the code accepts a potential update with
probability `gamma`. Thus `gamma = 1` gives roughly one accepted potential
refresh per sweep.

After each microstep, force directions may also fluctuate independently. For
force `a`, the direction flips with probability

```text
p_flip_a = ffr_a / N
```

clamped to `[0, 1]`. Thus `ffr_a = 1` means roughly one direction flip per
sweep for that force.

## Potential Objects

Potential types are constructed in `potentials.jl`.

The main containers are:

```text
Potential
    V                 scalar lattice potential
    fluctuation_mask  deterministic additive update mask
    fluctuation_sign  sign toggled by plus-minus/reflection style updates

IndependentFluctuatingPoints
    V                 scalar lattice potential
    indices           independently fluctuating sites
    magnitude         fluctuation scale
    fluctuation_statistics

ProfileSwitchPotential
    potentials        finite list of potential profiles
    probabilities     sampling weights
    V                 currently active profile
    current           active profile index
```

For a simple `Potential`, `potential_update!` adds
`fluctuation_mask * fluctuation_sign` and flips the sign. For
`IndependentFluctuatingPoints`, an RNG is required and selected sites are
resampled from the requested distribution. For `ProfileSwitchPotential`, the
active profile is randomly resampled according to `probabilities`.

## Fluctuating PmLr Potential

The 1D PmLr case uses:

```text
potential_type = ratchet_PmLr
fluctuation_type = profile_switch
```

`ratchet_PmLr` is a four-profile local potential ensemble:

```text
smudge
left_smudge
minus_smudge
left_minus_smudge
```

These are local two-site profiles near the middle of the ring. A profile-switch
update chooses one of the four profiles; it is a random resampling, not a
deterministic cycle. With the default weights, the four profiles are sampled
uniformly.

This scheme is useful for a localized fluctuating scalar potential. The
potential itself has zero average over the four left/right and plus/minus
profiles, but the particle hopping rates depend nonlinearly on potential
differences.

Representative managed run:

```text
diffusive_1d_pmlr_L2048_rho100_gamma1_V16
L = 2048
rho0 = 100
D = 1
T = 1
gamma = 1
potential strength = 16
forcing magnitude = 0
```

## Bond Force Objects

Directed localized forcing is represented by `BondForce`:

```text
BondForce
    bond_indices    ([site_left], [site_right]) in 1D
    direction_flag  true means endpoint 1 -> endpoint 2
    magnitude       force strength
```

The force is not a scalar potential. It directly modifies hopping rates on the
specified bond. In 1D, the helper `directed_bond_forcing_1d` returns:

```text
+f  for a hop along the active direction of the bond
-f  for a hop opposite to the active direction
 0  for hops on all other bonds
```

Two rate schemes exist:

```text
legacy_penalty
symmetric_normalized
```

The currently preferred diffusive force workflow uses `symmetric_normalized`.
For this scheme,

```text
force_prefactor(i -> j) =
    (D + F_directed(i -> j)) / (D + max_bond_force).
```

Some older workflows and the default settings in `run_active_objects.jl` use
`legacy_penalty` unless the config overrides it. In that scheme,

```text
force_prefactor(i -> j) = D - max(F_directed(i -> j), 0)
```

with no `D + max_bond_force` normalization. The final candidate probability is
still clamped to `[0, 1]` before accepting/rejecting the move. For force-driven
runs, always check the saved config or run spec before comparing results across
families.

Under `symmetric_normalized`, with `D = 1`, force magnitude `f = 1`, and zero
potential:

```text
unforced bonds:          p = 1 / 2
active forced direction: p = 1
reverse forced direction: p = 0
```

The force direction flips by toggling `direction_flag`. With `ffr = 1`, this
toggle happens about once per sweep.

## Single Origin Bond Force

The single-origin-bond run places one fluctuating force bond at the center of a
1D periodic system.

Representative managed run:

```text
single_origin_bond_L2048_rho100_f1_ffr1
L = 2048
rho0 = 100
D = 1
T = 1
gamma = 0
potential_type = zero
fluctuation_type = no-fluctuation
force bond = [1024, 1025]
force magnitude = 1
ffr = 1
forcing_rate_scheme = symmetric_normalized
```

This is a localized fluctuating current drive. The potential is zero. The only
drive is the alternating direction of the center bond.

## Two-Force-D Runs

The `two_force_d` family uses the same passive diffusive bath and the same
`BondForce` update rules, but it places two independent fluctuating force bonds
in the 1D ring. These runs are used to study how responses depend on the
separation between two localized fluctuating current sources.

Generated configs live under:

```text
run_and_tumble/configuration_files/two_force_d_sweep/warmup/d_<d>.yaml
run_and_tumble/configuration_files/two_force_d_sweep/production/d_<d>.yaml
```

They are generated by:

```text
run_and_tumble/cluster_scripts/generate_two_force_d_sweep_configs.sh
```

The default generated setup is:

```text
dim_num = 1
potential_type = zero
fluctuation_type = no-fluctuation
gamma = 0
D = 1
T = 1
forcing_magnitudes = [1, 1]
ffrs = [1, 1]
forcing_direction_flags = [true, true]
bond_pass_count_mode = all_forcing_bonds
```

The two bonds are placed symmetrically around the center of the ring. In the
generator, `d` is the site displacement from the right endpoint of the left
bond to the left endpoint of the right bond. Equivalently, there are `d - 1`
empty lattice sites between the two nearest bond endpoints.

For a system of length `L`, with `center = L / 2`, the generated positions are:

```text
left_bond_right = center - floor(d / 2)
left_bond_left  = left_bond_right - 1

right_bond_left  = center + ceil(d / 2)
right_bond_right = right_bond_left + 1
```

with periodic wrapping.

Each force has its own `direction_flag` and its own `ffr`. The two force
directions therefore fluctuate independently. During each particle move, the
directed forcing contribution is the sum of all active bond forces that match
the candidate hop. Since the two-force configs use distinct bonds, this usually
means that only one of the two forces affects a given hop.

The run family records bond-passage statistics for both force bonds. For each
sweep, the code counts forward and reverse passages through each tracked bond.
Those counts are later used for current-like observables and aggregate
analysis.

Conceptually, a `two_force_d` run is:

```text
periodic 1D independent diffusive particles
two localized fluctuating directed bonds
zero scalar potential
independent telegraph direction process on each bond
parameter d controls the distance between the two local drives
```

## Active Objects

`run_active_objects.jl` extends the same diffusive bath by making force bonds
mobile. Each active object is a nearest-neighbor force bond. The object position
is represented by the left endpoint of that bond:

```text
object at left_site x  <=>  force bond [x, x + 1]
```

The active-object module canonicalizes each force bond to this left-site
representation on the 1D ring. The force direction can still point either way
through that bond, depending on `direction_flag`.

The active-object runner currently supports only `dim_num = 1`.
Its default `forcing_rate_scheme` is `legacy_penalty`; cluster/config files may
override this, so the active-object force convention should be read from the
specific YAML used for the run.

### Bath Update

Each active-object sweep first updates the bath by calling the same diffusive
correlation/update path:

```text
update_and_compute_correlations!(state, param, ..., collect_statistics=true)
```

This means that during the bath update:

```text
particles hop with the passive-diffusive / potential / force rule
force directions may fluctuate according to ffr
bond passage counts are collected for the object bonds
```

After the bath sweep, the active-object code reads the latest passage counts:

```text
rightward_counts, leftward_counts = latest_bond_passage_counts(state)
```

These counts are then used to propose object moves.

### Object Motion Conventions

For object motion, the code converts bath passage counts into left/right object
move rates. The sign convention is:

```text
rightward bath passages through an object contribute to that object's left-move rate
leftward bath passages through an object contribute to that object's right-move rate
```

In other words, the object tends to move opposite to the bath flux through its
bond, with strength controlled by `object_kappa`, plus optional unbiased object
diffusion `object_D0`.

A proposed object move is sampled from rates:

```text
lambda_left
lambda_right
lambda_total = lambda_left + lambda_right
p_move = 1 - exp(-lambda_total)
```

If a move occurs, the object moves left with probability
`lambda_left / lambda_total`, otherwise right. The accepted displacement is
`-1`, `0`, or `+1` in the left-site coordinate.

When multiple objects exist, proposed moves are applied in a random object
order. A proposed move is rejected if another object's left site already
occupies the target left site. Accepted moves update the underlying force-bond
location immediately.

### Object Motion Schemes

The active-object module supports three object-motion schemes.

#### hard_refresh

The object accumulates bond-passage counts over a fixed window of
`object_refresh_sweeps`. At the end of the window:

```text
lambda_left  = object_D0 * object_refresh_sweeps + object_kappa * window_forward
lambda_right = object_D0 * object_refresh_sweeps + object_kappa * window_reverse
```

Then a move is sampled, the window counters are cleared, and a new window
begins.

This is the main scheme used by the hard-refresh histogram cluster scripts.

#### exponential_memory

The object maintains exponentially weighted memory of forward and reverse
passages:

```text
alpha = 1 / object_memory_sweeps
memory_forward = (1 - alpha) * memory_forward + alpha * rightward_counts
memory_reverse = (1 - alpha) * memory_reverse + alpha * leftward_counts
```

Every sweep, object move rates are computed from this memory:

```text
lambda_left  = object_D0 + object_kappa * memory_forward
lambda_right = object_D0 + object_kappa * memory_reverse
```

#### per_hop_probability

This scheme interprets `object_kappa` as a per-hop trigger probability. The code
maps it to an effective rate:

```text
hop_rate_scale = -log(1 - object_kappa)
```

and then uses:

```text
lambda_left  = object_D0 + hop_rate_scale * rightward_counts
lambda_right = object_D0 + hop_rate_scale * leftward_counts
```

The required range is:

```text
0 <= object_kappa < 1
```

### Two Active Objects

For two active objects, `run_active_objects.jl` can infer initial force-bond
locations from `forcing_distance_d`, using logic similar to the passive
two-force setup:

```text
left_site = center - floor((d + 1) / 2)
right_site = left_site + d + 1

object 1 bond = [left_site, left_site + 1]
object 2 bond = [right_site, right_site + 1]
```

Again, `d` is the displacement from the right endpoint of the left bond to the
left endpoint of the right bond; there are `d - 1` empty sites between the
nearest endpoints.

The cluster helper
`submit_active_objects_two_objects_hard_refresh_hist.sh` builds hard-refresh
two-object histogram runs. It sets:

```text
object_motion_scheme = hard_refresh
object_refresh_sweeps = ots
object_memory_sweeps = ots
object_history_interval = ots
object_kappa = kappa
forcing_distance_d = d
```

The active-object state records object history, including:

```text
history_sweeps
history_left_sites
history_move_deltas
history_rightward_counts
history_leftward_counts
history_forward_distance
history_min_distance
history_forward_gap
history_min_gap
history_min_edge_distance
```

These histories are what the histogram workflows aggregate.

## Statistics That Are Stored

For all these schemes, the state stores running averages of density observables
and optional bond/object observables. The most important fields for LLM-level
context are:

```text
rho
    instantaneous density field

rho_avg
    time-averaged density field

rho_matrix_avg_cuts
    time-averaged pair-density arrays or selected cuts

bond_pass_stats
    per-bond passage statistics when bond_pass_count_mode enables tracking

object_stats
    active-object memory, latest counts, latest moves, and history arrays
```

Warmup/production workflows may reset measurement statistics after warmup while
preserving the physical state. In active-object runs, the warmup reset preserves
the object memory state and the object positions.

## Compact Prompt for Another LLM

Use this if you want to ask another LLM for theoretical help:

```text
I simulate independent diffusive particles on a periodic 1D lattice. Particles
hop to nearest neighbors with Metropolis rates in a scalar potential and with
optional localized directed bond-force prefactors. Bond forces are telegraph
variables: each force has a direction flag and flips independently at rate ffr
per sweep. A fluctuating-potential run instead randomly switches among local
potential profiles at rate gamma per sweep. In two_force_d runs, two such
localized fluctuating directed bonds are placed at separation d. In
run_active_objects.jl, those force bonds become mobile active objects: after
each bath sweep, passage counts through each object bond are converted into
left/right object move rates, and accepted moves shift the force bond by one
lattice site. I want to reason about the continuum or hydrodynamic description
of these numerical schemes.
```

Useful follow-up questions:

```text
What continuum source term corresponds to a fluctuating directed bond force?

How does a zero-mean fluctuating scalar potential differ from a zero-mean
fluctuating bond force?

For two independent localized fluctuating bonds separated by d, what large-scale
density or current correlations should one expect?

For active objects whose motion is driven by passage counts through their bonds,
what effective stochastic dynamics should describe the object positions?
```
