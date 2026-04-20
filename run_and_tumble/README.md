# Simulation Runners

This README documents the local simulation entry points in this directory. It intentionally does not cover `cluster_scripts/` or `condor_scripts/`.

Everything below assumes `run_and_tumble/` itself is the working directory and project root.

## Setup

Julia `1.11.5` was used to verify the command-line interfaces documented here. `Project.toml` and a pinned `Manifest.toml` are now checked in, so a clean setup on another machine should be:

1. Enter this directory.
2. Instantiate the Julia environment once.
3. Run simulations with `julia --project=.` so Julia uses the local dependency set.

```bash
cd run_and_tumble
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

After that, use commands of the form:

```bash
julia --project=. <runner>.jl --config configuration_files/<file>.yaml
```

Notes:

- The first setup or first run on a fresh machine can take a while because Julia may download packages, precompile them, and fetch plotting artifacts.
- On headless machines, or if plotting causes backend/display issues, add `--performance_mode` to the run command.
- The checked-in Julia environment covers the simulation runners and common local plotting/analysis dependencies.

## Which runner should I use?

- `run_diffusive_no_activity.jl`
  Passive diffusive particles on the lattice. No run-and-tumble terms (`α` and `ϵ` are not used here). Supports 1D and 2D. This is the current runner to use for "run_diffusive".
- `run_and_tumble.jl`
  Run-and-tumble particles. Use this when you want nonzero tumbling/activity parameters such as `α` and `ϵ`. Supports 1D and 2D.
- `run_ssep.jl`
  Symmetric simple exclusion process (SSEP). Here `ρ₀` is a filling fraction, so keep `0 <= ρ₀ <= 1`. Supports the default discrete-sweep dynamics and the 1D continuous-time mode `simulation_mode: "ctmc_1d"`.
- `run_active_objects.jl`
  Diffusive particles coupled to moving active objects or moving force bonds. This runner is currently 1D only: `dim_num` must be `1`.

## Common command pattern

All local runs follow the same basic pattern:

```bash
julia --project=. <runner>.jl --config configuration_files/<file>.yaml
```

Useful flags that appear on one or more runners:

- `--config <path>`: load parameters from YAML.
- `--continue <state.jld2>`: continue a saved state from its current time.
- `--continue_sweeps <N>`: override the number of additional sweeps when continuing.
- `--initial_state <state.jld2>`: start from a saved configuration but reset accumulated statistics. Supported by `run_diffusive_no_activity.jl`, `run_and_tumble.jl`, and `run_ssep.jl`.
- `--performance_mode`: disable plotting and progress output for lean or headless runs.
- `--num_runs <N>`: launch multiple independent runs before aggregation. Supported by the diffusive, RTP, and SSEP runners.
- `--estimate_only`, `--estimate_runtime`, `--estimate_sample_size <N>`: runtime-estimation options. Supported by the diffusive runner, and partially by the active-object runner.

Saved states are written as `.jld2` files under the `save_dir` defined in the YAML.

## Current config keys

Most runners use the same core YAML structure:

- Geometry and density: `dim_num`, `L`, `ρ₀`, `D`, `T`
- Dynamics length: `n_sweeps`, `warmup_sweeps`
- Potential and fluctuations: `potential_type`, `fluctuation_type`, `potential_magnitude`, `γ`
- Initial condition: `ic`
- Forcing: `forcing_type`, `forcing_types`, `forcing_bond_pairs`, `forcing_distance_d`, `forcing_magnitude`, `forcing_magnitudes`, `ffr`, `ffrs`, `forcing_direction_flags`, `forcing_rate_scheme`, `bond_pass_count_mode`
- Output: `show_times`, `save_times`, `save_dir`, `description`

Mode-specific additions:

- `run_and_tumble.jl`: `α`, `ϵ`
- `run_ssep.jl`: `simulation_mode`, `correlation_observable_mode`, `correlation_cut_offsets`, `full_corr_tensor`
- `run_active_objects.jl`: `object_motion_scheme`, `object_refresh_sweeps`, `object_memory_sweeps`, `object_kappa`, `object_D0`, `object_history_interval`

## Quick examples

- Passive diffusion, 1D, zero potential:

```bash
julia --project=. run_diffusive_no_activity.jl \
  --config configuration_files/diffusive_no_activity.yaml
```

- Passive diffusion, 1D, single fluctuating bond:

```bash
julia --project=. run_diffusive_no_activity.jl \
  --config configuration_files/diffusive_single_forcing.yaml
```

- Run-and-tumble particles:

```bash
julia --project=. run_and_tumble.jl \
  --config configuration_files/params_1d_small.yaml
```

- SSEP, discrete sweep:

```bash
julia --project=. run_ssep.jl \
  --config configuration_files/ssep_single_forcing.yaml
```

- SSEP, 2D discrete sweep:

```bash
julia --project=. run_ssep.jl \
  --config configuration_files/ssep_single_forcing_2d.yaml
```

- SSEP, 1D CTMC:

```bash
julia --project=. run_ssep.jl \
  --config configuration_files/ssep_ctmc_single_center_bond_debug.yaml
```

## How to run the 2D diffusive mode

Use `run_diffusive_no_activity.jl`. There is no separate `run_diffusive.jl`; the passive/diffusive mode is selected by using this runner, and 2D is selected by setting `dim_num: 2` in the YAML.

### Minimal 2D diffusive config

Create a YAML file such as `configuration_files/diffusive_2d.yaml` with:

```yaml
dim_num: 2
L: 32
ρ₀: 1000
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: 10000
warmup_sweeps: 0
description: "diffusive_2d_example"

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_type: "center_bond_x"
forcing_magnitude: 1.0
ffr: 1.0
forcing_direction_flags: [true]
forcing_rate_scheme: "symmetric_normalized"
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "saved_states/diffusive_2d"
```

Then run:

```bash
julia --project=. run_diffusive_no_activity.jl \
  --config configuration_files/diffusive_2d.yaml
```

Useful variants:

- Headless or no plots:

```bash
julia --project=. run_diffusive_no_activity.jl \
  --config configuration_files/diffusive_2d.yaml \
  --performance_mode
```

- Runtime estimate only:

```bash
julia --project=. run_diffusive_no_activity.jl \
  --config configuration_files/diffusive_2d.yaml \
  --estimate_only \
  --estimate_sample_size 200
```

- Continue an existing saved state:

```bash
julia --project=. run_diffusive_no_activity.jl \
  --continue path/to/state.jld2 \
  --continue_sweeps 5000 \
  --performance_mode
```

Notes for this 2D mode:

- `forcing_type: "center_bond_x"` means the force is applied on the horizontal bond through the center. For even `L` this is `([L/2, L/2] -> [L/2 + 1, L/2])`.
- Use `forcing_type: "center_bond_y"` if you want the vertical center bond instead.
- If you want pure diffusion with no forcing, keep the runner the same and set `forcing_magnitude: 0.0` and `ffr: 0.0`.
- If `forcing_rate_scheme` is omitted in `run_diffusive_no_activity.jl`, the default is now `symmetric_normalized`.

### Forcing options in 2D

- `forcing_type: "center_bond_x"`: one horizontal bond through the center.
- `forcing_type: "center_bond_y"`: one vertical bond through the center.
- `forcing_types: ["center_bond_x", "center_bond_y"]`: multiple standard centered bonds.
- `forcing_bond_pairs`: explicit bonds. In 2D each entry must look like `[[x1, y1], [x2, y2]]`.
- `forcing_direction_flags`: one Boolean per force. `true` means the active direction is from the first endpoint to the second; `false` reverses it.
- `forcing_magnitude` or `forcing_magnitudes`: strength of each force.
- `ffr` or `ffrs`: expected direction flips per sweep per force. `ffr: 0.0` keeps the bond direction fixed; larger values flip it more often.

### `forcing_rate_scheme` in 2D

The scheme changes how the force modifies hop rates on forced bonds. In 2D the logic is the same as in 1D: only hops that cross a forced bond are affected; all other lattice hops still use the baseline diffusive rate `D`.

If a proposed hop crosses a forced bond, the code first assigns it a signed forcing:

- `+f` along the currently active bond direction
- `-f` against the currently active bond direction

That signed forcing is then converted into a hop-rate prefactor.

- `legacy_penalty`
  Aliases accepted by the code: `legacy`, `current`.
  Prefactor: `D - max(signed_force, 0)`.
  Essence: only one direction is penalized, while the opposite direction stays at the baseline rate.
  For a single forced bond with no potential, this gives `D - f` in the active direction and `D` in the opposite direction.
- `symmetric_normalized`
  Aliases accepted by the code: `symmetric`, `normalized_symmetric`.
  Prefactor: `(D + signed_force) / (D + f_max)`, where `f_max` is the largest total forcing magnitude on any bond.
  Essence: the two directions are treated symmetrically through a `D ± f` form, then normalized by `D + f_max` so the overall update scale stays bounded.
  For a single forced bond with no potential, this gives `1` in the active direction and `(D - f) / (D + f)` in the opposite direction.
  This is the default scheme for `run_diffusive_no_activity.jl`.

If you also have a nonzero potential, the same prefactor is multiplied by the usual Metropolis factor `min(1, exp(-ΔV/T))`.

Practical guidance:

- Use `legacy_penalty` when you explicitly want backward compatibility with older diffusive runs.
- Use `symmetric_normalized` when you want the default current diffusive behavior and a more explicit `D ± f` bias.
- Keeping `forcing_magnitude <= D` is the clearest regime. If `f > D`, one of the directional rates can go negative before the code clamps probabilities back into `[0, 1]`.

## Important compatibility note

The current runners read `ρ₀` and `γ` from YAML. Some older example files in `configuration_files/` still use older names such as `N` or `γ′`. Treat those files as legacy templates, not as guaranteed drop-in configs for the current runners.

In practice:

- For current local runs, prefer `ρ₀` over `N`.
- For current local runs, prefer `γ` over `γ′`.
- For SSEP only, remember that `ρ₀` is an occupancy fraction, not particles per site.

## Where to look next

- Use `julia --project=. <runner>.jl --help` to inspect the exact CLI flags for a runner.
- Use `load_and_plot.jl` and the scripts in `analysis/` after you have saved `.jld2` states.
