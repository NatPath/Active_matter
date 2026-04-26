# Run And Tumble Repo Map

This repo now has a split between:

- stable public entry points at the repo root
- implementation code under `src/`

The goal is to keep the public commands simple while making the implementation layout easier to navigate.

## Stable Entry Points

These top-level files are still the public interface:

- `run_diffusive_no_activity.jl`
- `run_active_objects.jl`
- `run_ssep.jl`
- `run_and_tumble.jl`
- `load_and_plot.jl`
- `load_and_plot_diffusive_current_stats.jl`
- `aggregate_results.jl`
- `run_benchmark.jl`
- `run_batch_analysis.jl`
- `cluster_scripts/*.sh`
- `configuration_files/*.yaml`

Important:

- The runner and analysis scripts at the repo root are real entry points, not wrappers.
- The implementation modules live under `src/`.
- The old top-level module files (`modules_*.jl`, `save_utils.jl`, `plot_utils.jl`, `potentials.jl`) are gone.

## Source Layout

Implementation code lives under:

- `src/common/`
  - shared utilities used by multiple simulation families
  - `potentials.jl`
  - `plot_utils.jl`
  - `save_utils.jl`
- `src/diffusive/`
  - diffusive non-interacting simulation code
- `src/active_objects/`
  - active-object dynamics on top of the diffusive bath
- `src/ssep/`
  - SSEP simulation code
- `src/run_and_tumble/`
  - run-and-tumble simulation code

## Where To Look First

If you come back after a while, this is the shortest path:

1. Pick the family:
   - diffusive: `run_diffusive_no_activity.jl`
   - active objects: `run_active_objects.jl`
   - ssep: `run_ssep.jl`
   - run-and-tumble: `run_and_tumble.jl`
2. Open the matching runner at the repo root.
3. From there, follow the matching module under `src/<family>/`.
4. Use `src/common/` for shared save/plot/potential logic.

## Configs

Configs still live under `configuration_files/`.

Current conventions:

- keep canonical configs with parameter names that are stable identifiers
- do not encode analysis cutoffs like `tr` in the config filename
- use suffixes only for workflow meaning, for example:
  - `_local`
  - `_smoke`
  - `_local_smoke`

## Cluster Workflow

The cluster directory is effectively a manually synced copy of this repo. That means:

1. Sync code first:
   - `bash cluster_scripts/copy_things_to_cluster.sh`
2. Submit jobs through `cluster_scripts/submit_*.sh`
3. Inspect runs on the cluster with the relevant `inspect_*.sh`
4. Pull back only what you need with `copy_data_from_cluster.sh`

Notes:

- Put machine-specific cluster defaults in a local ignored file:
  - `cluster_scripts/cluster_env.sh`
  - start from `cluster_scripts/cluster_env.example.sh`
- `copy_things_to_cluster.sh` now copies:
  - top-level entry points
  - `src/`
  - `utility_scripts/`
  - `cluster_scripts/`
  - top-level configs
  - root markdown docs
- `copy_data_from_cluster.sh --run_id ...` checks both configured roots by default:
  - `CLUSTER_DATA_ROOT`
  - `CLUSTER_CODE_ROOT`
  and fetches from the root where that `run_id` was actually found
- heavy work should go through submit wrappers, not direct execution on the login node

### Cluster file-count inspection

Use the read-only inspector to rank file-heavy directories before deleting anything:

```bash
bash cluster_scripts/inspect_file_usage.sh --depth 2 --top 20
```

It prints:

- shallow inode summaries for the resolved cluster/repo roots
- repo-specific cleanup candidates such as:
  - `two_force_d/add_repeats_jobs/*/dag_snippets`
  - `*/submit`
  - `active_objects/.../histograms/per_run`
  - `manual_aggregate_jobs`
  - aggregate archive directories
  - raw `topup_batches` / `repeat_batches`

The script does not delete anything. Treat `run_info.txt`, `manifest.csv`, configs, and the latest live aggregate/histogram outputs as the core provenance to keep.

## Common Workflows

### Local simulation

```bash
julia --startup-file=no run_active_objects.jl \
    --config configuration_files/active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_k5e-5.yaml
```

### Cluster active-object histogram run

```bash
bash cluster_scripts/submit_active_objects_histogram_dag.sh \
    --config configuration_files/active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_k5e-5.yaml \
    --num_replicas 10 \
    --n_sweeps 100000 \
    --tr 0
```

### Copy aggregate results back by `run_id`

```bash
bash cluster_scripts/copy_data_from_cluster.sh \
    --run_family active_objects \
    --run_id <run_id> \
    --sync_scope aggregation \
    --plot
```

### Rebuild active-object histograms from already saved states

```bash
bash cluster_scripts/submit_active_objects_saved_states_into_histograms.sh \
    --run_id <run_id>
```

## Safe Editing Rules

To avoid breaking workflows:

- keep the top-level public filenames stable
- move implementation changes into `src/`
- treat `cluster_scripts/submit_*.sh` as public interfaces
- when changing output naming or run metadata, preserve `run_id`-based workflows
- when adding new Julia helpers needed on the cluster, make sure `copy_things_to_cluster.sh` syncs them
- after structural changes, verify:
  - `run_active_objects.jl --help`
  - `run_diffusive_no_activity.jl --help`
  - `run_ssep.jl --help`
  - `load_and_plot.jl --help`
  - a `--no_submit` smoke generation for the affected cluster submit wrapper

## Recommended Comfort Improvements

These are low-risk habits that help under Condor:

- always run `copy_things_to_cluster.sh` right before submitting after code changes
- prefer `--no_submit` smoke generation for new submit wrappers
- keep one canonical config per physical setup, then derive local/smoke variants explicitly
- use aggregate-only recovery scripts when simulation states already exist
- inspect `run_info.txt`, `manifest.csv`, submit files, and logs before assuming a physics or code failure
- keep runtime estimation and aggregation submission separate in your head: simulation first, histogram/aggregation second
- when a run has many replicas, make the aggregation path independently rerunnable from saved states
