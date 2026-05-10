# Managed Diffusive Runs

One shared managed workflow now handles passive 1D diffusive cases through a
case switch:

- `diffusive_1d_pmlr`: ratchet PmLr potential, no forcing.
- `single_origin_bond`: zero potential, one centered bond `[L/2, L/2+1]`.
  Bond-passage/J accumulation is disabled for managed runs so large-L
  production segments do not pay that extra cost.

The stable pool lives under:

```text
runs/<case>/managed/<run_id>/
```

Each replica has one canonical checkpoint path,
`replicas/<replica_id>/current.jld2`. The checkpoint is replaced atomically.
The companion `current.meta` records committed `elapsed_sweeps` and production
`statistics_sweeps`.

## Initialize

PmLr:

```bash
bash cluster_scripts/managed_diffusive.sh init \
  --case diffusive_1d_pmlr \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --L 2048 \
  --rho 100 \
  --gamma 1 \
  --potential_strength 16 \
  --warmup_threshold 1000000 \
  --segment_sweeps 1000000 \
  --checkpoint_interval 100000 \
  --target_replicas 600
```

Centered fluctuating bond:

```bash
bash cluster_scripts/managed_diffusive.sh init \
  --case single_origin_bond \
  --run_id single_origin_bond_L2048_rho100_f1_ffr1 \
  --L 2048 \
  --rho 100 \
  --force_strength 1 \
  --ffr 1 \
  --warmup_threshold 1000000 \
  --segment_sweeps 1000000 \
  --checkpoint_interval 100000 \
  --target_replicas 600
```

## Advance

```bash
bash cluster_scripts/managed_diffusive.sh submit \
  --case <case> \
  --run_id <managed_run_id> \
  --slots 60
```

The scheduler first uses ready imported warmups, then balances production
replicas by lowest `statistics_sweeps`, then advances warmups closest to the
threshold, then creates new random warmups until `target_replica_count` is
reached.

Each submitted batch includes a DAG final notification node. It uses the same
`cluster_scripts/dag_completion_notify.sh` path as the older DAG workflows, so
ntfy is controlled by `cluster_scripts/cluster_env.sh`:

```bash
export NOTIFY_NTFY_TOPIC="your-topic"
export NOTIFY_NTFY_SERVER="https://ntfy.sh"
```

The per-batch notification status log is written under
`runs/<case>/managed/<run_id>/batches/<batch_id>/notification/`. Set
`NO_DAG_NOTIFICATION=true` to generate a DAG without the final notification
node.

## Status

```bash
bash cluster_scripts/managed_diffusive.sh status \
  --case <case> \
  --run_id <managed_run_id>
```

## Aggregate

Aggregation is on demand and uses only idle replicas by default.

```bash
bash cluster_scripts/managed_diffusive.sh aggregate \
  --case <case> \
  --run_id <managed_run_id> \
  --min_tstats 1
```

The heavy aggregation work is submitted as a job by the wrapper above. The
worker helper is `cluster_scripts/aggregate_managed_diffusive.sh`.

## Copy And Plot

Single-origin bond collapse plots use the bond center rather than the left
site as the collapse origin by default:

```bash
bash cluster_scripts/copy_data_from_cluster.sh \
  --run_id single_origin_bond_L2048_rho100_f1_ffr1 \
  --sync_scope aggregation \
  --plot \
  --collapse_indices 20:20:100
```

To compare the latest fetched local aggregates for a PmLr run and a
single-origin bond run on one auto-scaled 1D collapse plot:

```bash
julia --startup-file=no utility_scripts/plot_paired_1d_data_collapse.jl \
  diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  single_origin_bond_L2048_rho100_f1_ffr1 \
  --collapse_indices 20:20:100
```

By default, the fluctuating-potential amplitude is kept as the reference scale
and the fluctuating-force curves are multiplied by the robust peak ratio. The
script saves paired `full_data` and `antisymmetric` collapse figures, plus
matching individual figures for each input case. All collapse figures include
the gray `ref ~ (x/y)/(1+(x/y)^2)^2` curve. Fluctuating-potential curves are solid/circle
lines, fluctuating-force curves are dashed/diamond lines, and the same color is
reused for matching `y` values. Use `--scale_mode unit_peak` to normalize each
run by its own robust peak instead.

## Import Warmups

```bash
bash cluster_scripts/managed_diffusive.sh import-warmups \
  --case <case> \
  --run_id <managed_run_id> \
  --source_state_dir <warmup_states_dir>
```

By default this registers existing files without copying or renaming them.

## Recovery

Only reclaim claims after confirming the selected Condor jobs are not running:

```bash
bash cluster_scripts/managed_diffusive.sh reclaim \
  --case <case> \
  --run_id <managed_run_id> \
  --batch_id <batch_id> \
  --dry_run

bash cluster_scripts/managed_diffusive.sh reclaim \
  --case <case> \
  --run_id <managed_run_id> \
  --batch_id <batch_id>
```

Use `--all_running` only after confirming no managed jobs for this run are
active. Reclaiming a still-running job can create two writers for the same
replica. The worker preserves statistics already saved in `current.jld2`, so a
stale ledger value from a reclaim should not reset production averages.

For long-running production, this is the usual continuation sequence:

```bash
bash cluster_scripts/managed_diffusive.sh status \
  --case single_origin_bond \
  --run_id single_origin_bond_L2048_rho100_f1_ffr1

bash cluster_scripts/managed_diffusive.sh reclaim \
  --case single_origin_bond \
  --run_id single_origin_bond_L2048_rho100_f1_ffr1 \
  --all_running \
  --dry_run

bash cluster_scripts/managed_diffusive.sh reclaim \
  --case single_origin_bond \
  --run_id single_origin_bond_L2048_rho100_f1_ffr1 \
  --all_running

bash cluster_scripts/managed_diffusive.sh submit \
  --case single_origin_bond \
  --run_id single_origin_bond_L2048_rho100_f1_ffr1 \
  --slots 600 \
  --sweeps 2000000
```

For a clean run, `status` should show the dominant
`checkpoint_aware_warmup_gap_counts` entry at the warmup threshold. Larger gaps
mean those replicas have less production statistics than their elapsed sweep
count suggests; future submits still weight aggregation by the saved statistics
counter, so the extra elapsed sweeps are not counted as production samples.

The old PmLr script names are kept as thin compatibility wrappers.
