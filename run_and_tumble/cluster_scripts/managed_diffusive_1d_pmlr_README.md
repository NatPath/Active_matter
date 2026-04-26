# Managed Diffusive 1D PmLr Runs

This workflow keeps one stable replica pool under:

```text
runs/diffusive_1d_pmlr/managed/<run_id>/
```

Each replica has one canonical checkpoint path, `replicas/<replica_id>/current.jld2`.
The checkpoint is replaced atomically. The companion `current.meta` records the
committed `elapsed_sweeps` and production `statistics_sweeps`.

## Normal Operation

Initialize or refresh the managed run metadata:

```bash
bash cluster_scripts/init_managed_diffusive_1d_pmlr.sh \
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

Import existing warmup states without duplicating state files:

```bash
bash cluster_scripts/import_managed_diffusive_1d_pmlr_warmups.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --source_state_dir runs/diffusive_1d_pmlr/warmup/diffusive_1d_pmlr_L2048_rho100_gamma1_V16_warmup_ns1000000_nr600_20260422-234330/states
```

Submit the next allocation. The scheduler first uses ready imported warmups,
then balances production replicas by lowest `statistics_sweeps`, then advances
warmups closest to threshold, then creates new random warmups until
`target_replica_count` is reached.

```bash
bash cluster_scripts/submit_managed_diffusive_1d_pmlr_batch.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --slots 60
```

Check status:

```bash
bash cluster_scripts/status_managed_diffusive_1d_pmlr.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16
```

## Aggregation

Aggregation is on demand. It uses only idle replicas by default, so it does not
read a checkpoint while that replica is actively being updated.

```bash
bash cluster_scripts/submit_managed_diffusive_1d_pmlr_aggregate.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --min_tstats 1
```

Fetch the latest aggregate and metadata:

```bash
bash cluster_scripts/copy_data_from_cluster.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --sync_scope aggregation
```

## Recovery

If Condor jobs are killed or held, first confirm they are not still running.
Then clear stale claims. This salvages `replicas/<id>/current.meta` when present,
so the next submission continues from the last committed checkpoint.

```bash
bash cluster_scripts/reclaim_managed_diffusive_1d_pmlr_claims.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --batch_id <batch_id>
```

Use `--dry_run` first if unsure. Use `--all_running` only after confirming no
managed jobs for this run are still active.
