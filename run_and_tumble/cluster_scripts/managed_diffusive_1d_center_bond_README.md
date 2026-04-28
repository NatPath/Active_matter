# Managed Diffusive 1D Center-Bond Runs

This workflow keeps one stable replica pool under:

```text
runs/diffusive_1d_center_bond/managed/<run_id>/
```

The default physical setup is:

```text
L=2048, rho=100, potential_type=zero, f=1, ffr=1, forcing_type=center_bond_x
```

## Normal Operation

Initialize the managed run metadata:

```bash
bash cluster_scripts/init_managed_diffusive_1d_center_bond.sh \
  --run_id diffusive_1d_center_bond_L2048_rho100_f1_ffr1 \
  --L 2048 \
  --rho 100 \
  --force_strength 1 \
  --ffr 1 \
  --warmup_threshold 1000000 \
  --segment_sweeps 1000000 \
  --checkpoint_interval 100000 \
  --target_replicas 600
```

Submit the next allocation:

```bash
bash cluster_scripts/submit_managed_diffusive_1d_center_bond_batch.sh \
  --run_id diffusive_1d_center_bond_L2048_rho100_f1_ffr1 \
  --slots 60
```

Check status:

```bash
bash cluster_scripts/status_managed_diffusive_1d_center_bond.sh \
  --run_id diffusive_1d_center_bond_L2048_rho100_f1_ffr1
```

## Aggregation

Aggregation uses only idle replicas by default.

```bash
bash cluster_scripts/submit_managed_diffusive_1d_center_bond_aggregate.sh \
  --run_id diffusive_1d_center_bond_L2048_rho100_f1_ffr1 \
  --min_tstats 1
```

Fetch the latest aggregate and metadata:

```bash
bash cluster_scripts/copy_data_from_cluster.sh \
  --run_id diffusive_1d_center_bond_L2048_rho100_f1_ffr1 \
  --sync_scope aggregation
```

## Recovery

If jobs are killed or held, confirm they are not still running, then clear stale
claims:

```bash
bash cluster_scripts/reclaim_managed_diffusive_1d_center_bond_claims.sh \
  --run_id diffusive_1d_center_bond_L2048_rho100_f1_ffr1 \
  --batch_id <batch_id>
```

Use `--dry_run` first if unsure.
