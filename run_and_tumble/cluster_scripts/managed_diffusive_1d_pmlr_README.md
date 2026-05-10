# Managed Diffusive 1D PmLr Runs

The PmLr managed workflow now uses the shared managed diffusive scripts. See
`cluster_scripts/managed_diffusive_README.md` for the common commands.

The old PmLr entrypoints still work as compatibility wrappers, for example:

```bash
bash cluster_scripts/init_managed_diffusive_1d_pmlr.sh \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --L 2048 \
  --rho 100 \
  --gamma 1 \
  --potential_strength 16
```

Equivalent shared command:

```bash
bash cluster_scripts/managed_diffusive.sh init \
  --case diffusive_1d_pmlr \
  --run_id diffusive_1d_pmlr_L2048_rho100_gamma1_V16 \
  --L 2048 \
  --rho 100 \
  --gamma 1 \
  --potential_strength 16
```
