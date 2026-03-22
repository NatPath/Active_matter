#!/bin/bash

# Directory where config files will be stored
CONFIG_DIR="generated_configs"
mkdir -p "$CONFIG_DIR"
START_I=1000000
END_I=1000040
for SWEEPS in $(seq $START_I $END_I)
do
    CONFIG_FILE="$CONFIG_DIR/params_${SWEEPS}.yaml"
    cp "$CONFIG_DIR/params_template.yaml" "$CONFIG_FILE"  # Copy from the original template
    sed -i "s/^n_sweeps:.*/n_sweeps: ${SWEEPS}/" "$CONFIG_FILE"
done

echo "Generated new configs for sweeps $START_I to $END_I inside $CONFIG_DIR/"
