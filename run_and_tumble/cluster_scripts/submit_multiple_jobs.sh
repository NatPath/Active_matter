#!/bin/bash

# Loop over your sweeps
for SWEEPS in $(seq 1000000 1000040)
do
    echo "Submitting sweep ${SWEEPS}"

    # Create a temporary submit file for each sweep
    cat > "submit_${SWEEPS}.sub" <<EOL
Universe   = vanilla
Executable = rtp_from_config.sh
arguments  = generated_configs/params_${SWEEPS}.yaml
should_transfer_files = NO
output     = condor_logs/job_${SWEEPS}.out
error      = condor_logs/job_${SWEEPS}.err
log        = condor_logs/job_${SWEEPS}.log
request_cpus = 10
request_memory = 40 GB
queue
EOL

    # Submit the job
    condor_submit "submit_${SWEEPS}.sub"

done

echo "All jobs submitted."
