#!/bin/bash

# filepath: /p/home/jusers/zhou17/juwels/spare-ml/zhou17/NanoAdam/huggingface_glue_mnli/slurm/submit_jobs_with_lr.sh

# Define the array of learning rates

SEEDS=(123 456 789 1000)
LRS=(0.001 0.1 0.8)


job_script="example/cispa_cluster_job.sh"

# Loop over each learning rate
for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Submitting job with learning rate: $lr and seed: $seed"

    # Submit the job with the learning rate as an argument
    sbatch --exclude=xe8545-a100-06 $job_script $lr $seed

    # Wait for a short time to avoid overwhelming the scheduler
    sleep 2
    
done

echo "All jobs submitted."