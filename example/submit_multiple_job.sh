#!/bin/bash

# filepath: /p/home/jusers/zhou17/juwels/spare-ml/zhou17/NanoAdam/huggingface_glue_mnli/slurm/submit_jobs_with_lr.sh

# Define the array of learning rates

SEEDS=(7 42 1234)
LRS=(1e-1 7e-5)
wd=0.0
optimiser="adamw"
dataset="cola"
EPOCHS=(268 536 804 1072 1340)

job_script="example/cispa_cluster_job.sh"


# Loop over each learning rate
for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do
            echo "Submitting job with opt: $optimiser, learning rate: $lr, seed: $seed, weight decay: $wd"

            # # Submit the job with the learning rate as an argument
            # sbatch $job_script $lr $seed $optimiser

            # source ./example/run.sh $lr $seed $optimiser $wd

            source ./example/run_experiments_LLM.sh $lr $seed $epoch $optimiser $wd $dataset


            # python ./example/parameter_shift.py --lr $lr --seed $seed --opt $optimiser --weight_decay $wd --dataset $dataset
            # Wait for a short time to avoid overwhelming the scheduler
            sleep 2
        done
  
    done
done

echo "All jobs submitted."