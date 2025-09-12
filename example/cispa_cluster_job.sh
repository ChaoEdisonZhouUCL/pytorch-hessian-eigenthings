#!/bin/bash

#SBATCH --job-name=NLP
#SBATCH --output=/home/c01chzh/CISPA-projects/pt_network-2024/pytorch-hessian-eigenthings/tmp/job-%j.out
#SBATCH --error=/home/c01chzh/CISPA-projects/pt_network-2024/pytorch-hessian-eigenthings/tmp/job-%j.err
#SBATCH --partition=xe8545
## AAASBATCH --cpus-per-task=128
#SBATCH --time=6-23:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:4


if [ ! -f ~/.config/enroot/.credentials ]; then
        mkdir -p ~/.config/enroot/
        ln -s ~/CISPA-home/.config/enroot/.credentials ~/.config/enroot/.credentials
fi
export HF_HOME=$HOME/CISPA-projects/pt_network-2024/.huggingface_cache
WORK_DIR=$HOME/CISPA-projects/pt_network-2024/pytorch-hessian-eigenthings
JOBTMPDIR=$WORK_DIR/tmp/job-"$SLURM_JOB_ID"
srun mkdir -p "$JOBTMPDIR"

# Get the number of GPUs allocated to this job
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # If CUDA_VISIBLE_DEVICES is not set, try to get it from SLURM_GPUS
    if [ -n "$SLURM_GPUS" ]; then
        NUM_GPUS=$(echo $SLURM_GPUS | tr ',' '\n' | wc -l)
    else
        # If neither is set, try to query the node resources
        NUM_GPUS=$(scontrol show node $HOSTNAME | grep "Gres" | grep -o "gpu:[0-9]*" | cut -d':' -f2)
    fi
fi

echo "Number of GPUs allocated: $NUM_GPUS"

# Set environment variables for distributed training
export MASTER_PORT=$((1000 + RANDOM % 5000))  # 12345  # Set a free port for master node communication
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Get master node address

export WORLD_SIZE=$((SLURM_NNODES * 4))  # Total number of processes
export RANK=$SLURM_PROCID                         # Current process rank
export nproc_per_node=4
export WANDB_MODE=online

srun --container-image=projects.cispa.saarland:5005#c01chzh/nanoadam:latest --container-mounts="$WORK_DIR":/workspace\
    bash $WORK_DIR/example/run.sh $1 $2 $3
   
    
srun mv $WORK_DIR/tmp/job-"$SLURM_JOB_ID".out "$JOBTMPDIR"/out.txt
srun mv $WORK_DIR/tmp/job-"$SLURM_JOB_ID".err "$JOBTMPDIR"/err.txt
