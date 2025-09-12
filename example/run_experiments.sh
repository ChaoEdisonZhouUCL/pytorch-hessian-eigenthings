#!/bin/bash
# filepath: /usr/local1/chao/pytorch-hessian-eigenthings/example/run_experiments.sh

# Make script executable with: chmod +x run_experiments.sh
# Run with: ./run_experiments.sh

echo "HOSTNAME: $HOSTNAME"
# Set platform-specific configurations
if [[ "$HOSTNAME" == *"juwels"* ]]; then
    PLATFORM="juwels"
    ENV_ACTIVATION="micromamba run -n microadam"
    TORCHRUN_CMD="torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$nproc_per_node"
    export WANDB_MODE=offline
elif [[ "$HOSTNAME" == *"xe8545"* ]] || [[ "$HOSTNAME" == *"r6525"* ]]; then
    PLATFORM="cispa"
    ENV_ACTIVATION="conda run -n microadam"
    TORCHRUN_CMD="python"
    export WANDB_MODE=online
    $ENV_ACTIVATION python -m pip install --upgrade pip wandb
    $ENV_ACTIVATION pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings


else
    PLATFORM="rml"
    ENV_ACTIVATION=""
    TORCHRUN_CMD="python"
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_MODE=online
fi

# Configuration
lr=$1
seed=$2
epoch=$3
optimiser=$4
NUM_EIGENTHINGS=50
MODE="power_iter"
RESULTS_DIR="./experiment_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Results summary file
SUMMARY_FILE="$RESULTS_DIR/eigenvalue_summary.txt"
echo "Experiment Results Summary" > "$SUMMARY_FILE"
echo "=========================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Main experiment

echo "==============================================="
echo "Running experiment: SEED=$seed, LR=$lr"
echo "==============================================="

# Create experiment-specific directory
EXP_DIR="$RESULTS_DIR/${optimiser}_seed_${seed}_lr_${lr}_epoch_15"
mkdir -p "$EXP_DIR"

# Log file for this experiment
LOG_FILE="$EXP_DIR/experiment.log"
echo "Starting experiment: SEED=$seed, LR=$lr" > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Step 3: Find the generated checkpoint
# Assuming the checkpoint is saved in the format: weight_gradient_hist/finetune_resnet_cifar10_*/lr_${lr}_seed_${seed}/*_epoch_*.pth
CHECKPOINT_PATTERN="./weight_gradient_hist/finetune_resnet_cifar10_${optimiser}/lr_${lr}_epoch_15/_seed_${seed}/*_epoch_${epoch}.pth"
CHECKPOINT_DIR=$(find . -path "$CHECKPOINT_PATTERN" -type f | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint found for seed=$seed, lr=$lr" | tee -a "$LOG_FILE"
    echo "ERROR: No checkpoint found for seed=$seed, lr=$lr" >> "$SUMMARY_FILE"
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"

# Step 4: Run main.py to calculate eigenvalues
echo "Step 2: Calculating eigenvalues..." | tee -a "$LOG_FILE"
EIGENVAL_OUTPUT="$EXP_DIR/eigenvalues.txt"

$ENV_ACTIVATION $TORCHRUN_CMD example/main.py \
    --mode="$MODE" \
    --num_eigenthings="$NUM_EIGENTHINGS" \
    --seed="$seed" \
    --cuda \
    --checkpoint_dir="$CHECKPOINT_DIR" \
    --output_excel="$EXP_DIR/eigenvalues_seed${seed}_lr${lr}_epoch${epoch}.xlsx" \
    2>&1 | tee "$EIGENVAL_OUTPUT"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Eigenvalue calculation failed for seed=$seed, lr=$lr" | tee -a "$LOG_FILE"
    echo "ERROR: Eigenvalue calculation failed for seed=$seed, lr=$lr" >> "$SUMMARY_FILE"
fi

# Step 5: Extract and save results to summary
echo "Results for SEED=$seed, LR=$lr:" >> "$SUMMARY_FILE"
echo "Checkpoint: $CHECKPOINT_DIR" >> "$SUMMARY_FILE"
grep -A 10 "Eigenvals:" "$EIGENVAL_OUTPUT" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Completed experiment: SEED=$seed, LR=$lr" | tee -a "$LOG_FILE"
echo "Results saved to: $EXP_DIR" | tee -a "$LOG_FILE"
echo ""

echo "==============================================="
echo "All experiments completed!"
echo "Summary file: $SUMMARY_FILE"
echo "Individual results in: $RESULTS_DIR"
echo "==============================================="
