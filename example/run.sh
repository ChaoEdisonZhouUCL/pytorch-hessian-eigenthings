#!/bin/bash
# filepath: /usr/local1/chao/pytorch-hessian-eigenthings/example/run_experiments.sh

# Make script executable with: chmod +x run_experiments.sh
# Run with: ./run_experiments.sh

# Configuration


RESULTS_DIR="./experiment_results"
CONFIG_FILE="./yaml/finetune_cnn_cifar10.yaml"

# Create results directory
mkdir -p "$RESULTS_DIR"
lr=$1
seed=$2
optimiser=$3
wd=$4

echo "==============================================="
echo "Running experiment: SEED=$seed, LR=$lr, OPTIMISER=$optimiser, WD=$wd"
echo "==============================================="

Timestamp=$(date +"%Y%m%d_%H%M%S")
# Create experiment-specific directory
EXP_DIR="$RESULTS_DIR/${optimiser}_seed_${seed}_lr_${lr}_wd_${wd}_${Timestamp}"
mkdir -p "$EXP_DIR"

# Log file for this experiment
LOG_FILE="$EXP_DIR/experiment.log"
echo "Starting experiment: SEED=$seed, LR=$lr, OPTIMISER=$optimiser, WD=$wd" > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Step 1: Update config file with current seed
# Create a temporary config file with the current seed
TEMP_CONFIG="$EXP_DIR/temp_config.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Update seed in the temp config (assuming the config has a 'seed' field)
if grep -q "seed:" "$TEMP_CONFIG"; then
    sed -i "s/seed:.*/seed: $seed/" "$TEMP_CONFIG"
else
    echo "seed: $seed" >> "$TEMP_CONFIG"
fi

# Step 2: Run finetune_vision.py
echo "Step 1: Fine-tuning model with seed=$seed, lr=$lr, opt=$optimiser, wd=$wd..." | tee -a "$LOG_FILE"
python example/finetune_vision.py --config "$TEMP_CONFIG" --lr "$lr" --opt "$optimiser" --wd "$wd" 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Fine-tuning failed for seed=$seed, lr=$lr, opt=$optimiser" | tee -a "$LOG_FILE"
    echo "ERROR: Fine-tuning failed for seed=$seed, lr=$lr, opt=$optimiser" >> "$SUMMARY_FILE"
    continue
fi

# Clean up temporary config
rm "$TEMP_CONFIG"

echo "Completed experiment: SEED=$seed, LR=$lr, opt=$optimiser" | tee -a "$LOG_FILE"
echo "Results saved to: $EXP_DIR" | tee -a "$LOG_FILE"
echo ""


echo "==============================================="
echo "All experiments completed!"
echo "Summary file: $SUMMARY_FILE"
echo "Individual results in: $RESULTS_DIR"
echo "==============================================="