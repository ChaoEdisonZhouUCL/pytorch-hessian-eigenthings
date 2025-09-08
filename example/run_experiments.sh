#!/bin/bash
# filepath: /usr/local1/chao/pytorch-hessian-eigenthings/example/run_experiments.sh

# Make script executable with: chmod +x run_experiments.sh
# Run with: ./run_experiments.sh

# Configuration
SEEDS=(42 123 456 789 1000)
LRS=(0.001 0.1 0.8)
NUM_EIGENTHINGS=100
MODE="power_iter"
RESULTS_DIR="./experiment_results"
CONFIG_FILE="/usr/local1/chao/pytorch-hessian-eigenthings/yaml/finetune_cnn_cifar10.yaml"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Results summary file
SUMMARY_FILE="$RESULTS_DIR/eigenvalue_summary.txt"
echo "Experiment Results Summary" > "$SUMMARY_FILE"
echo "=========================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Main experiment loop
for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
        echo "==============================================="
        echo "Running experiment: SEED=$seed, LR=$lr"
        echo "==============================================="
        
        # Create experiment-specific directory
        EXP_DIR="$RESULTS_DIR/seed_${seed}_lr_${lr}"
        mkdir -p "$EXP_DIR"
        
        # Log file for this experiment
        LOG_FILE="$EXP_DIR/experiment.log"
        echo "Starting experiment: SEED=$seed, LR=$lr" > "$LOG_FILE"
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
        echo "Step 1: Fine-tuning model with seed=$seed, lr=$lr..." | tee -a "$LOG_FILE"
        python example/finetune_vision.py --config "$TEMP_CONFIG" --lr "$lr" 2>&1 | tee -a "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: Fine-tuning failed for seed=$seed, lr=$lr" | tee -a "$LOG_FILE"
            echo "ERROR: Fine-tuning failed for seed=$seed, lr=$lr" >> "$SUMMARY_FILE"
            continue
        fi
        
        # Step 3: Find the generated checkpoint
        # Assuming the checkpoint is saved in the format: weight_gradient_hist/finetune_resnet_cifar10_*/lr_${lr}_seed_${seed}/*_epoch_*.pth
        CHECKPOINT_PATTERN="weight_gradient_hist/finetune_resnet_cifar10_*/lr_${lr}_seed_${seed}/*_epoch_*.pth"
        CHECKPOINT_DIR=$(find . -path "$CHECKPOINT_PATTERN" -type f | head -1)
        
        if [ -z "$CHECKPOINT_DIR" ]; then
            echo "ERROR: No checkpoint found for seed=$seed, lr=$lr" | tee -a "$LOG_FILE"
            echo "ERROR: No checkpoint found for seed=$seed, lr=$lr" >> "$SUMMARY_FILE"
            continue
        fi
        
        echo "Found checkpoint: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
        
        # Step 4: Run main.py to calculate eigenvalues
        echo "Step 2: Calculating eigenvalues..." | tee -a "$LOG_FILE"
        EIGENVAL_OUTPUT="$EXP_DIR/eigenvalues.txt"
        
        python example/main.py \
            --mode="$MODE" \
            --num_eigenthings="$NUM_EIGENTHINGS" \
            --seed="$seed" \
            --cuda \
            --checkpoint_dir="$CHECKPOINT_DIR" \
            2>&1 | tee "$EIGENVAL_OUTPUT"
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: Eigenvalue calculation failed for seed=$seed, lr=$lr" | tee -a "$LOG_FILE"
            echo "ERROR: Eigenvalue calculation failed for seed=$seed, lr=$lr" >> "$SUMMARY_FILE"
            continue
        fi
        
        # Step 5: Extract and save results to summary
        echo "Results for SEED=$seed, LR=$lr:" >> "$SUMMARY_FILE"
        echo "Checkpoint: $CHECKPOINT_DIR" >> "$SUMMARY_FILE"
        grep -A 10 "Eigenvals:" "$EIGENVAL_OUTPUT" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        
        # Clean up temporary config
        rm "$TEMP_CONFIG"
        
        echo "Completed experiment: SEED=$seed, LR=$lr" | tee -a "$LOG_FILE"
        echo "Results saved to: $EXP_DIR" | tee -a "$LOG_FILE"
        echo ""
    done
done

echo "==============================================="
echo "All experiments completed!"
echo "Summary file: $SUMMARY_FILE"
echo "Individual results in: $RESULTS_DIR"
echo "==============================================="

# Create a CSV file for easy analysis
CSV_FILE="$RESULTS_DIR/eigenvalue_results.csv"
echo "seed,lr,checkpoint,top_eigenvalue,second_eigenvalue,third_eigenvalue" > "$CSV_FILE"

# Parse results and create CSV (basic extraction - adjust based on actual output format)
for seed in "${SEEDS[@]}"; do
    for lr in "${LRS[@]}"; do
        EXP_DIR="$RESULTS_DIR/seed_${seed}_lr_${lr}"
        if [ -f "$EXP_DIR/eigenvalues.txt" ]; then
            # Extract top 3 eigenvalues (adjust parsing based on actual output format)
            EIGENVALS=$(grep -A 3 "Eigenvals:" "$EXP_DIR/eigenvalues.txt" | tail -3 | tr '\n' ',' | sed 's/,$//')
            CHECKPOINT=$(grep "Found checkpoint:" "$EXP_DIR/experiment.log" | cut -d' ' -f3)
            echo "$seed,$lr,$CHECKPOINT,$EIGENVALS" >> "$CSV_FILE"
        fi
    done
done

echo "CSV results saved to: $CSV_FILE"