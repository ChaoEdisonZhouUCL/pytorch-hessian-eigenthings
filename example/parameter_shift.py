import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from collections import OrderedDict

def load_state_dict(path):
    """Load state dictionary from file"""
    return torch.load(path, map_location='cpu')

def filter_classifier_params(state_dict, classifier_keys=None):
    """Remove classifier layer parameters from state dict"""
    if classifier_keys is None:
        # Common classifier layer names for ResNet
        classifier_keys = ['fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias', 
                          'head.weight', 'head.bias']
    
    filtered_dict = OrderedDict()
    excluded_params = []
    
    for key, value in state_dict.items():
        is_classifier = any(clf_key in key for clf_key in classifier_keys)
        if not is_classifier:
            filtered_dict[key] = value.float()
        else:
            excluded_params.append(key)
    
    print(f"Excluded classifier parameters: {excluded_params}")
    return filtered_dict

def calculate_parameter_shift(pretrained_dict, finetuned_dict, include_classifier=True):
    """
    Calculate L1 and L2 parameter shifts between pretrained and finetuned models
    
    Args:
        pretrained_dict: State dict of pretrained model
        finetuned_dict: State dict of finetuned model  
        include_classifier: Whether to include classifier layer in calculation
        
    Returns:
        dict: Contains L1 and L2 norms and per-layer analysis
    """
    
    # Filter out classifier if requested
    if not include_classifier:
        pretrained_dict = filter_classifier_params(pretrained_dict)
        finetuned_dict = filter_classifier_params(finetuned_dict)
    
    # Ensure both dicts have same keys
    common_keys = set(pretrained_dict.keys()) & set(finetuned_dict.keys())
    
    if len(common_keys) != len(pretrained_dict.keys()):
        missing_in_finetuned = set(pretrained_dict.keys()) - common_keys
        missing_in_pretrained = set(finetuned_dict.keys()) - common_keys
        print(f"Warning: Key mismatch!")
        print(f"Missing in finetuned: {missing_in_finetuned}")
        print(f"Missing in pretrained: {missing_in_pretrained}")
    
    # Calculate shifts for each parameter
    per_layer_shifts = {}
    
    # Collect all parameter shifts for global calculation
    all_pretrained_params = []
    all_finetuned_params = []
    
    for key in sorted(common_keys):
        pretrained_param = pretrained_dict[key].float()
        finetuned_param = finetuned_dict[key].float()
        
        # Store flattened parameters for global calculation
        all_pretrained_params.append(pretrained_param.flatten())
        all_finetuned_params.append(finetuned_param.flatten())
        
        # Calculate parameter shift (delta) for this layer
        shift = finetuned_param - pretrained_param
        shift_flat = shift.flatten()
        pretrained_flat = pretrained_param.flatten()
        
        # Calculate norms for this layer
        l1_shift = torch.norm(shift_flat, p=1).item()
        l2_shift = torch.norm(shift_flat, p=2).item()
        l1_pretrained = torch.norm(pretrained_flat, p=1).item()
        l2_pretrained = torch.norm(pretrained_flat, p=2).item()
        
        per_layer_shifts[key] = {
            'l1_shift': l1_shift,
            'l2_shift': l2_shift,
            'l1_pretrained': l1_pretrained,
            'l2_pretrained': l2_pretrained,
            'num_params': shift_flat.numel()
        }
    
    # Calculate global shifts correctly
    # Concatenate all parameters into single vectors
    all_pretrained_concat = torch.cat(all_pretrained_params)
    all_finetuned_concat = torch.cat(all_finetuned_params)
    
    # Calculate global parameter shift
    global_shift = all_finetuned_concat - all_pretrained_concat
    
    # Calculate global norms
    global_l1_shift = torch.norm(global_shift, p=1).item()
    global_l2_shift = torch.norm(global_shift, p=2).item()
    global_l1_pretrained = torch.norm(all_pretrained_concat, p=1).item()
    global_l2_pretrained = torch.norm(all_pretrained_concat, p=2).item()
    
    results = {
        'global': {
            'l1_shift': global_l1_shift,
            'l2_shift': global_l2_shift,
            'l1_pretrained': global_l1_pretrained,
            'l2_pretrained': global_l2_pretrained,
            'total_params': all_pretrained_concat.numel()
        },
        'per_layer': per_layer_shifts,
        'include_classifier': include_classifier
    }
    
    return results

def print_results(results, title):
    """Print formatted results"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    global_stats = results['global']
    print(f"Total Parameters: {global_stats['total_params']:,}")
    print(f"Include Classifier: {results['include_classifier']}")
    print()
    
    print("GLOBAL PARAMETER SHIFTS:")
    print(f"L1 Shift (Absolute):     {global_stats['l1_shift']:.6f}")
    print(f"L2 Shift (Absolute):     {global_stats['l2_shift']:.6f}")
    print()
    
    print("PRETRAINED MODEL NORMS:")
    print(f"L1 Norm:                 {global_stats['l1_pretrained']:.6f}")
    print(f"L2 Norm:                 {global_stats['l2_pretrained']:.6f}")
    print()
    
def save_results_to_file(results, filename):
    """Save detailed results to text file"""
    with open(filename, 'w') as f:
        f.write("PARAMETER SHIFT ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")
        
        global_stats = results['global']
        f.write(f"Include Classifier: {results['include_classifier']}\n")
        f.write(f"Total Parameters: {global_stats['total_params']:,}\n\n")
        
        f.write("GLOBAL STATISTICS:\n")
        f.write(f"L1 Shift (Absolute): {global_stats['l1_shift']:.6f}\n")
        f.write(f"L2 Shift (Absolute): {global_stats['l2_shift']:.6f}\n")
        f.write(f"L1 Pretrained Norm: {global_stats['l1_pretrained']:.6f}\n")
        f.write(f"L2 Pretrained Norm: {global_stats['l2_pretrained']:.6f}\n\n")
        
        f.write("PER-LAYER STATISTICS:\n")
        f.write(f"{'Layer':<50} {'L1_Shift':<12} {'L2_Shift':<12} {'L1_Pretrained':<15} {'L2_Pretrained':<15} {'Params':<10}\n")
        f.write("-" * 120 + "\n")
        
        for layer_name, stats in results['per_layer'].items():
            f.write(f"{layer_name:<50} {stats['l1_shift']:<12.6f} {stats['l2_shift']:<12.6f} "
                   f"{stats['l1_pretrained']:<15.6f} {stats['l2_pretrained']:<15.6f} {stats['num_params']:<10}\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate L1 and L2 parameter shifts")
    parser.add_argument('--lr', type=float, default=0.001,required=False,
                       help="Learning rate used during finetuning (for reference)")
    parser.add_argument('--seed', type=int, default=42, required=False,
                       help="Random seed used during finetuning (for reference)")
    parser.add_argument('--output_dir', type=str, default='./parameter_shift_results',
                       help="Directory to save results")
    
    args = parser.parse_args()
    lr=args.lr
    seed=args.seed
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model weights
    print("Loading model weights...")
    pretrained_path = f'/usr/local1/chao/pytorch-hessian-eigenthings/weight_gradient_hist/finetune_resnet_cifar10_adamw/lr_{lr}_epoch_15/_seed_{seed}/resnet18_adamw_epoch_0.pth'
    finetuned_path = f'/usr/local1/chao/pytorch-hessian-eigenthings/weight_gradient_hist/finetune_resnet_cifar10_adamw/lr_{lr}_epoch_15/_seed_{seed}/resnet18_adamw_epoch_14.pth'
    pretrained_dict = load_state_dict(pretrained_path)
    finetuned_dict = load_state_dict(finetuned_path)
    
    print(f"Pretrained model has {len(pretrained_dict)} parameter groups")
    print(f"Finetuned model has {len(finetuned_dict)} parameter groups")
    
    # Case 1: Whole model (including classifier)
    print("\nCalculating shifts for whole model...")
    results_whole = calculate_parameter_shift(pretrained_dict, finetuned_dict, include_classifier=True)
    print_results(results_whole, "CASE 1: WHOLE MODEL (INCLUDING CLASSIFIER)")
    
    # Save results
    save_results_to_file(results_whole, os.path.join(args.output_dir, f"adamw_lr_{lr}_seed_{seed}_epoch15_whole_model_shifts.txt"))
    
    # Case 2: Model without classifier  
    print("\nCalculating shifts without classifier...")
    results_no_classifier = calculate_parameter_shift(pretrained_dict, finetuned_dict, include_classifier=False)
    print_results(results_no_classifier, "CASE 2: MODEL WITHOUT CLASSIFIER")
    
    # Save results
    save_results_to_file(results_no_classifier, os.path.join(args.output_dir, f"adamw_lr_{lr}_seed_{seed}_epoch15_no_classifier_shifts.txt"))
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Whole Model':<15} {'No Classifier':<15} {'Difference':<15}")
    print("-" * 70)
    
    whole_global = results_whole['global']
    no_clf_global = results_no_classifier['global']
    
    metrics = [
        ('L1 Shift (Absolute)', 'l1_shift'),
        ('L2 Shift (Absolute)', 'l2_shift'),
        ('Total Parameters', 'total_params'),
    ]
    
    for metric_name, metric_key in metrics:
        whole_val = whole_global[metric_key]
        no_clf_val = no_clf_global[metric_key]
        diff = whole_val - no_clf_val
        if metric_key == 'total_params':
            print(f"{metric_name:<25} {whole_val:<15,} {no_clf_val:<15,} {diff:<15,}")
        else:
            print(f"{metric_name:<25} {whole_val:<15.6f} {no_clf_val:<15.6f} {diff:<15.6f}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()