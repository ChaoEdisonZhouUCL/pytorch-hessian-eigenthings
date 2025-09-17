import pickle, re
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_gradients(pickle_file_path):
    """Load gradients from pickle file"""
    with open(pickle_file_path, 'rb') as f:
        gradients = pickle.load(f)
    return gradients

def convert_to_numpy(tensor):
    """Convert any tensor to numpy array, handling different dtypes"""
    if tensor is None:
        return None
        
    try:
        # Handle PyTorch tensors
        if hasattr(tensor, 'numpy'):
            try:
                return tensor.numpy()
            except TypeError:
                # Handle BFloat16 or other unsupported types
                return tensor.float().numpy()
        elif hasattr(tensor, 'cpu'):
            try:
                return tensor.cpu().numpy()
            except TypeError:
                return tensor.cpu().float().numpy()
        else:
            return np.array(tensor, dtype=np.float32)
    except Exception as e:
        print(f"Error converting tensor: {e}")
        return np.array(tensor, dtype=np.float32)


def process_conv_gradients_numpy(gradients, n_eigenvals=50):
    """
    Process conv weight gradients using pure NumPy/SciPy:
    1. Flatten
    2. L2 normalize
    3. Calculate eigenvalues efficiently
    """
    conv_eigenvalues = {}
    
    print(f"Using NumPy/SciPy for CPU computation")
    print(f"Target eigenvalues: {n_eigenvals}")
    
    for name, grad in gradients.items():
        # Skip if gradient is None or not a conv weight
        if grad is None or 'conv' not in name.lower() or 'weight' not in name.lower():
            continue

        if not ('layer1.0.conv1.weight' in name or 'layer2.0.conv1.weight' in name or 'layer4.0.conv1.weight' in name):
            continue

        print(f"Processing {name}, shape: {grad.shape}")
        
        # Convert to numpy
        grad_np = convert_to_numpy(grad)
        if grad_np is None:
            continue
            
        # Flatten the gradient
        grad_flat = grad_np.flatten().astype(np.float32)  # Use float16 for memory efficiency

        # L2 normalize
        grad_norm = np.linalg.norm(grad_flat)
        if grad_norm > 1e-12:
            grad_normalized = grad_flat / grad_norm
        else:
            print(f"Warning: Very small gradient norm for {name}: {grad_norm}")
            continue
        
       
        try:
            # For outer product v*v^T, the eigenvalues are [||v||^2, 0, 0, ...]
            # Since we normalized, the first eigenvalue is 1.0, rest are 0
            # But let's compute it properly for verification
            
            print(f"  Computing eigenvalues for matrix size {len(grad_normalized)}")         
            # For smaller matrices, compute outer product and full eigendecomposition
            outer_product = np.outer(grad_normalized, grad_normalized)
            
            # Add small regularization for numerical stability
            outer_product += 1e-12 * np.eye(outer_product.shape[0])
            
            # Use scipy for eigenvalues (faster than numpy for large matrices)
            # eigenvalues = scipy.linalg.eigvalsh(outer_product)
            # # largest k eigenvalues of symmetric A
            eigenvalues, _ = eigsh(outer_product, k=n_eigenvals, which="LM")    # or which="SA" for smallest
            eigenvalues = eigenvalues[::-1]  # Sort descending
                
            
            
            # Take top eigenvalues
            top_eigenvalues = eigenvalues[:50]            
            conv_eigenvalues[name] = top_eigenvalues.astype(np.float32)
            print(f"  Top 5 eigenvalues: {top_eigenvalues}")
            
        except Exception as e:
            print(f"Error computing eigenvalues for {name}: {e}")
    
    return conv_eigenvalues


# Keep plotting functions the same but add speedup
def plot_eigenvalue_heatmap(conv_eigenvalues, save_dir="./gradient_analysis"):
    """Plot heatmap of eigenvalues for all conv layers"""
    if not conv_eigenvalues:
        print("No conv layer eigenvalues to plot")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for heatmap
    layer_names = list(conv_eigenvalues.keys())
    max_eigenvals = max(len(eigenvals) for eigenvals in conv_eigenvalues.values())
    
    # Create matrix for heatmap
    eigenval_matrix = np.zeros((len(layer_names), max_eigenvals))
    
    for i, (name, eigenvals) in enumerate(conv_eigenvalues.items()):
        eigenval_matrix[i, :len(eigenvals)] = eigenvals
        if len(eigenvals) < max_eigenvals:
            eigenval_matrix[i, len(eigenvals):] = np.nan
    
    # Create heatmap
    plt.figure(figsize=(15, max(8, len(layer_names) * 0.5)))
    
    mask = np.isnan(eigenval_matrix)
    
    sns.heatmap(eigenval_matrix, 
                yticklabels=[name.replace('.', '\n') for name in layer_names],
                xticklabels=[f'λ{i+1}' for i in range(max_eigenvals)],
                cmap='viridis',
                mask=mask,
                cbar_kws={'label': 'Eigenvalue'},
                annot=False)
    
    plt.title('Eigenvalues of Conv Layer Gradient Outer Products (NumPy)')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Conv Layers')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'conv_gradient_eigenvalues_heatmap_numpy.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import time
    
    # Example usage
    gradient_file = "/usr/local1/chao/pytorch-hessian-eigenthings/weight_gradient_hist/finetune_resnet_cifar10_adamw/gradients/gradients_epoch_0_batch_0.pkl"
    
    if not os.path.exists(gradient_file):
        print(f"Gradient file not found: {gradient_file}")
        return
    
    m = re.search(r'epoch_(\d+)_batch_(\d+)\.pkl$', gradient_file)
    if not m:
        raise ValueError("Could not find 'epoch_#_batch_#.pkl' in the path")
    epoch, batch = map(int, m.groups())
    
    # Load gradients
    print(f"Loading gradients from: {gradient_file}")
    gradients = load_gradients(gradient_file)
    print(f"Loaded gradients for {len(gradients)} parameters")
        
    start_time = time.time()
    try:
        conv_eigenvalues = process_conv_gradients_numpy(gradients)
        end_time = time.time()
        
        print(f"✓ completed in {end_time - start_time:.2f} seconds")
        print(f"  Processed {len(conv_eigenvalues)} conv layers")
        
        # Save results for the fastest successful method
        if conv_eigenvalues:
            save_dir = os.path.join(os.path.dirname(gradient_file), f"epoch_{epoch}_batch_{batch}_eigenvalue_analysis")
            plot_eigenvalue_heatmap(conv_eigenvalues, save_dir)
            
            eigenval_file = os.path.join(save_dir, f"conv_eigenvalues.pkl")
            with open(eigenval_file, 'wb') as f:
                pickle.dump(conv_eigenvalues, f)
            print(f"  Saved to: {eigenval_file}")
            
    except Exception as e:
        end_time = time.time()
        print(f"✗ failed after {end_time - start_time:.2f} seconds: {e}")

if __name__ == "__main__":
    main()