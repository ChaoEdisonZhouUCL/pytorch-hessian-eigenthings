"""
A simple example to calculate the top eigenvectors for the hessian of
ResNet18 network for CIFAR-10
"""

import torch, random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from timm import create_model
import pandas as pd
import os

from hessian_eigenthings import compute_hessian_eigenthings

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extra_args(parser):
    parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", default="", type=str, help="path to checkpoint file")
    parser.add_argument(
        "--num_eigenthings",
        default=100,
        type=int,
        help="number of eigenvals/vecs to compute",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="train set batch size"
    )
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="test set batch size"
    )
    parser.add_argument(
        "--momentum", default=0.0, type=float, help="power iteration momentum term"
    )
    parser.add_argument(
        "--num_steps", default=50, type=int, help="number of power iter steps"
    )
    parser.add_argument("--max_samples", default=2048, type=int)
    parser.add_argument("--cuda", action="store_true", help="if true, use CUDA/GPUs")
    parser.add_argument(
        "--full_dataset",
        action="store_true",
        help="if true,\
                        loop over all batches in set for each gradient step",
    )
    parser.add_argument("--fname", default="", type=str)
    parser.add_argument("--mode", type=str, choices=["power_iter", "lanczos"])
    parser.add_argument("--output_excel", default="eigenvalues.xlsx", type=str, 
                        help="Output Excel filename for saving eigenvalues")


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # 1. CIFAR-10 dataset
    # ----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    # ----------------------------
    # 2. Pretrained ResNet-18
    # ----------------------------
    print("Loaded pretrained model")
    model = create_model('resnet18', pretrained=True, num_classes=10).to(device)
    # Load checkpoint
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = '/usr/local1/chao/pytorch-hessian-eigenthings/weight_gradient_hist/finetune_resnet_cifar10_sgd/lr_0.8/resnet18_sgd_epoch_30.pth'
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Load weights into model
    state_dict = torch.load(checkpoint_dir, map_location="cpu")  # or "cuda:0"
    model.load_state_dict(state_dict)
    
    criterion = torch.nn.CrossEntropyLoss()
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model,
        testloader,
        criterion,
        args.num_eigenthings,
        mode=args.mode,
        # power_iter_steps=args.num_steps,
        max_possible_gpu_samples=args.max_samples,
        # momentum=args.momentum,
        full_dataset=args.full_dataset,
        use_gpu=args.cuda,
    )
    print(f"resnet18 finetuned with sgd from final epoch has:")
    print("Eigenvecs:")
    print(eigenvecs)
    print("Eigenvals:")
    print(eigenvals)
    
    # Save eigenvalues to Excel
    eigenvals_np = eigenvals.cpu().numpy() if torch.is_tensor(eigenvals) else np.array(eigenvals)
    
    # Create a DataFrame with eigenvalues
    df = pd.DataFrame({
        'Index': range(len(eigenvals_np)),
        'Eigenvalue': eigenvals_np
    })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_excel) if os.path.dirname(args.output_excel) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Excel
    df.to_excel(args.output_excel, index=False, sheet_name='Eigenvalues')
    print(f"Eigenvalues saved to: {args.output_excel}")
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    extra_args(parser)
    args = parser.parse_args()
    main(args)
