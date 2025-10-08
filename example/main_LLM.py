"""
A simple example to calculate the top eigenvectors for the hessian of
ResNet18 network for CIFAR-10
"""

import torch, random
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset to use")
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
    parser.add_argument("--max_seq_length", default=128, type=int,required=False, help="maximum sequence length")

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"[INFO] Using seed: {args.seed}")
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # 1. load model and tokenizer
    # ----------------------------
    print("[INFO] Loaded tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    print("[INFO] Loaded model")
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir).to(device)
    # Define a wrapper so model accepts only input_ids tensor as in the HVP flow
    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        def forward(self, input_ids):
            # Derive attention mask to improve stability; still keeps input type as Tensor
            attention_mask = (input_ids != self.pad_id).to(input_ids.dtype)
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    model = ModelWrapper(model).to(device)
    
    # ----------------------------
    # 2. load dataset
    # ----------------------------
    raw = load_dataset("glue", args.dataset)
    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
        )
    tokenized = raw.map(tokenize_fn, batched=True)
    # Keep only what we need for the HVP flow
    keep_cols = [c for c in tokenized["train"].column_names if c in ["input_ids", "label"]]
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in keep_cols])
    tokenized.set_format(type="torch", columns=["input_ids", "label"])

    # Use validation split since GLUE test labels are not publicly available
    testset = tokenized["validation"]

    # Make batches be (Tensor, Tensor): (input_ids, labels)
    def collate_tuple(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        return input_ids, labels

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_tuple,
    )
    

    
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
