# Standard libraries
import argparse
import os
import random
import yaml

# Third-party libraries
import numpy as np
import wandb
from tqdm import tqdm

# PyTorch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Timm (PyTorch Image Models)
from timm import create_model
from timm.data import create_transform, resolve_model_data_config

def set_seed_full(seed):
    """Set seed for full reproducibility across all random number generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For distributed training reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set other library seeds
    random.seed(seed)
    np.random.seed(seed)

def load_config(args):
    # Correctly access the config file path from args
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Get SLURM job ID if available and add it to the config
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_job_id:
        config['slurm_job_id'] = slurm_job_id
    return config


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def reduce_tensor(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def param_groups_with_name(
    model: nn.Module,
):
    params = []
    names = []
    for name, param in model.named_parameters():
        params.append(param)
        names.append(name)
    return [{"params": params,  "names": names}]


def train_per_epoch(
        cur_epoch, total_epoch, model, train_loader, train_sampler, optimizer, scheduler, criterion, distributed, configs):
    if distributed:
        train_sampler.set_epoch(cur_epoch)
    model.train()
    running_loss, total_correct, total_samples = 0.0, 0, 0
    # bfloat16 does not need GradScaler, but kept for compatibility
    scaler = GradScaler(enabled=configs['bf16'])

    for images, labels in tqdm(train_loader, desc=f"Epoch {cur_epoch+1}/{total_epoch}"):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=configs['bf16'], dtype=torch.bfloat16 if configs['bf16'] else torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
    if scheduler is not None:
        scheduler.step()

    return running_loss, total_correct, total_samples


def eval_per_epoch(
        model, test_loader, criterion, configs):
    # Evaluation
    model.eval()
    running_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            with autocast(device_type='cuda', enabled=configs['bf16'], dtype=torch.bfloat16 if configs['bf16'] else torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    return running_loss, total_correct, total_samples




def task(rank, world_size, local_rank, configs):
    distributed = configs['ddp']
    
    # Set seed early and consistently across all processes
    base_seed = configs['seed']
    # Use different seeds per rank to avoid identical sampling
    process_seed = base_seed + rank
    set_seed_full(process_seed)
    
    print(f"Rank {rank}: using seed: {process_seed} (base: {base_seed})")        

    # Load pretrained model
    print("[INFO] Loading pretrained model...")
    if 'vit' in configs['model_name'].lower() or 'resnet' in configs['model_name'].lower():
        model = create_model(configs['model_name'], pretrained=configs['pretrained'],
                             num_classes=configs['num_classes']).cuda()
        data_config = resolve_model_data_config(model)
        transform_train = create_transform(**data_config, is_training=True)
        transform_eval = create_transform(**data_config, is_training=False)
    else:
        raise ValueError(
            f"Unsupported model architecture: {configs['model_name']}")

    # Ensure only rank 0 downloads the dataset
    if configs['dataset_name'].lower() == 'cifar10':
        datasets_class = torchvision.datasets.CIFAR10
        train_split = True
        test_split = False
    elif configs['dataset_name'].lower() == 'cifar100':
        datasets_class = torchvision.datasets.CIFAR100
        train_split = True
        test_split = False
    elif configs['dataset_name'].lower() == 'flowers':
        datasets_class = torchvision.datasets.Flowers102
        train_split = 'train'
        test_split = 'test'
    else:
        raise ValueError(
            f"Unsupported dataset: {configs['dataset_name']}. Supported datasets: cifar10")
    if local_rank == 0:
        os.makedirs(configs['data_root'], exist_ok=True)
        # download on main rank
        if configs['dataset_name'].lower() == 'cifar10' or configs['dataset_name'].lower() == 'cifar100':
            train_dataset = datasets_class(
                root=configs['data_root'], train=train_split, transform=transform_train, download=True)
            test_dataset = datasets_class(
                root=configs['data_root'], train=test_split, transform=transform_eval, download=True)
        else:  # flowers
            train_dataset = datasets_class(
                root=configs['data_root'], split=train_split, transform=transform_train, download=True)
            test_dataset = datasets_class(
                root=configs['data_root'], split=test_split, transform=transform_eval, download=True)
    
    # Synchronize all processes to wait until rank 0 is done
    if distributed:
        dist.barrier()  # Ensures all processes wait before proceeding

    # Load dataset
    if configs['dataset_name'].lower() == 'cifar10' or configs['dataset_name'].lower() == 'cifar100':
        train_dataset = datasets_class(
            root=configs['data_root'], train=train_split, transform=transform_train, download=False)
        test_dataset = datasets_class(
            root=configs['data_root'], train=test_split, transform=transform_eval, download=False)
    else:  # flowers
        train_dataset = datasets_class(
            root=configs['data_root'], split=train_split, transform=transform_train, download=False)
        test_dataset = datasets_class(
            root=configs['data_root'], split=test_split, transform=transform_eval, download=False)

    train_sampler = DistributedSampler(
        train_dataset, seed=base_seed) if distributed else None

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=configs['batch_size'],
                              sampler=train_sampler,
                              shuffle=(train_sampler is None)  # Only shuffle if no sampler
                              )
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=configs['batch_size'],
                             shuffle=False)

    if configs['bf16']:
        print("[INFO] Using bfloat16 precision...")
        model = model.to(dtype=torch.bfloat16)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model
    print("[INFO] Model architecture:")
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=configs['label_smoothing'])

    # optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'], weight_decay=configs['weight_decay'])
    if configs['optimizer_name'].lower() == "adamw":
        print(f"[INFO] Using AdamW optimizer...")
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=configs['learning_rate'],
                          weight_decay=configs['weight_decay'])
    elif configs['optimizer_name'].lower() == "sgd":
        print(f"[INFO] Using SGD optimizer...")
        from torch.optim import SGD
        optimizer = SGD(model.parameters(), lr=configs['learning_rate'],
                        weight_decay=configs['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {configs['optimizer_name']}")
    
    if configs['scheduler_name'] == "cosineannealinglr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs['num_epochs'])
    else:
        scheduler = None

    # Training loop
    if local_rank == 0:
        print("[INFO] Initializing Weights & Biases...")
        wandb.init(project=configs['wandb_project'],
                   name=configs['wandb_name'], config=configs)
        save_model(model, configs['save_dir'], configs, epoch=0)
        
    for epoch in range(configs['num_epochs']):
        train_loss, train_correct, train_samples = train_per_epoch(
            epoch, configs['num_epochs'], model, train_loader, train_sampler, optimizer, scheduler, criterion, distributed, configs)
        val_loss, val_correct, val_samples = eval_per_epoch(
            model, test_loader, criterion, configs)
       

        if distributed:
            syn_tensor = torch.tensor(
                [train_loss, train_correct, train_samples, val_loss,
                    val_correct, val_samples], device='cuda'
            )
            train_loss, train_correct, train_samples, val_loss, val_correct, val_samples = reduce_tensor(
                syn_tensor)
            dist.barrier()

        train_loss = train_loss/train_samples
        train_acc = train_correct/train_samples
        val_loss = val_loss/val_samples
        val_acc = val_correct/val_samples

        if local_rank == 0:
            wandb.log({"loss/strain": train_loss, "acc/train": train_acc,
                      "loss/val": val_loss, "acc/val": val_acc,
                       "steps/epoch": epoch, "steps/lr": optimizer.param_groups[0]['lr'],
                       })

            print(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")
            if (epoch + 1) % (configs['num_epochs']//2) == 0 or (epoch + 1) == configs['num_epochs']:
                save_model(model, configs['save_dir'], configs, epoch=epoch+1)
    if local_rank == 0:        
        print("[INFO] Finishing Weights & Biases...")
        wandb.finish()

    if distributed:
        print("[INFO] Synchronizing model state across processes...")
        dist.barrier()
        dist.destroy_process_group()

def save_model(model, save_path, configs, epoch):
    print(f"[INFO] Saving model at epoch {epoch}...")
    save_path = os.path.join(save_path, f"lr_{configs['learning_rate']}_epoch_{configs['num_epochs']}", f"_seed_{configs['seed']}")
    os.makedirs(save_path, exist_ok=True)
    
    # Handle DDP model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model.state_dict(), os.path.join(
        save_path, f"{configs['model_name']}_{configs['optimizer_name']}_epoch_{epoch}.pth"))
    print("[INFO] Model saved successfully.")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default="/usr/local1/chao/pytorch-hessian-eigenthings/yaml/finetune_cnn_cifar10.yaml",
                        help="Path to the config YAML file")
    parser.add_argument('--lr', type=float, default=None,
                        help="Learning rate for the optimizer")
    parser.add_argument('--opt', type=str, default=None,
                        help="Optimizer name")
    args = parser.parse_args()

    # Load the config from the passed path
    config = load_config(args)

    if args.lr is not None:
        config['learning_rate'] = args.lr
    if args.opt is not None:
        config['optimizer_name'] = args.opt
    config['wandb_name'] = f"{config['wandb_name']}_{config['dataset_name']}_{config['optimizer_name']}_lr{config['learning_rate']}_seed{config['seed']}"

    config['save_dir']=f"{config['save_dir']}_{config['dataset_name']}_{config['optimizer_name']}"
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config['ddp'] = True
    else:
        config['ddp'] = False

    # Now you can use the config in your training code
    print(f"Using config: {config}")

    distributed = config['ddp']
    if distributed:
        rank, world_size, local_rank = init_distributed_mode()
    else:
        rank, world_size, local_rank = 0, 1, 0

    # adjust batch size for distributed training
    config['batch_size'] = max(
        1, int(config['batch_size'] / world_size))

    task(rank, world_size, local_rank, config)


if __name__ == "__main__":
    
    main()
