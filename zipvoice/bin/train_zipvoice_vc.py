#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for ZipVoice Voice Conversion model

This script trains a ZipVoice model adapted for voice conversion using:
1. Pre-trained Zipformer as content encoder
2. Self-reconstruction training strategy
3. LibriTTS-100h dataset
"""

import argparse
import copy
import json
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from zipvoice.dataset.dataset_vc import VoiceConversionCollate, VoiceConversionDataset
from zipvoice.models.zipvoice_vc import ZipVoiceVC
from zipvoice.utils.common import setup_logger
from zipvoice.utils.feature import VocosFbank


def get_args():
    parser = argparse.ArgumentParser(description="Train ZipVoice Voice Conversion model")
    
    # Training configuration
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port for DDP training.",
    )
    parser.add_argument(
        "--use-fp16",
        type=int,
        default=1,
        help="Whether to use half precision training.",
    )
    
    # Model configuration
    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/zipvoice_vc_base.json",
        help="Path to model configuration file.",
    )
    parser.add_argument(
        "--pretrained-content-encoder",
        type=str,
        required=True,
        help="Path to pre-trained Zipformer ASR model for content encoder.",
    )
    parser.add_argument(
        "--freeze-content-encoder",
        type=int,
        default=1,
        help="Whether to freeze content encoder parameters.",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--train-manifest",
        type=str,
        required=True,
        help="Path to training manifest file.",
    )
    parser.add_argument(
        "--dev-manifest",
        type=str,
        required=True,
        help="Path to development manifest file.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=500.0,
        help="Maximum duration per batch (seconds).",
    )
    parser.add_argument(
        "--max-len",
        type=float,
        default=20.0,
        help="Maximum length of utterances (seconds).",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.8,
        help="Probability of masking frames in speech condition.",
    )
    parser.add_argument(
        "--mask-length",
        type=int,
        default=10,
        help="Length of consecutive masked segments.",
    )
    
    # Training parameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.001,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--save-every-n",
        type=int,
        default=5,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--valid-every-n",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )
    
    # Output configuration
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/zipvoice_vc",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    
    return parser.parse_args()


def setup_distributed(rank: int, world_size: int, master_port: int):
    """Setup distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    
    if torch.cuda.is_available():
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=world_size
        )
        torch.cuda.set_device(rank)
    else:
        torch.distributed.init_process_group(
            "gloo", rank=rank, world_size=world_size
        )


def cleanup_distributed():
    """Cleanup distributed training"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_model(config: Dict[str, Any]) -> ZipVoiceVC:
    """Create ZipVoiceVC model from configuration"""
    model_config = config["model"]
    return ZipVoiceVC(**model_config)


def create_datasets(args) -> Tuple[VoiceConversionDataset, VoiceConversionDataset]:
    """Create training and validation datasets"""
    
    # Load manifests
    train_cuts = load_manifest_lazy(args.train_manifest)
    dev_cuts = load_manifest_lazy(args.dev_manifest)
    
    # Create feature extractor
    feature_extractor = VocosFbank()
    
    # Create datasets
    train_dataset = VoiceConversionDataset(
        cuts=train_cuts,
        feature_extractor=feature_extractor,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
        max_duration=args.max_len,
        shuffle=True,
    )
    
    dev_dataset = VoiceConversionDataset(
        cuts=dev_cuts,
        feature_extractor=feature_extractor,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
        max_duration=args.max_len,
        shuffle=False,
    )
    
    return train_dataset, dev_dataset


def create_data_loaders(
    train_dataset: VoiceConversionDataset,
    dev_dataset: VoiceConversionDataset,
    args,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders with dynamic bucketing"""
    
    # Create samplers
    train_sampler = DynamicBucketingSampler(
        train_dataset.cuts,
        max_duration=args.max_duration,
        shuffle=True,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    
    dev_sampler = DynamicBucketingSampler(
        dev_dataset.cuts,
        max_duration=args.max_duration,
        shuffle=False,
        drop_last=False,
        rank=rank,
        world_size=world_size,
    )
    
    # Create collate function
    collate_fn = VoiceConversionCollate()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, dev_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    use_fp16: bool = True,
) -> float:
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        if use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(
                    source_features=batch['source_features'],
                    source_features_lens=batch['source_features_lens'],
                    target_features=batch['target_features'],
                    target_features_lens=batch['target_features_lens'],
                    speech_condition=batch['speech_condition'],
                    speech_condition_lens=batch['speech_condition_lens'],
                )
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                source_features=batch['source_features'],
                source_features_lens=batch['source_features_lens'],
                target_features=batch['target_features'],
                target_features_lens=batch['target_features_lens'],
                speech_condition=batch['speech_condition'],
                speech_condition_lens=batch['speech_condition_lens'],
            )
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress
        if batch_idx % 100 == 0:
            logging.info(
                f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
            )
            
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dev_loader: DataLoader,
    epoch: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    use_fp16: bool = True,
) -> float:
    """Validation"""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dev_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        source_features=batch['source_features'],
                        source_features_lens=batch['source_features_lens'],
                        target_features=batch['target_features'],
                        target_features_lens=batch['target_features_lens'],
                        speech_condition=batch['speech_condition'],
                        speech_condition_lens=batch['speech_condition_lens'],
                    )
                    loss = outputs['loss']
            else:
                outputs = model(
                    source_features=batch['source_features'],
                    source_features_lens=batch['source_features_lens'],
                    target_features=batch['target_features'],
                    target_features_lens=batch['target_features_lens'],
                    speech_condition=batch['speech_condition'],
                    speech_condition_lens=batch['speech_condition_lens'],
                )
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    logging.info(f"Validation Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss: float,
    exp_dir: str,
    is_best: bool = False,
):
    """Save checkpoint"""
    
    checkpoint = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(exp_dir, f'checkpoint-epoch-{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(exp_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def train_worker(rank: int, world_size: int, args):
    """Training worker for distributed training"""
    
    # Setup
    if world_size > 1:
        setup_distributed(rank, world_size, args.master_port)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Setup logging
    if rank == 0:
        setup_logger(filename=os.path.join(args.exp_dir, "train.log"))
        os.makedirs(args.exp_dir, exist_ok=True)
        
        # Setup tensorboard
        writer = SummaryWriter(os.path.join(args.exp_dir, "tensorboard"))
    else:
        writer = None
    
    # Set random seeds
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)
    
    # Load model configuration
    config = load_model_config(args.model_config)
    
    # Create model
    model = create_model(config)
    
    # Load pre-trained content encoder
    model.load_pretrained_content_encoder(
        args.pretrained_content_encoder, 
        freeze=bool(args.freeze_content_encoder)
    )
    
    model = model.to(device)
    
    # Setup DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Create datasets and data loaders
    train_dataset, dev_dataset = create_datasets(args)
    train_loader, dev_loader = create_data_loaders(
        train_dataset, dev_dataset, args, rank, world_size
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-6)
    
    # Learning rate scheduler
    total_steps = args.num_epochs * len(train_loader)
    scheduler = LinearLR(
        optimizer, 
        start_factor=1e-6, 
        end_factor=1.0, 
        total_iters=args.warmup_steps
    )
    
    # GradScaler for mixed precision
    scaler = GradScaler() if args.use_fp16 else None
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        if rank == 0:
            logging.info(f"Starting epoch {epoch}/{args.num_epochs}")
        
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, device, writer, args.use_fp16
        )
        
        # Update learning rate
        if epoch <= args.warmup_steps // len(train_loader):
            scheduler.step()
        
        # Validation
        if epoch % args.valid_every_n == 0:
            val_loss = validate(model, dev_loader, epoch, device, writer, args.use_fp16)
            
            # Save checkpoint
            if rank == 0:
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                if epoch % args.save_every_n == 0 or is_best:
                    save_checkpoint(
                        model, optimizer, scaler, epoch, val_loss, args.exp_dir, is_best
                    )
        
        if rank == 0:
            logging.info(f"Epoch {epoch} completed. Train Loss: {train_loss:.4f}")
    
    # Cleanup
    if writer is not None:
        writer.close()
    
    if world_size > 1:
        cleanup_distributed()


def main():
    args = get_args()
    
    if args.world_size > 1:
        mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size)
    else:
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
