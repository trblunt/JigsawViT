# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
# writer = SummaryWriter('runs/no-improvement-rotation')

def update_targets_based_on_indices(original_targets, ids_used):
    # Create a new tensor filled with -1 of the same shape as original_targets
    updated_targets = torch.full_like(original_targets, -1, device=original_targets.device)
    
    # Use torch.gather to select the values from the original targets based on ids_used
    values_to_keep = torch.gather(original_targets, 1, ids_used)

    # Now, we scatter these kept values back to the updated_targets at the same indices
    updated_targets.scatter_(1, ids_used, values_to_keep)

    return updated_targets

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_jigsaw', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_rotations', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    accumulated_loss = torch.tensor(0.0, device=device)
    num_batches = 0

    try:
        for samples in metric_logger.log_every(data_loader, 10, f'Epoch: [{epoch}]'):
            # Unpack data based on your data structure
            images = samples['images'].to(device)
            targets = samples['targets'].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)  # Assume DistillationLoss is now correctly handled

            loss.backward()
            optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            accumulated_loss += loss.detach()
            num_batches += 1

    except KeyboardInterrupt:
        print("Training interrupted")
        sys.exit(1)

    avg_loss = accumulated_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch completed: Avg Loss: {avg_loss:.4f}")

    return {'loss': avg_loss.item(), 'lr': optimizer.param_groups[0]["lr"]}




@torch.no_grad()
def evaluate(data_loader, model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, _ in metric_logger.log_every(data_loader, 10, header):
        images, shuffled_images, target_positions, target_rotations = samples
        images = images.to(device, non_blocking=True)
        shuffled_images = shuffled_images.to(device, non_blocking=True)
        target_positions = target_positions.to(device, non_blocking=True)
        target_rotations = target_rotations.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images, shuffled_images, eval=True)
            
            batch_indices = torch.arange(outputs.ids_masked.size(0), device=outputs.ids_masked.device).unsqueeze(-1)
            target_positions[batch_indices, outputs.ids_masked] = -1
            target_rotations[batch_indices, outputs.ids_masked] = -1

            loss_jigsaw = F.cross_entropy(outputs.pred_jigsaw.view(-1, outputs.pred_jigsaw.size(-1)), target_positions.view(-1), ignore_index=-1)
            loss_rotations = F.cross_entropy(outputs.pred_rot.view(-1, outputs.pred_rot.size(-1)), target_rotations.view(-1), ignore_index=-1)

            
            acc1_jigsaw, acc5_jigsaw = accuracy(outputs.pred_jigsaw.view(-1, outputs.pred_jigsaw.size(-1)), target_positions.view(-1), topk=(1, 5))
            acc1_rotations = accuracy(outputs.pred_rot.view(-1, outputs.pred_rot.size(-1)), target_rotations.view(-1), topk=(1,))
            
            metric_logger.update(loss_jigsaw=loss_jigsaw.item(), loss_rotations=loss_rotations.item())
            metric_logger.meters['acc1_jigsaw'].update(acc1_jigsaw.item(), n=images.size(0))
            metric_logger.meters['acc5_jigsaw'].update(acc5_jigsaw.item(), n=images.size(0))
            metric_logger.meters['acc1_rotations'].update(acc1_rotations[0].item(), n=images.size(0))

    metric_logger.synchronize_between_processes()
    print(' * Jigsaw Acc@1 {acc1_jigsaw.global_avg:.3f} Acc@5 {acc5_jigsaw.global_avg:.3f}'
          ' Rotation Acc@1 {acc1_rotations.global_avg:.3f}'
          .format(acc1_jigsaw=metric_logger.acc1_jigsaw, acc5_jigsaw=metric_logger.acc5_jigsaw,
                  acc1_rotations=metric_logger.acc1_rotations))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}