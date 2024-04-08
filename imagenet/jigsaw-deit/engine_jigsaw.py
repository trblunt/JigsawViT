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
    metric_logger.add_meter('lr_new', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_cls', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_jigsaw', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_rotations', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accumulated_loss = torch.tensor(0.0, device=device)
    accumulated_loss_cls = torch.tensor(0.0, device=device)
    accumulated_loss_jigsaw = torch.tensor(0.0, device=device)
    accumulated_loss_rotations = torch.tensor(0.0, device=device)
    num_batches = 0

    torch.cuda.set_sync_debug_mode(1)

    # profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    #                    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
    #                    record_shapes=True,
    #                    with_stack=False)

    try:
        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            images, shuffled_images, target_positions, target_rotations = samples

            images = images.to(device, non_blocking=True)
            shuffled_images = shuffled_images.to(device, non_blocking=True)
            target_positions = target_positions.to(device, non_blocking=True)
            target_rotations = target_rotations.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                with record_function("model_inference"):
                    outputs = model(images, shuffled_images)
                
                with record_function("loss_calculation"):
                    loss = criterion(images, outputs.pred_cls, targets)
                    accumulated_loss_cls += loss
                    if args.use_jigsaw:
                        target_positions = update_targets_based_on_indices(target_positions, outputs.ids_used)
                        target_rotations = update_targets_based_on_indices(target_rotations, outputs.ids_used)
                        loss_jigsaw = F.cross_entropy(outputs.pred_jigsaw.view(-1, outputs.pred_jigsaw.size(-1)), target_positions.view(-1), ignore_index=-1)
                        loss += loss_jigsaw * args.lambda_jigsaw
                        loss_rotations = F.cross_entropy(outputs.pred_rot.view(-1, outputs.pred_rot.size(-1)), target_rotations.view(-1), ignore_index=-1)
                        loss += loss_rotations * args.lambda_rotations
                        # writer.add_scalar('Loss/total', loss.item(), epoch * len(data_loader) + num_batches)
                        # writer.add_scalar('Loss/jigsaw', loss_jigsaw.item(), epoch * len(data_loader) + num_batches)
                        # writer.add_scalar('Loss/rotation', loss_rotations.item(), epoch * len(data_loader) + num_batches)

                        # For debugging, print the first 16 target_rotations and pred_rot
                        # print("target_rotations", target_rotations[:16])
                        # print("pred_rot", outputs.pred_rot[:16])

                    accumulated_loss += loss
                    accumulated_loss_jigsaw += loss_jigsaw
                    accumulated_loss_rotations += loss_rotations
                    num_batches += 1
                
            with record_function("loss_sync"):
                if num_batches % print_freq == 0:
                    # Calculate average losses
                    avg_loss = accumulated_loss / print_freq
                    avg_loss_cls = accumulated_loss_cls / print_freq
                    avg_loss_jigsaw = accumulated_loss_jigsaw / print_freq
                    avg_loss_rotations = accumulated_loss_rotations / print_freq

                    # Update metric logger (converts to CPU scalar here, implicitly calls .item())
                    metric_logger.meters['loss'].update(avg_loss.item(), n=print_freq)
                    metric_logger.meters['loss_cls'].update(avg_loss_cls.item(), n=print_freq)
                    metric_logger.meters['loss_jigsaw'].update(avg_loss_jigsaw.item(), n=print_freq)
                    metric_logger.meters['loss_rotations'].update(avg_loss_rotations.item(), n=print_freq)

                    # Reset accumulators
                    accumulated_loss.zero_()
                    accumulated_loss_cls.zero_()
                    accumulated_loss_jigsaw.zero_()
                    accumulated_loss_rotations.zero_()

                    # for idx, group in enumerate(optimizer.param_groups):
                    #     print("Learning rate for group {} is {}".format(idx, group['lr']))
                    #     if 'lr_scale' in group:
                    #         print("Learning rate scale for group {} is {}".format(idx, group['lr_scale']))

                    if not math.isfinite(avg_loss.item()):
                        print("Loss is {}, stopping training".format(avg_loss.item()))
                        sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            
            # for name, parameter in model.named_parameters():
            #     if parameter.grad is not None:
            #         writer.add_histogram(f"Gradients/{name}", parameter.grad, epoch * len(data_loader) + num_batches)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr_new=optimizer.param_groups[-1]["lr"])

            # # At the end of each batch iteration, log learning rates
            # for i, param_group in enumerate(optimizer.param_groups):
            #     writer.add_scalar(f'Learning_Rate/group_{i}', param_group['lr'], epoch * len(data_loader) + num_batches)

            #profiler.step()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, saving profiler data...")
        # writer.close()
        #profiler.step()
        sys.exit(1)
    # except Exception as e:
    #     print("Caught exception: {}".format(e))
    # finally:
    #     metric_logger.synchronize_between_processes()
    #     print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
