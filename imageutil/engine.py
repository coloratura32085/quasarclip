import math
import sys
from typing import Iterable

import torch

import imageutil.lr_sched as lr_svitae_prnched
import imageutil.misc as misc
from imageutil import lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    metric_logger, log_writer=None, args=None):
    model.train(True)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Training loop
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Adjust learning rate
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True, dtype=torch.float32)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer,clip_grad = args.clip_grad, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_one_epoch(model: torch.nn.Module,
                       val_loader: Iterable, device: torch.device, epoch: int,
                       metric_logger, log_writer=None, args=None):
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for val_samples in val_loader:
            val_samples = val_samples.to(device, non_blocking=True, dtype=torch.float32)

            with torch.cuda.amp.autocast():
                val_loss_batch, _, _ = model(val_samples, mask_ratio=args.mask_ratio)
                val_loss += val_loss_batch.item()

    val_loss /= len(val_loader)
    print(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")

    if log_writer is not None:
        epoch_1000x = int(epoch * 1000)
        log_writer.add_scalar('val_loss', val_loss, epoch_1000x)

    # Update the metric_logger for validation loss
    metric_logger.update(val_loss=val_loss)

    return {'val_loss' : val_loss}


def train_and_validate(model: torch.nn.Module,
                       train_loader: Iterable, val_loader: Iterable,
                       optimizer: torch.optim.Optimizer, device: torch.device,
                       epoch: int, loss_scaler, log_writer=None, args=None):
    # Initialize metric_logger outside the loop so it can be shared across both training and validation
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # Train for one epoch
    train_stats = train_one_epoch(
        model, train_loader, optimizer, device, epoch, loss_scaler, metric_logger, log_writer, args
    )

    # Validate after each epoch
    val_loss = None
    if val_loader is not None:
        val_loss = validate_one_epoch(
            model, val_loader, device, epoch, metric_logger, log_writer, args
        )

    # Combine training and validation stats into one dictionary
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", metric_logger)
    return stats

        # # Optionally: You could log additional information, such as epoch-level training stats.
        # print(f"Epoch {epoch}: Train Loss: {train_stats['loss']:.4f}, Validation Loss: {val_loss:.4f}")




