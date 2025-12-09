# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from datasets import load_from_disk

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import imageutil.misc as misc
from imageutil.misc import NativeScalerWithGradNormCount as NativeScaler

from models import models_mae_vitae

from imageutil.engine import train_one_epoch, train_and_validate

from dataset_util.FiveChannelDataset import FiveChannelDataset
from root_path import ROOT_PATH

from imageutil.trans import CustomRandomHorizontalFlip, CustomRandomRotation, CustomRandomVerticalFlip, CustomCenterCrop


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vitae_base_patch8', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=196, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default=f'{ROOT_PATH}/data/data_med/train_dataset', type=str,
                        help='train_dataset path')
    parser.add_argument('--test_path', default=f'{ROOT_PATH}/data/data_med/test_dataset', type=str,
                        help='test_dataset path')

    parser.add_argument('--output_dir', default=f'{ROOT_PATH}/outputs/img/pths',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=f'{ROOT_PATH}/outputs/img/log',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank','--local-rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--clip_grad', default=1.0, type=float,
                        help='Gradient clipping norm (default: 1.0). Set to 0 to disable.')
    
    # dataset
    # parser.add_argument('--dataset', default=None, type=str, choices=['image'], help='type of dataset')

    # gpu_num
    # TODO 我改成了1
    parser.add_argument("--gpu_num", default=1, type=int, help='number of gpus')

    parser.add_argument("--tag", default=None, type=int, help='different number of training samples')
    parser.add_argument('--distributed',default=True)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    best_loss = float('inf')
    # simple augmentation
    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        CustomRandomHorizontalFlip(p=0.4),
        CustomRandomVerticalFlip(p=0.4),
        CustomRandomRotation(degrees=45, resample='bilinear', p=0.4),
        # CustomCenterCrop(size=32),
        transforms.Normalize(mean=[0.00386281, 0.00558842, 0.0086468,  0.01220661, 0.01972715],
                             std=[0.4165482, 0.27164888, 0.38365453, 0.476246, 1.3442839])])
    
    # if args.dataset == 'image':
    # dataset_train = FiveChannelDataset(args.data_path, transform=transform_train)
    dataset_train = load_from_disk(args.train_path)
    dataset_train = FiveChannelDataset(dataset_train, transform=transform_train)
    dataset_test = load_from_disk(args.test_path)
    dataset_test = FiveChannelDataset(dataset_test, transform=transform_train)

    # dataset_train = dataset_train.with_transform(transform_train)
    # elif args.dataset == 'millionAID':
    #
    #     args.data_path = '../Dataset/millionaid/'
    #
    #     dataset_train = MillionAIDDataset(args.data_path, train=True, transform=transform_train, tag=args.tag)
    # else:
    #     raise NotImplementedError

    # output folder
    args.output_dir = os.path.join(args.output_dir, '_'+str(args.input_size),
    str(args.epochs)+'_'+str(args.mask_ratio)+'_'+str(args.blr)+'_'+str(args.weight_decay)+'_'+str(args.batch_size*args.gpu_num))

    os.makedirs(args.output_dir, exist_ok=True)

    print(dataset_train)
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    # if global_rank == 0 and
    # TODO 我改了
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    # if 'mae_vit_' in args.model:
    #     print('MAE pretraining ViT series model')
    #     model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    if 'mae_vitae_' in args.model:
        print('MAE pretraining ViTAE series model')
        model = models_mae_vitae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    else:
        raise NotImplementedError

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  # 累积iter, lr会增加

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    #TODO 改了
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_test.sampler.set_epoch(epoch)
        stats = train_and_validate(
            model, data_loader_train, data_loader_test,  # Added val_loader (data_loader_test)
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        current_loss = stats['val_loss']  # Assuming 'val_loss' is returned in test_stats
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"New best val loss: {best_loss:.4f}, saving model...")
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, filename="best_val.pth")
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
