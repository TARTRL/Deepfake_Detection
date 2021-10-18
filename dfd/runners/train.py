#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import argparse
import time
import yaml

import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP

    has_apex = False
assert has_apex

from dfd.timm.data import DeepFakeDataset_v3, create_deepfake_loader_v3, \
    resolve_data_config, FastCollateMixup, \
    mixup_batch, AugMixDataset
from dfd.timm.models import create_deepfake_model_v4, resume_checkpoint, convert_splitbn_model
from dfd.timm.utils import *
from dfd.timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from dfd.timm.optim import create_optimizer
from dfd.timm.scheduler import create_scheduler
from dfd.utils import get_base_name, check_dir, new_dir, check_file, del_file
from dfd.server_json import parse_server


torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('--data', default=None, type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--validation_frac', default=0.1, type=float,
                    help='validation fraction')
parser.add_argument('--train_frac', default=1, type=float,
                    help='validation fraction')
parser.add_argument('--label_balance', action='store_true', default=False,
                    help='Balance label')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--class_names', default=None, type=str,
                    help='the list of the names of classes')
parser.add_argument('--share_file', default='', type=str, metavar='SF',
                    help='share file')
parser.add_argument('--master_share_file', default='', type=str, metavar='SF',
                    help='master share file')
parser.add_argument('--json_file', type=str, metavar='JS',
                    help='json file')
parser.add_argument('--model-version', default=None, type=str, metavar='model version',
                    help='model version')
parser.add_argument('--input-size-v2', default=None, type=str,
                    help='input size("channel,height,width")')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--flicker', type=float, default=0.,
                    help='flicker probability')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--basic_lr', type=float, default=0.0000625, metavar='BLR',
                    help='basic learning rate (default: 0.0000625)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# Augmentation parameters
parser.add_argument('--blur_radiu', type=int, default=1,
                    help='Blur radiu (default: 1)')
parser.add_argument('--blur_prob', type=float, default=0,
                    help='Blur probability (default: 0)')
parser.add_argument('--color-jitter', type=float, default=0.2, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel", opt:"const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--rotate_range', type=int, default=0,
                    help='Rotate degree range (default: 0)')
parser.add_argument('--remax', type=float, default=0.02,
                    help='Max area of random erase')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')
# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='prec1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "prec1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main(rank, args, args_text):
    setup_default_logging()
    args.local_rank = rank
    args.whole_rank = rank + args.start_rank
    args.prefetcher = not args.no_prefetcher
    args.distributed = False

    if args.world_size > 1:
        args.distributed = True
        if args.distributed and args.num_gpu > 1:
            args.num_gpu = 1

    if args.distributed:
        assert args.local_size > args.local_rank
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank

        torch.cuda.set_device(args.device)

        if args.start_rank == 0:
            share_file = args.master_share_file
        else:
            share_file = args.share_file
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='file://{}'.format(share_file),
                                             world_size=args.world_size,
                                             rank=args.whole_rank)

    else:
        args.device = 'cuda:0'
        args.world_size = 1
        args.local_size = 1
        args.local_rank = 0  # global rank
        args.whole_rank = 0

    assert args.local_rank >= 0 and args.whole_rank >= 0

    if args.distributed:
        logging.info('local rank %d whole rank %d, local size %d world_size %d.'
                     % (args.local_rank, args.whole_rank, args.local_size, args.world_size))
    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.whole_rank)

    assert args.initial_checkpoint == '' or check_file(
        args.initial_checkpoint), 'checkpoint file {} doesn\'t exist!'.format(args.initial_checkpoint)

    if args.local_rank == 0: logging.info("creating model...")
    model = create_deepfake_model_v4(
        args.model,
        in_chans=12,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint,
        strict=True
    )


    if args.local_rank == 0: logging.info("creat model finished")
    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    if args.num_gpu > 1:
        if args.amp:
            logging.warning(
                'AMP does not work well with nn.DataParallel, disabling. Use distributed mode for multi-GPU AMP.')
            args.amp = False
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    if args.local_rank == 0: logging.info("creating optimizer")
    optimizer = create_optimizer(args, model)
    if args.local_rank == 0: logging.info("creat optimizer finished")
    use_amp = False
    if has_apex and args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        use_amp = True

    if args.local_rank == 0:
        logging.info('NVIDIA APEX {}. AMP {}.'.format(
            'installed' if has_apex else 'not installed', 'on' if use_amp else 'off'))

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(model, args.resume)
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
        if use_amp and 'amp' in resume_state and 'load_state_dict' in amp.__dict__:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')
            amp.load_state_dict(resume_state['amp'])
    del resume_state

    model_ema = None
    if args.model_ema:
        if args.local_rank == 0: logging.info('creating EMA model...')
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)
        if args.local_rank == 0: logging.info('create EMA model done')

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex:
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex:
            model = DDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            model = DDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    data_dirs = args.data.split(':')
    if args.local_rank == 0:
        print('load datas:{}'.format(data_dirs))
    time.sleep(1)

    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            logging.error('Training folder does not exist at: {}'.format(data_dir))
            exit(1)

    if args.local_rank == 0: logging.info('creating train dataset...')

    dataset_train = DeepFakeDataset_v3(data_dirs, class_names=args.class_names,
                                       train_split=True,
                                       is_training=True,
                                       train_ratio=args.train_frac,
                                       random_state=0,
                                       label_balance=args.label_balance,
                                       noise_fake=False)

    collate_fn = None
    if args.prefetcher and args.mixup > 0:
        assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
        collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)


    loader_train = create_deepfake_loader_v3(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        re_max=args.remax,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        fp16=True,
        pin_memory=args.pin_mem,
        flicker=args.flicker,
        rotate_range=args.rotate_range,
        blur_prob=args.blur_prob,
        blur_radiu=args.blur_radiu
    )

    if args.local_rank == 0: logging.info('create train dataset done')

    if args.local_rank == 0: logging.info('creating validation dataset...')


    dataset_eval = DeepFakeDataset_v3(data_dirs, class_names=args.class_names,
                                      train_split=True,
                                      is_training=False,
                                      train_ratio=args.train_frac, random_state=0, label_balance=args.label_balance)

    loader_eval = create_deepfake_loader_v3(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        fp16=True
    )
    if args.local_rank == 0: logging.info('create validation dataset done')

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    elif args.mixup > 0.:
        # smoothing is handled with mixup label transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        # print('using CrossEntropyLoss')
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        validate_loss_fn = train_loss_fn

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None

    if args.whole_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=args.output_dir, checkpoint_dir_bak=args.output_dir_bak,
                                decreasing=decreasing)
        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if args.whole_rank == 0:
        logging.info(
            'local rank:{} whole rank:{} train data:{} batches {} images, val data:{} batches {} images'.format(
                args.local_rank, args.whole_rank,
                len(loader_train),
                len(loader_train) * args.batch_size * args.world_size,
                len(loader_eval),
                len(loader_eval) * args.batch_size * args.world_size * args.validation_batch_size_multiplier))

    time.sleep(3)
    if args.local_rank == 0: logging.info('begin training!')

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=args.output_dir,
                use_amp=use_amp, model_ema=model_ema)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if args.whole_rank == 0:
                update_summary(
                    epoch, train_metrics, eval_metrics, filename=os.path.join(args.output_dir, 'summary.csv'),
                    plots_dir=os.path.join(args.output_dir, 'plots'),
                    write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, args,
                    epoch=epoch, model_ema=model_ema, metric=save_metric, use_amp=use_amp)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        if args.local_rank == 0: logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None):
    if args.prefetcher and args.mixup > 0 and loader.mixup_enabled:
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            loader.mixup_enabled = False

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if args.mixup > 0.:
                input, target = mixup_batch(
                    input, target,
                    alpha=args.mixup, num_classes=args.num_classes, smoothing=args.smoothing,
                    disable=args.mixup_off_epoch and epoch >= args.mixup_off_epoch)

        output = model(input)

        loss = loss_fn(output, target)
        prec1 = accuracy(output, target, topk=(1,))
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
        else:
            reduced_loss = loss.data

        optimizer.zero_grad()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        loss_value = reduced_loss.item()
        if not np.isnan(loss_value):
            losses_m.update(loss_value, input.size(0))

        prec1_m.update(prec1.item(), output.size(0))

        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)


            if args.local_rank == 0:
                ets_time = batch_time_m.avg * (len(loader) - batch_idx) / 60
                logging.info(
                    'Train:{} [{:>4d}/{}] '
                    'Loss:{loss.val:.5f}({loss.avg:.5f}) '
                    'Prec@1:{top1.val:>7.4f}({top1.avg:>7.4f}) '
                    'Time:{batch_time.val:.3f}({batch_time.avg:.3f})s/batch '
                    '{rate:.5f}({rate_avg:.5f})s/image '
                    'LR:{lr:.3e} '
                    'Data:{data_time.val:.3f}({data_time.avg:.3f})s/batch '
                    'ETS:{ets_time:.3f}min'.format(
                        epoch,
                        batch_idx, len(loader),
                        loss=losses_m,
                        top1=prec1_m,
                        batch_time=batch_time_m,
                        rate=batch_time_m.val / (input.size(0) * args.world_size),
                        rate_avg=batch_time_m.avg / (input.size(0) * args.world_size),
                        lr=lr,
                        data_time=data_time_m,
                        ets_time=ets_time))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    metrics = OrderedDict([('loss', losses_m.avg), ('prec1', prec1_m.avg), ('learning_rate', lr)])
    return metrics


def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)

            prec1 = accuracy(output, target, topk=(1,))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                prec1 = reduce_tensor(prec1, args.world_size)

            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            loss_value = reduced_loss.item()
            if not np.isnan(loss_value):
                losses_m.update(loss_value, input.size(0))

            prec1_m.update(prec1.item(), output.size(0))


            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                ets_time = batch_time_m.avg * (len(loader) - batch_idx) / 60
                logging.info(
                    '{0}:[{1:>4d}/{2}] '
                    'Loss:{loss.val:>.4f}({loss.avg:>.4f}) '
                    'Prec@1:{top1.val:>.4f}({top1.avg:>.4f}) '
                    'Time:{batch_time.val:.3f}({batch_time.avg:.3f})s/batch '
                    'est:{ets_time:.3f}min'.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, top1=prec1_m,
                        batch_time=batch_time_m,
                        ets_time=ets_time))

    metrics = OrderedDict([('loss', losses_m.avg), ('prec1', prec1_m.avg)])
    return metrics


def launch_main():
    args, args_text = _parse_args()
    args.hostname, gpus, args.world_size, args.local_size, args.start_rank = parse_server(args.json_file)
    assert not args.initial_checkpoint or check_file(args.initial_checkpoint), 'no checkpoint:{}'.format(
        args.initial_checkpoint)

    if args.start_rank == 0 and check_file(args.master_share_file):
        print('remove share file {}'.format(args.master_share_file))
        del_file(args.master_share_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_index) for gpu_index in gpus])

    print('{}: world size:{} local size:{} start rank {}'.format(args.hostname, args.world_size, args.local_size,
                                                                 args.start_rank))
    print('to use gpus:{}'.format(gpus))

    if args.start_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            args.model_version,
            get_base_name(args.data)
        ])

        output_dir = os.path.join(output_base, exp_name)
        if check_dir(output_dir):
            print('{} exists!'.format(output_dir))
            time.sleep(1)


        new_dir(output_dir)
        args.output_dir = output_dir

        output_dir_bak = output_dir
        while output_dir_bak[-1] == '/':
            output_dir_bak = output_dir_bak[:-1]
        output_dir_bak += '_bak'
        print('output bak dir: {}'.format(output_dir_bak))
        if not check_dir(output_dir_bak):
            new_dir(output_dir_bak)
        args.output_dir_bak = output_dir_bak
    else:
        args.output_dir = None
        args.output_dir_bak = None

    # ajust learning rate
    args.lr = args.batch_size * args.world_size * args.basic_lr
    print('learning rate:{}'.format(args.lr))
    mp.spawn(main, nprocs=args.local_size, args=(args, args_text))


if __name__ == '__main__':
    launch_main()