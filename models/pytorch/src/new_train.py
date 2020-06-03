#!/usr/bin/env python

import collections
import pathlib
import time
import json
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False
try:
    import apex
    is_apex_available = True
except Exception:
    is_apex_available = False

from dataloader import get_loader
import utils
from utils import str2bool, AverageMeter
import augmentations
from argparser import get_config

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str)
    parser.add_argument('--config', type=str)

    # model config (VGG)
    parser.add_argument('--n_channels', type=str)
    parser.add_argument('--n_layers', type=str)
    parser.add_argument('--use_bn', type=str2bool)
    #
    parser.add_argument('--base_channels', type=int)
    parser.add_argument('--block_type', type=str)
    parser.add_argument('--depth', type=int)
    # model config (ResNet-preact)
    parser.add_argument('--remove_first_relu', type=str2bool)
    parser.add_argument('--add_last_bn', type=str2bool)
    parser.add_argument('--preact_stage', type=str)
    # model config (WRN)
    parser.add_argument('--widening_factor', type=int)
    # model config (DenseNet)
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--compression_rate', type=float)
    # model config (WRN, DenseNet)
    parser.add_argument('--drop_rate', type=float)
    # model config (PyramidNet)
    parser.add_argument('--pyramid_alpha', type=int)
    # model config (ResNeXt)
    parser.add_argument('--cardinality', type=int)
    # model config (shake-shake)
    parser.add_argument('--shake_forward', type=str2bool)
    parser.add_argument('--shake_backward', type=str2bool)
    parser.add_argument('--shake_image', type=str2bool)
    # model config (SENet)
    parser.add_argument('--se_reduction', type=int)

    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test_first', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')

    # TensorBoard configuration
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_train_images', action='store_true')
    parser.add_argument('--tensorboard_test_images', action='store_true')
    parser.add_argument('--tensorboard_model_params', action='store_true')

    # configuration of optimizer
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ghost_batch_size', type=int)
    parser.add_argument(
        '--optimizer', type=str, choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--gradient_clip', type=float)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--no_weight_decay_on_bn', action='store_true')
    # configuration for SGD
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--nesterov', type=str2bool)
    # configuration for learning rate scheduler
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['none', 'multistep', 'cosine', 'sgdr'])
    # configuration for multi-step scheduler]
    parser.add_argument('--milestones', type=str)
    parser.add_argument('--lr_decay', type=float)
    # configuration for cosine-annealing scheduler and SGDR scheduler
    parser.add_argument('--lr_min', type=float, default=0)
    # configuration for SGDR scheduler
    parser.add_argument('--T0', type=int)
    parser.add_argument('--Tmult', type=int)
    # configuration for Adam
    parser.add_argument('--betas', type=str)
    # configuration for LARS
    parser.add_argument('--lars_eps', type=float, default=1e-9)
    parser.add_argument('--lars_thresh', type=float, default=1e-2)

    # configuration of data loader
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'CIFAR102', 'MNIST', 'FashionMNIST', 'KMNIST'])
#     parser.add_argument(
#         '--testset',
#         type=str,
#         default='CIFAR10',
#         choices=['CIFAR10', 'CIFAR101', 'CIFAR102', 'MNIST', 'FashionMNIST', 'KMNIST'])
    parser.add_argument('--num_workers', type=int, default=7)
    # standard data augmentation
    parser.add_argument('--use_random_crop', type=str2bool)
    parser.add_argument('--random_crop_padding', type=int, default=4)
    parser.add_argument('--use_horizontal_flip', type=str2bool)
    # (dual-)cutout configuration
    parser.add_argument('--use_cutout', action='store_true', default=False)
    parser.add_argument(
        '--use_dual_cutout', action='store_true', default=False)
    parser.add_argument('--cutout_size', type=int, default=16)
    parser.add_argument('--cutout_prob', type=float, default=1)
    parser.add_argument('--cutout_inside', action='store_true', default=False)
    parser.add_argument('--dual_cutout_alpha', type=float, default=0.1)
    # random erasing configuration
    parser.add_argument(
        '--use_random_erasing', action='store_true', default=False)
    parser.add_argument('--random_erasing_prob', type=float, default=0.5)
    parser.add_argument(
        '--random_erasing_area_ratio_range', type=str, default='[0.02, 0.4]')
    parser.add_argument(
        '--random_erasing_min_aspect_ratio', type=float, default=0.3)
    parser.add_argument('--random_erasing_max_attempt', type=int, default=20)
    # mixup configuration
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', type=float, default=1)
    # RICAP configuration
    parser.add_argument('--use_ricap', action='store_true', default=False)
    parser.add_argument('--ricap_beta', type=float, default=0.3)
    # label smoothing configuration
    parser.add_argument(
        '--use_label_smoothing', action='store_true', default=False)
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1)
    # fp16
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False
    if not is_apex_available:
        args.use_amp = False
    if args.use_amp:
        args.fp16 = True

    config = get_config(args)

    return config


def train(epoch, model, optimizer, scheduler, criterion, train_loader, config,
          writer, amp_handle):
    global global_step

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']
    device = torch.device(run_config['device'])

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if data_config['use_mixup']:
            data, targets = augmentations.mixup.mixup(
                data, targets, data_config['mixup_alpha'],
                data_config['n_classes'])
        elif data_config['use_ricap']:
            data, targets = augmentations.ricap.ricap(
                data, targets, data_config['ricap_beta'],
                data_config['n_classes'])

        if run_config['tensorboard_train_images']:
            if step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Train/Image', image, epoch)

        if data_config['use_dual_cutout']:
            w = data.size(3) // 2
            data1 = data[:, :, :, :w]
            data2 = data[:, :, :, w:]

        if run_config['fp16'] and not run_config['use_amp']:
            if data_config['use_dual_cutout']:
                data1 = data1.half()
                data2 = data2.half()
            else:
                data = data.half()

        if run_config['tensorboard']:
            if optim_config['scheduler'] != 'none':
                lr = scheduler.get_lr()[0]
            else:
                lr = optim_config['base_lr']
            writer.add_scalar('Train/LearningRate', lr, global_step)

        if torch.cuda.device_count() == 1:
            if data_config['use_dual_cutout']:
                data1 = data1.to(device)
                data2 = data2.to(device)
            else:
                data = data.to(device)

        if data_config['use_mixup']:
            t1, t2, lam = targets
            targets = (t1.to(device), t2.to(device), lam)
        elif data_config['use_ricap']:
            labels, weights = targets
            labels = [label.to(device) for label in labels]
            targets = (labels, weights)
        else:
            targets = targets.to(device)

        optimizer.zero_grad()

        if 'ghost_batch_size' not in optim_config.keys():
            if data_config['use_dual_cutout']:
                outputs1 = model(data1)
                outputs2 = model(data2)
                outputs = (outputs1, outputs2)
            else:
                outputs = model(data)
            loss = criterion(outputs, targets)
            if amp_handle is not None:
                with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if 'gradient_clip' in optim_config.keys():
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               optim_config['gradient_clip'])
            optimizer.step()
        else:
            batch_size = optim_config['batch_size']
            ghost_batch_size = optim_config['ghost_batch_size']
            n_split = batch_size // ghost_batch_size

            if data_config['use_dual_cutout']:
                data1_chunks = data1.chunk(n_split)
                data2_chunks = data2.chunk(n_split)
                target_chunks = targets.chunk(n_split)
                outputs1 = []
                outputs2 = []
                for data1_chunk, data2_chunk, target_chunk in zip(
                        data1_chunks, data2_chunks, target_chunks):
                    output1_chunk = model(data1_chunk)
                    output2_chunk = model(data2_chunk)
                    outputs1.append(output1_chunk)
                    outputs2.append(output2_chunk)
                    output_chunk = (output1_chunk, output2_chunk)
                    loss = criterion(output_chunk, target_chunk)
                    if amp_handle is not None:
                        with amp_handle.scale_loss(loss,
                                                   optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                outputs1 = torch.cat(outputs1)
                outputs2 = torch.cat(outputs2)
                outputs = (outputs1, outputs2)
            else:
                data_chunks = data.chunk(n_split)
                if data_config['use_mixup']:
                    targets1, targets2, lam = targets
                    target_chunks = [
                        (chunk1, chunk2, lam) for chunk1, chunk2 in zip(
                            targets1.chunk(n_split), targets2.chunk(n_split))
                    ]
                elif data_config['use_ricap']:
                    target_list, weights = targets
                    target_list_chunks = list(
                        zip(*[target.chunk(n_split)
                              for target in target_list]))
                    target_chunks = [(chunk, weights)
                                     for chunk in target_list_chunks]
                else:
                    target_chunks = targets.chunk(n_split)
                outputs = []
                for data_chunk, target_chunk in zip(data_chunks,
                                                    target_chunks):
                    output_chunk = model(data_chunk)
                    outputs.append(output_chunk)
                    loss = criterion(output_chunk, target_chunk)
                    if amp_handle is not None:
                        with amp_handle.scale_loss(loss,
                                                   optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                outputs = torch.cat(outputs)
            if 'gradient_clip' in optim_config.keys():
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               optim_config['gradient_clip'])
            for param in model.parameters():
                param.grad.data.div_(n_split)
            optimizer.step()
        
        if optim_config['scheduler'] in ['multistep', 'sgdr']:
            scheduler.step(epoch - 1)
        elif optim_config['scheduler'] == 'cosine':
            scheduler.step()
            
        loss_ = loss.item()
        num = data.size(0)
        if data_config['use_mixup']:
            targets1, targets2, lam = targets
            accuracy = lam * utils.accuracy(outputs, targets1)[0].item() + (
                1 - lam) * utils.accuracy(outputs, targets2)[0].item()
        elif data_config['use_ricap']:
            accuracy = sum([
                weight * utils.accuracy(outputs, labels)[0].item()
                for labels, weight in zip(*targets)
            ])
        elif data_config['use_dual_cutout']:
            accuracy = utils.accuracy((outputs1 + outputs2) / 2,
                                      targets)[0].item()
        else:
            accuracy = utils.accuracy(outputs, targets)[0].item()

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

    train_log = collections.OrderedDict({
        'epoch':
        epoch,
        'train':
        collections.OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy_meter.avg,
            'time': elapsed,
        }),
    })
    return train_log


def test(epoch, model, criterion, test_loaders, run_config, writer):
    logger.info('Test {}'.format(epoch))

    device = torch.device(run_config['device'])

    model.eval()
    
    test_log = collections.OrderedDict({'epoch': epoch, 'test': collections.OrderedDict()}) #initiate the test log
    for dataset_name, test_loader in test_loaders.items():
        logger.info('Testset {}'.format(dataset_name))
        loss_meter = AverageMeter()
        correct_meter = AverageMeter()
        start = time.time()
        with torch.no_grad():
            for step, (data, targets) in enumerate(test_loader):
                if run_config['tensorboard_test_images']:
                    if epoch == 0 and step == 0:
                        image = torchvision.utils.make_grid(
                            data, normalize=True, scale_each=True)
                        writer.add_image('Test/Image', image, epoch)

                if run_config['fp16'] and not run_config['use_amp']:
                    data = data.half()

                data = data.to(device)
                targets = targets.to(device)

                outputs = model(data)
                loss = criterion(outputs, targets)

                _, preds = torch.max(outputs, dim=1)

                loss_ = loss.item()
                correct_ = preds.eq(targets).sum().item()
                num = data.size(0)

                loss_meter.update(loss_, num)
                correct_meter.update(correct_, 1)

            accuracy = correct_meter.sum / len(test_loader.dataset)

            logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch, loss_meter.avg, accuracy))

            elapsed = time.time() - start
            logger.info('Elapsed {:.2f}'.format(elapsed))

        if run_config['tensorboard']:
            if epoch > 0:
                writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/Accuracy', accuracy, epoch)
            writer.add_scalar('Test/Time', elapsed, epoch)

        if run_config['tensorboard_model_params']:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step)
        
        single_test_log = collections.OrderedDict({
            'epoch': epoch,
            'test': collections.OrderedDict({dataset_name:
                collections.OrderedDict({
                    'loss': loss_meter.avg,
                    'accuracy': accuracy,
                    'time': elapsed,
                }),
            }),
        })
        test_log['test'].update(single_test_log['test'])
        
    return test_log


def update_state(state, epoch, accuracy, model, optimizer):
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['epoch'] = epoch
    state['accuracy'] = accuracy

    # update best accuracy
    if accuracy > state['best_accuracy']:
        state['best_accuracy'] = accuracy
        state['best_epoch'] = epoch

    return state


def main():
    # parse command line argument and generate config dictionary
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    # TensorBoard SummaryWriter
    if run_config['tensorboard']:
        writer = SummaryWriter(run_config['outdir'])
    else:
        writer = None

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    epoch_seeds = np.random.randint(
        np.iinfo(np.int32).max // 2, size=optim_config['epochs'])

    # create output directory
    outdir = pathlib.Path(run_config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    # save config as json file in output directory
    outpath = outdir / 'config.json'
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # load data loaders
    train_loader, test_loaders = get_loader(config['data_config'])

    # load model
    logger.info('Loading model...')
    model = utils.load_model(config['model_config'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    if run_config['fp16'] and not run_config['use_amp']:
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    device = torch.device(run_config['device'])
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    logger.info('Done')

    train_criterion, test_criterion = utils.get_criterion(
        config['data_config'])

    # create optimizer
    if optim_config['no_weight_decay_on_bn']:
        params = [
            {
                'params': [
                    param for name, param in model.named_parameters()
                    if 'bn' not in name
                ]
            },
            {
                'params': [
                    param for name, param in model.named_parameters()
                    if 'bn' in name
                ],
                'weight_decay':
                0
            },
        ]
    else:
        params = model.parameters()
    optim_config['steps_per_epoch'] = len(train_loader)
    optimizer, scheduler = utils.create_optimizer(params, optim_config)

    # for mixed-precision
    amp_handle = apex.amp.init(
        enabled=run_config['use_amp']) if is_apex_available else None

    # run test before start training
    if run_config['test_first']:
        test(0, model, test_criterion, test_loaders, run_config, writer)

    state = {
        'config': config,
        'state_dict': None,
        'optimizer': None,
        'epoch': 0,
        'accuracy': 0,
        'best_accuracy': 0,
        'best_epoch': 0,
    }
    epoch_logs = []
    for epoch, seed in zip(range(1, optim_config['epochs'] + 1), epoch_seeds):
        np.random.seed(seed)
        # train
        train_log = train(epoch, model, optimizer, scheduler, train_criterion,
                          train_loader, config, writer, amp_handle)

        # test
        test_log = test(epoch, model, test_criterion, test_loaders, run_config,
                        writer)

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        utils.save_epoch_logs(epoch_logs, outdir)

        # update state dictionary
        state = update_state(state, epoch, epoch_log['test'][config['data_config']['dataset']]['accuracy'],
                             model, optimizer)

        # save model
        utils.save_checkpoint(state, outdir)


if __name__ == '__main__':
    main()
