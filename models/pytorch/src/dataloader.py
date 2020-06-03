import pathlib
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

import augmentations
import transforms

from cifar10x_loader import *


class Dataset:
    def __init__(self, config):
        self.config = config
        dataset_rootdir = pathlib.Path('~/.torchvision/datasets').expanduser()

        if self.config['dataset'] in ['CIFAR10_10K', 'CIFAR101', 'CIFAR102', 'CIFAR10_30K','CIFAR102_30K',]:
            self.dataset_dir = '../'
        else:
            self.dataset_dir = dataset_rootdir / config['dataset']

        self.local_testset_dir = '../'

        self._train_transforms = []
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def get_datasets(self):
        # load training set
        if self.config['dataset'] in ['CIFAR10_10K', 'CIFAR101', 'CIFAR102', 'CIFAR10_30K','CIFAR102_30K',]:
            train_dataset = CIFAR10X(self.dataset_dir, train=True,
                                     transform=self.train_transform,
                                     trainset=self.config['dataset'])
        else:
            train_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                self.dataset_dir,
                train=True,
                transform=self.train_transform,
                download=True)
         # load testset
        test_datasets = {}
        # load cifar-10 testset
        test_datasets['CIFAR10'] = getattr(torchvision.datasets, 'CIFAR10')(
            self.dataset_dir, train=False,
            transform=self.test_transform, download=True)
        # load cifar10x testset
        test_datasets['CIFAR102_30K'] = CIFAR10X(self.local_testset_dir,train=False,
                                    transform=self.test_transform,
                                    testset='CIFAR102_30K')
        test_datasets['CIFAR102'] = CIFAR10X(self.local_testset_dir,train=False,
                                    transform=self.test_transform,
                                    testset='CIFAR102')
        test_datasets['CIFAR101'] = CIFAR10X(self.local_testset_dir,train=False,
                                    transform=self.test_transform,
                                    testset='CIFAR101')
        test_datasets['CIFAR10_30K'] = CIFAR10X(self.local_testset_dir,train=False,
                                    transform=self.test_transform,
                                    testset='CIFAR10_30K')

        if self.config['dataset'] != 'CIFAR10':
            print('Mean per Channel:', train_dataset.data.mean(axis=(0,1,2))/255)
            print('Standard Deviation per Channel:', train_dataset.data.std(axis=(0,1,2))/255)
        return train_dataset, test_datasets

    def _add_random_crop(self):
        transform = torchvision.transforms.RandomCrop(
            self.size, padding=self.config['random_crop_padding'])
        self._train_transforms.append(transform)

    def _add_horizontal_flip(self):
        self._train_transforms.append(
            torchvision.transforms.RandomHorizontalFlip())

    def _add_normalization(self):
        self._train_transforms.append(
            transforms.Normalize(self.mean, self.std))

    def _add_to_tensor(self):
        self._train_transforms.append(transforms.ToTensor())

    def _add_random_erasing(self):
        transform = augmentations.random_erasing.RandomErasing(
            self.config['random_erasing_prob'],
            self.config['random_erasing_area_ratio_range'],
            self.config['random_erasing_min_aspect_ratio'],
            self.config['random_erasing_max_attempt'])
        self._train_transforms.append(transform)

    def _add_cutout(self):
        transform = augmentations.cutout.Cutout(self.config['cutout_size'],
                                                self.config['cutout_prob'],
                                                self.config['cutout_inside'])
        self._train_transforms.append(transform)

    def _add_dual_cutout(self):
        transform = augmentations.cutout.DualCutout(
            self.config['cutout_size'], self.config['cutout_prob'],
            self.config['cutout_inside'])
        self._train_transforms.append(transform)

    def _get_train_transform(self):
        if self.config['use_random_crop']:
            self._add_random_crop()
        if self.config['use_horizontal_flip']:
            self._add_horizontal_flip()
        self._add_normalization()
        if self.config['use_random_erasing']:
            self._add_random_erasing()
        if self.config['use_cutout']:
            self._add_cutout()
        elif self.config['use_dual_cutout']:
            self._add_dual_cutout()
        self._add_to_tensor()
        return torchvision.transforms.Compose(self._train_transforms)

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            transforms.Normalize(self.mean, self.std),
            transforms.ToTensor(),
        ])
        return transform


class CIFAR(Dataset):
    def __init__(self, config):
        self.size = 32
        if config['dataset'] == 'CIFAR10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])
        elif config['dataset'] == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])
        elif config['dataset'] == 'CIFAR102':
            self.mean = np.array([0.50332078, 0.48892964, 0.45191514])
            self.std = np.array([0.25841826, 0.25530315, 0.26965219])
        elif config['dataset'] == 'CIFAR10_10K':
            self.mean = np.array([0.49341091, 0.48464332, 0.44945974])
            self.std = np.array([0.24795955, 0.2443223,  0.26241769])
        elif config['dataset'] == 'CIFAR10_30K':
            self.mean = np.array([0.49197699, 0.48254069, 0.4467944])
            self.std = np.array([0.2469906, 0.24341879, 0.26154817])
        elif config['dataset'] == 'CIFAR102_30K':
            self.mean = np.array([0.50125684, 0.48950628, 0.45377465])
            self.std = np.array([0.25489271, 0.2515715,  0.26774696])
        super(CIFAR, self).__init__(config)


class MNIST(Dataset):
    def __init__(self, config):
        self.size = 28
        if config['dataset'] == 'MNIST':
            self.mean = np.array([0.1307])
            self.std = np.array([0.3081])
        elif config['dataset'] == 'FashionMNIST':
            self.mean = np.array([0.2860])
            self.std = np.array([0.3530])
        elif config['dataset'] == 'KMNIST':
            self.mean = np.array([0.1904])
            self.std = np.array([0.3475])
        super(MNIST, self).__init__(config)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_loader(config):
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']

    dataset_name = config['dataset']
    assert dataset_name in [
        'CIFAR10', 'CIFAR101', 'CIFAR102', 'CIFAR10_10K', 'CIFAR10_30K', 'CIFAR102_30K', 'KMNIST'
    ]

    if dataset_name in ['CIFAR10', 'CIFAR101', 'CIFAR102','CIFAR10_10K', 'CIFAR10_30K', 'CIFAR102_30K',]:
        dataset = CIFAR(config)
    elif dataset_name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset = MNIST(config)

    train_dataset, test_datasets = dataset.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loaders = {}
    for testset_name, test_dataset in test_datasets.items():
        test_loaders[testset_name] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=use_gpu,
            drop_last=False,)
    return train_loader, test_loaders
