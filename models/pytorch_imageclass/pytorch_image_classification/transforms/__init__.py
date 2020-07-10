from typing import Callable, Tuple

import numpy as np
import torchvision
import yacs.config

from .transforms import (
    CenterCrop,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizeCrop,
    Resize,
    ToTensor,
)

from .cutout import Cutout, DualCutout
from .random_erasing import RandomErasing


def _get_dataset_stats(
        config: yacs.config.CfgNode) -> Tuple[np.ndarray, np.ndarray]:
    name = config.dataset.name
    if name == 'CIFAR10':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif name == 'CIFAR101':   # Added by W210 Team
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])  # Should be based on training dataset
        std = np.array([0.2470, 0.2435, 0.2616])
    elif name == 'CIFAR10_CM_1':   # Added by W210 Team
        # RGB
        mean = np.array([0.48900609, 0.47970164, 0.47680734])  # Should be based on training dataset
        std = np.array([0.16860414, 0.17559971, 0.18220738])
        
    elif name == 'CIFAR10_RA_2_5':   # Added by W210 Team
        # RGB
        mean = np.array([0.48169277, 0.47972626, 0.46786197])  # Should be based on training dataset
        std = np.array([0.14300863, 0.14176982, 0.14451227])

    elif name == 'CIFAR10_RA_2_20':   # Added by W210 Team
        # RGB
        mean = np.array([0.59741131, 0.59679988, 0.58773509])  # Should be based on training dataset
        std = np.array([0.13812994, 0.13592779, 0.14246847])

    elif name == 'CIFAR10_RA_3_20':   # Added by W210 Team
        # RGB
        mean = np.array([0.60252478, 0.6022783, 0.59686454])  # Should be based on training dataset
        std = np.array([0.11540394, 0.11270143, 0.11942429])

    elif name == 'CIFAR101_RA_1_20':  # Added by W210 Team
        std = np.array([0.5720768, 0.56465674, 0.54534674])
        mean = np.array([0.16938294, 0.16641179, 0.17129586])
    elif name == 'CIFAR101_RA_2_20':  # Added by W210 Team
        std = np.array([0.60339547, 0.60351885, 0.59763299])
        mean = np.array([0.13662655, 0.13409996, 0.1415314])
    elif name == 'CIFAR101_RA_3_20':  # Added by W210 Team
        std = np.array([0.60806673, 0.6066123, 0.60138855])
        mean = np.array([0.11522382, 0.1132553,  0.1194451])
    elif name == 'CIFAR101_RA_2_5':  # Added by W210 Team
        std = np.array([0.48399701, 0.47999693, 0.46704372])
        mean = np.array([0.15088125, 0.14923236, 0.15124402])
        
    elif name == 'CIFAR10_CM_1':  # Added by W210 Team
        std = np.array([0.48900609, 0.47970164, 0.47680734])
        mean = np.array([0.16860414, 0.17559971, 0.18220738])
    elif name == 'CIFAR10_CM_.5':  # Added by W210 Team
        std = np.array([0.4890093, 0.47970309, 0.4768041])
        mean = np.array([0.16148799, 0.16867248, 0.17584336])
    elif name == 'CIFAR10_CM_.25':  # Added by W210 Team
        std = np.array([0.48900969, 0.47970441, 0.47680566])
        mean = np.array([0.15818763, 0.16574074, 0.17315236])
        
    elif name == 'CIFAR100':
        # RGB
        mean = np.array([0.5071, 0.4865, 0.4409])
        std = np.array([0.2673, 0.2564, 0.2762])
    elif name == 'MNIST':
        mean = np.array([0.1307])
        std = np.array([0.3081])
    elif name == 'FashionMNIST':
        mean = np.array([0.2860])
        std = np.array([0.3530])
    elif name == 'KMNIST':
        mean = np.array([0.1904])
        std = np.array([0.3475])
    elif name == 'ImageNet':
        # RGB
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError()
    return mean, std


def create_transform(config: yacs.config.CfgNode, is_train: bool) -> Callable:
    if config.model.type == 'cifar':
        return create_cifar_transform(config, is_train)
    elif config.model.type == 'imagenet':
        return create_imagenet_transform(config, is_train)
    else:
        raise ValueError


def create_cifar_transform(config: yacs.config.CfgNode,
                           is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.use_dual_cutout:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)


def create_imagenet_transform(config: yacs.config.CfgNode,
                              is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        else:
            transforms.append(CenterCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.use_dual_cutout:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = []
        if config.tta.use_resize:
            transforms.append(Resize(config))
        if config.tta.use_center_crop:
            transforms.append(CenterCrop(config))
        transforms += [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)
