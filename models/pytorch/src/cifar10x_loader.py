from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle

class CIFAR10X(data.Dataset):
    base_folder = 'cifar10x'
    train_list = {'CIFAR102':'cifar102_min_overlap_train_v4',
                  'CIFAR10_10K':'cifar10_10k_train',
                  'CIFAR10_30K':'cifar10_30k_train',
                  'CIFAR102_30K':'cifar102_30k_train',
                 }
    test_list = {'CIFAR102':'cifar102_min_overlap_test_v4',
                 'CIFAR10':'cifar10_test_batch',
                 'CIFAR101':'cifar101_v6_test_batch',
                 'CIFAR10_10K':'cifar10_10k_test',
                 'CIFAR10_30K':'cifar10_30k_test',
                 'CIFAR102_30K':'cifar102_30k_test',
                }

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, trainset='CIFAR10_10K',testset='CIFAR102'):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = [self.train_list[trainset]]
        else:
            downloaded_list = [self.test_list[testset]]

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                try:
                    self.data.append(entry[b'data'])
                except:
                    self.data.append(entry['data'])
                try:
                    self.targets.extend(entry[b'labels'])
                except:
                    self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
