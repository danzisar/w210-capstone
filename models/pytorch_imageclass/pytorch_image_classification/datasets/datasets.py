from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config

from PIL import Image   # Added by W210 Team
import numpy as np      # Added by W210 Team
import io               # Added by W210 Team
import boto3            # Added by W210 Team

from torch.utils.data import Dataset

from pytorch_image_classification import create_transform


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)


# Added by W210 Team
# Class to define custom torchvision dataset for CIFAR 10.1 test set
class CIFAR101_Dataset(Dataset):
    def __init__(self, file, s3bucket, s3path, transform=None):
        self.samples = list(range(1, 1001))
        label_filename = s3path + file + '_labels.npy'
        image_filename = s3path + file + '_data.npy'
           
        bucket='sagemaker-may29'
        s3_cifar101 = "sagemaker/cifar101/"
        filename = 'cifar10.1_v6'
        
        s3 = boto3.resource('s3')
        
        obj = s3.Object(bucket, label_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            labels = np.load(f)
            self.y = labels.astype('long')
            
        obj = s3.Object(bucket, image_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            self.X = np.load(f)
            
        #transforms.Compose([transforms.ToTensor()])
        self.transforms = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        img = Image.fromarray(data)
        
        if self.transforms:
            img = self.transforms(img) 
            
        if self.y is not None:
            return (img, self.y[i])
            #return self.transform(img), self.transform(self.y[index])
        
        else:
            return img
        
# Added by W210 Team
# Class to define custom torchvision dataset for CIFAR 10.1 test set
class CIFAR10_RA_Dataset(Dataset):
    def __init__(self, file, s3bucket, s3path, transform=None):
        self.samples = list(range(1, 50001))
        label_filename = s3path + 'cifar10_labels.npy'
        image_filename = s3path + file 
        
        s3 = boto3.resource('s3')
        
        obj = s3.Object(s3bucket, label_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            labels = np.load(f)
            self.y = labels.astype('long').flatten()
            
        obj = s3.Object(s3bucket, image_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            self.X = np.load(f)
            
        #transforms.Compose([transforms.ToTensor()])
        print(self.X.shape)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        img = Image.fromarray(data)
        
        if self.transform:
            print("self.transforms:", self.transform)
            img = self.transform(img) 
            
        if self.y is not None:
            return (img, self.y[i])
            #return self.transform(img), self.transform(self.y[index])
        
        else:
            return img
        
        

def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST',
            'KMNIST',
    ]:
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    elif config.dataset.name == 'ImageNet':
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = create_transform(config, is_train=True)
        val_transform = create_transform(config, is_train=False)
        train_dataset = torchvision.datasets.ImageFolder(
            dataset_dir / 'train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                       transform=val_transform)
        return train_dataset, val_dataset
    # ELIF added by W210 Team
    elif config.dataset.name == 'CIFAR101':
        print("CIFAR 10.1")
        cifar101_transform = create_transform(config, is_train=True)
        dataset = CIFAR101_Dataset('cifar10.1_v6', 
                                   'sagemaker-may29',
                                   'sagemaker/cifar101/',
                                   cifar101_transform)
        return dataset
    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_RA_2_5":
        if is_train:
            ra_transform = create_transform(config, is_train=True)
            dataset = CIFAR10_RA_Dataset('cifar10_ra_2_5.npy',
                                         'sagemaker-may29',
                                         'sagemaker/RandAugmentation/',
                                         transform=None)
            val_ratio = config.train.val_ratio
            assert val_ratio < 1
            val_num = int(len(dataset) * val_ratio)
            train_num = len(dataset) - val_num
            lengths = [train_num, val_num]
            train_subset, val_subset = torch.utils.data.dataset.random_split(dataset, lengths)
            train_transform = create_transform(config, is_train=True)
            val_transform = create_transform(config, is_train=False)
            train_dataset = SubsetDataset(train_subset, train_transform)
            val_dataset = SubsetDataset(val_subset, val_transform)
            return train_dataset, val_dataset
        
    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_RA_3_20":
        if is_train:
            ra_transform = create_transform(config, is_train=True)
            dataset = CIFAR10_RA_Dataset('cifar10_ra_3_20.npy',
                                         'sagemaker-may29',
                                         'sagemaker/RandAugmentation/',
                                         transform=None)
            val_ratio = config.train.val_ratio
            assert val_ratio < 1
            val_num = int(len(dataset) * val_ratio)
            train_num = len(dataset) - val_num
            lengths = [train_num, val_num]
            train_subset, val_subset = torch.utils.data.dataset.random_split(dataset, lengths)
            train_transform = create_transform(config, is_train=True)
            val_transform = create_transform(config, is_train=False)
            train_dataset = SubsetDataset(train_subset, train_transform)
            val_dataset = SubsetDataset(val_subset, val_transform)
            return train_dataset, val_dataset
        
        else:
            print("CIFAR 10 Random Augmentation N=3 M=20")
            ra_transform = create_transform(config, is_train=False)
            dataset = CIFAR10_RA_Dataset('cifar10_ra_3_20.npy',
                                         'sagemaker-may29',
                                         'sagemaker/RandAugmentation/',
                                         ra_transform)
            return dataset
        
    else:
        raise ValueError()
