from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config

from PIL import Image   # Added by W210 Team
import numpy as np      # Added by W210 Team
import io               # Added by W210 Team
import boto3            # Added by W210 Team
import torchvision.transforms as transforms  # Added by W210 Team
import random           # Added by W210 Team

from torch.utils.data import Dataset

from pytorch_image_classification import create_transform


# One variable to control the bucket setup for all W210 Data Sources
# Must be set based on the Amazon Instance you are running on 
#w210_bucket = 'sagemaker-june29' # Added by W210 Team
w210_bucket = 'sagemaker-may29' # Added by W210 Team

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
# Class that randomly selects 'class_size' elements from CIFAR10 testset to use 
class CIFAR10_2k_Dataset(Dataset):
    def __init__(self, class_size, transform=None):
       
        module = getattr(torchvision.datasets, 'CIFAR10')
        testset = module('~/.torch/datasets/CIFAR10',
                         train=False,
                         transform=None,
                         download=True)
        
        class_lists = {0: [], 1: [], 2: [], 3: [], 4: [], 
                       5: [], 6: [], 7: [], 8: [], 9: []
                      }
        
        for l,i in zip(testset.targets, testset.data):
            class_lists[l].append(i)
            
        new_testset_data = []
        new_testset_labels = []
        
        # For each class, randomly select class_size from each 
        for i in range(10):
            new_testset_data += random.sample(class_lists[i], class_size)
            new_testset_labels += ([i] * class_size)
            
        self.y = np.array(new_testset_labels).astype('long')
        self.X = np.array(new_testset_data)
            
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
# Class to define custom torchvision dataset for CIFAR 10 CutMix dataset
class CutMix_Dataset(Dataset):
    def __init__(self, file, s3bucket, s3path, transform=None):
        self.samples = list(range(1, 1001))
        label_filename = s3path + file + '_labels.npy'
        image_filename = s3path + file + '_data.pt'
        
        s3 = boto3.resource('s3')
        
        obj = s3.Object(s3bucket, label_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            y = np.load(f)
            #t1 = np.array([np.long(x[0]) for x in y])
            #t2 = np.array([np.long(x[1]) for x in y])
            #l = np.array([x[2] for x in y])
            #self.y = [t1, t2, l]
            self.y = y 
            
        obj = s3.Object(s3bucket, image_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            x = torch.load(f)
            x = [torch.squeeze(item) for item in x]
            x = [transforms.ToPILImage()(item) for item in x]
            self.X = [np.array(item) for item in x]
            
        self.transforms = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        img = Image.fromarray(data)
        
        if self.transforms:
            img = self.transforms(img) 
            
        if self.y is not None:
            #y = np.array([self.y[0][i], self.y[1][i], self.y[2][i]])
            #return (img, y)
            return(img, self.y[i])
            #return self.transform(img), self.transform(self.y[index])
        
        else:
            return img
    

# Added by W210 Team
# Class to define custom torchvision dataset for CIFAR 10.1 test set
class CIFAR101_Dataset(Dataset):
    def __init__(self, file, s3bucket, s3path, transform=None):
        self.samples = list(range(1, 1001))
        label_filename = s3path + file + '_labels.npy'
        image_filename = s3path + file + '_data.npy'
           
        s3 = boto3.resource('s3')
        
        obj = s3.Object(s3bucket, label_filename)
        with io.BytesIO(obj.get()["Body"].read()) as f:
            f.seek(0)
            labels = np.load(f)
            self.y = labels.astype('long')
            
        obj = s3.Object(s3bucket, image_filename)
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
# Class to define custom torchvision dataset for CIFAR Random Augment data sets
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
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        img = Image.fromarray(data)
        
        if self.transform:
            img = self.transform(img) 
            
        if self.y is not None:
            return (img, self.y[i])
        
        else:
            return img

# Added by W210 Team
class CIFAR10_RA_TestDataset(Dataset):
    def __init__(self, file, s3bucket, s3path, transform=None):
        self.samples = list(range(1, 50001))
        label_filename = s3path + 'cifar10_test_labels.npy'
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
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        img = Image.fromarray(data)
        
        if self.transform:
            img = self.transform(img) 
            
        if self.y is not None:
            return (img, self.y[i])
        
        else:
            return img  
        
        
# Added by W210 Team
# Class to define custom torchvision dataset for CIFAR 10.1 test set
class CIFAR101_RA_Dataset(Dataset):
    def __init__(self, file, s3bucket, s3path, transform=None):
        label_filename = s3path + 'cifar10.1_v6_labels.npy'
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
            
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        img = Image.fromarray(data)
        
        if self.transform:
            img = self.transform(img) 
            
        if self.y is not None:
            return (img, self.y[i])
        
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
        cifar101_transform = create_transform(config, is_train=False)
        dataset = CIFAR101_Dataset('cifar10.1_v6', 
                                   w210_bucket,
                                   'sagemaker/cifar101/',
                                   cifar101_transform)
        return dataset
    
        # ELIF added by W210 Team
    elif config.dataset.name == 'CIFAR101_RA_2_5':
        print("CIFAR101_RA_2_5")
        cifar101_transform = create_transform(config, is_train=False)
        dataset = CIFAR101_RA_Dataset('c101_test_ra_2_5.npy',
                                      w210_bucket,
                                      'sagemaker/RandAugmentation/',
                                      cifar101_transform)
        return dataset
    
        # ELIF added by W210 Team
    elif config.dataset.name == 'CIFAR101_RA_2_20':
        print("CIFAR101_RA_2_20")
        cifar101_transform = create_transform(config, is_train=False)
        dataset = CIFAR101_RA_Dataset('c101_test_ra_2_20.npy',
                                      w210_bucket,
                                      'sagemaker/RandAugmentation/',
                                      cifar101_transform)
        return dataset
    
        # ELIF added by W210 Team
    elif config.dataset.name == 'CIFAR101_RA_3_20':
        print("CIFAR101_RA_3_20")
        cifar101_transform = create_transform(config, is_train=False)
        dataset = CIFAR101_RA_Dataset('c101_test_ra_3_20.npy',
                                      w210_bucket,
                                      'sagemaker/RandAugmentation/',
                                      cifar101_transform)
        return dataset

    elif config.dataset.name == 'CIFAR101_RA_1_20':
        print("CIFAR101_RA_1_20")
        cifar101_transform = create_transform(config, is_train=False)
        dataset = CIFAR101_RA_Dataset('c101_test_ra_1_20.npy',
                                      w210_bucket,
                                      'sagemaker/RandAugmentation/',
                                      cifar101_transform)
        return dataset
    
    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_CM_1":
        if is_train:
            dataset = CutMix_Dataset('cifar10_cm_beta1_prob1',
                                     w210_bucket,
                                     'sagemaker/CutMix/',
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
            module = getattr(torchvision.datasets, 'CIFAR10')
            c10_dataset = module('~/.torch/datasets/CIFAR10',
                                 train=is_train,
                                 transform=None,
                                 download=True)
            c10_train_subset, val_subset = torch.utils.data.dataset.random_split(
                    c10_dataset, lengths)
            val_dataset = SubsetDataset(val_subset, val_transform)
            
            return train_dataset, val_dataset

    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_CM_.5":
        if is_train:
            dataset = CutMix_Dataset('cifar10_cm_beta1_prob.5',
                                     w210_bucket,
                                     'sagemaker/CutMix/',
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
            module = getattr(torchvision.datasets, 'CIFAR10')
            c10_dataset = module('~/.torch/datasets/CIFAR10',
                                 train=is_train,
                                 transform=None,
                                 download=True)
            c10_train_subset, val_subset = torch.utils.data.dataset.random_split(
                    c10_dataset, lengths)
            val_dataset = SubsetDataset(val_subset, val_transform)
            
            return train_dataset, val_dataset        
        
    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_CM_.25":
        if is_train:
            dataset = CutMix_Dataset('cifar10_cm_beta1_prob.25',
                                     w210_bucket,
                                     'sagemaker/CutMix/',
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
            module = getattr(torchvision.datasets, 'CIFAR10')
            c10_dataset = module('~/.torch/datasets/CIFAR10',
                                 train=is_train,
                                 transform=None,
                                 download=True)
            c10_train_subset, val_subset = torch.utils.data.dataset.random_split(
                    c10_dataset, lengths)
            val_dataset = SubsetDataset(val_subset, val_transform)
            
            return train_dataset, val_dataset  
        
        
    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_RA_2_5":
        if is_train:
            ra_transform = create_transform(config, is_train=True)
            dataset = CIFAR10_RA_Dataset('cifar10_ra_2_5.npy',
                                         w210_bucket,
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
            dataset = CIFAR10_RA_TestDataset('c10_test_ra_2_5.npy',
                                             w210_bucket,
                                             'sagemaker/RandAugmentation/',
                                             ra_transform)
            return dataset
        
    # ELIF added by W210 Team
    elif config.dataset.name == "CIFAR10_RA_3_20":
        if is_train:
            ra_transform = create_transform(config, is_train=True)
            dataset = CIFAR10_RA_Dataset('cifar10_ra_3_20.npy',
                                         w210_bucket,
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
            dataset = CIFAR10_RA_TestDataset('c10_test_ra_3_20.npy',
                                             w210_bucket,
                                             'sagemaker/RandAugmentation/',
                                             ra_transform)
            return dataset

    
    # ELIF added by W210 Team to add randAugment with N=2, M=20
    elif config.dataset.name == "CIFAR10_RA_2_20":
        if is_train:
            ra_transform = create_transform(config, is_train=True)
            dataset = CIFAR10_RA_Dataset('cifar10_ra_2_20.npy',
                                         w210_bucket,
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
            print("CIFAR 10 Random Augmentation N=2 M=20")
            ra_transform = create_transform(config, is_train=False)
            dataset = CIFAR10_RA_TestDataset('c10_test_ra_2_20.npy',
                                             w210_bucket,
                                             'sagemaker/RandAugmentation/',
                                             ra_transform)
            return dataset

    # ELIF added by W210 Team to add randAugment with N=1, M=20
    elif config.dataset.name == "CIFAR10_RA_1_20":
        if is_train:
            dataset = CIFAR10_RA_Dataset('cifar10_ra_1_20.npy',
                                         w210_bucket,
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
            print("CIFAR 10 Random Augmentation N=1 M=20")
            ra_transform = create_transform(config, is_train=False)
            dataset = CIFAR10_RA_TestDataset('c10_test_ra_1_20.npy',
                                             w210_bucket,
                                             'sagemaker/RandAugmentation/',
                                             ra_transform)
            return dataset
        
    elif config.dataset.name == "CIFAR10_2k":
        print("CIFAR 10, 2k subset")
        cifar102k_transform = create_transform(config, is_train=False)
        dataset = CIFAR10_2k_Dataset(200, cifar102k_transform)
        return dataset    
    
    else:
        raise ValueError()
