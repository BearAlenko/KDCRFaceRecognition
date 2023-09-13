import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.io import read_image
import os
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset


cudnn.benchmark = True

class IdentityAgeDataset(Dataset):
    def __init__(self, img_dir, datafile, transform=None):
        self.img_labels_ages = pd.read_csv(datafile, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels_ages)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels_ages.iloc[idx, 0])
        #print(img_path)
        image = read_image(img_path)
        label = self.img_labels_ages.iloc[idx, 1]
        age = self.img_labels_ages.iloc[idx, 2]
        #age_out = int(age/10)
        #if age_out > 6:
        #    age_out = 6
        age_out = age
        if self.transform:
            image = self.transform(image)
        return {'image':image, 'label':label, 'age':age_out}

class pairsDataset(Dataset):
    def __init__(self, img_dir, datafile, transform=None):
        self.img_labels_ages = pd.read_csv(datafile, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels_ages)

    def __getitem__(self, idx):
        img_path0 = os.path.join(self.img_dir, self.img_labels_ages.iloc[idx, 0])
        img_path1 = os.path.join(self.img_dir, self.img_labels_ages.iloc[idx, 1])
        image0 = read_image(img_path0)
        image1 = read_image(img_path1)
        label = self.img_labels_ages.iloc[idx, 2]
        if self.transform:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
        return {'image0': image0, 'image1':image1, 'label':label}


def get_train_valid_loader(train_root,
                          test_root, train_dir, val_dir, test_dir, test_large_dir,
                           batch_size,
                           augment,
                           random_seed,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False, img_size=112):
    """

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    # define transforms
    valid_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size,img_size)),
                                      #transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize])
    if augment:
        train_transform = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((img_size,img_size)),
        #transforms.RandomRotation(degrees=(-20, 20)),
        #transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize])
    else:
        train_transform = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((img_size,img_size)),
                                      transforms.ToTensor(),
                                      normalize])

    # load the dataset
    train_dataset = IdentityAgeDataset(datafile=train_dir, img_dir = train_root, transform=train_transform)

    valid_dataset = IdentityAgeDataset(datafile=val_dir, img_dir = train_root, transform=valid_transform)
    test_dataset = pairsDataset(datafile=test_dir,img_dir = test_root, transform=valid_transform)
    testlarge_dataset = pairsDataset(datafile=test_large_dir,img_dir = test_root, transform=valid_transform)
    #th_dataset = pairsDataset(datafile=th_dir,img_dir = train_root, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=num_workers, pin_memory=pin_memory
    )
    test_large_loader = torch.utils.data.DataLoader(
        testlarge_dataset, num_workers=num_workers, pin_memory=pin_memory
    )
    #th_loader = torch.utils.data.DataLoader(
    #    th_dataset, num_workers=num_workers, pin_memory=pin_memory
    #)
    return (train_loader, valid_loader, test_loader, test_large_loader, len(train_dataset), len(valid_dataset), len(test_dataset), len(test_large_loader)
    )