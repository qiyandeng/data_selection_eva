from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch import tensor, long
import numpy as np
from torch.utils.data import random_split

def CIFAR10(data_path, balance=True):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    # Data transformations with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),     # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load full training and test datasets
    full_train_dataset = datasets.CIFAR10(data_path, train=True, download=False, transform=transform_train)
    full_test_dataset = datasets.CIFAR10(data_path, train=False, download=False, transform=transform_test)

    # extract pool and train datasets
    # pool_size = int(0.99 * len(full_train_dataset))
    # train_size = len(full_train_dataset) - pool_size
    # dst_pool, dst_train = random_split(full_train_dataset, [pool_size, train_size])
    dst_train = full_train_dataset
    
    
    # Split test dataset into validation and test datasets
    val_idx = []
    test_idx = []
    
    for c in range(num_classes):
        idx = np.where(np.array(full_test_dataset.targets) == c)[0]
        len_c = len(idx)
        c_val_size = int(0.5 * len_c)
        c_test_size = len_c - c_val_size
        c_val_idx, c_test_idx = random_split(idx, [c_val_size, c_test_size])
        val_idx.extend(c_val_idx)
        test_idx.extend(c_test_idx)
        
    dst_val = Subset(full_test_dataset, val_idx)
    dst_test = Subset(full_test_dataset, test_idx)

    # Define class names
    class_names = full_train_dataset.classes

    # Function to set targets and labels
    def set_targets_and_labels(dataset, full_dataset):
        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset.targets = tensor([full_dataset.targets[i] for i in indices], dtype=long)
            dataset.labels = [full_dataset.targets[i] for i in indices]
        else:
            dataset.targets = tensor(full_dataset.targets, dtype=long)
            dataset.labels = full_dataset.targets

    # Set targets and labels for training, validation, and test datasets
    set_targets_and_labels(dst_train, full_train_dataset)
    # set_targets_and_labels(dst_pool, full_train_dataset)
    set_targets_and_labels(dst_val, full_test_dataset)
    set_targets_and_labels(dst_test, full_test_dataset)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test