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

    if balance:
        # Split the full training set into training and validation sets (80/20 split)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        dst_train, dst_val = random_split(full_train_dataset, [train_size, val_size])
        dst_test = full_test_dataset
    else:
        # 训练集均匀划分
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        dst_train, dst_val_base = random_split(full_train_dataset, [train_size, val_size])

        # 获取原始数据集的 targets（从 Subset 的 indices 映射到原始数据集中）
        val_targets = np.array(full_train_dataset.targets)

        # 获取验证集对应的索引
        val_indices_base = dst_val_base.indices

        # 根据索引提取验证集的 targets
        val_targets_subset = val_targets[val_indices_base]

        # 获取不同类别的样本索引
        class_0_indices = [i for i, target in zip(val_indices_base, val_targets_subset) if target == 0]
        class_1_indices = [i for i, target in zip(val_indices_base, val_targets_subset) if target == 1]
        # other_class_indices = [i for i, target in zip(val_indices_base, val_targets_subset) if target != 0 and target != 1]

        # 合并验证集的索引
        val_indices = list(class_0_indices) + list(class_1_indices)

        # 创建验证集
        dst_val = Subset(full_train_dataset, val_indices)

        # 测试集的 targets
        test_targets = np.array(full_test_dataset.targets)

        # 获取测试集的不同类别索引
        class_0_test_indices = [i for i, target in enumerate(test_targets) if target == 0]
        class_1_test_indices = [i for i, target in enumerate(test_targets) if target == 1]

        # 合并测试集的索引
        test_indices = list(class_0_test_indices) + list(class_1_test_indices)

        # 创建测试集
        dst_test = Subset(full_test_dataset, test_indices)

    # Set validation dataset transform to test transform (no data augmentation)
    dst_val.dataset.transform = transform_test

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
    set_targets_and_labels(dst_val, full_train_dataset)
    set_targets_and_labels(dst_test, full_test_dataset)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test