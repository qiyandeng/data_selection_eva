from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch import tensor, long
import numpy as np
from torch.utils.data import random_split

def CIFAR100(data_path, balance=True):
    channel = 3
    im_size = (32, 32)
    num_classes = 100

    #optuna2
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    # 增强的数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    # optuna3
    # mean = [0.5071, 0.4866, 0.4409]
    # std = [0.2673, 0.2564, 0.2762]


    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(size=(32, 32), padding=4),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    # ])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载完整数据集
    full_dataset = datasets.CIFAR100(data_path, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100(data_path, train=False, download=False, transform=transform_val_test)

    # 按类别分层分割训练集和验证集
    train_idx = []
    val_idx = []
    
    for c in range(num_classes):
        class_idx = np.where(np.array(full_dataset.targets) == c)[0]
        np.random.shuffle(class_idx)
        split = int(0.8 * len(class_idx))
        train_idx.extend(class_idx[:split])
        val_idx.extend(class_idx[split:])

    # 创建Subset数据集
    dst_train = Subset(full_dataset, train_idx)
    dst_val = Subset(full_dataset, val_idx)
    dst_test = test_dataset  # 使用完整测试集

    # 保持验证集预处理一致（不包含数据增强）
    dst_val.dataset.transform = transform_val_test

    # 设置targets和labels属性
    def set_attributes(subset, fullset):
        subset.targets = tensor([fullset.targets[i] for i in subset.indices], dtype=long)
        subset.labels = [fullset.targets[i] for i in subset.indices]
        subset.classes = fullset.classes

    set_attributes(dst_train, full_dataset)
    set_attributes(dst_val, full_dataset)

    return channel, im_size, num_classes, full_dataset.classes, mean, std, dst_train, dst_val, dst_test