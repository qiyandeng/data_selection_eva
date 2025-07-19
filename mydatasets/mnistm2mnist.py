from torchvision import datasets, transforms
from torch import tensor, long
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from PIL import Image
import os
import io


class MNISTM(Dataset):
    def __init__(self, parquet_path, transform=None):
        self.data = pd.read_parquet(parquet_path)
        self.images = self.data['image']
        self.labels = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_dict = self.images[idx]
        image_bytes = image_dict["bytes"]
        image = Image.open(io.BytesIO(image_bytes))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def MnistM2Mnist(data_path, permuted=False, permutation_seed=None):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mnist_mean = [0.1307, 0.1307, 0.1307]
    mnist_std = [0.3081, 0.3081, 0.3081]
    mnistm_mean = [0.5, 0.5, 0.5]
    mnistm_std = [0.5, 0.5, 0.5]

    mnist_transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    mnistm_transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mnistm_mean, std=mnistm_std)
    ])

    # 加载 MNIST-M 作为训练集
    mnistm_train_path = "/data/dengqiyan/EVA/data/mnistm/mnistm_train.parquet"
    full_train_dataset = MNISTM(parquet_path=mnistm_train_path, transform=mnistm_transform)

    # 加载 MNIST 作为测试集
    full_test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=mnist_transform)

    # 拆分 MNIST-M 为训练集和验证集（80/20）
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    dst_train = train_dataset
    dst_val = val_dataset
    dst_test = full_test_dataset

    class_names = [str(c) for c in range(num_classes)]

    def set_labels_to_targets(dataset, full_dataset):
        if hasattr(full_dataset, 'targets'):
            full_labels = full_dataset.targets
        else:
            full_labels = full_dataset.labels

        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset.labels = [full_labels[i] for i in indices]
            dataset.targets = tensor([full_labels[i] for i in indices], dtype=long)
        else:
            dataset.labels = full_labels
            dataset.targets = tensor(full_labels, dtype=long)

    # 设置标签
    set_labels_to_targets(dst_train, full_train_dataset)
    set_labels_to_targets(dst_val, full_train_dataset)
    set_labels_to_targets(dst_test, full_test_dataset)

    return channel, im_size, num_classes, class_names, mnistm_mean, mnistm_std, dst_train, dst_val, dst_test
