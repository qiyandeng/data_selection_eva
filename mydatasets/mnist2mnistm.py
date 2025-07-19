# from torchvision import datasets, transforms
# from torch import tensor, long
# from torch.utils.data import random_split
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# import pandas as pd
# from PIL import Image
# import os
# import io


# class MNISTM(Dataset):
#     def __init__(self, parquet_path, transform=None):
#         """
#         初始化 MNIST-M 数据集
#         :param parquet_path: MNIST-M 数据集路径 (.parquet 文件)
#         :param transform: 图像变换
#         """
#         self.data = pd.read_parquet(parquet_path)
#         self.images = self.data['image']
#         self.labels = self.data['label']
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # 从字典解析图像
#         image_dict = self.images[idx]
#         image_bytes = image_dict["bytes"]
#         image = Image.open(io.BytesIO(image_bytes))  # 从字节中加载图像
#         if self.transform:
#             image = self.transform(image)
#         label = self.labels[idx]
#         return image, label


# def Mnist2MnistM(data_path, permuted=False, permutation_seed=None):
#     channel = 3  # 扩展为 3 通道
#     im_size = (32, 32)  # 调整尺寸到 32x32
#     num_classes = 10
#     mnist_mean = [0.1307, 0.1307, 0.1307]  # MNIST 数据集均值
#     mnist_std = [0.3081, 0.3081, 0.3081]  # MNIST 数据集标准差
#     mnistm_mean = [0.5, 0.5, 0.5]  # MNIST-M 数据集假定均值
#     mnistm_std = [0.5, 0.5, 0.5]  # MNIST-M 数据集假定标准差

#     # 定义数据转换
#     mnist_transform = transforms.Compose([
#         transforms.Resize(im_size),  # 调整尺寸到 32x32
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道扩展为三通道
#         transforms.Normalize(mean=mnist_mean, std=mnist_std)
#     ])
#     mnistm_transform = transforms.Compose([
#         transforms.Resize(im_size),  # 调整尺寸到 32x32
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mnistm_mean, std=mnistm_std)
#     ])

#     # 加载 MNIST 训练集
#     full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=mnist_transform)

#     mnistm_test_path = "./data/mnistm/mnistm_test.parquet"
#     # 加载 MNIST-M 测试集
#     full_test_dataset = MNISTM(parquet_path=mnistm_test_path, transform=mnistm_transform)

#     # 将 MNIST 测试集拆分为验证集和测试集（50/50 拆分）
#     test_size = len(full_test_dataset)
#     val_size = test_size // 2
#     test_size = test_size - val_size
#     dst_val, dst_test = random_split(full_test_dataset, [val_size, test_size])
#     dst_train = full_train_dataset

#     # 定义类别名称
#     class_names = [str(c) for c in range(num_classes)]

#     # Function to set targets and labels
#     def set_targets_to_labels(dataset, full_dataset):
#         if isinstance(dataset, Subset):
#             indices = dataset.indices
#             dataset.targets = tensor([full_dataset.targets[i] for i in indices], dtype=long)
#             dataset.labels = [full_dataset.targets[i] for i in indices]
#         else:
#             dataset.targets = tensor(full_dataset.targets, dtype=long)
#             dataset.labels = full_dataset.targets
    
#     def set_labels_to_targets(dataset, full_dataset):
#         if isinstance(dataset, Subset):
#             indices = dataset.indices
#             dataset.labels = [full_dataset.labels[i] for i in indices]
#             dataset.targets = tensor([full_dataset.labels[i] for i in indices], dtype=long)
#         else:
#             dataset.labels = full_dataset.targets
#             dataset.targets = tensor(full_dataset.targets, dtype=long)

#     # Set targets and labels for training, validation, and test datasets
#     set_targets_to_labels(dst_train, full_train_dataset)
#     set_labels_to_targets(dst_val, full_test_dataset)
#     set_labels_to_targets(dst_test, full_test_dataset)

#     return channel, im_size, num_classes, class_names, mnist_mean, mnist_std, dst_train, dst_val, dst_test

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
        """
        初始化 MNIST-M 数据集
        :param parquet_path: MNIST-M 数据集路径 (.parquet 文件)
        :param transform: 图像变换
        """
        self.data = pd.read_parquet(parquet_path)
        self.images = self.data['image']
        self.labels = self.data['label']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 从字典解析图像
        image_dict = self.images[idx]
        image_bytes = image_dict["bytes"]
        image = Image.open(io.BytesIO(image_bytes))  # 从字节中加载图像
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def Mnist2MnistM(data_path, permuted=False, permutation_seed=None):
    channel = 3  # 扩展为 3 通道
    im_size = (32, 32)  # 调整尺寸到 32x32
    num_classes = 10
    mnist_mean = [0.1307, 0.1307, 0.1307]  # MNIST 数据集均值
    mnist_std = [0.3081, 0.3081, 0.3081]  # MNIST 数据集标准差
    mnistm_mean = [0.5, 0.5, 0.5]  # MNIST-M 数据集假定均值
    mnistm_std = [0.5, 0.5, 0.5]  # MNIST-M 数据集假定标准差

    # 定义数据转换
    mnist_transform = transforms.Compose([
        transforms.Resize(im_size),  # 调整尺寸到 32x32
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道扩展为三通道
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    mnistm_transform = transforms.Compose([
        transforms.Resize(im_size),  # 调整尺寸到 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=mnistm_mean, std=mnistm_std)
    ])

    # 加载 MNIST 训练集
    full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=mnist_transform)

    mnistm_test_path = "/data/dengqiyan/EVA/data/mnistm/mnistm_test.parquet"
    # 加载 MNIST-M 测试集
    full_test_dataset = MNISTM(parquet_path=mnistm_test_path, transform=mnistm_transform)

    # 将完整的训练集拆分为训练集和验证集（80/20 拆分）
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    dst_train = train_dataset
    dst_val = val_dataset
    dst_test = full_test_dataset

    # 定义类别名称
    class_names = [str(c) for c in range(num_classes)]

    # Function to set targets and labels
    def set_targets_to_labels(dataset, full_dataset):
        if hasattr(full_dataset, 'targets'):
            full_labels = full_dataset.targets
        else:
            full_labels = full_dataset.labels

        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset.targets = tensor([full_labels[i] for i in indices], dtype=long)
            dataset.labels = [full_labels[i] for i in indices]
        else:
            dataset.targets = tensor(full_labels, dtype=long)
            dataset.labels = full_labels

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

    # Set targets and labels for training, validation, and test datasets
    set_targets_to_labels(dst_train, full_train_dataset)
    set_targets_to_labels(dst_val, full_train_dataset)  # 使用训练集的标签
    set_labels_to_targets(dst_test, full_test_dataset)

    return channel, im_size, num_classes, class_names, mnist_mean, mnist_std, dst_train, dst_val, dst_test
