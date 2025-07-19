import os
import pickle
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torch import tensor, long

def load_pkl_data(file_path):
    """加载存储在 pickle 文件中的数据和标签"""
    with open(file_path, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels

from PIL import Image
import numpy as np

class TinyImageNetDataset(Dataset):
    """自定义 Dataset 类以便与 DataLoader 一起使用"""
    def __init__(self, data, labels, class_names, transform=None):
        self.data = data
        self.labels = labels
        self.class_names = class_names  # 类别名称列表
        self.transform = transform

        # 创建类别名称到索引的映射
        self.label_map = {name: idx for idx, name in enumerate(class_names)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # 确保标签是类别名称字符串
        if isinstance(label, torch.Tensor):
            label = label.item()  # 将 Tensor 转为标量

        # 将整数标签转换为类别名称
        label_name = self.class_names[label]

        # 使用 label_map 将类别名称映射到类别索引
        label = self.label_map[label_name]

        # 如果 img 是 torch.Tensor 类型，转换为 PIL.Image 类型
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img, label


def TinyImageNet(data_path, downsize=True, balance=True):
    # 加载数据
    data_path = "/data/dengqiyan/EVA/data/TinyImageNet"
    train_data, train_labels = load_pkl_data(os.path.join(data_path, 'train.pkl'))
    val_data, val_labels = load_pkl_data(os.path.join(data_path, 'valid.pkl'))
    test_data, test_labels = load_pkl_data(os.path.join(data_path, 'test.pkl'))

    channel = 3
    im_size = (32, 32) if downsize else (64, 64)
    num_classes = 200
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    # 创建类别名称列表
    class_names = list(set(train_labels))  # 假设每个类名唯一，构建一个从标签字符串到类别索引的映射

    # 定义转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    if downsize:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # 使用自定义 Dataset 类
    train_dataset = TinyImageNetDataset(train_data, train_labels, class_names, transform=transform)
    val_dataset = TinyImageNetDataset(val_data, val_labels, class_names, transform=transform)
    test_dataset = TinyImageNetDataset(test_data, test_labels, class_names, transform=transform)

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset