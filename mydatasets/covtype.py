import numpy as np
import joblib
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os

def load_covertype_dataset(data_dir):
    """
    加载Covertype数据集的特征和标签。

    Args:
        data_dir (str): 数据集根目录路径。

    Returns:
        tuple: (特征数据, 标签)
    """
    samples_path = os.path.join(data_dir, "covertype", "samples_py3")
    targets_path = os.path.join(data_dir, "covertype", "targets_py3")

    # Load the data from the files
    X = joblib.load(samples_path)
    y = joblib.load(targets_path)
    y = y - 1  # Adjust labels to start from 0

    return X, y

def classwise_split(dataset_targets, ratio, num_classes):
    """
    按类别比例拆分数据集为训练集和验证集。

    Args:
        dataset_targets (array-like): 标签数组。
        ratio (float): 训练集比例。
        num_classes (int): 类别数量。

    Returns:
        tuple: (训练集索引, 验证集索引)
    """
    train_val_label = np.array(dataset_targets)
    train_indexes = []
    val_indexes = []

    for class_id in range(num_classes):
        indexes = np.where(train_val_label == class_id)[0]
        np.random.shuffle(indexes)
        train_num = int(len(indexes) * ratio)
        train_indexes.extend(indexes[:train_num])
        val_indexes.extend(indexes[train_num:])

    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes

class ConvertypeDataset(Dataset):
    def __init__(self, data, labels, indexes):
        """
        初始化 ConvertypeDataset。

        Args:
            data (array-like): 特征数据。
            labels (array-like): 标签数据。
            indexes (array-like): 要包含的数据索引。
        """
        self.num_classes = 7  # 定义类别数量

        # 定义数据和标签，使用提供的索引
        self.data = np.array(data)[indexes]
        self.labels = np.array(labels)[indexes]

    def __getitem__(self, index):
        """
        获取指定索引的数据和标签。

        Args:
            index (int): 数据索引。

        Returns:
            tuple: (数据, 标签)
        """
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self) -> int:
        """
        返回数据集的大小。

        Returns:
            int: 数据集大小。
        """
        return len(self.labels)

def covtype(data_path, balance=True):
    """
    加载Covertype数据集，按balance参数划分训练集和验证集，计算标准化参数，并返回相关参数和数据集对象。

    Args:
        data_path (str): 数据集根目录路径。
        balance (bool): 是否在验证集中均衡类别分布。

    Returns:
        tuple: 包含channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset的元组。
    """
    # Load and clean the data
    data, labels = load_covertype_dataset(data_path)

    # 提取所有训练集的标签
    train_labels = np.array(labels)

    if balance:
        # 使用分层拆分，确保训练集和验证集的类别分布与原始数据一致
        train_indices, val_indices = classwise_split(
            dataset_targets=train_labels,
            ratio=0.8,
            num_classes=7
        )
        print(f"Balance=True: Split into {len(train_indices)} training and {len(val_indices)} validation samples with balanced classes.")
    else:
        # 当 balance=False 时，验证集仅包含类别为0的数据
        # 首先，获取类别为0的样本索引
        class_0_indices = np.where(train_labels == 0)[0]
        # 获取所有非类别0的样本索引
        other_class_indices = np.where(train_labels != 0)[0]

        # 将类别为0的样本按80/20划分
        class_0_train_indices, class_0_val_indices = train_test_split(
            class_0_indices,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        # 训练集包含所有非类别0的样本和80%的类别0样本
        train_indices = np.concatenate([other_class_indices, class_0_train_indices])
        # 验证集仅包含20%的类别0样本
        val_indices = class_0_val_indices

        print(f"Balance=False: Split into {len(train_indices)} training and {len(val_indices)} validation samples with validation set containing only class 0.")

    # 创建训练和验证数据集
    dst_train = ConvertypeDataset(data, labels, train_indices)
    dst_val = ConvertypeDataset(data, labels, val_indices)

    # Test data can be the same as the validation data in this setup
    dst_test = dst_val

    # 定义类别名称
    class_names = [f"Class {i}" for i in range(7)]

    # 计算均值和标准差，仅基于训练集
    mean = np.mean(data[train_indices], axis=0)
    std = np.std(data[train_indices], axis=0)
    # print(f"Computed mean: {mean}")
    # print(f"Computed std: {std}")

    # 标准化训练和验证数据
    dst_train.data = (dst_train.data - mean) / (std + 1e-8)
    dst_val.data = (dst_val.data - mean) / (std + 1e-8)
    dst_test.data = (dst_test.data - mean) / (std + 1e-8)

    # 定义其他参数
    channel = 1  # For tabular data, typically set to 1
    im_size = data.shape[1]  # 确保 im_size 是整数，表示特征数量
    num_classes = 7

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test