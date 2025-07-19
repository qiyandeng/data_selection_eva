import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch import tensor, float32, long
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch


# 自定义 HR 数据集类
class HRDataset(Dataset):
    def __init__(self, file_path, preprocessor, fit_scaler=False):
        # 加载数据
        data = pd.read_csv(file_path)
        # 预处理数据
        features = preprocessor.transform(data.iloc[:, :-1])  # 前 13 列为属性
        
        # 如果是稀疏矩阵，转换为密集矩阵
        if hasattr(features, "toarray"):
            features = features.toarray()
        
        self.features = tensor(features, dtype=float32)
        self.labels = tensor(data.iloc[:, -1].values, dtype=long)  # 最后一列为标签
        self.targets = self.labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def HR(data_path, balance=True):
    data_path = "/data/dengqiyan/EVA/data/HR"
    # 数据文件路径
    train_path = f"{data_path}/train.csv"
    valid_path = f"{data_path}/valid.csv"
    test_path = f"{data_path}/test.csv"

    # 列类型：连续特征和分类特征
    continuous_columns = ['Age', 'HoursPerWeek', 'fnlgwt', 'CapitalLoss', 'EdNum', 'CapitalGain']
    categorical_columns = ['Education', 'Sex', 'Race', 'Relationship', 'MaritalStatus', 'Country', 'Occupation']

    # 创建一个 ColumnTransformer 用于处理不同类型的特征
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    # 创建预处理器并拟合训练数据
    train_data = pd.read_csv(train_path)
    preprocessor.fit(train_data.iloc[:, :-1])

    # 加载数据集
    train_dataset = HRDataset(train_path, preprocessor, fit_scaler=True)
    test_dataset = HRDataset(test_path, preprocessor, fit_scaler=False)

    # 计算验证集的大小为 20% 的训练集
    valid_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - valid_size

    # 使用 random_split 将训练集分割成训练集和验证集
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    # 获取数据维度
    channel = 1  # HR 数据是 tabular 数据，假设单通道
    im_size = (train_dataset.dataset.features.shape[1],)  # 输入的特征维度
    num_classes = len(np.unique(train_dataset.dataset.labels))  # 根据训练集的标签推断类别数
    class_names = [str(i) for i in range(num_classes)]  # 类别名称假设为字符串

    # 进行标准化
    scaler = StandardScaler()
    train_dataset.dataset.features = tensor(
        scaler.fit_transform(train_dataset.dataset.features), dtype=float32
    )
    valid_dataset.dataset.features = tensor(
        scaler.transform(valid_dataset.dataset.features), dtype=float32
    )
    test_dataset.features = tensor(
        scaler.transform(test_dataset.features), dtype=float32
    )
    mean = scaler.mean_
    std = scaler.scale_

    def set_subset_labels(subset):
        if isinstance(subset, torch.utils.data.Subset):
            subset.labels = [subset.dataset.labels[i] for i in subset.indices]
            subset.targets = tensor(subset.labels, dtype=long)
        else:
            subset.labels = subset.labels.tolist() if isinstance(subset.labels, torch.Tensor) else subset.labels
            subset.targets = tensor(subset.labels, dtype=long)

    set_subset_labels(train_dataset)
    set_subset_labels(valid_dataset)
    set_subset_labels(test_dataset)

    # 返回必要的信息和数据集
    return channel, im_size, num_classes, class_names, mean, std, train_dataset, valid_dataset, test_dataset

