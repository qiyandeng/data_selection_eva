import numpy as np
from torch.utils.data import Dataset
from torch import tensor, float32, long
from sklearn.preprocessing import StandardScaler

# 自定义 IMDB 数据集类
class IMDBLargeDataset(Dataset):
    def __init__(self, features_path, labels_path):
        # 加载数据
        self.features = tensor(np.load(features_path), dtype=float32)
        self.labels = tensor(np.load(labels_path).reshape(-1), dtype=long)

        # 修正标签范围：将 -1 替换为 5
        self.labels[self.labels == -1] = 5

        # self.targets 等于 self.labels
        self.targets = self.labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 数据加载和预处理函数
def IMDBLarge(data_path, balance=True):
    # 数据文件路径
    train_features_path = "/data/dengqiyan/EVA/data/IMDB/IMDBLargeC5-train-X.npy"
    train_labels_path = "/data/dengqiyan/EVA/data/IMDB/IMDBLargeC5-train-y.npy"
    val_features_path = "/data/dengqiyan/EVA/data/IMDB/IMDBLargeC5-val-X.npy"
    val_labels_path = "/data/dengqiyan/EVA/data/IMDB/IMDBLargeC5-val-y.npy"
    test_features_path = "/data/dengqiyan/EVA/data/IMDB/IMDBLargeC5-test-X.npy"
    test_labels_path = "/data/dengqiyan/EVA/data/IMDB/IMDBLargeC5-test-y.npy"

    # 加载数据集
    train_dataset = IMDBLargeDataset(train_features_path, train_labels_path)
    val_dataset = IMDBLargeDataset(val_features_path, val_labels_path)
    test_dataset = IMDBLargeDataset(test_features_path, test_labels_path)

    # 数据标准化
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_dataset.features)
    val_features = scaler.transform(val_dataset.features)
    test_features = scaler.transform(test_dataset.features)

    # 更新标准化后的特征
    train_dataset.features = tensor(train_features, dtype=float32)
    val_dataset.features = tensor(val_features, dtype=float32)
    test_dataset.features = tensor(test_features, dtype=float32)

    # 获取数据维度
    channel = 1  # 假设数据为单通道
    im_size = (train_dataset.features.shape[1],)  # 输入特征维度
    num_classes = len(np.unique(train_dataset.labels.numpy()))  # 根据训练集的标签推断类别数
    class_names = [str(i) for i in range(num_classes)]  # 类别名称假设为字符串

    # 返回必要的信息和数据集
    mean = scaler.mean_
    std = scaler.scale_
    return channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset