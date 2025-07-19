import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import tensor, float32
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HouseDataset(Dataset):
    def __init__(self, features, labels, feature_scaler=None, label_scaler=None):
        # 特征标准化
        if feature_scaler is not None:
            self.features = tensor(feature_scaler.transform(features), dtype=float32)
        else:
            self.features = tensor(features, dtype=float32)
        
        # 标签标准化
        if label_scaler is not None:
            # 保存原始标签用于后续逆变换
            self.raw_labels = labels.copy()
            self.labels = tensor(label_scaler.transform(labels.reshape(-1, 1)), dtype=float32).squeeze()
        else:
            self.labels = tensor(labels, dtype=float32)
            self.raw_labels = labels.copy()
            
        # 保存scaler引用
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def House(data_path, balance=True):
    data_path = "/data/dengqiyan/EVA/data/House"
    
    # 加载训练数据
    train_data = pd.read_csv(f"{data_path}/train.csv")
    test_data = pd.read_csv(f"{data_path}/test.csv")

    # 拆分特征和标签
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values.reshape(-1, 1)  # 保持二维结构
    
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

    # 创建标准化器
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # 训练集标准化
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = label_scaler.fit_transform(y_train)

    # 验证集拆分和标准化（使用训练集的scaler）
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )

    # 测试集标准化（使用训练集的scaler）
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = label_scaler.transform(y_test)

    # 创建数据集（不再需要后期处理）
    train_dataset = HouseDataset(
        X_train_split, y_train_split, 
        feature_scaler=None,  # 已经提前处理
        label_scaler=None
    )
    
    valid_dataset = HouseDataset(
        X_val_split, y_val_split,
        feature_scaler=None,
        label_scaler=None
    )

    test_dataset = HouseDataset(
        X_test_scaled, y_test_scaled,
        feature_scaler=None,
        label_scaler=None
    )

    # 保存scaler用于后续逆变换
    train_dataset.feature_scaler = feature_scaler
    train_dataset.label_scaler = label_scaler
    valid_dataset.label_scaler = label_scaler
    test_dataset.label_scaler = label_scaler

    # 数据维度信息
    channel = 1
    im_size = (X_train.shape[1],)
    num_classes = 1
    class_names = ["price"]

    return channel, im_size, num_classes, class_names, feature_scaler.mean_, feature_scaler.scale_, train_dataset, valid_dataset, test_dataset