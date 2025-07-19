import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

def load_adult_dataset(data_dir):
    samples_path = data_dir + "/adult/train.csv"
    targets_path = data_dir + "/adult/test.csv"
    print(f"Loading data from:\n- Train: {samples_path}\n- Test: {targets_path}")  # 确认路径

    # 加载数据
    df_train = pd.read_csv(samples_path, header=None, skiprows=0, 
                           names=["age","workclass","fnlwgt","education","education_num",
                                  "marital-status","occupation","relationship","race","sex",
                                  "capital-gain","capital-loss","hours-per-week",
                                  "native-country","money"])
    df_test = pd.read_csv(targets_path, header=None, skiprows=0, 
                          names=["age","workclass","fnlwgt","education","education_num",
                                 "marital-status","occupation","relationship","race","sex",
                                 "capital-gain","capital-loss","hours-per-week",
                                 "native-country","money1"])
    print(f"Raw train shape: {df_train.shape}, test shape: {df_test.shape}")  # 检查原始数据大小
    return df_train, df_test

def dataCleaning(dataSet):
    # 删除不相关的列
    dataSet.drop(['education'], axis=1, inplace=True)
    dataSet = dataSet.replace(' ?', np.nan)
    
    # 处理缺失值，使用众数填充
    dataSet.fillna(value={
        'workclass': dataSet['workclass'].mode()[0],
        'occupation': dataSet['occupation'].mode()[0],
        'native-country': dataSet['native-country'].mode()[0]
    }, inplace=True)  

    # 类别数据映射
    workclass = {' State-gov': 0, ' Self-emp-not-inc': 1, ' Private': 2, ' Federal-gov': 3,
                 ' Local-gov': 4, ' Self-emp-inc': 5, ' Without-pay': 6, ' Never-worked': 7}
    maritalStatus = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 2,
                     ' Married-spouse-absent': 3, ' Separated': 4, 
                     ' Married-AF-spouse': 5, ' Widowed': 6}
    occupation = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2, 
                  ' Prof-specialty': 3, ' Other-service': 4, ' Sales': 5, 
                  ' Craft-repair': 6, ' Transport-moving': 7, ' Farming-fishing': 8, 
                  ' Machine-op-inspct': 9, ' Tech-support': 10, ' Protective-serv': 11,
                  ' Armed-Forces': 12, ' Priv-house-serv': 13}
    relationship = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, 
                    ' Own-child': 3, ' Unmarried': 4, ' Other-relative': 5}
    race = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, 
            ' Amer-Indian-Eskimo': 3, ' Other': 4}
    sex = {' Male': 0, ' Female': 1}
    nativeCountry = {' United-States': 0, ' Cuba': 1, ' Jamaica': 2, ' India': 3, 
                     ' Mexico': 4, ' South': 5, ' Puerto-Rico': 6, ' Honduras': 7, 
                     ' England': 8, ' Canada': 9, ' Germany': 10, ' Iran': 11, 
                     ' Philippines': 12, ' Italy': 13, ' Poland': 14, ' Columbia': 15, 
                     ' Cambodia': 16, ' Thailand': 17, ' Ecuador': 18, ' Laos': 19, 
                     ' Taiwan': 20, ' Haiti': 21, ' Portugal': 22, 
                     ' Dominican-Republic': 23, ' El-Salvador': 24, ' France': 25, 
                     ' Guatemala': 26, ' China': 27, ' Japan': 28, ' Yugoslavia': 29, 
                     ' Peru': 30, ' Outlying-US(Guam-USVI-etc)': 31, ' Scotland': 32, 
                     ' Trinadad&Tobago': 33, ' Greece': 34, ' Nicaragua': 35, 
                     ' Vietnam': 36, ' Hong': 37, ' Ireland': 38, ' Hungary': 39, 
                     ' Holand-Netherlands': 40}
    # money = {' <=50K': 0, ' >50K': 1}
    # money1 = {' <=50K.': 0, ' >50K.': 1}

    dataSet['workclass'] = dataSet['workclass'].map(workclass)
    dataSet['marital-status'] = dataSet['marital-status'].map(maritalStatus)
    dataSet['occupation'] = dataSet['occupation'].map(occupation)
    dataSet['relationship'] = dataSet['relationship'].map(relationship)
    dataSet['race'] = dataSet['race'].map(race)
    dataSet['sex'] = dataSet['sex'].map(sex)
    dataSet['native-country'] = dataSet['native-country'].map(nativeCountry)
    
    # if 'money' in dataSet.columns:
    #     dataSet['money'] = dataSet['money'].map(money)
    # if 'money1' in dataSet.columns:
    #     dataSet['money1'] = dataSet['money1'].map(money1)

    return dataSet

class AdultsDataset(Dataset):
    def __init__(self, data, is_train=True, mean=None, std=None):
        # 定义类别数量
        self.num_classes = 2
        self.is_train = is_train

        # 分离特征和标签
        if 'money' in data.columns:
            self.data = data.iloc[:, :-1].values.astype(np.float32)
            self.labels = data['money'].values.astype(np.int64)
        elif 'money1' in data.columns:
            self.data = data.iloc[:, :-1].values.astype(np.float32)
            self.labels = data['money1'].values.astype(np.int64)
        else:
            raise ValueError("标签列不存在！")

        # 计算均值和标准差（仅训练集）
        if self.is_train:
            if mean is None or std is None:
                self.mean = np.mean(self.data, axis=0)
                self.std = np.std(self.data, axis=0)
            else:
                self.mean = mean
                self.std = std
            self.data = (self.data - self.mean) / (self.std + 1e-8)
        else:
            if mean is not None and std is not None:
                self.mean = mean
                self.std = std
                self.data = (self.data - self.mean) / (self.std + 1e-8)
            else:
                self.mean = None
                self.std = None

    def __getitem__(self, index):
        # print(f"Index type: {type(index)}, value: {index}")
        if torch.is_tensor(index):
            index = index.item()  # 将标量张量转为整数
        if isinstance(index, (list, np.ndarray, torch.Tensor)):
            if len(index) > 0:
                index = index[0]
            else:
                # 如果 index 是空的，返回 None 或者引发异常
                raise IndexError("Received empty index list in __getitem__.")
        index = int(index)
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.labels)

def adult(data_path, balance=True):
    # 加载和清洗数据
    df_train, df_test = load_adult_dataset(data_dir=data_path)
    df_train = dataCleaning(df_train)
    df_test = dataCleaning(df_test)

    

    if balance:
        # 平衡划分训练集和验证集
        # 按照标签的比例划分训练集和验证集
        data_label_0 = df_train[df_train['money'] == 0]
        data_label_1 = df_train[df_train['money'] == 1]
        df_label_0_train, df_label_0_val = train_test_split(data_label_0, test_size=0.2, random_state=42)
        df_label_1_train, df_label_1_val = train_test_split(data_label_1, test_size=0.2, random_state=42)
        df_train = pd.concat([df_label_0_train, df_label_1_train]).reset_index(drop=True)
        df_val = pd.concat([df_label_0_val, df_label_1_val]).reset_index(drop=True)
        
    else:
        # 提取标签为0的数据
        df_label_0 = df_train[df_train['money'] == 0]
        df_label_1 = df_train[df_train['money'] == 1]

        # 将标签为0的数据按80/20划分为训练集和验证集
        df_label_0_train, df_label_0_val = train_test_split(df_label_0, test_size=0.2, random_state=42)
        df_label_1_train, df_label_1_val = train_test_split(df_label_1, test_size=0.2, random_state=42)

        # 训练集包含所有标签为1的数据和80%的标签为0的数据
        df_train = pd.concat([df_label_1_train, df_label_0_train]).reset_index(drop=True)

        # 验证集只包含剩余20%的标签为0的数据和10%的标签为1的数据
        df_label_1_val_1, df_label_1_val_2 = train_test_split(df_label_1_val, test_size=0.5, random_state=42)
        df_val = df_label_0_val

        #测试集
        #提取标签为0的数据
        df_label_0_test = df_test[df_test['money1'] == 0]
        df_label_1_test = df_test[df_test['money1'] == 1]
        df_test = df_label_0_test

    # 定义类别名称
    class_names = ['<=50K', '>50K']

    # 创建训练数据集，计算均值和标准差
    train_dataset = AdultsDataset(df_train, is_train=True)
    mean = train_dataset.mean
    std = train_dataset.std

    # 创建验证和测试数据集，使用训练集的均值和标准差
    val_dataset = AdultsDataset(df_val, is_train=False, mean=mean, std=std)
    test_dataset = AdultsDataset(df_test, is_train=False, mean=mean, std=std)

    # 定义其他参数
    channel = 1  # 对于表格数据，通常设为1
    im_size = train_dataset.data.shape[1]
    num_classes = train_dataset.num_classes

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset