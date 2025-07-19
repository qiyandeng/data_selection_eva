# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImdbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        初始化数据集。

        Args:
            texts (list): 文本列表。
            labels (list or numpy.ndarray): 标签列表。
            tokenizer (BertTokenizer): BERT 分词器。
            max_length (int): 最大序列长度。
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        # 对文本进行分词和编码
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将编码后的数据转换为字典
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去掉 batch 维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)


def imdb_bert(data_path, balance=True):
    """
    加载IMDB数据集，准备数据用于直接微调BERT模型。

    Args:
        data_path (str): 数据集根目录路径。
        balance (bool): 是否在验证集中均衡类别分布。

    Returns:
        tuple: 包含channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset的元组。
    """
    print("Loading IMDB dataset...")

    def load_data(data_dir):
        """
        从指定目录加载数据。

        Args:
            data_dir (str): 数据目录路径。

        Returns:
            tuple: 包含文本列表和标签列表的元组。
        """
        texts = []
        labels = []
        for label_dir, label in [('pos', 1), ('neg', 0)]:
            dir_path = os.path.join(data_dir, label_dir)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

            for filename in tqdm(file_list, desc=f"Reading {label_dir} files"):
                file_path = os.path.join(dir_path, filename)
                if not os.path.isfile(file_path):
                    print(f"Skipping non-file: {file_path}")
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        texts.append(content)
                        labels.append(label)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            print(f"Completed processing directory: {dir_path}")
        return texts, labels

    # 加载训练数据
    train_dir = os.path.join(data_path, 'aclImdb', 'train')
    train_texts, train_labels = load_data(train_dir)

    # 根据参数 balance 划分训练集和验证集
    if balance:
        # 将数据和标签一起打乱
        combined = list(zip(train_texts, train_labels))
        np.random.shuffle(combined)
        train_texts[:], train_labels[:] = zip(*combined)
        # 划分训练集和验证集
        split_ratio = 0.8
        split_index = int(len(train_texts) * split_ratio)
        train_texts_split = train_texts[:split_index]
        train_labels_split = train_labels[:split_index]
        valid_texts_split = train_texts[split_index:]
        valid_labels_split = train_labels[split_index:]

        # 加载测试数据
        test_dir = os.path.join(data_path, 'aclImdb', 'test')
        test_texts, test_labels = load_data(test_dir)

        # 在balance=True时，测试集保持原分布
        test_texts_final = test_texts
        test_labels_final = test_labels
    else:
        # 提取标签为0的数据索引
        label_0_indices = [i for i, label in enumerate(train_labels) if label == 0]
        # 打乱标签为0的数据索引
        np.random.shuffle(label_0_indices)
        # 计算切分点
        split_index = int(len(label_0_indices) * 0.8)
        # 划分训练集和验证集索引
        label_0_train_indices = label_0_indices[:split_index]
        label_0_val_indices = label_0_indices[split_index:]
        # 提取标签为1的数据索引
        label_1_indices = [i for i, label in enumerate(train_labels) if label == 1]
        # 训练集包含所有的标签为1的数据和80%的标签为0的数据
        train_indices = label_1_indices + label_0_train_indices
        # 验证集只包含20%的标签为0的数据
        val_indices = label_0_val_indices
        # 提取训练集和验证集的数据和标签
        train_texts_split = [train_texts[i] for i in train_indices]
        train_labels_split = [train_labels[i] for i in train_indices]
        valid_texts_split = [train_texts[i] for i in val_indices]
        valid_labels_split = [train_labels[i] for i in val_indices]

        # 加载测试数据
        test_dir = os.path.join(data_path, 'aclImdb', 'test')
        test_texts, test_labels = load_data(test_dir)

        # 处理测试集，使其与验证集具有相同的类别分布
        # 即，训练集包含所有标签为1和80%的标签为0
        # 验证集包含20%的标签为0
        # 测试集也将包含所有标签为1和20%的标签为0

        # 提取标签为0的测试集索引
        label_0_test_indices = [i for i, label in enumerate(test_labels) if label == 0]
        # 打乱标签为0的测试集索引
        np.random.shuffle(label_0_test_indices)
        # 计算测试集中的20%标签为0的索引
        split_index_test = int(len(label_0_test_indices) * 0.2)
        label_0_test_val_indices = label_0_test_indices[:split_index_test]
        # 提取标签为1的测试集索引
        label_1_test_indices = [i for i, label in enumerate(test_labels) if label == 1]
        # 合并标签为1和20%标签为0的测试集索引
        test_indices = label_0_test_val_indices
        # 提取最终的测试集文本和标签
        test_texts_split = [test_texts[i] for i in test_indices]
        test_labels_split = [test_labels[i] for i in test_indices]

        test_texts_final = test_texts_split
        test_labels_final = test_labels_split

    # 初始化 BERT tokenizer 和 model
    try:
        tokenizer = BertTokenizer.from_pretrained('/data/wyh-dataset/text-dataset/bert-base-uncase')
    except Exception as e:
        print(f"Error loading BERT tokenizer: {e}")
        raise e

    # 创建 Dataset 对象
    train_dataset = ImdbDataset(train_texts_split, train_labels_split, tokenizer)
    val_dataset = ImdbDataset(valid_texts_split, valid_labels_split, tokenizer)
    test_dataset = ImdbDataset(test_texts_final, test_labels_final, tokenizer)

    # 定义其他参数
    channel = 1  # 对于文本数据，通常设为1
    im_size = 768  # BERT 的隐藏层维度
    num_classes = 2
    class_names = ['neg', 'pos']
    mean = 0.0  # 直接微调 BERT 不需要标准化
    std = 1.0   # 直接微调 BERT 不需要标准化

    print(f"Channel: {channel}, Image Size: {im_size}, Num Classes: {num_classes}")
    print(f"Class Names: {class_names}")

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset