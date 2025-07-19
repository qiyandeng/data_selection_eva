# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImdbDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        初始化数据集。

        Args:
            embeddings (numpy.ndarray): 预计算的BERT嵌入。
            labels (list or numpy.ndarray): 标签列表。
        """
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def compute_embeddings(texts, bert_model, tokenizer, batch_size=2048):
    """
    计算文本的BERT嵌入。

    Args:
        texts (list): 文本列表。
        bert_model (BertModel): 预训练的BERT模型。
        tokenizer (BertTokenizer): BERT分词器。
        batch_size (int, optional): 批处理大小。默认为32。

    Returns:
        numpy.ndarray: 嵌入矩阵。
    """
    bert_model.eval()
    embeddings = []

    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Computing embeddings"):
            # Tokenize批量文本
            encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            # 获取BERT输出
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_outputs = outputs.pooler_output.cpu().numpy()

            embeddings.append(pooled_outputs)

    # 合并所有批次的嵌入
    embeddings = np.vstack(embeddings)
    return embeddings


def imdb(data_path):
    """
    加载IMDB数据集，计算BERT嵌入，进行标准化，并返回相关参数和数据集对象。

    Args:
        data_path (str): 数据集根目录路径。

    Returns:
        tuple: 包含channel, im_size, num_classes, class_names, mean, std, train_dataset, test_dataset的元组。
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
            print(f"Processing directory: {dir_path}")
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
            print(f"Found {len(file_list)} files in {dir_path}")

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
    print(f"Loading training data from {train_dir}...")
    train_texts, train_labels = load_data(train_dir)
    print(f"Loaded {len(train_texts)} training samples.")

    # 划分训练集和验证集
    split_ratio = 0.8
    split_index = int(len(train_texts) * split_ratio)
    train_texts_split = train_texts[:split_index]
    train_labels_split = train_labels[:split_index]
    valid_texts_split = train_texts[split_index:]
    valid_labels_split = train_labels[split_index:]
    print(f"Split into {len(train_texts_split)} training and {len(valid_texts_split)} validation samples.")

    # 加载测试数据
    test_dir = os.path.join(data_path, 'aclImdb', 'test')
    print(f"Loading test data from {test_dir}...")
    test_texts, test_labels = load_data(test_dir)
    print(f"Loaded {len(test_texts)} test samples.")

    # 初始化 BERT tokenizer 和 model
    print("Initializing BERT model and tokenizer...")
    try:
        bert_model = BertModel.from_pretrained('/data/wyh-dataset/text-dataset/bert-base-uncase')  # 使用在线预训练模型
        tokenizer = BertTokenizer.from_pretrained('/data/wyh-dataset/text-dataset/bert-base-uncase')
    except Exception as e:
        print(f"Error loading BERT model or tokenizer: {e}")
        raise e

    bert_model.to(device)
    bert_model.eval()
    print("Initialized BERT model and tokenizer.")

    # 计算训练集嵌入
    print("Creating training dataset and computing embeddings...")
    train_embeddings = compute_embeddings(train_texts_split, bert_model, tokenizer, batch_size=32)
    print("Computed training embeddings.")

    # 计算均值和标准差
    mean = np.mean(train_embeddings, axis=0)
    std = np.std(train_embeddings, axis=0)
    print("Computed mean and standard deviation.")

    # 标准化训练集嵌入
    train_embeddings = (train_embeddings - mean) / (std + 1e-8)
    print("Standardized training embeddings.")

    # 计算验证集嵌入
    print("Creating validation dataset and computing embeddings...")
    val_embeddings = compute_embeddings(valid_texts_split, bert_model, tokenizer, batch_size=32)
    val_embeddings = (val_embeddings - mean) / (std + 1e-8)
    print("Computed and standardized validation embeddings.")

    # 计算测试集嵌入
    print("Creating test dataset and computing embeddings...")
    test_embeddings = compute_embeddings(test_texts, bert_model, tokenizer, batch_size=32)
    test_embeddings = (test_embeddings - mean) / (std + 1e-8)
    print("Computed and standardized test embeddings.")

    # 创建 Dataset 对象
    train_dataset = ImdbDataset(train_embeddings, train_labels_split)
    val_dataset = ImdbDataset(val_embeddings, valid_labels_split)
    test_dataset = ImdbDataset(test_embeddings, test_labels)

    # 定义其他参数
    channel = 1  # 对于文本数据，通常设为1
    im_size = train_embeddings.shape[1]  # BERT pooler_output 维度通常为768
    num_classes = 2
    class_names = ['neg', 'pos']
    print(f"Channel: {channel}, Image Size: {im_size}, Num Classes: {num_classes}")
    print(f"Class Names: {class_names}")

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, test_dataset