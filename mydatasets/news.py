# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NewsDataset(Dataset):
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


def compute_embeddings(texts, bert_model, tokenizer, batch_size=32):
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


def save_embeddings(embeddings, labels, embedding_path, labels_path):
    """
    保存嵌入和标签到指定路径。

    Args:
        embeddings (numpy.ndarray): 嵌入矩阵。
        labels (list or numpy.ndarray): 标签列表。
        embedding_path (str): 嵌入保存路径。
        labels_path (str): 标签保存路径。
    """
    np.save(embedding_path, embeddings)
    np.save(labels_path, labels)
    print(f"Saved embeddings to {embedding_path} and labels to {labels_path}.")


def load_embeddings(embedding_path, labels_path):
    """
    从指定路径加载嵌入和标签。

    Args:
        embedding_path (str): 嵌入文件路径。
        labels_path (str): 标签文件路径。

    Returns:
        tuple: (嵌入矩阵, 标签列表)
    """
    embeddings = np.load(embedding_path)
    labels = np.load(labels_path)
    print(f"Loaded embeddings from {embedding_path} and labels from {labels_path}.")
    return embeddings, labels


def news(data_path, balance=True, categories=None):
    """
    加载20 Newsgroups数据集，计算或加载BERT嵌入，进行标准化，并返回相关参数和数据集对象。

    Args:
        data_path (str): 数据集根目录路径。
        balance (bool): 是否在验证和测试集中均衡类别分布。
        categories (list, optional): 20 Newsgroups的类别列表。默认为预定义的类别。

    Returns:
        tuple: 包含channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset的元组。
    """
    if categories is None:
        categories = [
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc'
        ]

    print("Loading 20 Newsgroups dataset...")

    # 定义嵌入保存目录
    embedding_dir = "./embedding"
    os.makedirs(embedding_dir, exist_ok=True)

    # 定义嵌入和标签的文件路径
    if balance:      
        train_embedding_path = os.path.join(embedding_dir, "balance_news_train_embeddings.npy")
        train_labels_path = os.path.join(embedding_dir, "balance_news_train_labels.npy")
        val_embedding_path = os.path.join(embedding_dir, "balance_news_val_embeddings.npy")
        val_labels_path = os.path.join(embedding_dir, "balance_news_val_labels.npy")
        test_embedding_path = os.path.join(embedding_dir, "balance_news_test_embeddings.npy")
        test_labels_path = os.path.join(embedding_dir, "balance_news_test_labels.npy")
        mean_path = os.path.join(embedding_dir, "balance_news_mean.npy")
        std_path = os.path.join(embedding_dir, "balance_news_std.npy")
    else :    
        train_embedding_path = os.path.join(embedding_dir, "unbalance_news_train_embeddings.npy")
        train_labels_path = os.path.join(embedding_dir, "unbalance_news_train_labels.npy")
        val_embedding_path = os.path.join(embedding_dir, "unbalance_news_val_embeddings.npy")
        val_labels_path = os.path.join(embedding_dir, "unbalance_news_val_labels.npy")
        test_embedding_path = os.path.join(embedding_dir, "unbalance_news_test_embeddings.npy")
        test_labels_path = os.path.join(embedding_dir, "unbalance_news_test_labels.npy")
        mean_path = os.path.join(embedding_dir, "unbalance_news_mean.npy")
        std_path = os.path.join(embedding_dir, "unbalance_news_std.npy")

    # 加载训练和测试数据
    # train_dataset_raw = fetch_20newsgroups(
    #     data_home=data_path,
    #     subset='train',
    #     shuffle=True,
    #     categories=categories,
    #     remove=('headers', 'footers', 'quotes')  # 可选：移除邮件头部、尾部和引用内容
    # )
    # test_dataset_raw = fetch_20newsgroups(
    #     data_home=data_path,
    #     subset='test',
    #     shuffle=True,
    #     categories=categories,
    #     remove=('headers', 'footers', 'quotes')  # 可选
    # )
    train_dataset_raw = load_files(
        container_path=os.path.join(data_path, "/data/dengqiyan/wyh-dataset/text-dataset/News/20news_home/20news-bydate/20news-bydate-train"),
        categories=categories,
        encoding="latin1",
        decode_error="ignore"
    )

    test_dataset_raw = load_files(
        container_path=os.path.join(data_path, "/data/dengqiyan/wyh-dataset/text-dataset/News/20news_home/20news-bydate/20news-bydate-test"),
        categories=categories,
        encoding="latin1",
        decode_error="ignore"
    )

    train_texts = train_dataset_raw.data
    train_labels = train_dataset_raw.target
    test_texts = test_dataset_raw.data
    test_labels = test_dataset_raw.target
    class_names = train_dataset_raw.target_names
    num_classes = len(class_names)

    print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples.")

    # 根据 balance 参数划分训练集和验证集
    if balance:
        # 使用分层拆分，确保训练集和验证集的类别分布与原始数据一致
        train_texts_split, valid_texts_split, train_labels_split, valid_labels_split = train_test_split(
            train_texts,
            train_labels,
            test_size=0.2,
            random_state=42,
            stratify=train_labels
        )
        print(f"Balance=True: Split into {len(train_texts_split)} training and {len(valid_texts_split)} validation samples with balanced classes.")

        # 测试集保持原分布
        test_texts_final = test_texts
        test_labels_final = test_labels

    else:
        # 当 balance=False 时，验证集仅包含类别为0的数据
        # 首先，获取类别为0的样本
        class_0_indices = np.where(train_labels == 0)[0]
        class_non_0_indices = np.where(train_labels != 0)[0]

        # 将类别为0的样本按80/20划分
        class_0_train_indices, class_0_val_indices = train_test_split(
            class_0_indices,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )

        # 训练集包含所有非类别0的样本和80%的类别0样本
        train_indices = np.concatenate([class_non_0_indices, class_0_train_indices])
        # 验证集仅包含20%的类别0样本
        val_indices = class_0_val_indices

        # 选择相应的文本和标签
        train_texts_split = [train_texts[i] for i in train_indices]
        train_labels_split = train_labels[train_indices]
        valid_texts_split = [train_texts[i] for i in val_indices]
        valid_labels_split = train_labels[val_indices]

        print(f"Balance=False: Split into {len(train_texts_split)} training and {len(valid_texts_split)} validation samples with validation set containing only class 0.")

        # 处理测试集，使其与验证集具有相同的类别分布
        # 即，测试集也仅包含类别为0的样本
        test_class_0_indices = np.where(test_labels == 0)[0]
        test_texts_final = [test_texts[i] for i in test_class_0_indices]
        test_labels_final = test_labels[test_class_0_indices]

        print(f"Balance=False: Test set contains {len(test_texts_final)} samples with only class 0.")

    # 初始化 BERT tokenizer 和 model
    print("Initializing BERT model and tokenizer...")
    try:
        bert_model = BertModel.from_pretrained('/data/dengqiyan/wyh-dataset/text-dataset/bert-base-uncase')  # 请确保路径正确
        tokenizer = BertTokenizer.from_pretrained('/data/dengqiyan/wyh-dataset/text-dataset/bert-base-uncase')
    except Exception as e:
        print(f"Error loading BERT model or tokenizer: {e}")
        raise e

    bert_model.to(device)
    bert_model.eval()
    print("Initialized BERT model and tokenizer.")

    # 检查是否存在训练集嵌入
    if (os.path.exists(train_embedding_path) and
        os.path.exists(train_labels_path) and
        os.path.exists(mean_path) and
        os.path.exists(std_path)):
        print("Loading existing training embeddings, labels, mean, and std...")
        train_embeddings, train_labels_split = load_embeddings(train_embedding_path, train_labels_path)
        mean = np.load(mean_path)
        std = np.load(std_path)
        print(f"Loaded mean from {mean_path} and std from {std_path}.")
    else:
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

        # 保存训练集嵌入和标签
        save_embeddings(train_embeddings, train_labels_split, train_embedding_path, train_labels_path)

        # 保存均值和标准差
        np.save(mean_path, mean)
        np.save(std_path, std)
        print(f"Saved mean to {mean_path} and std to {std_path}.")

    # 如果均值和标准差已经存在但训练集嵌入未加载，则加载它们
    if 'mean' not in locals() and os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        print(f"Loaded mean from {mean_path} and std from {std_path}.")
    elif 'mean' not in locals() and not (os.path.exists(mean_path) and os.path.exists(std_path)):
        # 如果训练集嵌入是通过加载而来的，均值和标准差已经计算并保存
        mean = np.load(mean_path)
        std = np.load(std_path)
        print(f"Loaded mean from {mean_path} and std from {std_path}.")

    # 检查是否存在验证集嵌入
    if os.path.exists(val_embedding_path) and os.path.exists(val_labels_path):
        print("Loading existing validation embeddings and labels...")
        val_embeddings, valid_labels_split = load_embeddings(val_embedding_path, val_labels_path)
    else:
        # 计算验证集嵌入
        print("Creating validation dataset and computing embeddings...")
        val_embeddings = compute_embeddings(valid_texts_split, bert_model, tokenizer, batch_size=32)
        print("Computed validation embeddings.")

        # 标准化验证集嵌入
        val_embeddings = (val_embeddings - mean) / (std + 1e-8)
        print("Standardized validation embeddings.")

        # 保存验证集嵌入和标签
        save_embeddings(val_embeddings, valid_labels_split, val_embedding_path, val_labels_path)

    # 检查是否存在测试集嵌入
    if os.path.exists(test_embedding_path) and os.path.exists(test_labels_path):
        print("Loading existing test embeddings and labels...")
        test_embeddings, test_labels_final = load_embeddings(test_embedding_path, test_labels_path)
    else:
        # 计算测试集嵌入
        print("Creating test dataset and computing embeddings...")
        test_embeddings = compute_embeddings(test_texts_final, bert_model, tokenizer, batch_size=32)
        print("Computed test embeddings.")

        # 标准化测试集嵌入
        test_embeddings = (test_embeddings - mean) / (std + 1e-8)
        print("Standardized test embeddings.")

        # 保存测试集嵌入和标签
        save_embeddings(test_embeddings, test_labels_final, test_embedding_path, test_labels_path)

    # 创建 Dataset 对象
    train_dataset = NewsDataset(train_embeddings, train_labels_split)
    val_dataset = NewsDataset(val_embeddings, valid_labels_split)
    test_dataset = NewsDataset(test_embeddings, test_labels_final)

    # 定义其他参数
    channel = 1  # 对于文本数据，通常设为1
    im_size = train_embeddings.shape[1]  # BERT pooler_output 维度通常为768
    # num_classes = num_classes 已经定义
    print(f"Channel: {channel}, Image Size: {im_size}, Num Classes: {num_classes}")
    print(f"Class Names: {class_names}")

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset