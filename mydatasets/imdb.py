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
    # print(f"Loaded embeddings from {embedding_path} and labels from {labels_path}.")
    return embeddings, labels


def imdb(data_path, balance=True):
    """
    加载IMDB数据集，计算或加载BERT嵌入，进行标准化，并返回相关参数和数据集对象。

    Args:
        data_path (str): 数据集根目录路径。
        balance (bool): 是否在验证集中均衡类别分布。

    Returns:
        tuple: 包含channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset的元组。
    """
    print("Loading IMDB dataset...")

    # 定义嵌入保存目录
    embedding_dir = "./embedding"
    os.makedirs(embedding_dir, exist_ok=True)

    # 定义嵌入和标签的文件路径
    if balance:      
        train_embedding_path = os.path.join(embedding_dir, "balance_imdb_train_embeddings.npy")
        train_labels_path = os.path.join(embedding_dir, "balance_imdb_train_labels.npy")
        val_embedding_path = os.path.join(embedding_dir, "balance_imdb_val_embeddings.npy")
        val_labels_path = os.path.join(embedding_dir, "balance_imdb_val_labels.npy")
        test_embedding_path = os.path.join(embedding_dir, "balance_imdb_test_embeddings.npy")
        test_labels_path = os.path.join(embedding_dir, "balance_imdb_test_labels.npy")
        mean_path = os.path.join(embedding_dir, "balance_imdb_mean.npy")
        std_path = os.path.join(embedding_dir, "balance_imdb_std.npy")
    else :    
        train_embedding_path = os.path.join(embedding_dir, "unbalance_imdb_train_embeddings.npy")
        train_labels_path = os.path.join(embedding_dir, "unbalance_imdb_train_labels.npy")
        val_embedding_path = os.path.join(embedding_dir, "unbalance_imdb_val_embeddings.npy")
        val_labels_path = os.path.join(embedding_dir, "unbalance_imdb_val_labels.npy")
        test_embedding_path = os.path.join(embedding_dir, "unbalance_imdb_test_embeddings.npy")
        test_labels_path = os.path.join(embedding_dir, "unbalance_imdb_test_labels.npy")
        mean_path = os.path.join(embedding_dir, "unbalance_imdb_mean.npy")
        std_path = os.path.join(embedding_dir, "unbalance_imdb_std.npy")

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
            # print(f"Processing directory: {dir_path}")
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
            # print(f"Found {len(file_list)} files in {dir_path}")

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
        bert_model = BertModel.from_pretrained('/data/dengqiyan/wyh-dataset/text-dataset/bert-base-uncase')  # 使用预训练模型
        tokenizer = BertTokenizer.from_pretrained('/data/dengqiyan/wyh-dataset/text-dataset/bert-base-uncase')
    except Exception as e:
        print(f"Error loading BERT model or tokenizer: {e}")
        raise e

    bert_model.to(device)
    bert_model.eval()

    # 检查是否存在训练集嵌入
    if os.path.exists(train_embedding_path) and os.path.exists(train_labels_path):
        print("Loading existing training embeddings and labels...")
        train_embeddings, train_labels_split = load_embeddings(train_embedding_path, train_labels_path)
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
    train_dataset = ImdbDataset(train_embeddings, train_labels_split)
    val_dataset = ImdbDataset(val_embeddings, valid_labels_split)
    test_dataset = ImdbDataset(test_embeddings, test_labels_final)

    # 定义其他参数
    channel = 1  # 对于文本数据，通常设为1
    im_size = train_embeddings.shape[1]  # BERT pooler_output 维度通常为768
    num_classes = 2
    class_names = ['neg', 'pos']
    print(f"Channel: {channel}, Image Size: {im_size}, Num Classes: {num_classes}")
    print(f"Class Names: {class_names}")

    return channel, im_size, num_classes, class_names, mean, std, train_dataset, val_dataset, test_dataset