# selection/ias.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class IncrementalTrainer:
    """增量训练器，用于 IAS 算法"""
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        # 初始化累积梯度 z 和平方梯度 n
        self.z = [torch.zeros_like(p.data) for p in model.parameters()]
        self.n = [torch.zeros_like(p.data) for p in model.parameters()]
        self.epsilon = 1e-8

    def update(self, inputs, labels):
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # 计算梯度
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        
        # 更新累积梯度 z 和平方梯度 n
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                self.z[i] += grads[i]
                self.n[i] += grads[i] ** 2

                # 根据 Algorithm 1 更新参数
                adjusted_lr = self.optimizer.param_groups[0]['lr'] / (torch.sqrt(self.n[i]) + self.epsilon)
                param.data -= adjusted_lr * self.z[i]

        return loss.item()

def calculate_adaptive_score(rewards, selected_counts, gamma, diversity_scores, alpha):
    """计算自适应得分：开发 + 探索 + 多样性"""
    exploitation = rewards
    exploration = gamma / (selected_counts + 1)
    diversity = alpha * diversity_scores
    return exploitation + exploration + diversity

class IASSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               n_clusters: int = 10,
               gamma: float = 0.1,
               alpha: float = 0.1,
               lambda_: float = 0.9,
               learning_rate: float = 0.01,
               batch_size: int = 32
               ) -> DataLoader:
        """
        使用 IAS (Incremental Active Selection) 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于增量训练的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - n_clusters:      聚类数量
          - gamma:           探索参数
          - alpha:           多样性权重
          - lambda_:         奖励衰减因子
          - learning_rate:   学习率
          - batch_size:      每轮选择的批次大小

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        
        # 2. 计算目标选择数量
        n = len(train_set)
        target_size = int(n * selection_fraction)
        logging.info(f"IASSelection: total={n}, target={target_size}")

        # 3. 特征提取和聚类
        logging.info("提取特征并进行聚类...")
        features = []
        labels = []
        
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        for inputs, targets in tqdm(train_loader, desc="提取特征"):
            features.append(inputs.view(inputs.size(0), -1).numpy())
            labels.extend(targets.numpy())
        
        features = np.vstack(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
        cluster_labels = kmeans.labels_

        # 计算簇中心和整体均值
        cluster_centroids = kmeans.cluster_centers_
        overall_mean = np.mean(features, axis=0)
        cluster_distances = np.linalg.norm(cluster_centroids - overall_mean, axis=1)

        # 计算每个数据点到其簇中心的距离，作为多样性得分
        diversity_scores = np.zeros(len(train_set))
        for idx in range(len(train_set)):
            cluster = cluster_labels[idx]
            data_point = features[idx]
            diversity_scores[idx] = np.linalg.norm(data_point - cluster_centroids[cluster])

        # 4. 初始化增量训练器
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        trainer = IncrementalTrainer(net, optimizer, criterion, device)

        # 5. 初始化自适应得分
        data_indices = np.arange(len(train_set))
        rewards = np.zeros(len(train_set))
        selected_counts = np.zeros(len(train_set))
        scores = np.zeros(len(train_set))
        selected_data_indices = set()

        # 6. IAS 选择过程
        epoch = 0
        pbar = tqdm(total=target_size, desc="IAS Selection")
        
        while len(selected_data_indices) < target_size:
            if epoch == 0:
                # 第一轮：选择与整体分布最不同的簇的数据点
                selected_cluster = np.argmax(cluster_distances)
                cluster_indices = data_indices[cluster_labels == selected_cluster]
                unselected_indices = list(set(cluster_indices) - selected_data_indices)
                selected_indices = unselected_indices[:min(batch_size, len(unselected_indices))]
            else:
                # 后续轮次：基于自适应得分采样
                unselected_indices = list(set(data_indices) - selected_data_indices)
                if len(unselected_indices) == 0:
                    break
                
                unselected_scores = scores[unselected_indices]
                num_to_select = min(batch_size, len(unselected_indices), target_size - len(selected_data_indices))
                selected_indices = np.array(unselected_indices)[np.argsort(unselected_scores)[-num_to_select:]]

            # 更新已选数据点集合
            selected_data_indices.update(selected_indices)
            selected_counts[selected_indices] += 1

            # 增量更新模型
            selected_loader = DataLoader(Subset(train_set, selected_indices), batch_size=len(selected_indices), shuffle=False)
            
            for inputs, labels_batch in selected_loader:
                loss_value = trainer.update(inputs, labels_batch)
                loss_value = loss_value / len(selected_indices)
                
                # 更新选中数据点的奖励
                for idx in selected_indices:
                    rewards[idx] = lambda_ * rewards[idx] + (1 - lambda_) * (-loss_value)

            # 计算所有数据点的自适应得分
            scores = calculate_adaptive_score(rewards, selected_counts, gamma, diversity_scores, alpha)

            pbar.update(len(selected_indices))
            epoch += 1

        pbar.close()
        
        selected_indices = list(selected_data_indices)
        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 7. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 8. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 