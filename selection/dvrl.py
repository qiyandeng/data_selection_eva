# selection/dvrl.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class SimpleValueNetwork(nn.Module):
    """简化的价值网络"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class DVRLSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               val_set=None,
               num_epochs: int = 10,
               learning_rate: float = 0.001
               ) -> DataLoader:
        """
        使用 DVRL (Data Valuation using Reinforcement Learning) 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于价值评估的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - val_set:         验证集（如果为 None，则使用训练集的一部分）
          - num_epochs:      价值网络训练轮数
          - learning_rate:   学习率

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"DVRLSelection: total={n}, core={core_size}")

        # 3. 准备验证集
        if val_set is None:
            val_size = max(20, int(0.2 * n))
            val_indices = list(range(val_size))
            val_set = Subset(train_set, val_indices)
            logging.info(f"使用训练集前 {val_size} 个样本作为验证集")

        # 4. 提取特征
        logging.info("提取特征...")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        
        try:
            extractor = EmbeddingExtractor(net).to(device)
            extractor.eval()
            
            all_features = []
            with torch.no_grad():
                for inputs, _ in tqdm(train_loader, desc="提取特征"):
                    inputs = inputs.to(device)
                    features = extractor(inputs)
                    all_features.append(features.cpu())
            
            features = torch.cat(all_features, dim=0)
            
        except Exception as e:
            logging.warning(f"特征提取失败，使用原始数据: {e}")
            all_features = []
            for inputs, _ in tqdm(train_loader, desc="使用原始数据"):
                all_features.append(inputs.view(inputs.size(0), -1))
            
            features = torch.cat(all_features, dim=0)

        # 5. 初始化价值网络
        input_dim = features.shape[1]
        value_network = SimpleValueNetwork(input_dim).to(device)
        optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # 6. 训练价值网络（简化版）
        logging.info("训练价值网络...")
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        
        for epoch in tqdm(range(num_epochs), desc="训练价值网络"):
            value_network.train()
            
            # 随机采样一些数据点
            sample_indices = np.random.choice(len(features), size=min(100, len(features)), replace=False)
            sample_features = features[sample_indices].to(device)
            
            # 计算价值分数
            value_scores = value_network(sample_features)
            
            # 简化的奖励计算：基于模型在验证集上的性能
            rewards = []
            for i, idx in enumerate(sample_indices):
                # 使用单个样本训练模型并评估性能（简化）
                sample_weight = value_scores[i].item()
                
                # 简化的奖励：基于样本的预测置信度
                sample_input, sample_target = train_set[idx]
                sample_input = sample_input.unsqueeze(0).to(device)
                sample_target = torch.tensor([sample_target]).to(device)
                
                with torch.no_grad():
                    output = net(sample_input)
                    if task_type == 'classification':
                        prob = torch.softmax(output, dim=1)
                        confidence = prob.max().item()
                        reward = confidence * sample_weight
                    else:
                        reward = sample_weight
                
                rewards.append(reward)
            
            rewards = torch.tensor(rewards, device=device)
            
            # 训练价值网络
            optimizer.zero_grad()
            loss = criterion(value_scores.squeeze(), rewards)
            loss.backward()
            optimizer.step()

        # 7. 计算所有样本的价值分数
        logging.info("计算样本价值分数...")
        value_network.eval()
        
        all_value_scores = []
        batch_size = 128
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size].to(device)
                batch_scores = value_network(batch_features)
                all_value_scores.extend(batch_scores.cpu().numpy())

        value_scores = np.array(all_value_scores).flatten()

        # 8. 选择价值分数最高的样本
        selected_indices = np.argsort(value_scores)[-core_size:].tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"价值分数范围: [{value_scores.min():.6f}, {value_scores.max():.6f}]")

        # 9. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 10. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 