# selection/tmcs.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_similarity_matrix(features, device):
    """计算特征矩阵的相似度矩阵"""
    features = features.to(device)
    similarity_matrix = torch.mm(features, features.t())
    return similarity_matrix

def compute_shapley_value_simple(features, device, num_iterations=50):
    """简化的 Shapley 值计算，基于特征相似度"""
    num_samples = features.size(0)
    shapley_values = torch.zeros(num_samples, device=device)
    
    # 预先计算全局相似度矩阵
    similarity_matrix = compute_similarity_matrix(features, device)
    
    for t in tqdm(range(num_iterations), desc="计算 TMCS Shapley 值"):
        # 随机排列数据点
        permutation = torch.randperm(num_samples)
        
        for j, idx in enumerate(permutation):
            if j == 0:
                # 第一个数据点，边际贡献为自相似度
                marginal_contribution = similarity_matrix[idx, idx].item()
            else:
                # 当前数据点与已加入子集的数据点的相似度之和
                prev_indices = permutation[:j]
                marginal_contribution = (similarity_matrix[idx, idx].item() + 
                                       similarity_matrix[idx, prev_indices].sum().item())

            # 更新 Shapley 值（递归平均）
            shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (t + 1)

    # 归一化 Shapley 值
    if shapley_values.max() > shapley_values.min():
        shapley_values = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min())
    
    return shapley_values

class TMCSSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               num_iterations: int = 20,
               use_features: bool = True
               ) -> DataLoader:
        """
        使用 TMCS (Truncated Monte Carlo Shapley) 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于特征提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - num_iterations:  Shapley 值计算迭代次数
          - use_features:    是否使用模型特征（否则使用原始数据）

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"TMCSSelection: total={n}, core={core_size}")

        # 3. 提取特征
        logging.info("提取特征...")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        
        if use_features:
            # 尝试使用模型提取特征
            try:
                extractor = EmbeddingExtractor(net).to(device)
                extractor.eval()
                
                all_features = []
                with torch.no_grad():
                    for inputs, _ in tqdm(train_loader, desc="提取模型特征"):
                        inputs = inputs.to(device)
                        features = extractor(inputs)
                        all_features.append(features.cpu())
                
                features = torch.cat(all_features, dim=0)
                logging.info(f"使用模型特征，维度: {features.shape}")
                
            except Exception as e:
                logging.warning(f"模型特征提取失败，使用原始数据: {e}")
                use_features = False
        
        if not use_features:
            # 使用原始数据展平
            all_features = []
            for inputs, _ in tqdm(train_loader, desc="展平原始数据"):
                all_features.append(inputs.view(inputs.size(0), -1))
            
            features = torch.cat(all_features, dim=0)
            logging.info(f"使用原始数据特征，维度: {features.shape}")

        # 4. 计算 Shapley 值
        logging.info("计算 TMCS Shapley 值...")
        shapley_values = compute_shapley_value_simple(
            features, device, num_iterations=num_iterations
        )

        # 5. 选择 Shapley 值最高的样本
        selected_indices = torch.topk(shapley_values, core_size).indices.tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"Shapley 值范围: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")

        # 6. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 7. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 