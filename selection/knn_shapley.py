# selection/knn_shapley.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_knn_shapley(features, labels, k=5, num_iterations=50):
    """计算基于 KNN 的 Shapley 值"""
    n = len(features)
    shapley_values = np.zeros(n)
    
    # 构建 KNN 模型
    knn = NearestNeighbors(n_neighbors=min(k+1, n), metric='euclidean')
    knn.fit(features)
    
    for iteration in tqdm(range(num_iterations), desc="计算 KNN-Shapley 值"):
        # 随机排列所有样本
        permutation = np.random.permutation(n)
        
        for j, idx in enumerate(permutation):
            if j == 0:
                # 第一个样本的边际贡献
                marginal_contribution = 1.0
            else:
                # 计算当前样本对 KNN 性能的边际贡献
                current_sample = features[idx:idx+1]
                current_label = labels[idx]
                
                # 找到当前样本的 k 个最近邻
                distances, indices = knn.kneighbors(current_sample)
                neighbor_indices = indices[0][1:]  # 排除自己
                
                # 计算与已加入样本的重叠度
                added_samples = set(permutation[:j])
                neighbor_overlap = len(set(neighbor_indices) & added_samples)
                
                # 边际贡献基于邻居重叠度和标签一致性
                if len(neighbor_indices) > 0:
                    neighbor_labels = labels[neighbor_indices]
                    label_consistency = np.mean(neighbor_labels == current_label)
                    marginal_contribution = label_consistency * (1 + neighbor_overlap / len(neighbor_indices))
                else:
                    marginal_contribution = 1.0
            
            # 更新 Shapley 值
            shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (iteration + 1)
    
    # 归一化
    if shapley_values.max() > shapley_values.min():
        shapley_values = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min())
    
    return shapley_values

class KNNShapleySelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               k: int = 5,
               num_iterations: int = 30
               ) -> DataLoader:
        """
        使用 KNN-Shapley 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于特征提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - k:               KNN 中的 k 值
          - num_iterations:  Shapley 值计算迭代次数

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"KNNShapleySelection: total={n}, core={core_size}, k={k}")

        # 3. 提取特征和标签
        logging.info("提取特征和标签...")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        
        try:
            extractor = EmbeddingExtractor(net).to(device)
            extractor.eval()
            
            all_features = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(train_loader, desc="提取特征"):
                    inputs = inputs.to(device)
                    features = extractor(inputs)
                    all_features.append(features.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            features = np.vstack(all_features)
            labels = np.array(all_labels)
            
        except Exception as e:
            logging.warning(f"特征提取失败，使用原始数据: {e}")
            all_features = []
            all_labels = []
            
            for inputs, labels_batch in tqdm(train_loader, desc="使用原始数据"):
                all_features.append(inputs.view(inputs.size(0), -1).numpy())
                all_labels.extend(labels_batch.numpy())
            
            features = np.vstack(all_features)
            labels = np.array(all_labels)

        # 4. 计算 KNN Shapley 值
        logging.info("计算 KNN Shapley 值...")
        shapley_values = compute_knn_shapley(features, labels, k=k, num_iterations=num_iterations)

        # 5. 选择 Shapley 值最高的样本
        selected_indices = np.argsort(shapley_values)[-core_size:].tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"KNN-Shapley 值范围: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")

        # 6. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(labels[i]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 7. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 