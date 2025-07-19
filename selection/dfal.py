# selection/dfal.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
from sklearn.cluster import KMeans

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class DFALSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               n_clusters: int = 10,
               diversity_weight: float = 0.5
               ) -> DataLoader:
        """
        使用 DFAL (Deep Feature Active Learning) 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于特征提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - n_clusters:      聚类数量
          - diversity_weight: 多样性权重

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"DFALSelection: total={n}, core={core_size}")

        # 3. 提取深度特征
        logging.info("提取深度特征...")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        
        try:
            extractor = EmbeddingExtractor(net).to(device)
            extractor.eval()
            
            all_features = []
            all_predictions = []
            
            with torch.no_grad():
                for inputs, _ in tqdm(train_loader, desc="提取深度特征"):
                    inputs = inputs.to(device)
                    features = extractor(inputs)
                    predictions = net(inputs)
                    
                    all_features.append(features.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
            
            features = np.vstack(all_features)
            predictions = np.vstack(all_predictions)
            
        except Exception as e:
            logging.warning(f"深度特征提取失败，使用原始数据: {e}")
            all_features = []
            all_predictions = []
            
            with torch.no_grad():
                for inputs, _ in tqdm(train_loader, desc="使用原始数据"):
                    inputs = inputs.to(device)
                    predictions = net(inputs)
                    
                    all_features.append(inputs.view(inputs.size(0), -1).cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
            
            features = np.vstack(all_features)
            predictions = np.vstack(all_predictions)

        # 4. 计算不确定性分数
        if task_type == 'classification':
            # 分类任务：使用预测熵
            probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
            uncertainty_scores = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        else:
            # 回归任务：使用预测方差
            uncertainty_scores = np.var(predictions, axis=1)

        # 5. 计算多样性分数（基于特征聚类）
        logging.info("计算多样性分数...")
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_
        
        # 计算每个样本到其聚类中心的距离作为多样性分数
        diversity_scores = np.zeros(len(features))
        for i, (feature, label) in enumerate(zip(features, cluster_labels)):
            diversity_scores[i] = np.linalg.norm(feature - cluster_centers[label])

        # 6. 组合不确定性和多样性分数
        # 归一化分数
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)
        diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        
        # 组合分数
        combined_scores = (1 - diversity_weight) * uncertainty_scores + diversity_weight * diversity_scores

        # 7. 选择分数最高的样本
        selected_indices = np.argsort(combined_scores)[-core_size:].tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"不确定性分数范围: [{uncertainty_scores.min():.6f}, {uncertainty_scores.max():.6f}]")
        logging.info(f"多样性分数范围: [{diversity_scores.min():.6f}, {diversity_scores.max():.6f}]")

        # 8. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 9. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 