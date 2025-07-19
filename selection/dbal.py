# selection/dbal.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def enable_dropout(model):
    """启用模型中的 dropout 层"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def compute_uncertainty_with_dropout(model, data_loader, device, num_samples=10):
    """使用 Monte Carlo Dropout 计算不确定性"""
    model.eval()
    enable_dropout(model)  # 启用 dropout
    
    all_predictions = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="计算 DBAL 不确定性"):
            inputs = inputs.to(device)
            
            # 多次前向传播
            predictions = []
            for _ in range(num_samples):
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
            
            # 计算预测的均值和方差
            predictions = np.array(predictions)  # [num_samples, batch_size, num_classes]
            mean_pred = np.mean(predictions, axis=0)  # [batch_size, num_classes]
            var_pred = np.var(predictions, axis=0)   # [batch_size, num_classes]
            
            # 计算不确定性（使用预测熵和方差）
            entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
            variance = np.sum(var_pred, axis=1)
            uncertainty = entropy + variance  # 组合不确定性
            
            all_predictions.extend(uncertainty)
    
    return np.array(all_predictions)

class DBALSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               num_mc_samples: int = 10
               ) -> DataLoader:
        """
        使用 DBAL (Deep Bayesian Active Learning) 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于不确定性计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - num_mc_samples:  Monte Carlo 采样次数

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"DBALSelection: total={n}, core={core_size}")

        # 3. 计算不确定性分数
        logging.info("使用 Monte Carlo Dropout 计算不确定性...")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
        
        if task_type == 'classification':
            uncertainty_scores = compute_uncertainty_with_dropout(
                net, train_loader, device, num_samples=num_mc_samples
            )
        else:
            # 回归任务的简化实现
            uncertainty_scores = []
            net.eval()
            enable_dropout(net)
            
            with torch.no_grad():
                for inputs, _ in tqdm(train_loader, desc="计算回归不确定性"):
                    inputs = inputs.to(device)
                    
                    predictions = []
                    for _ in range(num_mc_samples):
                        outputs = net(inputs)
                        predictions.append(outputs.cpu().numpy())
                    
                    predictions = np.array(predictions)
                    variance = np.var(predictions, axis=0).mean(axis=1)
                    uncertainty_scores.extend(variance)
            
            uncertainty_scores = np.array(uncertainty_scores)

        # 4. 选择不确定性最高的样本
        selected_indices = np.argsort(uncertainty_scores)[-core_size:].tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"不确定性分数范围: [{np.min(uncertainty_scores):.6f}, {np.max(uncertainty_scores):.6f}]")

        # 5. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 6. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 