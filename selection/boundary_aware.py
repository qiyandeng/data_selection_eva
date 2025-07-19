import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_uncertainty_scores(model, data_loader, device):
    """计算模型预测的不确定性分数（简化版本，使用预测熵）"""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="计算不确定性分数"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 计算预测概率的熵作为不确定性度量
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            scores.extend(entropy.cpu().numpy())
    
    return np.array(scores)

class BoundaryAwareSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               active_fraction_ratio: float = 0.1,
               min_selection_size: int = 50
               ) -> DataLoader:
        """
        使用边界感知方法选择数据点构建核心集（简化版本）
        
        这个简化版本使用预测熵作为不确定性度量，选择最不确定的样本

        参数：
          - model:           已实例化且将用于不确定性计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - active_fraction_ratio: 每轮选择的比例
          - min_selection_size: 每轮最小选择数量

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()

        # 2. 计算目标选择数量
        n = len(train_set)
        target_core_set_size = int(n * selection_fraction)
        target_core_set_size_per_selection = max(
            min_selection_size, 
            int(target_core_set_size * active_fraction_ratio)
        )
        
        logging.info(f"BoundaryAwareSelection: total={n}, target_core={target_core_set_size}")
        logging.info(f"每轮选择数量: {target_core_set_size_per_selection}")

        # 3. 主动学习选择过程
        core_indices = set()
        
        while len(core_indices) < target_core_set_size:
            # 创建当前未选择样本的数据加载器
            remaining_indices = [i for i in range(n) if i not in core_indices]
            if not remaining_indices:
                break
                
            remaining_subset = Subset(train_set, remaining_indices)
            remaining_loader = DataLoader(remaining_subset, batch_size=128, shuffle=False)
            
            # 计算不确定性分数
            logging.info("计算边界感知不确定性分数...")
            if task_type == 'classification':
                uncertainty_scores = compute_uncertainty_scores(net, remaining_loader, device)
            else:
                # 回归任务：使用预测方差作为不确定性
                uncertainty_scores = []
                with torch.no_grad():
                    for inputs, _ in tqdm(remaining_loader, desc="计算预测方差"):
                        inputs = inputs.to(device)
                        outputs = net(inputs)
                        variance = torch.var(outputs, dim=1).cpu().numpy()
                        uncertainty_scores.extend(variance)
                uncertainty_scores = np.array(uncertainty_scores)

            # 4. 选择不确定性最高的样本
            sorted_indices = np.argsort(uncertainty_scores)[::-1]  # 降序排列
            
            # 选择新样本
            remaining_to_select = min(
                target_core_set_size_per_selection,
                target_core_set_size - len(core_indices),
                len(remaining_indices)
            )
            
            new_selection_local = sorted_indices[:remaining_to_select]
            new_selection_global = [remaining_indices[i] for i in new_selection_local]
            core_indices.update(new_selection_global)
            
            logging.info(f"当前已选择: {len(core_indices)}/{target_core_set_size} 个样本")

        selected_indices = list(core_indices)
        logging.info(f"最终选择了 {len(selected_indices)} 个样本")

        # 5. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 6. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 