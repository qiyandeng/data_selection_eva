import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class SVPMaxEntropySelection(DataSelection):
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
        使用最大熵主动学习方法选择数据点构建核心集

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
        
        logging.info(f"SVPMaxEntropySelection: total={n}, target_core={target_core_set_size}")
        logging.info(f"每轮选择数量: {target_core_set_size_per_selection}")

        # 3. 主动学习选择过程
        core_indices = set()
        
        while len(core_indices) < target_core_set_size:
            # 计算不确定性分数
            train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
            scores, indices = [], []

            with torch.no_grad():
                for i, (datas, targets) in enumerate(tqdm(train_loader, desc="计算不确定性分数")):
                    datas = datas.to(device)
                    outputs = net(datas)
                    
                    if task_type == 'classification':
                        # 计算熵分数
                        preds = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        entropy_scores = -np.sum(preds * np.log(preds + 1e-6), axis=1)
                        scores.extend(entropy_scores)
                    else:
                        # 回归任务：使用预测方差作为不确定性
                        variance_scores = torch.var(outputs, dim=1).cpu().numpy()
                        scores.extend(variance_scores)
                    
                    indices.extend(range(i * 128, min((i + 1) * 128, len(train_set))))

            # 4. 选择不确定性最高的样本
            scores = np.array(scores)
            sorted_indices = np.argsort(scores)[::-1]  # 降序排列，选择不确定性最高的
            
            # 过滤掉已经选择的样本
            filtered_indices = [idx for idx in sorted_indices if idx not in core_indices]
            
            # 选择新样本
            remaining_to_select = min(
                target_core_set_size_per_selection,
                target_core_set_size - len(core_indices),
                len(filtered_indices)
            )
            
            new_selection = filtered_indices[:remaining_to_select]
            core_indices.update(new_selection)
            
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