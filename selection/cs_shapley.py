# selection/cs_shapley.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_class_conditional_shapley(features, labels, num_classes, num_iterations=50):
    """计算类条件 Shapley 值"""
    n = len(features)
    shapley_values = np.zeros(n)
    
    # 按类别分组
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[int(label)].append(i)
    
    for iteration in tqdm(range(num_iterations), desc="计算 CS-Shapley 值"):
        # 为每个类别计算 Shapley 值
        for class_id in range(num_classes):
            if class_id not in class_indices:
                continue
                
            indices = class_indices[class_id]
            if len(indices) <= 1:
                continue
            
            # 随机排列当前类别的样本
            permutation = np.random.permutation(indices)
            
            # 计算类内相似度矩阵
            class_features = features[indices]
            similarity_matrix = np.dot(class_features, class_features.T)
            
            # 计算边际贡献
            for j, idx in enumerate(permutation):
                if j == 0:
                    marginal_contribution = similarity_matrix[indices.index(idx), indices.index(idx)]
                else:
                    prev_indices = [indices.index(pi) for pi in permutation[:j]]
                    current_idx = indices.index(idx)
                    marginal_contribution = (similarity_matrix[current_idx, current_idx] + 
                                           similarity_matrix[current_idx, prev_indices].sum())
                
                # 更新 Shapley 值
                shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (iteration + 1)
    
    # 归一化
    if shapley_values.max() > shapley_values.min():
        shapley_values = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min())
    
    return shapley_values

class CSShapleySelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               num_classes: int = None,
               num_iterations: int = 30,
               balance_classes: bool = True
               ) -> DataLoader:
        """
        使用 CS-Shapley (Class-conditional Shapley) 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于特征提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - num_classes:     类别数量（如果为 None，则自动推断）
          - num_iterations:  Shapley 值计算迭代次数
          - balance_classes: 是否在各类别间平衡选择

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"CSShapleySelection: total={n}, core={core_size}")

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

        # 4. 推断类别数量
        if num_classes is None:
            num_classes = len(np.unique(labels))
        
        logging.info(f"检测到 {num_classes} 个类别")

        # 5. 计算类条件 Shapley 值
        if task_type == 'classification':
            logging.info("计算类条件 Shapley 值...")
            shapley_values = compute_class_conditional_shapley(
                features, labels, num_classes, num_iterations
            )
        else:
            # 回归任务：使用简化的 Shapley 值计算
            logging.warning("回归任务使用简化的 Shapley 值计算")
            similarity_matrix = np.dot(features, features.T)
            shapley_values = np.diag(similarity_matrix)
            shapley_values = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min() + 1e-8)

        # 6. 选择样本
        if balance_classes and task_type == 'classification':
            # 平衡选择：每个类别按比例选择
            logging.info("执行平衡类别选择...")
            selected_indices = []
            
            class_indices = defaultdict(list)
            for i, label in enumerate(labels):
                class_indices[int(label)].append(i)
            
            for class_id in range(num_classes):
                if class_id not in class_indices:
                    continue
                
                indices = class_indices[class_id]
                class_core_size = max(1, int(len(indices) * selection_fraction))
                class_shapley = shapley_values[indices]
                
                # 选择该类别中 Shapley 值最高的样本
                top_indices = np.argsort(class_shapley)[-class_core_size:]
                selected_indices.extend([indices[i] for i in top_indices])
            
            # 如果选择的样本数不够，补充剩余的高 Shapley 值样本
            if len(selected_indices) < core_size:
                remaining_indices = [i for i in range(n) if i not in selected_indices]
                remaining_shapley = shapley_values[remaining_indices]
                additional_count = core_size - len(selected_indices)
                additional_indices = np.argsort(remaining_shapley)[-additional_count:]
                selected_indices.extend([remaining_indices[i] for i in additional_indices])
        else:
            # 直接选择 Shapley 值最高的样本
            selected_indices = np.argsort(shapley_values)[-core_size:].tolist()

        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"Shapley 值范围: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")

        # 7. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(labels[i]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 8. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 