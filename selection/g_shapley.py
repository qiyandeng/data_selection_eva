import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import copy

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_shapley_value(model, train_dataset, device, num_iterations=50, learning_rate=0.001, batch_size=32):
    """使用梯度近似方法计算每个样本的 Shapley 值"""
    num_samples = len(train_dataset)
    shapley_values = torch.zeros(num_samples, device=device)
    T = num_iterations

    criterion = nn.CrossEntropyLoss()

    for t in tqdm(range(1, T+1), desc="计算 Shapley 值"):
        # 随机排列数据点
        permutation = torch.randperm(num_samples)
        
        # 创建模型副本
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        model_copy.train()
        
        optimizer = optim.SGD(model_copy.parameters(), lr=learning_rate)

        # 设置 BatchNorm 层为评估模式
        def set_bn_eval(m):
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
        model_copy.apply(set_bn_eval)

        # 初始化性能
        model_copy.eval()
        with torch.no_grad():
            val_loss = 0
            count = 0
            for i in range(min(100, num_samples)):  # 使用小样本评估性能
                input_i, label_i = train_dataset[i]
                input_i = input_i.unsqueeze(0).to(device)
                label_i = torch.tensor([label_i]).to(device)
                output = model_copy(input_i)
                loss = criterion(output, label_i)
                val_loss += loss.item()
                count += 1
            v_prev = -val_loss / count if count > 0 else 0

        model_copy.train()
        model_copy.apply(set_bn_eval)

        # 将 permutation 分成批次
        batch_indices_list = [permutation[i:i+batch_size] for i in range(0, num_samples, batch_size)]

        for batch_indices in batch_indices_list:
            # 获取当前批次的数据
            inputs = []
            labels = []
            for idx in batch_indices:
                input_j, label_j = train_dataset[idx]
                inputs.append(input_j)
                labels.append(label_j)
            
            if len(inputs) == 0:
                continue
                
            batch_inputs = torch.stack(inputs).to(device)
            batch_labels = torch.tensor(labels).to(device)

            # 梯度更新参数
            optimizer.zero_grad()
            output = model_copy(batch_inputs)

            # 检查模型输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                continue

            loss = criterion(output, batch_labels)

            # 检查损失
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=5.0)
            optimizer.step()

            # 计算当前性能
            model_copy.eval()
            with torch.no_grad():
                val_loss = 0
                count = 0
                for i in range(min(100, num_samples)):
                    input_i, label_i = train_dataset[i]
                    input_i = input_i.unsqueeze(0).to(device)
                    label_i = torch.tensor([label_i]).to(device)
                    output = model_copy(input_i)
                    loss = criterion(output, label_i)
                    val_loss += loss.item()
                    count += 1
                v_current = -val_loss / count if count > 0 else 0

            model_copy.train()
            model_copy.apply(set_bn_eval)

            # 计算边际贡献
            marginal_contribution = v_current - v_prev

            # 更新当前批次中数据点的 Shapley 值
            for idx in batch_indices:
                shapley_values[idx] = ((t - 1) / t) * shapley_values[idx] + (1 / t) * (marginal_contribution / len(batch_indices))

            v_prev = v_current

    # 归一化 Shapley 值
    shapley_min = shapley_values.min()
    shapley_max = shapley_values.max()
    if shapley_max > shapley_min:
        shapley_values = (shapley_values - shapley_min) / (shapley_max - shapley_min + 1e-8)

    return shapley_values.cpu()

class GShapleySelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               num_iterations: int = 20,
               learning_rate: float = 0.001,
               batch_size: int = 32
               ) -> DataLoader:
        """
        使用梯度 Shapley 值方法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于 Shapley 值计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - num_iterations:  Shapley 值计算迭代次数
          - learning_rate:   学习率
          - batch_size:      批处理大小

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        
        # 2. 计算 Shapley 值
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"GShapleySelection: total={n}, core={core_size}")
        logging.info(f"计算 Shapley 值，迭代次数: {num_iterations}")

        shapley_values = compute_shapley_value(
            net, train_set, device, 
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            batch_size=batch_size
        )

        # 3. 选择 Shapley 值最高的样本
        selected_indices = torch.topk(shapley_values, core_size).indices.tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 4. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 5. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 