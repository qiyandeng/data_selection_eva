import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class FacilityLocation:
    def __init__(self, train_gradients, val_loss_gradient, device='cpu', batch_size=100):
        self.train_gradients = train_gradients  # 保留在 CPU
        self.val_loss_gradient = val_loss_gradient  # 验证集损失梯度
        self.device = device
        self.batch_size = batch_size  # 分批处理大小

    def calc_gain(self, idx_gain):
        idx_gain = np.array(idx_gain)
        valid_indices = idx_gain[idx_gain < self.train_gradients.size(0)]  # 检查索引范围

        # 分批计算增益
        gains = []
        for i in range(0, len(valid_indices), self.batch_size):
            batch_indices = valid_indices[i:i + self.batch_size]
            train_subset_gradient = self.train_gradients[batch_indices].to(self.device)
            batch_gain = torch.matmul(train_subset_gradient, self.val_loss_gradient)
            gains.append(batch_gain.cpu().numpy())

            # 清理显存
            del train_subset_gradient, batch_gain
            torch.cuda.empty_cache()

        return np.concatenate(gains)

def calc_gradient(data_loader, model, criterion, device):
    """计算每个样本的梯度"""
    model.eval()
    all_gradients = []
    total_samples = len(data_loader.dataset)

    # 提取需要梯度的参数列表
    params = [param for param in model.parameters() if param.requires_grad]

    for inputs, targets in tqdm(data_loader, desc="计算梯度"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)

        for i in range(batch_size):
            x = inputs[i].unsqueeze(0)  # [1, C, H, W]
            y = targets[i].unsqueeze(0)  # [1]
            outputs = model(x)
            loss = criterion(outputs, y)
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            flat_grads = torch.cat([g.view(-1) for g in grads], dim=0)  # [total_param_size]
            all_gradients.append(flat_grads.cpu())

            # 清理显存
            del x, y, outputs, loss, grads, flat_grads
            torch.cuda.empty_cache()

    gradients = torch.stack(all_gradients, dim=0)  # [total_samples, total_param_size]

    # 检查是否缺失样本
    if gradients.size(0) != total_samples:
        missing_samples = total_samples - gradients.size(0)
        logging.warning(f"缺失 {missing_samples} 个样本的梯度计算")
        # 用零填充缺失样本
        gradients = torch.cat([gradients, torch.zeros(missing_samples, gradients.size(1))], dim=0)

    return gradients

class GlisterSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               val_set=None,
               batch_size: int = 100
               ) -> DataLoader:
        """
        使用 GLISTER 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于梯度计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - val_set:         验证集（如果为None，使用训练集的一部分）
          - batch_size:      批处理大小

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()

        # 2. 准备数据加载器
        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
        
        if val_set is None:
            # 如果没有验证集，使用训练集的一小部分作为验证集
            val_size = min(1000, len(train_set) // 10)
            val_indices = np.random.choice(len(train_set), val_size, replace=False)
            val_subset = Subset(train_set, val_indices)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        else:
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

        # 3. 计算损失函数
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # 4. 计算训练集梯度
        logging.info("计算训练集梯度...")
        train_gradients = calc_gradient(train_loader, net, criterion, device)

        # 5. 计算验证集损失梯度
        logging.info("计算验证集损失梯度...")
        val_gradients = calc_gradient(val_loader, net, criterion, device)
        val_loss_gradient = val_gradients.mean(dim=0).to(device)

        # 6. GLISTER 选择
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"GlisterSelection: total={n}, core={core_size}")

        # 初始化 FacilityLocation
        facility_location = FacilityLocation(
            train_gradients=train_gradients,
            val_loss_gradient=val_loss_gradient,
            device=device,
            batch_size=batch_size
        )

        # 贪心选择
        selected_indices = []
        remaining_indices = list(range(n))

        for i in tqdm(range(core_size), desc="GLISTER 选择"):
            if len(remaining_indices) == 0:
                break
                
            # 计算增益
            gains = facility_location.calc_gain(remaining_indices)
            
            # 选择增益最大的样本
            best_idx_in_remaining = np.argmax(gains)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        selected_indices = np.array(selected_indices)
        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 7. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 8. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 