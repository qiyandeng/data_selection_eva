# selection/craig.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def euclidean_dist_pair_np(x):
    """计算欧氏距离矩阵"""
    (rowx, colx) = x.shape
    xy = np.dot(x, x.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
    return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))

class FacilityLocation:
    """简化的 Facility Location 子模函数"""
    def __init__(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix
        self.n = similarity_matrix.shape[0]
        self.cur_max = np.zeros(self.n, dtype=np.float32)

    def calc_gain(self, idx_gain, selected):
        """计算增益"""
        gains = np.maximum(0., self.similarity_matrix[:, idx_gain] - self.cur_max.reshape(-1, 1)).sum(axis=0)
        return gains

    def update_state(self, new_selection):
        """更新状态"""
        self.cur_max = np.maximum(self.cur_max, np.max(self.similarity_matrix[:, new_selection], axis=1))

class LazyGreedy:
    """简化的 Lazy Greedy 优化器"""
    def __init__(self, budget):
        self.budget = budget

    def select(self, gain_function, update_function, n):
        """贪心选择"""
        selected = []
        remaining = list(range(n))
        
        for _ in tqdm(range(self.budget), desc="CRAIG 贪心选择"):
            if not remaining:
                break
                
            # 计算所有剩余样本的增益
            gains = gain_function(remaining, selected)
            
            # 选择增益最大的样本
            best_idx_in_remaining = np.argmax(gains)
            best_idx = remaining[best_idx_in_remaining]
            
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # 更新状态
            update_function([best_idx])
        
        return selected

def compute_gradients_simple(model, data_loader, device):
    """简化的梯度计算"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    all_gradients = []
    
    # 获取最后一层参数
    last_layer = None
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            last_layer = param
            break
    
    if last_layer is None:
        # 如果找不到最后一层，使用所有参数
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [last_layer]
    
    for inputs, targets in tqdm(data_loader, desc="计算梯度"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            x = inputs[i].unsqueeze(0)
            y = targets[i].unsqueeze(0)
            
            outputs = model(x)
            loss = criterion(outputs, y).sum()
            
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            flat_grads = torch.cat([g.view(-1) for g in grads], dim=0)
            all_gradients.append(flat_grads.cpu().numpy())
            
            # 清理显存
            del x, y, outputs, loss, grads, flat_grads
            torch.cuda.empty_cache()
    
    return np.array(all_gradients)

class CraigSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device
               ) -> DataLoader:
        """
        使用 CRAIG 算法选择数据点构建核心集（简化版本）

        参数：
          - model:           已实例化且将用于梯度计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()

        # 2. 计算梯度
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"CraigSelection: total={n}, core={core_size}")

        train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
        
        logging.info("计算样本梯度...")
        all_gradients = compute_gradients_simple(net, train_loader, device)

        # 3. 计算相似度矩阵
        logging.info("计算相似度矩阵...")
        similarity_matrix = euclidean_dist_pair_np(all_gradients)
        
        # 转换为相似度（距离越小，相似度越大）
        similarity_matrix = -1.0 * similarity_matrix
        similarity_matrix -= (np.min(similarity_matrix) - 1e-3)

        # 4. 使用 Facility Location + Lazy Greedy 选择
        logging.info("执行子模优化...")
        facility_location = FacilityLocation(similarity_matrix)
        lazy_greedy = LazyGreedy(core_size)
        
        selected_indices = lazy_greedy.select(
            gain_function=facility_location.calc_gain,
            update_function=facility_location.update_state,
            n=n
        )

        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 5. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 6. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 