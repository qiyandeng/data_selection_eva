# selection/lissa.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def grad_z(z, t, model, criterion, device):
    """计算单个样本的梯度"""
    model.eval()
    z, t = z.to(device), t.to(device)
    y = model(z)
    loss = criterion(y, t)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    return grads

def hvp(loss, params, v):
    """计算 Hessian-vector 乘积"""
    grad1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    elem_prod = sum(torch.sum(g1 * v1) for g1, v1 in zip(grad1, v))
    hv = torch.autograd.grad(elem_prod, params, retain_graph=True)
    return hv

def lissa_simple(model, train_loader, grad_z_test, criterion, device, recursion_depth=50, damping=0.01, scale=25.0):
    """简化的 LiSSA 算法实现"""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    h_estimate = [torch.zeros_like(p) for p in params]

    for i in tqdm(range(recursion_depth), desc="LiSSA 迭代"):
        for z, t in train_loader:
            z, t = z.to(device), t.to(device)
            y = model(z)
            loss = criterion(y, t)
            hv = hvp(loss, params, h_estimate)

            hv = [h / len(z) for h in hv]
            h_estimate = [
                _v + (1 - damping) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(grad_z_test, h_estimate, hv)
            ]
            h_estimate = [h.detach().clone() for h in h_estimate]
            break  # 仅采样一个批次

    return h_estimate

class LiSSASelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               val_set=None,
               recursion_depth: int = 30,
               damping: float = 0.01,
               scale: float = 25.0,
               num_test_samples: int = 5
               ) -> DataLoader:
        """
        使用 LiSSA 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于影响函数计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - val_set:         验证集（如果为 None，则使用训练集的一部分）
          - recursion_depth: LiSSA 递归深度
          - damping:         阻尼参数
          - scale:           缩放参数
          - num_test_samples: 测试样本数量

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"LiSSASelection: total={n}, core={core_size}")

        # 3. 准备测试集
        if val_set is None:
            # 如果没有提供验证集，使用训练集的前几个样本作为测试集
            test_indices = list(range(min(num_test_samples, n)))
            test_set = Subset(train_set, test_indices)
            logging.info(f"使用训练集前 {len(test_indices)} 个样本作为测试集")
        else:
            test_set = val_set

        test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
        criterion = nn.CrossEntropyLoss()

        # 4. 计算测试样本的 s_test
        logging.info("计算测试样本的 LiSSA 向量...")
        s_test_samples = []
        
        for i, (z_test, t_test) in enumerate(test_loader):
            if i >= num_test_samples:
                break
                
            z_test, t_test = z_test.to(device), t_test.to(device)
            grad_z_test = grad_z(z_test, t_test, net, criterion, device)
            
            train_loader_sample = DataLoader(train_set, batch_size=32, shuffle=True)
            s_test_sample = lissa_simple(
                net, train_loader_sample, grad_z_test, criterion, device,
                recursion_depth=recursion_depth, damping=damping, scale=scale
            )
            s_test_samples.append(s_test_sample)

        # 5. 计算每个训练样本的影响分数
        logging.info("计算影响分数...")
        influence_scores = []
        
        train_loader_single = DataLoader(train_set, batch_size=1, shuffle=False)
        
        for idx, (z, t) in enumerate(tqdm(train_loader_single, desc="计算影响分数")):
            z, t = z.to(device), t.to(device)
            grad_z_vec = grad_z(z, t, net, criterion, device)
            
            influence = 0.0
            for s_test_vec in s_test_samples:
                influence += -sum([
                    torch.sum(k * j).item() for k, j in zip(grad_z_vec, s_test_vec)
                ]) / len(train_set)
            
            influence_scores.append((idx, influence))

        # 6. 选择影响分数最高的样本
        influence_scores.sort(key=lambda x: -x[1])
        selected_indices = [idx for idx, score in influence_scores[:core_size]]
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        scores = [score for _, score in influence_scores[:core_size]]
        logging.info(f"影响分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 8. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 