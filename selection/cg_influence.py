# selection/cg_influence.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_gradient(model, inputs, targets, criterion, device):
    """计算单个样本的梯度"""
    model.eval()
    inputs, targets = inputs.to(device), targets.to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return torch.cat([g.view(-1) for g in grads])

def conjugate_gradient_simple(A_func, b, x0=None, max_iter=50, tol=1e-6):
    """简化的共轭梯度算法"""
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    
    r = b - A_func(x)
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        Ap = A_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        
        if torch.sqrt(rsnew) < tol:
            break
            
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x

class CGInfluenceSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               val_set=None,
               max_cg_iter: int = 30,
               damping: float = 0.01
               ) -> DataLoader:
        """
        使用共轭梯度影响函数算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于影响函数计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - val_set:         验证集（如果为 None，则使用训练集的一部分）
          - max_cg_iter:     共轭梯度最大迭代次数
          - damping:         阻尼参数

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"CGInfluenceSelection: total={n}, core={core_size}")

        # 3. 准备验证集
        if val_set is None:
            val_size = max(10, int(0.1 * n))
            val_indices = list(range(val_size))
            val_set = Subset(train_set, val_indices)
            logging.info(f"使用训练集前 {val_size} 个样本作为验证集")

        criterion = nn.CrossEntropyLoss()

        # 4. 计算验证集梯度
        logging.info("计算验证集梯度...")
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        val_gradients = []
        
        for inputs, targets in tqdm(val_loader, desc="计算验证集梯度"):
            grad = compute_gradient(net, inputs, targets, criterion, device)
            val_gradients.append(grad)
        
        # 平均验证集梯度
        avg_val_grad = torch.stack(val_gradients).mean(dim=0)

        # 5. 定义 Hessian-vector 乘积函数（简化版）
        def hvp_func(v):
            # 简化的 Hessian-vector 乘积计算
            train_loader_sample = DataLoader(train_set, batch_size=32, shuffle=True)
            
            hvp_result = torch.zeros_like(v)
            count = 0
            
            for inputs, targets in train_loader_sample:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                
                grads = torch.autograd.grad(loss, net.parameters(), create_graph=True)
                flat_grads = torch.cat([g.view(-1) for g in grads])
                
                grad_v_prod = torch.dot(flat_grads, v)
                hvp = torch.autograd.grad(grad_v_prod, net.parameters(), retain_graph=True)
                hvp_flat = torch.cat([h.view(-1) for h in hvp])
                
                hvp_result += hvp_flat
                count += 1
                
                if count >= 3:  # 限制计算量
                    break
            
            return hvp_result / count + damping * v

        # 6. 使用共轭梯度求解影响函数
        logging.info("使用共轭梯度求解影响函数...")
        s_test = conjugate_gradient_simple(hvp_func, avg_val_grad, max_iter=max_cg_iter)

        # 7. 计算每个训练样本的影响分数
        logging.info("计算影响分数...")
        influence_scores = []
        
        train_loader_single = DataLoader(train_set, batch_size=1, shuffle=False)
        
        for idx, (inputs, targets) in enumerate(tqdm(train_loader_single, desc="计算影响分数")):
            train_grad = compute_gradient(net, inputs, targets, criterion, device)
            influence = -torch.dot(train_grad, s_test).item()
            influence_scores.append((idx, influence))

        # 8. 选择影响分数最高的样本
        influence_scores.sort(key=lambda x: -x[1])
        selected_indices = [idx for idx, score in influence_scores[:core_size]]
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        scores = [score for _, score in influence_scores[:core_size]]
        logging.info(f"影响分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 9. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 10. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 