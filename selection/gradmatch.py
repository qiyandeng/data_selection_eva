import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
from scipy.linalg import solve
from scipy.optimize import nnls
from numpy.linalg import norm, lstsq

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def OrthogonalMP_REG(A, b, tol=1E-4, nnz=None, positive=False, lam=1):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit with regularization'''
    AT = A.T
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = np.zeros(n)
    resid = np.copy(b)
    normb = norm(b)
    indices = []

    for i in range(nnz):
        if norm(resid) / normb < tol:
            break
        projections = AT.dot(resid)
        if positive:
            index = np.argmax(projections)
        else:
            index = np.argmax(abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / np.dot(A_i, A_i)
        else:
            A_i = A[:, indices]
            if positive:
                # 使用非负最小二乘
                x_i, _ = nnls(A_i, b)
            else:
                # 使用带正则化的最小二乘
                x_i = lstsq(A_i, b, rcond=None)[0]
        resid = b - A_i.dot(x_i)

    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x

def compute_gradients(model, data_loader, device):
    """计算每个样本的梯度"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 提取需要梯度的参数列表
    params = [param for param in model.parameters() if param.requires_grad]
    
    all_gradients = []
    
    for inputs, targets in tqdm(data_loader, desc="计算梯度"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            x = inputs[i].unsqueeze(0)
            y = targets[i].unsqueeze(0)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            flat_grads = torch.cat([g.view(-1) for g in grads], dim=0)
            all_gradients.append(flat_grads.cpu().numpy())
            
            # 清理显存
            del x, y, outputs, loss, grads, flat_grads
            torch.cuda.empty_cache()
    
    return np.array(all_gradients)

class GradMatchSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               val_set=None,
               num_classes: int = None,
               balance: bool = True
               ) -> DataLoader:
        """
        使用 GradMatch 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于梯度计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - val_set:         验证集（如果为None，使用训练集的一部分）
          - num_classes:     类别数（分类任务需要）
          - balance:         是否进行平衡的类别选择

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

        # 3. 计算梯度
        logging.info("计算训练集梯度...")
        train_gradients = compute_gradients(net, train_loader, device)
        
        logging.info("计算验证集梯度...")
        val_gradients = compute_gradients(net, val_loader, device)
        val_gradient_mean = val_gradients.mean(axis=0)

        # 4. GradMatch 选择
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"GradMatchSelection: total={n}, core={core_size}")

        if balance and task_type == 'classification' and num_classes is not None:
            logging.info("执行平衡的 GradMatch 选择")
            selected_indices = []
            
            # 获取标签
            labels = []
            for i in range(len(train_set)):
                _, label = train_set[i]
                labels.append(int(label))
            labels = np.array(labels)
            
            for c in range(num_classes):
                class_indices = np.where(labels == c)[0]
                if len(class_indices) == 0:
                    continue
                    
                class_core_size = max(1, int(core_size * (len(class_indices) / n)))
                class_gradients = train_gradients[class_indices]
                
                # 使用 OMP 求解权重
                weights = OrthogonalMP_REG(
                    class_gradients.T, 
                    val_gradient_mean, 
                    nnz=class_core_size,
                    positive=True
                )
                
                # 选择权重最大的样本
                top_indices = np.argsort(weights)[-class_core_size:]
                selected_class_indices = class_indices[top_indices]
                selected_indices.extend(selected_class_indices.tolist())
                
            selected_indices = np.array(selected_indices)
        else:
            logging.info("执行全局 GradMatch 选择")
            # 使用 OMP 求解权重
            weights = OrthogonalMP_REG(
                train_gradients.T, 
                val_gradient_mean, 
                nnz=core_size,
                positive=True
            )
            
            # 选择权重最大的样本
            selected_indices = np.argsort(weights)[-core_size:]

        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 5. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 6. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 