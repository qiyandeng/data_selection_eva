import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

def compute_tracin_scores_simple(model, train_loader, val_loader, device, num_epochs=3):
    """简化的 TracIn 分数计算"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # 计算验证集梯度
    val_gradients = []
    for inputs, labels in tqdm(val_loader, desc="计算验证集梯度"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 收集梯度
        grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.view(-1))
        
        if grad_vec:
            val_gradients.append(torch.cat(grad_vec).detach().cpu())
    
    # 计算训练集每个样本的影响分数
    influence_scores = []
    
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc="计算 TracIn 分数")):
        inputs, labels = inputs.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 收集训练样本梯度
        train_grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                train_grad_vec.append(param.grad.view(-1))
        
        if train_grad_vec:
            train_grad = torch.cat(train_grad_vec).detach().cpu()
            
            # 计算与验证集梯度的内积
            influence = 0.0
            for val_grad in val_gradients:
                influence += torch.dot(train_grad, val_grad).item()
            
            influence_scores.append(influence / len(val_gradients))
        else:
            influence_scores.append(0.0)
    
    return np.array(influence_scores)

class TracInSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               val_set=None,
               num_epochs: int = 3,
               learning_rate: float = 0.01
               ) -> DataLoader:
        """
        使用 TracIn 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于影响函数计算的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - val_set:         验证集（如果为 None，则使用训练集的一部分）
          - num_epochs:      预训练轮数
          - learning_rate:   学习率

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        
        # 2. 计算目标选择数量
        n = len(train_set)
        core_size = int(n * selection_fraction)
        logging.info(f"TracInSelection: total={n}, core={core_size}")

        # 3. 准备验证集
        if val_set is None:
            # 如果没有提供验证集，使用训练集的前10%作为验证集
            val_size = max(10, int(0.1 * n))
            val_indices = list(range(val_size))
            val_set = Subset(train_set, val_indices)
            logging.info(f"使用训练集前 {val_size} 个样本作为验证集")
        
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        # 4. 简单预训练模型
        logging.info("预训练模型...")
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_loader_pretrain = DataLoader(train_set, batch_size=128, shuffle=True)
        
        for epoch in tqdm(range(num_epochs), desc="预训练"):
            net.train()
            for inputs, labels in train_loader_pretrain:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

        # 5. 计算 TracIn 分数
        logging.info("计算 TracIn 影响分数...")
        train_loader_single = DataLoader(train_set, batch_size=1, shuffle=False)
        
        influence_scores = compute_tracin_scores_simple(
            net, train_loader_single, val_loader, device, num_epochs
        )

        # 6. 选择影响分数最高的样本
        selected_indices = np.argsort(influence_scores)[-core_size:].tolist()
        
        logging.info(f"选择了 {len(selected_indices)} 个样本")
        logging.info(f"影响分数范围: [{np.min(influence_scores):.6f}, {np.max(influence_scores):.6f}]")

        # 7. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 8. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 