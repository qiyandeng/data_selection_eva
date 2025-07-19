import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class QuadTreeNode:
    def __init__(self, point_indices, bounds, depth=0):
        self.point_indices = point_indices
        self.bounds = bounds  # [(min_dim0, max_dim0), (min_dim1, max_dim1), ...]
        self.children = []
        self.depth = depth

    def subdivide(self, data, max_depth, min_points=10):
        if self.depth >= max_depth or len(self.point_indices) <= min_points:
            return

        # 选择方差最大的维度进行划分
        variances = np.var(data[self.point_indices], axis=0)
        split_dim = np.argmax(variances)
        split_value = np.median(data[self.point_indices, split_dim])

        # 划分点集
        left_mask = data[self.point_indices, split_dim] <= split_value
        right_mask = data[self.point_indices, split_dim] > split_value
        left_indices = self.point_indices[left_mask]
        right_indices = self.point_indices[right_mask]

        # 确保子节点分布均匀，避免单侧分裂
        if len(left_indices) < min_points or len(right_indices) < min_points:
            return

        # 定义子节点的边界
        left_bounds = self.bounds.copy()
        left_bounds[split_dim] = (self.bounds[split_dim][0], split_value)
        right_bounds = self.bounds.copy()
        right_bounds[split_dim] = (split_value, self.bounds[split_dim][1])

        # 创建子节点
        left_child = QuadTreeNode(left_indices, left_bounds, self.depth + 1)
        right_child = QuadTreeNode(right_indices, right_bounds, self.depth + 1)
        self.children = [left_child, right_child]

def build_quadtree(data, max_depth=15, min_points=10):
    """高效地构建四叉树，以递归方式划分数据空间"""
    initial_bounds = [(np.min(data[:, dim]), np.max(data[:, dim])) for dim in range(data.shape[1])]
    root = QuadTreeNode(np.array(range(data.shape[0])), initial_bounds, depth=0)
    stack = [root]

    while stack:
        node = stack.pop()
        node.subdivide(data, max_depth, min_points)
        if node.children:
            stack.extend(node.children)
    return root

def select_representatives_quadtree(root, data, core_set_size, epsilon):
    """基于四叉树选择代表点，并调整权重以构建 Coreset"""
    representative_indices = []
    weights = np.ones(data.shape[0])  # 初始权重分配

    def traverse(node):
        nonlocal representative_indices, core_set_size, epsilon, weights
        if len(representative_indices) >= core_set_size:
            return
        if not node.children:
            if len(node.point_indices) == 0:
                return
            # 选择与均值距离最小的点作为代表点
            points = data[node.point_indices]
            mean = np.mean(points, axis=0)
            distances = np.linalg.norm(points - mean, axis=1)
            sorted_indices = np.argsort(distances)
            num_points_to_select = min(len(sorted_indices), core_set_size - len(representative_indices))
            if num_points_to_select > 0:
                selected_points = sorted_indices[:num_points_to_select]
                selected_indices = node.point_indices[selected_points]
                representative_indices.extend(selected_indices.tolist())
                # 调整权重
                weights[selected_indices] *= (1 + epsilon)
        else:
            for child in node.children:
                traverse(child)
                if len(representative_indices) >= core_set_size:
                    break

    traverse(root)
    return representative_indices, weights

class LowDimensionSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               epsilon: float = 0.1,
               max_depth: int = 15,
               min_points: int = 10
               ) -> DataLoader:
        """
        使用低维度四叉树方法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于 embedding 提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - epsilon:         权重调整参数
          - max_depth:       四叉树最大深度
          - min_points:      叶节点最小点数

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()

        # 2. 尝试用 embedding 提取
        try:
            extractor = EmbeddingExtractor(net).to(device)
            extractor.eval()

            loader = DataLoader(train_set, batch_size=128, shuffle=False)
            feats, labels = [], []
            with torch.no_grad():
                for x, y in tqdm(loader, desc="提取 Embedding"):
                    emb = extractor(x.to(device))
                    feats.append(emb.cpu().numpy())
                    labels.append(y.numpy())
            data = np.concatenate(feats, axis=0)
            labels = np.concatenate(labels, axis=0)

            # 清理显存
            del extractor
            torch.cuda.empty_cache()

        except Exception as e:
            logging.warning(f"Embedding 提取失败，回退 flatten：{e}")
            loader = DataLoader(train_set, batch_size=128, shuffle=False)
            feats, labels = [], []
            for x, y in tqdm(loader, desc="平展原始数据"):
                feats.append(x.view(x.size(0), -1).numpy())
                labels.append(y.numpy())
            data = np.concatenate(feats, axis=0)
            labels = np.concatenate(labels, axis=0)

        # 3. 四叉树选择
        n = data.shape[0]
        core_size = int(n * selection_fraction)
        logging.info(f"LowDimensionSelection: total={n}, core={core_size}")

        # 构建四叉树
        logging.info("构建四叉树...")
        root = build_quadtree(data, max_depth=max_depth, min_points=min_points)

        # 选择代表点
        logging.info("选择代表点...")
        representative_indices, weights = select_representatives_quadtree(root, data, core_size, epsilon)

        # 确保选择的样本数量正确
        if len(representative_indices) > core_size:
            representative_indices = representative_indices[:core_size]
        
        representative_indices = np.array(representative_indices)
        logging.info(f"选择了 {len(representative_indices)} 个样本")

        # 4. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in representative_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 5. 构造并返回 DataLoader
        subset = Subset(train_set, representative_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 