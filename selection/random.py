import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class RandomSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               epsilon: float = 0.1,
               random_seed: int = 42
               ) -> DataLoader:
        """
        随机选择数据点构建核心集

        参数：
          - model:           已实例化且将用于 embedding 提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - epsilon:         权重调整参数
          - random_seed:     随机种子

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

        # 3. 随机选择
        n = data.shape[0]
        core_size = int(n * selection_fraction)
        logging.info(f"RandomSelection: total={n}, core={core_size}")

        if random_seed is not None:
            np.random.seed(random_seed)
        
        representative_indices = np.random.choice(n, size=core_size, replace=False)

        # 4. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in representative_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 5. 构造并返回 DataLoader
        subset = Subset(train_set, representative_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 