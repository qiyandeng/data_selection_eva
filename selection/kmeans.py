# methods/kmeans.py
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from methods.data_selection import DataSelection
from nets.utils import EmbeddingExtractor

class KMeansSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               k: int,
               train_set,
               task_type: str,
               device: torch.device
               ) -> DataLoader:
        """
        使用传入的 nn.Module 实例提取 embedding，再做 MiniBatchKMeans 聚类选点；
        若 embedding 提取失败，则回退到原始输入 flatten 后聚类。

        参数：
          - model:           已实例化且将用于 embedding 提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - k:               聚类簇数
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device

        返回：
          - DataLoader(subset)
        """
        # 1. 准备模型
        net = model.to(device)
        net.eval()

        ##############################################################
        ###############  embedding提取根据具体任务确定    ###############
        ##############################################################
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

        # 3. 聚类相关计算
        n = data.shape[0]
        core_size = int(n * selection_fraction)
        logging.info(f"KMeansSelection: total={n}, core={core_size}, clusters={k}")

        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096)
        for _ in tqdm(range(10), desc="MiniBatch 聚类"):
            kmeans.partial_fit(data)
        centers = kmeans.cluster_centers_
        assign = kmeans.labels_

        # 4. 每簇内选离中心最近的 per_cluster 个点
        reps = []
        per_cluster = max(1, core_size // k)
        for c in range(k):
            idxs = np.where(assign == c)[0]
            dists = np.linalg.norm(data[idxs] - centers[c], axis=1)
            chosen = idxs[np.argsort(dists)[:per_cluster]]
            reps.extend(chosen.tolist())

        # 5. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in reps]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 6. 构造并返回 DataLoader
        subset = Subset(train_set, reps)
        return DataLoader(subset, batch_size=128, shuffle=True)
