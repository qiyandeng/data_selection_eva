import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor
from methods.methods_utils.euclidean import euclidean_dist

class HerdingSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               num_classes: int = None,
               balance: bool = True
               ) -> DataLoader:
        """
        使用 Herding 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于 embedding 提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - num_classes:     类别数（分类任务需要）
          - balance:         是否进行平衡的类别选择

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

        # 3. Herding 选择
        n = data.shape[0]
        core_set_size = int(n * selection_fraction)
        logging.info(f"HerdingSelection: total={n}, core={core_set_size}")

        if balance and task_type == 'classification' and num_classes is not None:
            logging.info("执行平衡的 Herding 选择")
            selected_indices = []
            for c in range(num_classes):
                class_indices = np.where(labels == c)[0]
                if len(class_indices) == 0:
                    continue
                    
                class_data = torch.tensor(data[class_indices], dtype=torch.float32, device=device)
                class_mu = torch.mean(class_data, dim=0)
                selected_class_embeddings = torch.zeros_like(class_mu, device=device)
                selected_class_indices = np.zeros(len(class_indices), dtype=bool)
                class_core_set_size = max(1, int(core_set_size * (np.sum(labels == c) / n)))

                for i in tqdm(range(class_core_set_size), desc=f"Herding 选择类别 {c}"):
                    target = (i + 1) * class_mu - selected_class_embeddings
                    unselected_mask = ~selected_class_indices
                    if np.sum(unselected_mask) == 0:
                        break
                    dist = euclidean_dist(target.view(1, -1), class_data[unselected_mask])
                    if dist.numel() == 0:
                        break
                    p = torch.argmax(dist).item()
                    unselected_indices = np.where(unselected_mask)[0]
                    actual_index = class_indices[unselected_indices[p]]
                    selected_indices.append(actual_index)
                    selected_class_indices[unselected_indices[p]] = True
                    selected_class_embeddings += class_data[unselected_indices[p]]

            selected_indices = np.array(selected_indices)
        else:
            logging.info("执行全局 Herding 选择")
            matrix = torch.tensor(data, dtype=torch.float32, device=device)
            mu = torch.mean(matrix, dim=0)
            selected_embeddings = torch.zeros_like(mu, device=device)
            representative_indices = np.zeros(n, dtype=bool)

            for i in tqdm(range(core_set_size), desc="Herding 选择"):
                target = (i + 1) * mu - selected_embeddings
                unselected_mask = ~representative_indices
                if np.sum(unselected_mask) == 0:
                    break
                dist = euclidean_dist(target.view(1, -1), matrix[unselected_mask])
                if dist.numel() == 0:
                    break
                p = torch.argmax(dist).item()
                actual_index = np.where(unselected_mask)[0][p]
                representative_indices[actual_index] = True
                selected_embeddings += matrix[actual_index]

            selected_indices = np.where(representative_indices)[0]

        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 4. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 5. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 