import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from selection.data_selection import DataSelection
from nets.utils import EmbeddingExtractor
from methods.methods_utils.euclidean import euclidean_dist

def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None):
    """K-Center Greedy 算法实现"""
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]

class KCenterGreedySelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               num_classes: int = None,
               balance: bool = True,
               random_seed: int = 42
               ) -> DataLoader:
        """
        使用 K-Center Greedy 算法选择数据点构建核心集

        参数：
          - model:           已实例化且将用于 embedding 提取的 nn.Module
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set:       支持 __getitem__ 返回 (x, y) 的 Dataset
          - task_type:       'classification' 或 'regression'
          - device:          torch.device
          - num_classes:     类别数（分类任务需要）
          - balance:         是否进行平衡的类别选择
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

        # 3. K-Center Greedy 选择
        n = data.shape[0]
        core_set_size = int(n * selection_fraction)
        logging.info(f"KCenterGreedySelection: total={n}, core={core_set_size}")

        if balance and task_type == 'classification' and num_classes is not None:
            logging.info("执行平衡的 K-Center Greedy 选择")
            selected_indices = []
            for c in range(num_classes):
                class_indices = np.where(labels == c)[0]
                if len(class_indices) == 0:
                    continue
                    
                class_data = torch.tensor(data[class_indices], dtype=torch.float32, device=device)
                class_core_set_size = max(1, int(core_set_size * (np.sum(labels == c) / n)))

                # 使用 k-Center Greedy 选择样本
                selected_class_indices = k_center_greedy(
                    class_data,
                    budget=class_core_set_size,
                    metric=euclidean_dist,
                    device=device,
                    random_seed=random_seed,
                    index=class_indices,
                    already_selected=[]
                )
                selected_indices.extend(selected_class_indices.tolist())

            selected_indices = np.array(selected_indices)
        else:
            logging.info("执行全局 K-Center Greedy 选择")
            matrix = torch.tensor(data, dtype=torch.float32, device=device)
            selected_indices = k_center_greedy(
                matrix,
                budget=core_set_size,
                metric=euclidean_dist,
                device=device,
                random_seed=random_seed,
                index=np.arange(n),
                already_selected=[]
            )

        logging.info(f"选择了 {len(selected_indices)} 个样本")

        # 4. （分类）打印标签分布
        if task_type == 'classification':
            from collections import Counter
            core_lbs = [int(train_set[i][1]) for i in selected_indices]
            logging.info(f"核心集标签分布: {Counter(core_lbs)}")

        # 5. 构造并返回 DataLoader
        subset = Subset(train_set, selected_indices)
        return DataLoader(subset, batch_size=128, shuffle=True) 