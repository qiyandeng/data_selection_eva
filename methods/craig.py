import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import nets  # 确保您有一个名为 nets 的模块，包含所需的模型定义
import torch.optim.lr_scheduler as lr_scheduler
from functorch import grad, vmap
import logging
import os
import time
from collections import Counter
from scipy.sparse import csr_matrix
from torch.cuda.amp import GradScaler, autocast


# 自定义数据集返回样本权重
class WeightedSubsetDataset(Dataset):
    def __init__(self, subset, weights):
        self.subset = subset
        self.weights = weights  # 保持在 CPU 上

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, target = self.subset[idx]
        weight = self.weights[idx]  # 保持在 CPU 上
        return data, target, weight


# 自定义优化器基类
class OptimizerBase(object):
    def __init__(self, args, index, budget: int, already_selected=[]):
        self.args = args
        self.index = index

        if budget <= 0 or budget > len(index):
            raise ValueError("Illegal budget for optimizer.")

        self.n = len(index)
        self.budget = budget
        self.already_selected = already_selected


# LazyGreedy 优化器，记录每个选中样本的 Gain 值
class LazyGreedy(OptimizerBase):
    def __init__(self, args, index, budget: int, already_selected=[]):
        super(LazyGreedy, self).__init__(args, index, budget, already_selected)
        self.selected_gains = []  # 用于存储每个选中样本的 Gain 值

    def select(self, gain_function, update_state=None, **kwargs):
        assert callable(gain_function)
        if update_state is not None:
            assert callable(update_state)
        selected = np.zeros(self.n, dtype=bool)
        selected[self.already_selected] = True

        greedy_gain = np.zeros(len(self.index))
        greedy_gain[~selected] = gain_function(~selected, selected, **kwargs)
        greedy_gain[selected] = -np.inf

        for i in range(sum(selected), self.budget):
            best_gain = -np.inf
            last_max_element = -1
            while True:
                cur_max_element = greedy_gain.argmax()
                if last_max_element == cur_max_element:
                    # 选择当前最大的 Gain 元素到子集中
                    selected[cur_max_element] = True
                    selected_gain = greedy_gain[cur_max_element]
                    self.selected_gains.append(selected_gain)  # 记录 Gain 值
                    greedy_gain[cur_max_element] = -np.inf

                    if update_state is not None:
                        update_state(np.array([cur_max_element]), selected, **kwargs)
                    break
                new_gain = gain_function(np.array([cur_max_element]), selected, **kwargs)[0]
                greedy_gain[cur_max_element] = new_gain
                if new_gain >= best_gain:
                    best_gain = new_gain
                    last_max_element = cur_max_element
        return self.index[selected], self.selected_gains  # 返回选中的索引和对应的 Gain 值


# 子模函数基类
class SubmodularFunction(object):
    def __init__(self, index, similarity_kernel=None, similarity_matrix=None, already_selected=[]):
        self.index = index
        self.n = len(index)
        self.already_selected = already_selected

        assert similarity_kernel is not None or similarity_matrix is not None

        if similarity_kernel is not None:
            assert callable(similarity_kernel)
            self.similarity_kernel = self._similarity_kernel(similarity_kernel)
        else:
            assert similarity_matrix.shape[0] == self.n and similarity_matrix.shape[1] == self.n
            self.similarity_matrix = similarity_matrix
            self.similarity_kernel = lambda a, b: self.similarity_matrix[np.ix_(a, b)]

    def _similarity_kernel(self, similarity_kernel):
        return similarity_kernel


# FacilityLocation 子模函数
class FacilityLocation(SubmodularFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if len(self.already_selected) == 0:
            self.cur_max = np.zeros(self.n, dtype=np.float32)
        else:
            self.cur_max = np.max(self.similarity_kernel(np.arange(self.n), self.already_selected), axis=1)

        self.all_idx = np.ones(self.n, dtype=bool)

    def _similarity_kernel(self, similarity_kernel):
        # 初始化相似度矩阵
        self.sim_matrix = np.zeros([self.n, self.n], dtype=np.float32)
        self.if_columns_calculated = np.zeros(self.n, dtype=bool)

        def _func(a, b):
            if not np.all(self.if_columns_calculated[b]):
                if b.dtype != bool:
                    temp = ~self.all_idx
                    temp[b] = True
                    b = temp
                not_calculated = b & ~self.if_columns_calculated
                self.sim_matrix[:, not_calculated] = similarity_kernel(self.all_idx, not_calculated)
                self.if_columns_calculated[not_calculated] = True
            return self.sim_matrix[np.ix_(a, b)]
        return _func

    def calc_gain(self, idx_gain, selected, **kwargs):
        gains = np.maximum(0., self.similarity_kernel(self.all_idx, idx_gain) - self.cur_max.reshape(-1, 1)).sum(axis=1)
        return gains

    def calc_gain_batch(self, idx_gain, selected, **kwargs):
        batch_idx = ~self.all_idx
        batch_idx[0:kwargs["batch"]] = True
        gains = np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1, 1)).sum(axis=1)
        for i in range(kwargs["batch"], self.n, kwargs["batch"]):
            batch_idx = ~self.all_idx
            batch_idx[i * kwargs["batch"]:(i + 1) * kwargs["batch"]] = True
            gains += np.maximum(0., self.similarity_kernel(batch_idx, idx_gain) - self.cur_max[batch_idx].reshape(-1, 1)).sum(axis=1)
        return gains

    def update_state(self, new_selection, total_selected, **kwargs):
        self.cur_max = np.maximum(self.cur_max, np.max(self.similarity_kernel(self.all_idx, new_selection), axis=1))


# Euclidean 距离计算
def euclidean_dist_pair_np(x):
    (rowx, colx) = x.shape
    xy = np.dot(x, x.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowx, axis=1)
    return np.sqrt(np.clip(x2 + x2.T - 2. * xy, 1e-12, None))


# CRAIG 子集选择函数
# def craig_select_optimized(model, data_loader, device, subset_size, args):
#     model.eval()
#     all_gradients = []
#     all_indices = []

#     # 定义用于选择的损失函数（reduction='none'）
#     criterion = nn.CrossEntropyLoss(reduction='none').to(device)

#     # 获取最后一层参数
#     last_layer_params = list(model.parameters())[-1:]

#     # 定义对单个样本计算损失的函数
#     def compute_loss_single_sample(params, input_, target_):
#         input_ = input_.unsqueeze(0)  # [1,C,H,W]
#         target_ = target_.unsqueeze(0)  # [1]
#         outputs = model(input_)
#         loss = criterion(outputs, target_)  # [1]张量
#         return loss.sum()  # 返回单标量

#     # 对单样本梯度计算函数求梯度
#     grad_single = grad(compute_loss_single_sample)

#     # 使用 vmap 进行批处理梯度计算
#     grad_batch = vmap(grad_single, in_dims=(None, 0, 0))

#     # 计算每个样本的最后一层梯度
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         if inputs.size(0) == 0:
#             continue

#         # 使用 vmap 对整批数据计算梯度
#         try:
#             batch_gradients = grad_batch(last_layer_params, inputs, targets)
#         except Exception as e:
#             print(f"Error computing gradients: {e}, skipping batch {batch_idx}.")
#             continue
#         grad_tensor = batch_gradients[0]

#         start_idx = batch_idx * data_loader.batch_size
#         for i in range(inputs.size(0)):
#             # 对单个样本的梯度 flatten
#             flattened_grads = grad_tensor[i].view(-1).cpu().numpy()
#             all_gradients.append(flattened_grads)
#             all_indices.append(start_idx + i)

#     # 若没有任何梯度数据，直接返回空列表
#     if len(all_gradients) == 0:
#         print("No gradients computed")
#         return [], []

#     all_gradients = np.vstack(all_gradients)
#     n = len(all_gradients)

#     # 使用欧氏距离计算相似度矩阵
#     matrix = euclidean_dist_pair_np(all_gradients)

#     # 按照官方代码进行矩阵变换
#     matrix = -1.0 * matrix
#     matrix -= (np.min(matrix) - 1e-3)

#     # 使用自定义的 FacilityLocation 子模函数
#     submod_function = FacilityLocation(index=np.arange(n), similarity_matrix=matrix)

#     len_data_loader = len(data_loader)

#     # 确保 subset_size 不为 0 且不超过数据集大小
#     subset_size = int(len_data_loader * args.selection_fraction)
#     subset_size = max(1, subset_size)  # 确保 subset_size 至少为 1
#     subset_size = min(subset_size, len_data_loader)  # 确保 subset_size 不超过数据集大小

#     print(f"subset_size: {subset_size}, len_data_loader: {len_data_loader}")

#     # 使用自定义的 LazyGreedy 优化器
#     submod_optimizer = LazyGreedy(args=args, index=np.arange(n), budget=subset_size)

#     # 进行子模子集选择，返回选中的索引和对应的 Gain 值
#     selected_result, selected_gains = submod_optimizer.select(
#         gain_function=submod_function.calc_gain,
#         update_state=submod_function.update_state
#     )

#     # 将选择的索引映射回原数据集的索引
#     selected_indices = [all_indices[idx] for idx in selected_result]

#     # 计算 Gamma 权重，直接使用 selected_gains
#     gamma_normalized = np.array(selected_gains)

#     # 使用 Min-Max 归一化将权重缩放到 [0.5, 2.0] 之间
#     min_gain = np.min(gamma_normalized)
#     max_gain = np.max(gamma_normalized)
#     if max_gain - min_gain > 1e-8:
#         gamma_normalized = (gamma_normalized - min_gain) / (max_gain - min_gain)
#         gamma_normalized = gamma_normalized * 1.5 + 0.5  # 缩放到 [0.5, 2.0]
#     else:
#         # 如果所有 gain 相同，设置权重为 1
#         gamma_normalized = np.ones_like(gamma_normalized)

#     # 将权重限制在 [0.5, 2.0] 之间，避免过大或过小
#     gamma_normalized = np.clip(gamma_normalized, 0.5, 2.0)

#     gamma_normalized = torch.FloatTensor(gamma_normalized).flatten()

#     # 确保 gamma_weights 的长度与 selected_indices 一致
#     assert gamma_normalized.size(0) == len(selected_indices), "gamma_weights 的长度与 selected_indices 不一致"

#     return selected_indices, gamma_normalized

def craig_select_optimized(model, data_loader, device, subset_size, args):
    model.eval()
    all_gradients = []
    all_indices = []

    # 定义用于选择的损失函数（reduction='none'）
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    # 获取最后一层参数
    last_layer_params = list(model.parameters())[-1:]

    # 定义对单个样本计算损失的函数
    def compute_loss_single_sample(params, input_, target_):
        input_ = input_.unsqueeze(0)  # [1,C,H,W]
        target_ = target_.unsqueeze(0)  # [1]
        outputs = model(input_)
        loss = criterion(outputs, target_)  # [1]张量
        return loss.sum()  # 返回单标量

    # 对单样本梯度计算函数求梯度
    grad_single = grad(compute_loss_single_sample)

    # 使用 vmap 进行批处理梯度计算
    grad_batch = vmap(grad_single, in_dims=(None, 0, 0))

    # 计算每个样本的最后一层梯度
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if inputs.size(0) == 0:
            continue

        # 使用 vmap 对整批数据计算梯度
        try:
            batch_gradients = grad_batch(last_layer_params, inputs, targets)
        except Exception as e:
            print(f"Error computing gradients: {e}, skipping batch {batch_idx}.")
            continue
        grad_tensor = batch_gradients[0]

        start_idx = batch_idx * data_loader.batch_size
        for i in range(inputs.size(0)):
            # 对单个样本的梯度 flatten
            flattened_grads = grad_tensor[i].view(-1).cpu().numpy()
            all_gradients.append(flattened_grads)
            all_indices.append(start_idx + i)

    # 若没有任何梯度数据，直接返回空列表
    if len(all_gradients) == 0:
        print("No gradients computed")
        return [], []

    all_gradients = np.vstack(all_gradients)
    n = len(all_gradients)

    # 使用欧氏距离计算相似度矩阵
    matrix = euclidean_dist_pair_np(all_gradients)

    # 按照官方代码进行矩阵变换
    matrix = -1.0 * matrix
    matrix -= (np.min(matrix) - 1e-3)

    # 使用自定义的 FacilityLocation 子模函数
    submod_function = FacilityLocation(index=np.arange(n), similarity_matrix=matrix)

    total_samples = len(data_loader.dataset)  # 或者直接用 len(class_subset)
    subset_size = int(total_samples * args.selection_fraction)
    subset_size = max(1, subset_size)
    subset_size = min(subset_size, total_samples)


    print(f"subset_size: {subset_size}, len_data_loader: {total_samples}")

    # 使用自定义的 LazyGreedy 优化器
    submod_optimizer = LazyGreedy(args=args, index=np.arange(n), budget=subset_size)

    # 进行子模子集选择，返回选中的索引和对应的 Gain 值
    selected_result, selected_gains = submod_optimizer.select(
        gain_function=submod_function.calc_gain,
        update_state=submod_function.update_state
    )

    # 将选择的索引映射回原数据集的索引
    selected_indices = [all_indices[idx] for idx in selected_result]

    # 计算 Gamma 权重，直接使用 selected_gains
    gamma_normalized = np.array(selected_gains)

    # 使用 Min-Max 归一化将权重缩放到 [0.5, 2.0] 之间
    min_gain = np.min(gamma_normalized)
    max_gain = np.max(gamma_normalized)
    if max_gain - min_gain > 1e-8:
        gamma_normalized = (gamma_normalized - min_gain) / (max_gain - min_gain)
        gamma_normalized = gamma_normalized * 1.5 + 0.5  # 缩放到 [0.5, 2.0]
    else:
        # 如果所有 gain 相同，设置权重为 1
        gamma_normalized = np.ones_like(gamma_normalized)

    # 将权重限制在 [0.5, 2.0] 之间，避免过大或过小
    gamma_normalized = np.clip(gamma_normalized, 0.5, 2.0)

    gamma_normalized = torch.FloatTensor(gamma_normalized).flatten()

    # 确保 gamma_weights 的长度与 selected_indices 一致
    assert gamma_normalized.size(0) == len(selected_indices), "gamma_weights 的长度与 selected_indices 不一致"

    return selected_indices, gamma_normalized


# 创建带权重的子集 DataLoader，基于 Gain 权重
def create_weighted_subset_loader(dst_train, selected_indices, gamma_weights, batch_size=128, num_workers=4, pin_memory=True):
    # 创建 Subset 数据集
    subset_data = Subset(dst_train, selected_indices)

    # 创建带权重的自定义数据集，权重保持在 CPU 上
    weighted_subset = WeightedSubsetDataset(subset_data, gamma_weights)

    # 创建 DataLoader，设置 shuffle=True 以打乱数据
    subset_loader = DataLoader(
        weighted_subset,
        batch_size=batch_size, 
        shuffle=True,  
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return subset_loader


# 评估模型性能
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, precision, recall, f1


# 训练模型，应用 Gain 权重
def train_model(model, subset_loader, gamma_weights, optimizer, scheduler, num_epochs, device, val_loader, best_model_path, patience=10, best_acc=0.0):
    scaler = GradScaler()
    trigger_times = 0

    # 定义一个新的损失函数，使用 reduction='none' 以获取每个样本的损失
    per_sample_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    # 打印 gamma_weights 的统计信息
    print(f"gamma_weights min: {gamma_weights.min().item()}, max: {gamma_weights.max().item()}, mean: {gamma_weights.mean().item()}")

    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        running_loss = 0.0

        for batch_num, (inputs, targets, weights) in enumerate(subset_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            weights = weights.to(device)  # 将权重移动到 GPU
            optimizer.zero_grad()

            # 使用混合精度
            with autocast():
                outputs = model(inputs)
                # 使用新的损失函数计算每个样本的损失
                loss = per_sample_criterion(outputs, targets)  # [batch_size]

                # 确保 weights 的形状与 loss 的形状一致
                assert weights.dim() == 1, f"weights 应为 1-D 张量，但当前维度为 {weights.dim()}"
                assert weights.size(0) == loss.size(0), f"weights 的长度 {weights.size(0)} 与 loss 的长度 {loss.size(0)} 不一致"

                # 对损失应用权重
                weighted_loss = loss * weights  # [batch_size]
                loss_mean = weighted_loss.mean()

            # 反向传播和优化步骤
            scaler.scale(loss_mean).backward()
            # 添加梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss_mean.item()

        scheduler.step()

        # 在验证集上进行评估
        acc, precision, recall, f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, acc={acc:.4f}, precession={precision:.4f}, recall={recall:.4f}, F1 score={f1:.4f}")
        # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(subset_loader):.4f} - Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"save best model, acc: {best_acc:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("early stoping")
                break
    
    return best_acc


# 子集选择和训练流程（基于间隔）
def interval_selection_and_train(args, model, dst_train, dst_val, dst_test, optimizer, scheduler, device, best_model_path):
    accumulated_core_indices = set()
    best_acc = 0.0
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        balance = True
        selected_indices = []
        gamma_weights = []

        # 获取类别数和标签
        try:
            num_classes = len(set(int(target.item()) for target in dst_train.targets))
            train_labels = dst_train.targets
        except AttributeError:
            num_classes = len(set(int(target.item()) for target in dst_train.labels))
            train_labels = dst_train.labels

        if balance:
            for c in range(num_classes):
                # 获取当前类别的样本索引
                class_indices = np.where(np.array(train_labels) == c)[0]
                if len(class_indices) == 0:
                    print(f"类别 {c} 没有样本，跳过。")
                    continue

                # 计算当前类别的子集大小
                class_subset_size = int(len(class_indices) * args.selection_fraction)
                class_subset_size = min(class_subset_size, len(class_indices))

                flag = 0

                # 调整 class_subset_size 以确保是 batch_size 的整数倍
                adjusted_class_subset_size = (class_subset_size // args.selection_batch) * args.selection_batch
                if adjusted_class_subset_size < args.selection_batch and len(class_indices) >= args.selection_batch:
                    print(f"类别 {c}: 调整后的子集大小 {adjusted_class_subset_size} 小于 batch_size {args.selection_batch}")
                    flag = 1
                    # continue
                class_subset_size = adjusted_class_subset_size

                # 创建当前类别的数据加载器，设置 drop_last=True
                class_subset = Subset(dst_train, class_indices)
                if flag == 1:  
                    class_loader = DataLoader(
                        class_subset,
                        batch_size=args.selection_batch, 
                        shuffle=False,
                        # drop_last=True, 
                        drop_last=False, 
                        num_workers=args.workers,
                        pin_memory=True
                )
                else:
                    class_loader = DataLoader(
                    class_subset,
                    batch_size=args.selection_batch, 
                    shuffle=False,
                    drop_last=True, 
                    num_workers=args.workers,
                    pin_memory=True
                )
                    

                # 使用 CRAIG 子集选择函数，获取选中索引和对应的 Gain 权重
                class_selected, class_gamma = craig_select_optimized(
                    model,
                    class_loader,
                    device,
                    class_subset_size,
                    args
                )

                if not class_selected:
                    print(f"类别 {c}: 没有选择到任何样本。")
                    continue

                # 将选择的样本索引映射回全数据集
                selected_class_indices = [class_indices[idx] for idx in class_selected]
                selected_indices.extend(selected_class_indices)
                gamma_weights.extend(class_gamma.tolist())  # 收集所有 Gain 权重

        else:
            full_loader = DataLoader(
                dst_train,
                batch_size=args.selection_batch,  
                shuffle=False,
                drop_last=True, 
                num_workers=args.workers,
                pin_memory=True
            )
            selected_indices, gamma_weights = craig_select_optimized(model, full_loader, device, target_core_set_size, args)
            # 确保没有重复选择
            assert len(selected_indices) == len(set(selected_indices)), "selected_indices 中存在重复的样本"

        # 确保没有重复选择
        selected_indices = list(set(selected_indices))
        
        gamma_weights = gamma_weights[:len(selected_indices)]

        # 确保 gamma_weights 不为空
        assert len(gamma_weights) > 0, "gamma_weights 为空"

        print(f"Final number of selected samples: {len(selected_indices)}")

        accumulated_core_indices.update(selected_indices)

        # 转换 gamma_weights 为张量，并确保其在正确的设备上
        gamma_tensor = torch.FloatTensor(gamma_weights)  # 保持在 CPU 上

        # 修改归一化方式：使用 Min-Max 归一化将权重缩放到 [0.5, 2.0] 之间
        if gamma_tensor.mean() > 0:
            min_weight = gamma_tensor.min()
            max_weight = gamma_tensor.max()
            if max_weight - min_weight > 1e-8:
                gamma_tensor = (gamma_tensor - min_weight) / (max_weight - min_weight)
                gamma_tensor = gamma_tensor * 1.5 + 0.5  # 缩放到 [0.5, 2.0]
            else:
                # 如果所有权重相同，设置为1
                gamma_tensor = torch.ones_like(gamma_tensor)

        # 将权重限制在 [0.5, 2.0] 之间，避免过大或过小
        gamma_tensor = torch.clamp(gamma_tensor, min=0.5, max=2.0)

        assert gamma_tensor.size(0) == len(selected_indices), "gamma_weights 的长度与 selected_indices 不一致"

        subset_loader = create_weighted_subset_loader(
            dst_train, selected_indices, gamma_tensor,
            batch_size=args.selection_batch, 
            num_workers=args.workers,
            pin_memory=True
        )

        # 调用 train_model，不再传递 criterion
        best_acc = train_model(model, subset_loader, gamma_tensor, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")


# 子集选择和训练流程（一次性选择）
def once_selection_and_train(args, model, dst_train, dst_val, dst_test, optimizer, scheduler, device, best_model_path):
    subset_size = int(len(dst_train) * args.selection_fraction)
    balance = True
    selected_indices = []
    gamma_weights = []
    full_loader = DataLoader(dst_train, batch_size=args.selection_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

    select_start_time = time.time()

    # 获取数据集类别数和标签
    try:
        num_classes = len(set(int(target.item()) for target in dst_train.targets))
        train_labels = dst_train.targets
    except AttributeError:
        num_classes = len(set(int(target.item()) for target in dst_train.labels))
        train_labels = dst_train.labels

    print(f"Num classes: {num_classes}")

    if balance:
        for c in tqdm(range(num_classes), desc="Class"):
            # 获取当前类别的样本索引
            class_indices = np.where(np.array(train_labels) == c)[0]
            if len(class_indices) == 0:
                continue

            class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))

            # 创建当前类别的数据加载器，设置 drop_last=True
            class_subset = Subset(dst_train, class_indices)
            class_loader = DataLoader(
                class_subset,
                batch_size=args.selection_batch,  
                shuffle=False,
                drop_last=True,
                num_workers=args.workers,
                pin_memory=True
            )

            print("len(class_subset_size):", class_subset_size)
            # 使用 CRAIG 子集选择函数，获取选中索引和对应的 Gain 权重
            class_selected, class_gamma = craig_select_optimized(
                model, class_loader, device, class_subset_size, args
            )
            if not class_selected:
                print(f"类别 {c} 没有选择到任何样本。")
                continue

            # 将选择的样本索引映射回全数据集
            selected_class_indices = [class_indices[idx] for idx in class_selected]
            selected_indices.extend(selected_class_indices)
            gamma_weights.extend(class_gamma.tolist())  # 收集所有 Gain 权重

            # 确保没有重复选择
            assert len(selected_indices) == len(set(selected_indices)), "selected_indices 中存在重复的样本"
    else:
        selected_indices, gamma_weights = craig_select_optimized(model, full_loader, device, subset_size, args)
        # 确保没有重复选择
        assert len(selected_indices) == len(set(selected_indices)), "selected_indices 中存在重复的样本"

    # 确保没有重复选择
    selected_indices = list(set(selected_indices))
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    gamma_weights = gamma_weights[:len(selected_indices)]

    # 确保 gamma_weights 不为空
    assert len(gamma_weights) > 0, "gamma_weights 为空"

    print(f"Final number of selected samples: {len(selected_indices)}")

    # 转换 gamma_weights 为张量，并确保其在正确的设备上
    gamma_tensor = torch.FloatTensor(gamma_weights)  # 保持在 CPU 上

    # 修改归一化方式：使用 Min-Max 归一化将权重缩放到 [0.5, 2.0] 之间
    if gamma_tensor.mean() > 0:
        min_weight = gamma_tensor.min()
        max_weight = gamma_tensor.max()
        if max_weight - min_weight > 1e-8:
            gamma_tensor = (gamma_tensor - min_weight) / (max_weight - min_weight)
            gamma_tensor = gamma_tensor * 1.5 + 0.5  # 缩放到 [0.5, 2.0]
        else:
            # 如果所有权重相同，设置为1
            gamma_tensor = torch.ones_like(gamma_tensor)

    # 将权重限制在 [0.5, 2.0] 之间，避免过大或过小
    gamma_tensor = torch.clamp(gamma_tensor, min=0.5, max=2.0)

    assert gamma_tensor.size(0) == len(selected_indices), "gamma_weights 的长度与 selected_indices 不一致"

    subset_loader = create_weighted_subset_loader(
        dst_train, selected_indices, gamma_tensor,
        batch_size=args.selection_batch, 
        num_workers=args.workers,
        pin_memory=True
    )
    select_end_time = time.time()
    selection_time = select_end_time - select_start_time
    print(f"Selection time: {selection_time:.2f} seconds")

    # 调用 train_model，不再传递 criterion
    train_model(model, subset_loader, gamma_tensor, optimizer, scheduler, args.epochs, device, val_loader, best_model_path, patience=10, best_acc=0.0)
    
    return selection_time


# 主函数，执行 CRAIG 选择和训练
def craig_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    start_time = time.time()
    # log_path = os.path.join(args.save_log_path, args.dataset, args.method, args.selection_framework, str(args.selection_fraction), f"{args.balance_target}.log")
    # best_model_path = os.path.join(args.save_best_model_path, args.dataset, args.method, args.selection_framework, str(args.selection_fraction), f"{args.balance_target}.pth")
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  args.model + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
    
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    # 替换全局 print 为日志输出
    global print
    print = lambda *args: logging.info(" ".join(map(str, args)))
    try:
        # 打印初始日志信息
        print(f"Log path: {log_path}")
        print(f"method: {args.method}")
        print(f"network: {args.model}")
        print(f"dataset: {args.dataset}")
        print(f"selection_fraction: {args.selection_fraction}")
        if hasattr(args, 'proxy_model') and args.proxy_model != 'none':
            print(f"proxy_model: {args.proxy_model}")

        print(f"数据集信息：通道数={channel}，图像大小={im_size}，类别数={num_classes}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA 不可用")

        # 数据集标签分布日志
        try:
            train_labels = [int(dst_train[i][1]) for i in range(len(dst_train))]
        except:
            train_labels = [int(dst_train[i].labels) for i in range(len(dst_train))]
        print("train set label distribution:", Counter(train_labels))

        try:
            val_labels = [int(dst_val[i][1]) for i in range(len(dst_val))]
        except:
            val_labels = [int(dst_val[i].labels) for i in range(len(dst_val))]
        print("val set label distribution:", Counter(val_labels))

        try:
            test_labels = [int(dst_test[i][1]) for i in range(len(dst_test))]
        except:
            test_labels = [int(dst_test[i].labels) for i in range(len(dst_test))]
        print("test set label distribution:", Counter(test_labels))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print("CUDA is available:", torch.cuda.is_available())

        # 初始化模型
        model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)

        # 初始化优化器和学习率调度器
        # 将学习率调整为 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.selection_weight_decay)  # 使用 AdamW 优化器，与 craig_four.py 保持一致
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 测试集 DataLoader
        test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

        # 根据选择框架运行相应的子集选择和训练流程
        if args.selection_framework == 'interval':
            interval_selection_and_train(args, model, dst_train, dst_val, dst_test, optimizer, scheduler, device, best_model_path)
        elif args.selection_framework == 'once':
            selection_time = once_selection_and_train(args, model, dst_train, dst_val, dst_test, optimizer, scheduler, device, best_model_path)

        # 加载最佳模型并在测试集上进行最终评估
        end_time = time.time()
        model.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
        print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        os.remove(best_model_path)
        print(f"Total time: {time.time() - start_time:.4f}s")


        # 记录最终评估结果到csv文件中
        import pandas as pd


        elapsed_time = end_time - start_time
        csv_path = os.path.join(args.save_log_path, args.dataset, args.model + '_new.csv')
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))

        selection_ratio = str(args.selection_fraction)
        metrics = ['precision', 'recall', 'f1', 'selection_time', 'time']
        columns = pd.MultiIndex.from_product([[selection_ratio], metrics])

        # 转为字符串以支持追加
        precision_str = f"{precision:.4f}"
        recall_str = f"{recall:.4f}"
        f1_str = f"{f1:.4f}"
        selection_time_str = f"{selection_time:.2f}"
        time_str = f"{elapsed_time:.2f}"

        new_data = pd.DataFrame([[precision_str, recall_str, f1_str, selection_time_str, time_str]], index=[args.method], columns=columns)

        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            updated_df = old_df.copy()

            # 添加新列（如果没有）
            for col in new_data.columns:
                if col not in updated_df.columns:
                    updated_df[col] = pd.NA

            # 添加新行（如果没有）
            if args.method not in updated_df.index:
                updated_df.loc[args.method] = pd.NA

            # 更新数据：追加 | 分隔的值
            for col in new_data.columns:
                old_val = str(updated_df.loc[args.method, col]) if pd.notna(updated_df.loc[args.method, col]) else ""
                new_val = new_data.loc[args.method, col]
                if old_val:
                    if new_val not in old_val.split('|'):  # 避免重复
                        updated_df.loc[args.method, col] = old_val + '|' + new_val
                else:
                    updated_df.loc[args.method, col] = new_val

            updated_df.to_csv(csv_path)
        else:
            new_data.to_csv(csv_path)


    except Exception as e:
        # 记录异常信息到日志
        logging.error(f"An error occurred in Arnoldi_selection: {str(e)}")
        logging.error("Stack trace: ", exc_info=True)  # 记录完整的错误堆栈信息
