import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse
from torchvision import transforms
import nets  # 请确保您有一个名为 nets 的模块，包含您需要的模型
from torch.optim import lr_scheduler
from functorch import grad, vmap
from scipy.linalg import solve
from scipy.optimize import nnls
from numpy.linalg import norm, lstsq
import logging
import os

# 自定义 WeightedSubset 类，将样本与权重关联
class WeightedSubset(Dataset):
    def __init__(self, subset, gammas):
        """
        初始化 WeightedSubset
        Parameters:
            subset (Subset): 原始子集
            gammas (list or np.array): 对应子集的权重
        """
        self.subset = subset
        self.gammas = gammas

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, target = self.subset[idx]
        gamma = self.gammas[idx]
        return data, target, gamma

# 改进后的 Orthogonal Matching Pursuit (OMP) 实现
def OrthogonalMP_REG(A, b, tol=1E-4, nnz=None, positive=False, lam=1):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit with regularization
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz: maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
      lam: regularization coefficient
    Returns:
       vector of length n
    '''
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

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, device, val_loader, best_model_path, patience=10, best_acc=0.0):
    trigger_times = 0

    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        for inputs, labels_batch, gammas in data_loader:
            inputs, labels_batch, gammas = inputs.to(device), labels_batch.to(device), gammas.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 使用 gammas 对 loss 进行加权平均
            # 因为 criterion 的 reduction='none'，返回的是 [batch_size] 的loss向量
            # 先加权，再mean，确保最终是标量。
            loss = (criterion(outputs, labels_batch) * gammas).mean()

            loss.backward()
            optimizer.step()

        scheduler.step()
        acc, precision, recall, f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, acc={acc:.4f}, precession={precision:.4f}, recall={recall:.4f}, F1 score={f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_acc

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path):
    core_indices = set()
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    full_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    accumulated_core_indices = set()
    best_acc = 0.0

    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        balance = True
        selected_indices = []
        selected_gammas = []

        try:
            num_classes = len(set(int(target.item()) for target in dst_train.targets))
            train_labels = dst_train.targets
        except:
            num_classes = len(set(int(target.item()) for target in dst_train.labels))
            train_labels = dst_train.labels

        if balance:
            for c in range(num_classes):
                # 获取当前类别的样本索引
                class_indices = np.where(np.array(dst_train.labels) == c)[0]
                class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))

                # 创建当前类别的数据加载器
                class_subset = Subset(dst_train, class_indices)
                class_loader = DataLoader(class_subset, batch_size=128, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

                # 使用 GradMatch 子集选择函数
                class_selected, class_gammas = gradmatch_select_optimized(model, full_loader, class_loader, device, class_subset_size, args)
                # 将选择的样本索引映射回全数据集
                selected_class_indices = [class_indices[idx] for idx in class_selected]
                selected_indices.extend(selected_class_indices)
                selected_gammas.extend(class_gammas)
        else:
            class_loader = DataLoader(dst_train, batch_size=128, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
            subset_size = int(len(dst_train) * args.selection_fraction)
            selected_indices, selected_gammas = gradmatch_select_optimized(model, full_loader, class_loader, device, subset_size, args)
            core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
            print("Core set label distribution:", Counter(core_labels))
        
        # 打印选择的信息
        # print(f"Epoch {epoch + 1}/{args.epochs} - Selected {len(selected_indices)} samples")

        print(f"Final number of selected samples: {len(selected_indices)}")
        accumulated_core_indices.update(selected_indices)

        # 创建带有权重的子集数据加载器
        subset_data = WeightedSubset(Subset(dst_train, selected_indices), selected_gammas)
        subset_loader = DataLoader(subset_data, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)       
        if best_acc > 0:
                try:
                    model.load_state_dict(torch.load(best_model_path))
                except:
                    pass
        best_acc = train_model(model, subset_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    select_start = time.time()
    subset_size = int(len(dst_train) * args.selection_fraction)
    full_loader = DataLoader(dst_train, batch_size=128, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    # print(f"Epoch {epoch + 1}/{args.epochs} - Subset size: {subset_size}")
    balance = True
    # 对每个类别分别进行子集选择以处理类别不平衡
    selected_indices = []
    selected_gammas = []

    # try:
    #     num_classes = len(set(int(target.item()) for target in dst_train.targets))
    #     train_labels = dst_train.targets
    # except:
    #     num_classes = len(set(int(target.item()) for target in dst_train.labels))
    #     train_labels = dst_train.labels
    train_labels = [int(dst_train[i][1]) for i in range(len(dst_train))]
    num_classes = len(set(train_labels))

    
    if balance:
        print("Balanced selection")
        for c in tqdm(range(num_classes),desc="Class"):
            # 获取当前类别的样本索引
            class_indices = np.where(np.array(dst_train.labels) == c)[0]
            class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))

            # 创建当前类别的数据加载器
            class_subset = Subset(dst_train, class_indices)
            class_loader = DataLoader(class_subset, batch_size=128, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

            # 使用 GradMatch 子集选择函数
            class_selected, class_gammas = gradmatch_select_optimized(model, full_loader, class_loader, device, class_subset_size, args)
            # 将选择的样本索引映射回全数据集
            selected_class_indices = [class_indices[idx] for idx in class_selected]
            selected_indices.extend(selected_class_indices)
            selected_gammas.extend(class_gammas)
    else:
        class_loader = DataLoader(dst_train, batch_size=128, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
        subset_size = int(len(dst_train) * args.selection_fraction)
        selected_indices, selected_gammas = gradmatch_select_optimized(model, full_loader, class_loader, device, subset_size, args)
        
    
    # 打印选择的信息
    # print(f"Epoch {epoch + 1}/{args.epochs} - Selected {len(selected_indices)} samples")
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    # 创建带有权重的子集数据加载器
    subset_data = WeightedSubset(Subset(dst_train, selected_indices), selected_gammas)
    subset_loader = DataLoader(subset_data, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)    

    select_end = time.time()   
    selection_time = select_end - select_start
    print(f"selection time:{selection_time}")
    train_model(model, subset_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)
    return selection_time


# 改进后的 GradMatch 子集选择函数
def gradmatch_select_optimized(model, full_loader, data_loader, device, subset_size, args):
    """
    GradMatch subset selection using Orthogonal Matching Pursuit with regularization.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    full_loader : DataLoader
        DataLoader for the full dataset (train or validation).
    data_loader : DataLoader
        DataLoader for candidate data.
    device : torch.device
        The device to perform computation on.
    subset_size : int
        The size of the subset to select.
    args : argparse.Namespace or similar
        Contains parameters like lambda_reg and print_freq.

    Returns
    -------
    selected_indices : list
        List of selected sample indices.
    gammas : np.array
        Array containing weights for each selected sample.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    model.zero_grad()

    # 定义计算单个样本损失的函数
    def compute_loss(params, x, y):
        outputs = model(x.unsqueeze(0))  # 添加批次维度
        loss = criterion(outputs, y.unsqueeze(0))  # 计算单个样本的损失
        return loss.squeeze()

    # 定义计算单个样本梯度的函数
    def compute_single_grad(params, x, y):
        return grad(compute_loss)(params, x, y)

    last_layer_params = list(model.parameters())[-1:]  # 仅计算最后一层的梯度
    
    # 使用 vmap 对整个批次计算梯度
    compute_grad_batch = vmap(compute_single_grad, in_dims=(None, 0, 0))

    # Step 1: 计算完整数据集梯度（目标梯度）
    full_grad_sum = None
    for batch_idx, (inputs, labels) in enumerate(full_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_gradients = compute_grad_batch(last_layer_params, inputs, labels)  # (batch_size, grad_dim)

        # 将每个参数的梯度展平并连接起来
        flattened_grads = torch.cat([g.view(g.size(0), -1) for g in batch_gradients], dim=1).cpu().numpy()  # (batch_size, grad_dim)

        # 累加每个批次的梯度总和
        batch_sum = np.sum(flattened_grads, axis=0).astype(np.float32)  # (grad_dim,)
        if full_grad_sum is None:
            full_grad_sum = batch_sum
        else:
            full_grad_sum += batch_sum

    full_gradients = full_grad_sum  # (grad_dim,)

    # Step 2: 计算候选样本梯度
    sample_gradients = []
    sample_indices = []

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_gradients = compute_grad_batch(last_layer_params, inputs, labels)  # (batch_size, grad_dim)

        # 将每个参数的梯度展平并连接起来
        flattened_grads = torch.cat([g.view(g.size(0), -1) for g in batch_gradients], dim=1).cpu().numpy()  # (batch_size, grad_dim)

        batch_size = inputs.size(0)

        for i in range(batch_size):
            try:
                # 确保所有梯度张量都有第i个样本
                if all(g.size(0) > i for g in batch_gradients):
                    flattened_grad = np.concatenate([g[i].view(-1).cpu().numpy() for g in batch_gradients]).astype(np.float32)
                    sample_gradients.append(flattened_grad)
                    sample_indices.append(batch_idx * data_loader.batch_size + i)
                else:
                    pass  # 可选：统计被跳过的样本数量
            except Exception as e:
                print(f"Error processing sample {i} in batch {batch_idx}: {e}. Skipping.")

    if sample_gradients:
        sample_gradients = np.vstack(sample_gradients).astype(np.float32)  # (num_samples, grad_dim)
    else:
        print("No valid sample gradients were collected.")
        return [], np.array([])

    # Step 3: 检查子集大小
    if len(sample_gradients) == 0:
        raise ValueError("Candidate gradients are empty. Check your data loader or input data.")
    subset_size = min(subset_size, len(sample_gradients))
    if len(sample_gradients) < subset_size:
        print(f"Warning: Not enough candidates. Adjusting subset size to {len(sample_gradients)}")
        subset_size = len(sample_gradients)

    # Step 4: 应用正则化的 OMP 进行选择
    A = sample_gradients.T  # (grad_dim, num_samples)
    b = full_gradients  # (grad_dim,)

    # 使用 OrthogonalMP_REG 进行选择
    x = OrthogonalMP_REG(A, b, tol=1E-4, nnz=subset_size, positive=False, lam=args.lambda_reg)

    # 提取选中的样本索引和权重
    selected_indices = np.where(x > 0)[0].tolist()
    gammas = x[selected_indices]

    # 如果选择的样本数量不足，随机选择剩余的样本
    if len(selected_indices) < subset_size:
        remaining = subset_size - len(selected_indices)
        all_indices = set(range(len(sample_gradients)))
        selected_set = set(selected_indices)
        remain_set = list(all_indices - selected_set)
        if len(remain_set) >= remaining:
            additional_indices = np.random.choice(remain_set, size=remaining, replace=False).tolist()
        else:
            additional_indices = remain_set
        selected_indices.extend(additional_indices)
        gammas = np.concatenate([gammas, np.ones(len(additional_indices))])

    # 打乱选中的样本和权重
    perm = np.random.permutation(len(selected_indices))
    selected_indices = np.array(selected_indices)[perm].tolist()
    gammas = np.array(gammas)[perm]

    # 映射回原始样本索引
    selected_sample_indices = [sample_indices[idx] for idx in selected_indices]
    gammas = gammas.astype(np.float32)
    
    

    return selected_sample_indices, gammas

# GradMatch 选择主函数
def gradmatch_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    start_time = time.time()
    # log_path = args.save_log_path + args.dataset + '/' +  args.method + '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +  '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
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
        if args.proxy_model != 'none':
            print(f"proxy_model: {args.proxy_model}")

        print(f"数据集信息：通道数={channel}，图像大小={im_size}，类别数={num_classes}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

        # 数据集标签分布日志
        print("train set label distribution:", Counter([int(dst_train[idx][1]) for idx in range(len(dst_train))]))
        print("val set label distribution:", Counter([int(dst_val[idx][1]) for idx in range(len(dst_val))]))
        print("test set label distribution:", Counter([int(dst_test[idx][1]) for idx in range(len(dst_test))]))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print("CUDA is available:", torch.cuda.is_available())
        best_model_path = f"{args.dataset}_{args.method}_best_model.pth"

        # 初始化模型
        model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay)

        # 定义学习率调度器
        def linear_warmup_lr(epoch):
            if epoch < args.pretrain_epochs:
                return epoch / args.pretrain_epochs
            else:
                return 1.0

        warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup_lr)
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.pretrain_epochs)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 定义不同的数据加载器
        full_loader = DataLoader(dst_train, batch_size=128, shuffle=True, drop_last=False, num_workers=1, pin_memory=True)
        val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        candidate_loader = DataLoader(dst_train, batch_size=128, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
        # 注意：如果候选集与全数据集不同，定义不同的加载器

        best_acc = 0.0

        random_size = int(len(dst_train) * args.selection_fraction)
        random_indices = torch.randperm(len(dst_train))[:random_size]
        random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)


        # Warm-start 初始化阶段
        print("Starting warm-start training on the full dataset...")
        for epoch in tqdm(range(args.pretrain_epochs), desc="Warm-start Training"):
            model.train()
            running_loss = 0.0
            for inputs, labels in random_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 检查损失是否为 NaN
                if torch.isnan(loss).any():
                    print("NaN detected in loss during warm-start. Skipping backward pass.")
                    continue

                loss = loss.mean()  # 计算平均损失
                loss.backward()

                # 添加梯度裁剪以防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                running_loss += loss.item()
            warmup_scheduler.step()

            # 每个warm-start epoch后评估模型
            acc, precision, recall, f1 = evaluate_model(model, val_loader, device)
            print(f"Warm-start Epoch [{epoch+1}/{args.pretrain_epochs}] - Loss: {running_loss/len(full_loader):.4f} - Acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_model_path)

        # 主训练阶段
        if args.selection_framework == 'interval':
            interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
        elif args.selection_framework == 'once':
            selection_time = once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)

        
        # 加载最佳模型并在测试集上评估
        end_time = time.time()
        model.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
        print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        print(f"Total time: {time.time() - start_time:.2f}s")
        os.remove(best_model_path)

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


