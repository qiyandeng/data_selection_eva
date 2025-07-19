import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import nets
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time


def compute_knn_shapley_optimized(features, labels, k_neighbors=5, num_permutations=10):
    """
    根据 Improved MC Approach 计算 KNN-Shapley 值。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 样本的标签，形状为 (N,)。
        k_neighbors (int): 最近邻的数量。
        num_permutations (int): 随机排列的次数。
    
    Returns:
        torch.Tensor: 每个样本的 Shapley 值，形状为 (N,)。
    """
    n_samples = features.size(0)
    shapley_values = torch.zeros(n_samples, device=features.device)
    labels = labels.cpu().numpy()

    # 利用 sklearn.NearestNeighbors 构建最近邻模型
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
    nn_model.fit(features.cpu().numpy())

    for _ in tqdm(range(num_permutations), desc="Calculating KNN-Shapley"):
        # 生成一个随机排列
        permutation = np.random.permutation(n_samples)

        # 初始化 max-heap 模拟最近邻
        max_heap = []
        marginal_contributions = torch.zeros(n_samples, device=features.device)

        for i, idx in enumerate(permutation):
            # 插入当前样本到 max-heap
            current_neighbors = nn_model.kneighbors(features[idx:idx + 1].cpu().numpy(), return_distance=False)[0][1:]
            
            # 计算效用增量
            if len(current_neighbors) > 0:
                correct_matches = (labels[current_neighbors] == labels[idx]).sum()
                marginal_contributions[idx] = correct_matches / k_neighbors

            if i > 0:
                marginal_contributions[idx] -= marginal_contributions[permutation[i - 1]]

        # 更新 Shapley 值
        shapley_values += marginal_contributions / num_permutations

    return shapley_values


def knn_shapley_selection(features, labels, num_samples, k_neighbors=5, num_permutations=10):
    """
    使用改进的 KNN-Shapley 方法选择样本。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 样本标签。
        num_samples (int): 需要选择的样本数量。
        k_neighbors (int): 最近邻的数量。
        num_permutations (int): 随机排列的次数。
    
    Returns:
        List[int]: 选择的样本索引。
    """
    shapley_values = compute_knn_shapley_optimized(features, labels, k_neighbors, num_permutations)
    selected_indices = torch.argsort(shapley_values, descending=True)[:num_samples].tolist()
    return selected_indices

def extract_features_and_labels(model, data_loader, device):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model(inputs)  # 假设模型最后一层输出特征
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)


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
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
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
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    accumulated_core_indices = set()

    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader

    batch_gradients_cache = []
    checkpoint_cache = []
    # interval_core_loader = core_loader

    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        # 预训练模型并缓存梯度（随机选择部分样本）
        features, labels = extract_features_and_labels(model, core_loader, device)

        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_core_set_size}")

        selected_indices = knn_shapley_selection(features, labels, target_core_set_size)

        accumulated_core_indices.update(selected_indices)

        print(f"Selected {len(selected_indices)} samples for core set.")
        

        core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        if best_acc > 0:
                try:
                    model.load_state_dict(torch.load(best_model_path))
                except:
                    pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")
    
def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    select_start = time.time()
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader
    
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    features, labels = extract_features_and_labels(model, core_loader, device)

    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_core_set_size}")

    selected_indices = knn_shapley_selection(features, labels, target_core_set_size)

    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    print(f"Selected {len(selected_indices)} samples for core set.")

    core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    select_end = time.time()
    selection_time = select_end-select_start
    print(f"selection time:{selection_time}")
    train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)
    return selection_time


def KNN_Shapley_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, balance=True, k_neighbors=5, num_permutations=10):
    import logging
    from collections import Counter
    import os

    
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

    # 替换全局 print
    global print
    print = lambda *args: logging.info(" ".join(map(str, args)))
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

    # best_model_path = args.save_best_model_path

    # 预训练模型
    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 随机选择一部分数据进行预训练
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader

    def pretrain_model(model, data_loader, criterion, optimizer, scheduler, device, num_epochs=3):
        print("Pretraining model...")
        model.train()
        for epoch in tqdm(range(num_epochs), desc="Pretraining"):
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
        print("Pretraining completed.")

    pretrain_model(model, core_loader, criterion, optimizer, scheduler, device)

    if args.selection_framework == 'interval':
        interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path)
    elif args.selection_framework == 'once':
        selection_time = once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)


    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True),device)
    print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    os.remove(best_model_path)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")

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

