import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import copy
from collections import Counter
import nets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from collections import defaultdict
import logging
import os
import time

# 定义一个字典来存储钩子捕获的特征
activation = defaultdict(list)

def compute_similarity_matrix(features, device):
    """
    计算特征矩阵的相似度矩阵。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        device (str): 设备类型。

    Returns:
        torch.Tensor: 相似度矩阵，形状为 (N, N)。
    """
    # 简单使用内积作为相似度度量
    features = features.to(device)
    similarity_matrix = torch.mm(features, features.t())
    return similarity_matrix

def compute_shapley_value_with_feature(feature, label,model,train_loader,  device, num_iterations=100, performance_tolerance=1e-5):
    num_samples = feature.size(0)
    shapley_values = torch.zeros(num_samples, device=device)
    T = num_iterations
    
    features, labels = extract_features_and_labels_util_feature(model, train_loader, device)

    # 预先计算全局相似度矩阵
    similarity_matrix = compute_similarity_matrix(features, device)
    total_performance = similarity_matrix.sum().item()

    for t in tqdm(range(T), desc="Calculating Shapley Values using TMCS with feature"):
        # 随机排列数据点
        permutation = torch.randperm(num_samples)
        previous_performance = 0.0

        for j, idx in enumerate(permutation):
            if j == 0:
                # 第一个数据点，边际贡献为自相似度
                marginal_contribution = similarity_matrix[idx, idx].item()
            else:
                # 当前数据点与已加入子集的数据点的相似度之和
                prev_indices = permutation[:j]
                marginal_contribution = similarity_matrix[idx, idx].item() + similarity_matrix[idx, prev_indices].sum().item()

            # 更新 Shapley 值（递归平均）
            shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (t + 1)

            # 累积当前子集的性能
            previous_performance += marginal_contribution

            # 截断条件
            if abs(total_performance - previous_performance) < performance_tolerance:
                break

    # 归一化 Shapley Value
    shapley_values = (shapley_values - shapley_values.min()) / (shapley_values.max() - shapley_values.min())
    return shapley_values

def compute_shapley_value_with_model(features, labels, model, val_loader, device, num_iterations=100, performance_tolerance=1e-5):
    """
    使用模型验证集性能计算 Shapley 值。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 标签张量。
        model (torch.nn.Module): 训练模型。
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器。
        device (str): 设备类型。
        num_iterations (int): 采样次数。
        performance_tolerance (float): 性能截断容差。

    Returns:
        torch.Tensor: Shapley 值向量，形状为 (N,)。
    """
    num_samples = features.size(0)
    shapley_values = torch.zeros(num_samples, device=device)

    for t in tqdm(range(num_iterations), desc="Calculating Shapley Values with Model evaluating"):
        # 随机排列样本
        permutation = torch.randperm(num_samples, device=device)
        # print("len(permutation) : ",len(permutation))
        subset_performance = 0.0

        for i, idx in enumerate(permutation):
            current_subset = permutation[:i + 1]  # 当前子集索引
            subset_dataset = Subset(TensorDataset(features, labels), current_subset.tolist())
            subset_loader = DataLoader(subset_dataset, batch_size=128, shuffle=True)

            # 在当前子集上训练模型
            trained_model = train_model_on_subset(model, subset_loader, device)

            # 使用验证集评估性能
            current_performance = evaluate_model(trained_model, val_loader, device)[0]  # 返回验证集准确率
            # print("current_performance: ",current_performance)

            # 边际贡献
            marginal_contribution = current_performance - subset_performance

            # 更新 Shapley 值
            shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (t + 1)

            # 截断条件
            if abs(subset_performance - current_performance) < performance_tolerance:
                # print(i)
                break
                
            # 更新子集性能
            subset_performance = current_performance

    # 归一化 Shapley 值
    shapley_values -= shapley_values.min()
    shapley_values /= shapley_values.max() - shapley_values.min()
    return shapley_values

def train_model_on_subset(model, subset_loader, device, epochs=5):
    """
    在子集上训练模型。

    Args:
        model (torch.nn.Module): 待训练模型。
        subset_loader (torch.utils.data.DataLoader): 子集数据加载器。
        device (str): 设备类型。
        epochs (int): 训练轮数。

    Returns:
        torch.nn.Module: 训练后的模型。
    """
    model = copy.deepcopy(model)  # 防止修改原模型
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for inputs, labels in subset_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def tmcs_shapley_selection(args, features, labels,model,train_loader, val_loader, target_core_set_size, device):
    """
    使用 Shapley Value 选择核心数据集。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 标签张量，形状为 (N,)。
        target_core_set_size (int): 核心数据集的目标大小。
        device (str): 设备类型。

    Returns:
        List[int]: 核心数据集的索引。
    """
    if args.shapley_util == "model":
        print("shapley_util : model")
        shapley_values = compute_shapley_value_with_model(features, labels, model, val_loader, device)
    elif args.shapley_util == "feature":
        print("shapley_util : feature")
        shapley_values = compute_shapley_value_with_feature(features, labels,model,train_loader,  device)
    elif args.shapley_util == "uncertainty":
        print("shapley_util : uncertainty")
        shapley_values = compute_shapley_value_with_uncertainty(features, labels, model, val_loader, device)
    selected_indices = torch.topk(shapley_values, target_core_set_size).indices.tolist()
    return selected_indices

def extract_features_and_labels(model, data_loader, device):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # features = model(inputs)  # 假设模型最后一层输出特征
            all_features.append(inputs.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)

def extract_features_and_labels_util_feature(model, data_loader, device):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model(inputs)  # 假设模型最后一层输出特征
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)

def evaluate_uncertainty(model, data_loader, device):
    """
    基于不确定性（预测熵）评估模型性能。

    Args:
        model (torch.nn.Module): 模型。
        data_loader (torch.utils.data.DataLoader): 数据加载器。
        device (str): 设备类型。

    Returns:
        float: 平均不确定性（熵）。
    """
    model.eval()
    uncertainty_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)  # 熵计算
            uncertainty_scores.extend(entropy.cpu().numpy())
    return float(np.mean(uncertainty_scores))

def compute_shapley_value_with_uncertainty(features, labels, model, val_loader, device, num_iterations=100, performance_tolerance=1e-5):
    """
    使用模型的不确定性计算 Shapley 值。

    Args:
        features (torch.Tensor): 特征张量，形状为 (N, D)。
        labels (torch.Tensor): 标签张量。
        model (torch.nn.Module): 训练模型。
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器。
        device (str): 设备类型。
        num_iterations (int): 采样次数。
        performance_tolerance (float): 性能截断容差。

    Returns:
        torch.Tensor: Shapley 值向量，形状为 (N,)。
    """
    num_samples = features.size(0)
    shapley_values = torch.zeros(num_samples, device=device)

    for t in tqdm(range(num_iterations), desc="Calculating Shapley Values with Uncertainty"):
        # 随机排列样本
        permutation = torch.randperm(num_samples, device=device)
        subset_performance = 0.0

        for i, idx in enumerate(permutation):
            current_subset = permutation[:i + 1]  # 当前子集索引
            subset_dataset = Subset(TensorDataset(features, labels), current_subset.tolist())
            subset_loader = DataLoader(subset_dataset, batch_size=128, shuffle=True)

            # 在当前子集上训练模型
            trained_model = train_model_on_subset(model, subset_loader, device)

            # 使用验证集评估性能（基于不确定性）
            current_performance = evaluate_uncertainty(trained_model, val_loader, device)  # 返回不确定性得分

            # 边际贡献
            marginal_contribution = current_performance - subset_performance

            # 更新 Shapley 值
            shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (t + 1)

            # 截断条件
            if abs(subset_performance - current_performance) < performance_tolerance:
                break

            # 更新子集性能
            subset_performance = current_performance

    # 归一化 Shapley 值
    shapley_values -= shapley_values.min()
    shapley_values /= shapley_values.max() - shapley_values.min()
    return shapley_values

def evaluate_uncertainty(model, data_loader, device):
    """
    基于不确定性（预测熵）评估模型性能。

    Args:
        model (torch.nn.Module): 模型。
        data_loader (torch.utils.data.DataLoader): 数据加载器。
        device (str): 设备类型。

    Returns:
        float: 平均不确定性（熵）。
    """
    model.eval()
    uncertainty_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)  # 熵计算
            uncertainty_scores.extend(entropy.cpu().numpy())
    return float(np.mean(uncertainty_scores))

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
    #初始化一个最佳模型，避免之后没有
    torch.save(model.state_dict(), best_model_path)

    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 数值稳定性处理：防止输出过大或过小
            outputs = torch.clamp(outputs, min=-10, max=10)

            loss = criterion(outputs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

        scheduler.step()
        acc, precision, recall, f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, acc={acc:.4f}, precession={precision:.4f}, recall={recall:.4f}, F1 score={f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            # print("best_acc is ",best_acc)
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
    
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
    accumulated_core_indices = set()
    core_loader = random_loader

    batch_gradients_cache = []
    checkpoint_cache = []
    # interval_core_loader = core_loader

    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        # 预训练模型并缓存梯度（随机选择部分样本）
        features, labels = extract_features_and_labels(model, train_loader, device)

        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_core_set_size}")

        selected_indices = tmcs_shapley_selection(args, features, labels, model,train_loader,  val_loader, target_core_set_size, device)

        print(f"Selected {len(selected_indices)} samples for core set.")
        accumulated_core_indices.update(selected_indices)

        core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")
    
def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path):
    select_start = time.time()
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    features, labels = extract_features_and_labels(model, train_loader, device)

    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_core_set_size}")

    selected_indices = tmcs_shapley_selection(args, features, labels ,model,train_loader,  val_loader, target_core_set_size, device)

    print(f"Selected {len(selected_indices)} samples for core set.")
    

    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    select_end = time.time()
    selection_time = select_end-select_start
    print(f"selection_time:{selection_time}")
    train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path, patience=10, best_acc=0.0)
    return selection_time

# 替换 TMCS_selection 中的逻辑
def TMCS_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, k=10):

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

    try:
        # 打印初始日志信息
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
            selection_time = once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path)


        model.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True),device)
        print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f}s")
        
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