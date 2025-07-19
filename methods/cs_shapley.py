import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import nets
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
import time

def get_label(dataset, idx):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if hasattr(dataset, 'labels'):
        return dataset.labels[idx]
    elif hasattr(dataset, 'targets'):
        return dataset.targets[idx]
    else:
        raise AttributeError("Dataset has no 'labels' or 'targets' attribute.")

def cs_shapley_tmc(trnX, trnY, devX, devY, label, num_classes, device, T=50):
    """
    使用 Truncated Monte Carlo Shapley (TMC-Shapley) 方法计算指定类别的 CS-Shapley 值。

    优化点：
    1. 针对输入数据的维度，动态调整代理模型的架构，支持图像和表格数据。
    2. 将非目标类别的数据训练移动到循环外部，只训练一次，减少重复计算。
    3. 预先将数据转换为张量，避免在循环中重复转换。
    4. 使用模型的 state_dict 保存和加载，避免重复训练非目标类别的数据。

    Args:
        trnX (np.ndarray): 训练集特征，形状为 [N, C, H, W] 或 [N, D]。
        trnY (np.ndarray): 训练集标签。
        devX (np.ndarray): 验证集特征，形状为 [M, C, H, W] 或 [M, D]。
        devY (np.ndarray): 验证集标签。
        label (int): 目标类别标签。
        num_classes (int): 类别数量。
        device: 计算设备。
        T (int): 迭代次数（采样的排列数量）。

    Returns:
        np.ndarray: 计算得到的 Shapley 值数组。
        np.ndarray: 对应的原始数据索引。
    """
    # 选择目标类别和非目标类别的数据
    orig_indices = np.where(trnY == label)[0]
    trnX_label = trnX[orig_indices]
    trnY_label = trnY[orig_indices]
    trnX_nonlabel = trnX[trnY != label]
    trnY_nonlabel = trnY[trnY != label]
    devX_label = devX[devY == label]
    devY_label = devY[devY == label]
    devX_nonlabel = devX[devY != label]
    devY_nonlabel = devY[devY != label]

    N = trnX_label.shape[0]
    shapley_values = np.zeros(N)
    criterion = nn.CrossEntropyLoss()

    # 预先将数据转换为张量，避免在循环中重复转换
    trnX_label_tensor = torch.from_numpy(trnX_label).float()
    trnY_label_tensor = torch.from_numpy(trnY_label).long()
    devX_label_tensor = torch.from_numpy(devX_label).float().to(device)
    devY_label_tensor = torch.from_numpy(devY_label).long().to(device)
    devX_nonlabel_tensor = torch.from_numpy(devX_nonlabel).float().to(device)
    devY_nonlabel_tensor = torch.from_numpy(devY_nonlabel).long().to(device)

    input_dim = trnX.shape[1:]
    is_image_data = len(input_dim) == 3  # 检查是否为图像数据

    # 使用简单的代理模型
    def proxy_model_class():
        if is_image_data:
            # 图像数据模型
            C, H, W = input_dim
            return nn.Sequential(
                nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * H * W, num_classes)
            )
        else:
            # 表格数据模型
            D = input_dim[0]
            return nn.Sequential(
                nn.Linear(D, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

    # 首先在非目标类别数据上训练模型，并保存其参数
    base_model = proxy_model_class().to(device)
    optimizer = optim.SGD(base_model.parameters(), lr=0.01)
    base_model.train()

    if len(trnX_nonlabel) > 0:
        trnX_nonlabel_tensor = torch.from_numpy(trnX_nonlabel).float()
        trnY_nonlabel_tensor = torch.from_numpy(trnY_nonlabel).long()
        train_dataset_nonlabel = TensorDataset(trnX_nonlabel_tensor, trnY_nonlabel_tensor)
        train_loader_nonlabel = DataLoader(train_dataset_nonlabel, batch_size=64, shuffle=True)
        for inputs, labels in train_loader_nonlabel:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 保存非目标类别数据训练后的模型参数
    base_model_state_dict = base_model.state_dict()

    for t in tqdm(range(T), desc=f"Calculating CS-Shapley for label {label}"):
        # 随机排列索引
        permutation = np.random.permutation(N)
        # 加载预训练的模型参数
        model = proxy_model_class().to(device)
        model.load_state_dict(base_model_state_dict)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.train()

        # 评估初始模型在验证集上的性能（类内和类间）
        prev_in_performance = evaluate_model_simple(model, devX_label_tensor, devY_label_tensor)
        prev_out_performance = evaluate_model_simple(model, devX_nonlabel_tensor, devY_nonlabel_tensor)

        # 逐步添加目标类别的数据点
        for pos, idx in enumerate(permutation):
            data_point = trnX_label_tensor[idx:idx+1].to(device)
            label_point = trnY_label_tensor[idx:idx+1].to(device)

            # 增量训练：只训练新加入的数据点
            optimizer.zero_grad()
            outputs = model(data_point)
            loss = criterion(outputs, label_point)
            loss.backward()
            optimizer.step()

            # 评估模型性能（类内和类间）
            current_in_performance = evaluate_model_simple(model, devX_label_tensor, devY_label_tensor)
            current_out_performance = evaluate_model_simple(model, devX_nonlabel_tensor, devY_nonlabel_tensor)

            # 计算边际贡献，考虑类内和类间的性能变化
            # marginal_contribution = (current_in_performance - prev_in_performance) - (current_out_performance - prev_out_performance)
            # shapley_values[idx] += (marginal_contribution - shapley_values[idx]) / (t + 1)
            # 考虑类间性能的权重，以类内性能的效果为基准
            marginal_contribution = current_in_performance - prev_in_performance
            weighted_out_contribution = np.exp(current_out_performance - prev_out_performance)
            shapley_values[idx] += (marginal_contribution * weighted_out_contribution - shapley_values[idx]) / (t + 1)

            # 更新前一步的性能
            prev_in_performance = current_in_performance
            prev_out_performance = current_out_performance

    # 平均化 Shapley 值
    shapley_values /= T

    return shapley_values, orig_indices

def evaluate_model_simple(model, devX_tensor, devY_tensor):
    """评估模型在指定验证集上的准确率"""
    model.eval()
    with torch.no_grad():
        inputs = devX_tensor.to(next(model.parameters()).device)
        labels = devY_tensor.to(next(model.parameters()).device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        total_correct = (preds == labels).sum().item()
        total_samples = labels.size(0)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy

# 将数据集转换为 numpy 数组
def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=1)
    all_features = []
    all_labels = []
    for inputs, labels in loader:
        all_features.append(inputs.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


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
    # interval_core_loader = core_loader
    # 准备训练集和验证集
    trnX, trnY = dataset_to_numpy(dst_train)
    devX, devY = dataset_to_numpy(dst_val)
    num_classes = len(set(trnY))
    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    accumulated_core_indices = set()
    T = 50
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        # 计算每个类别的 Shapley 值
        all_vals = []
        all_indices = []
        
        for label in range(num_classes):
            print(f"Calculating CS-Shapley values for label {label}...")
            val, orig_indices = cs_shapley_tmc(trnX, trnY, devX, devY, label, num_classes, device, T=T)
            all_vals.extend(val)
            all_indices.extend(orig_indices)

        all_vals = np.array(all_vals)
        all_indices = np.array(all_indices)

        # 归一化 Shapley 值
        if np.max(all_vals) - np.min(all_vals) > 0:
            all_vals = (all_vals - np.min(all_vals)) / (np.max(all_vals) - np.min(all_vals))
        else:
            all_vals = np.zeros_like(all_vals)

        # 根据 Shapley 值选择样本
        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_core_set_size}")
        selected_indices = all_indices[np.argsort(-all_vals)[:target_core_set_size]]
        
        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        core_indices.update(selected_indices)
        accumulated_core_indices.update(selected_indices)

        print(f"Selected {len(selected_indices)} samples for core set.")

        core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        if best_acc > 0:
                try:
                    model.load_state_dict(torch.load(best_model_path))
                except:
                    pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")
    

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    # 准备训练集和验证集
    trnX, trnY = dataset_to_numpy(dst_train)
    devX, devY = dataset_to_numpy(dst_val)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    T = 50

    s_sart = time.time()

    # 计算每个类别的 Shapley 值
    all_vals = []
    all_indices = []
    num_classes = len(set(trnY))
    for label in range(num_classes):
        print(f"Calculating CS-Shapley values for label {label}...")
        val, orig_indices = cs_shapley_tmc(trnX, trnY, devX, devY, label, num_classes, device, T=T)
        all_vals.extend(val)
        all_indices.extend(orig_indices)

    all_vals = np.array(all_vals)
    all_indices = np.array(all_indices)

    # 归一化 Shapley 值
    if np.max(all_vals) - np.min(all_vals) > 0:
        all_vals = (all_vals - np.min(all_vals)) / (np.max(all_vals) - np.min(all_vals))
    else:
        all_vals = np.zeros_like(all_vals)

    # 根据 Shapley 值选择样本
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_core_set_size}")
    selected_indices = all_indices[np.argsort(-all_vals)[:target_core_set_size]]
    
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    # try:
    #     core_labels = [dst_train.targets[idx] for idx in selected_indices]
    # except:
    #     core_labels = [dst_train.labels[idx] for idx in selected_indices]
    # print("Core set label distribution:", Counter(core_labels))

    print(f"Selected {len(selected_indices)} samples for core set.")

    core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

    e_time = time.time()
    selection_time = e_time - s_sart
    print(f"selection time:{selection_time}")

    train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)

    return selection_time


def CS_Shapley_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, T=50):
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
    try:
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
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
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


        end_time = time.time()
        model.load_state_dict(torch.load(best_model_path))
        test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
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
