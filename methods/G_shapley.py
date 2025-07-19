import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import copy
from collections import Counter
import nets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def compute_shapley_value(model_instantiator, train_dataset, performance_metric, device, num_iterations=100, learning_rate=0.001, batch_size=32):
    """
    使用梯度近似方法计算每个样本的 Shapley 值（使用批量更新）。
    """
    num_samples = len(train_dataset)
    shapley_values = torch.zeros(num_samples, device=device)
    T = num_iterations

    criterion = nn.CrossEntropyLoss()

    for t in tqdm(range(1, T+1), desc="Calculating Shapley Values using Gradient Shapley with Mini-batch"):
        # 随机排列数据点
        permutation = torch.randperm(num_samples)
        v_prev = None

        # 初始化模型参数
        model = model_instantiator.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # 设置模型为训练模式
        model.train()

        # 将 BatchNorm 层设置为评估模式，以避免批大小过小的问题
        def set_bn_eval(m):
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()
        model.apply(set_bn_eval)

        # 初始化性能或训练损失
        v_prev = performance_metric(model)

        # 将 permutation 分成批次
        batch_indices_list = [permutation[i:i+batch_size] for i in range(0, num_samples, batch_size)]

        for batch_indices in batch_indices_list:
            # 获取当前批次的数据
            inputs = []
            labels = []
            for idx in batch_indices:
                input_j, label_j = train_dataset[idx]
                inputs.append(input_j)
                labels.append(label_j)
            batch_inputs = torch.stack(inputs).to(device)
            batch_labels = torch.tensor(labels).to(device)

            # 梯度更新参数
            optimizer.zero_grad()
            output = model(batch_inputs)

            # 检查模型输出是否包含 NaN 或无穷大
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Warning: Model output contains NaN or Inf at iteration {t}")
                continue

            loss = criterion(output, batch_labels)

            # 检查损失是否为 NaN 或无穷大
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: Loss is NaN or Inf at iteration {t}")
                continue

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # 计算当前性能
            v_current = performance_metric(model)

            # 计算边际贡献（使用性能的变化）
            marginal_contribution = v_current - v_prev  # 注意这里取差值

            # 更新当前批次中数据点的 Shapley 值（递归平均）
            for idx in batch_indices:
                shapley_values[idx] = ((t - 1) / t) * shapley_values[idx] + (1 / t) * (marginal_contribution / len(batch_indices))

            # 更新前一性能
            v_prev = v_current

    # 归一化 Shapley 值
    shapley_min = shapley_values.min()
    shapley_max = shapley_values.max()
    shapley_values = (shapley_values - shapley_min) / (shapley_max - shapley_min + 1e-8)

    return shapley_values.cpu()

def gradient_shapley_selection(model_instantiator, train_dataset, target_core_set_size, performance_metric, device):
    """
    使用梯度近似 Shapley 值选择核心数据集。
    """
    shapley_values = compute_shapley_value(model_instantiator, train_dataset, performance_metric, device)
    selected_indices = torch.topk(shapley_values, target_core_set_size).indices.tolist()
    return selected_indices

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
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    batch_gradients_cache = []
    checkpoint_cache = []
    # interval_core_loader = core_loader

    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        # 定义性能评估函数
        def performance_metric(model):
            model.eval()
            total_loss = 0
            test_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
            avg_loss = total_loss / len(dst_val)
            return -avg_loss  # 使用负的平均损失作为性能指标

        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_core_set_size}")

        # 调整 batch_size
        batch_size = 32  # 可以根据实际情况调整
        selected_indices = gradient_shapley_selection(model, dst_train, target_core_set_size, performance_metric, device)

        core_labels = [dst_train[idx][1] for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        print(f"Selected {len(selected_indices)} samples for core set.")

        core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

def once_selection_and_train(args, model, dst_train, dst_val, dst_test,criterion, optimizer, scheduler, device,best_model_path):
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    # 定义性能评估函数
    def performance_metric(model):
        model.eval()
        total_loss = 0
        test_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dst_val)
        return -avg_loss  # 使用负的平均损失作为性能指标

    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_core_set_size}")

    # 调整 batch_size
    batch_size = 32  # 可以根据实际情况调整
    selected_indices = gradient_shapley_selection(model, dst_train, target_core_set_size, performance_metric, device)

    core_labels = [dst_train[idx][1] for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    print(f"Selected {len(selected_indices)} samples for core set.")

    core_loader = DataLoader(Subset(dst_train, selected_indices), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)


def G_shapley_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, balance=True, k_neighbors=5, num_permutations=10):
    import logging
    from collections import Counter
    import os
    import time
    start_time = time.time()

    # log_path = args.save_log_path + args.dataset + '/' +  args.method + '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +  '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    log_path = args.save_log_path + args.dataset + '/' +  str(args.selection_fraction)  +'/' + args.method + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
    
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

    train_loader = DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 随机选择一部分数据进行预训练
    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

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
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path)


    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=4, pin_memory=True),device)
    print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    os.remove(best_model_path)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")

