import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
import nets
import time

class IncrementalTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        # 初始化累积梯度 z 和平方梯度 n
        self.z = [torch.zeros_like(p.data) for p in model.parameters()]
        self.n = [torch.zeros_like(p.data) for p in model.parameters()]
        self.epsilon = 1e-8  # 防止除以零

    def update(self, inputs, labels):
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        # 计算梯度
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        
        # 更新累积梯度 z 和平方梯度 n
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                self.z[i] += grads[i]
                self.n[i] += grads[i] ** 2

                # 根据 Algorithm 1 更新参数
                adjusted_lr = self.optimizer.param_groups[0]['lr'] / (torch.sqrt(self.n[i]) + self.epsilon)
                param.data -= adjusted_lr * self.z[i]

        return loss.item()

def calculate_adaptive_score(rewards, selected_counts, gamma, diversity_scores, alpha):
    """计算自适应得分：开发 + 探索 + 多样性"""
    exploitation = rewards
    exploration = gamma / (selected_counts + 1)
    diversity = alpha * diversity_scores
    return exploitation + exploration + diversity

def IAS_score(train_dataset, val_loader, model, target_size, batch_size, gamma, lr, device, n_clusters, lambda_, alpha):
    # 数据集的特征提取和聚类
    features = []
    labels = []
    for inputs, targets in DataLoader(train_dataset, batch_size=128, shuffle=False):
        features.append(inputs.view(inputs.size(0), -1).numpy())
        labels.extend(targets.numpy())
    features = np.vstack(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    cluster_labels = kmeans.labels_

    # 计算簇中心和整体均值
    cluster_centroids = kmeans.cluster_centers_
    overall_mean = np.mean(features, axis=0)
    cluster_distances = np.linalg.norm(cluster_centroids - overall_mean, axis=1)

    # 计算每个数据点到其簇中心的距离，作为多样性得分
    diversity_scores = np.zeros(len(train_dataset))
    for idx in range(len(train_dataset)):
        cluster = cluster_labels[idx]
        data_point = features[idx]
        diversity_scores[idx] = np.linalg.norm(data_point - cluster_centroids[cluster])

    # 初始化模型和优化器
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainer = IncrementalTrainer(model, optimizer, criterion, device)

    # 初始化自适应得分
    data_indices = np.arange(len(train_dataset))
    rewards = np.zeros(len(train_dataset))  # 每个数据点的奖励
    selected_counts = np.zeros(len(train_dataset))  # 每个数据点的选择次数
    scores = np.zeros(len(train_dataset))

    selected_data_indices = set()  # 已选数据点的索引集合

    epoch = 0
    pbar = tqdm(total=target_size, desc="IAS Selection")
    while len(selected_data_indices) < target_size:
        if epoch == 0:
            # 第一轮：选择与整体分布最不同的簇的数据点
            selected_cluster = np.argmax(cluster_distances)
            cluster_indices = data_indices[cluster_labels == selected_cluster]
            # 从该簇中选取未被选过的数据点
            unselected_indices = list(set(cluster_indices) - selected_data_indices)
            selected_indices = unselected_indices[:batch_size]
        else:
            # 后续轮次：基于自适应得分采样
            # 从未被选过的数据中，根据scores选择得分最高的batch_size个数据点
            unselected_indices = list(set(data_indices) - selected_data_indices)
            if len(unselected_indices) == 0:
                break  # 所有数据都已被选取
            unselected_scores = scores[unselected_indices]
            selected_indices = np.array(unselected_indices)[np.argsort(unselected_scores)[-batch_size:]]

        # 更新已选数据点集合
        selected_data_indices.update(selected_indices)

        # 更新选择次数
        selected_counts[selected_indices] += 1

        selected_loader = DataLoader(Subset(train_dataset, selected_indices), batch_size=batch_size, shuffle=False)

        # 增量更新模型，并计算每个数据点的loss用于奖励
        for inputs, labels_batch in selected_loader:
            loss_value = trainer.update(inputs, labels_batch)
            loss_value = loss_value / len(selected_indices)
            # 更新选中数据点的奖励
            for idx in selected_indices:
                # 假设奖励为负的loss
                rewards[idx] = lambda_ * rewards[idx] + (1 - lambda_) * (-loss_value)

        # 评估验证集性能（可选，如果需要的话）
        # ...

        # 计算所有数据点的自适应得分
        scores = calculate_adaptive_score(
            rewards,
            selected_counts,
            gamma,
            diversity_scores,
            alpha
        )

        # 更新进度条
        pbar.update(len(selected_indices))
        epoch += 1

    pbar.close()
    return list(selected_data_indices), scores

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
    

def IAS_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    import logging
    from collections import Counter
    import os

    start_time = time.time()

    # log_path = args.save_log_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target) + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  args.model + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
    
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))
    # print(f"Log path: {log_path}")

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

    # 初始化模型
    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)

    # 定义数据加载器
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 创建验证集（从测试集中随机选择60%的数据）
    # if args.balance_target:
    #     val_set = dst_val
    # else:
    #     # 创建验证集（可以根据需要调整验证集的比例）
    #     val_size_ratio = 1
    #     val_class_size = []
    #     # 检查 dst_test 是否有 'labels' 属性，如果没有，则使用 'targets'
    #     if hasattr(dst_test, 'labels'):
    #         test_labels = dst_test.labels
    #     else:
    #         test_labels = dst_test.targets

    #     for c in range(num_classes):
    #         num_c = len(np.where(np.array(test_labels) == c)[0])
    #         val_class_size.append(int(num_c * val_size_ratio))

    #     val_indices = []
    #     for c in range(num_classes):
    #         class_indices = np.where(np.array(test_labels) == c)[0]
    #         selected_indices = np.random.choice(class_indices, val_class_size[c], replace=False)
    #         val_indices.extend(selected_indices)

    #     val_set = Subset(dst_test, val_indices)
    #     val_labels = [int(dst_test[idx][1]) for idx in val_indices]
    #     print("Validation set label distribution:", Counter(val_labels))

    val_set = dst_val

    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    select_start = time.time()

    # 设置 IAS_score 所需的参数
    batch_size = args.ias_batch_size if hasattr(args, 'ias_batch_size') else 128
    gamma = args.gamma if hasattr(args, 'gamma') else 0.1
    lr = args.ias_lr if hasattr(args, 'ias_lr') else 0.01
    lambda_ = args.lambda_ if hasattr(args, 'lambda_') else 0.9  # 遗忘因子
    alpha = args.alpha if hasattr(args, 'alpha') else 0.5  # 多样性权重

    # 设置目标核心集大小
    target_number = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_number}")

    # 计算每个样本的得分并选取核心集
    selected_data_indices, scores = IAS_score(
        dst_train,
        val_loader,
        model,
        target_size=target_number,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        device=device,
        n_clusters=num_classes,
        lambda_=lambda_,
        alpha=alpha
    )


    # 将选择的数据点按类别比例进行平衡
    # train_labels = np.array([int(dst_train[idx][1]) for idx in range(len(dst_train))])
    # class_counts = Counter(train_labels)
    # total_samples = len(dst_train)

    # # 计算训练集中每个类别的样本数量
    # core_set_per_class = {}
    # for c in range(num_classes):
    #     proportion = class_counts[c] / total_samples
    #     num_samples_c = int(target_number * proportion)
    #     core_set_per_class[c] = num_samples_c

    # # 调整由于取整导致的样本数量差异
    # allocated_samples = sum(core_set_per_class.values())
    # remaining_samples = target_number - allocated_samples
    # if remaining_samples > 0:
    #     # 将剩余的样本分配给样本数量较多的类别
    #     sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    #     for c, count in sorted_classes:
    #         core_set_per_class[c] += 1
    #         remaining_samples -= 1
    #         if remaining_samples == 0:
    #             break

    # # 按类别选取每个类别的得分最高的样本
    # core_set_indices = []
    # selected_data_indices = np.array(list(selected_data_indices))
    # selected_scores = scores[selected_data_indices]
    # selected_labels = train_labels[selected_data_indices]

    # for c in range(num_classes):
    #     class_indices = selected_data_indices[selected_labels == c]
    #     class_scores = selected_scores[selected_labels == c]
    #     sorted_class_indices = class_indices[np.argsort(class_scores)[::-1]]
    #     num_to_select = core_set_per_class[c]
    #     selected_indices = sorted_class_indices[:num_to_select]
    #     core_set_indices.extend(selected_indices)

    # 按照scores排序选取核心集
    selected_data_indices = np.array(list(selected_data_indices))
    selected_scores = scores[selected_data_indices]
    sorted_indices = selected_data_indices[np.argsort(selected_scores)[::-1]]
    core_set_indices = sorted_indices[:target_number]
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in core_set_indices]
    print("Core set label distribution:", Counter(core_labels))

    core_labels = [int(dst_train[idx][1]) for idx in core_set_indices]
    print("Core set label distribution:", Counter(core_labels))

    core_loader = DataLoader(Subset(dst_train, core_set_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

    # 使用核心集重新训练模型
    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    
    select_end = time.time()
    selection_time = select_end-select_start
    print(f"selection time:{selection_time}")
                    
    train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
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