import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets
from methods.methods_utils import euclidean_dist
from torchvision import transforms
from copy import deepcopy

class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        # Exclude the last fully connected layer
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def active_learning_selection(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device , best_model_path):
    print("Active learning selection")
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    target_core_set_size_per_selection = int(target_core_set_size * args.active_fraction_ratio)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    target_core_set_size_per_selection = max(100, target_core_set_size_per_selection)  # 最小选择 50 个样本

    best_acc = 0.0
    trained_epochs = 0
    # 进行多轮挑选
    while len(core_indices) < target_core_set_size:
        model.eval()
        # Use the model to calculate uncertainty scores
        train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
        scores, labels, indices = [], [], []

        with torch.no_grad():
            for i, (datas, targets) in enumerate(tqdm(train_loader,desc="Selecting samples")):
                datas = datas.to(device)
                outputs = model(datas)
                # Calculate Entropy scores
                preds = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                entropy_scores = -np.sum(preds * np.log(preds + 1e-6), axis=1)
                scores.extend(entropy_scores)
                labels.extend(targets.numpy())
                indices.extend(range(i * 128, min((i + 1) * 128, len(dst_train))))

        # Convert scores to numpy array and select based on least confidence
        scores = np.array(scores)
        sorted_indices = np.argsort(scores)
        # 过滤掉已经存在于core_indices中的索引
        filtered_indices = [idx for idx in sorted_indices if idx not in core_indices]
        # 从过滤后的索引中选取前 target_core_set_size_per_selection 个
        new_selection = filtered_indices[:target_core_set_size_per_selection]
        # 更新core_indices
        core_indices.update(new_selection)
        
        best_acc = train_model(model, DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True), criterion, optimizer, scheduler, 1, device, val_loader, best_model_path, patience=10, best_acc=best_acc)
        print(f"Selecting {len(core_indices)}/{target_core_set_size} samples")
    
    return core_indices
    
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

def svp_max_entropy_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train,dst_val, dst_test, balance=False):
    import logging
    from collections import Counter
    import os
    import time
    
    start_time = time.time()
    # log_path = args.save_log_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target) + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
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
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("CUDA is available:", torch.cuda.is_available())

    select_start = time.time()
    # Initialize model for embedding extraction
    proxy_model = nets.__dict__[args.proxy_model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)

    # 使用优化器和损失函数定义训练循环
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(proxy_model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    core_indicies = active_learning_selection(args, proxy_model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    os.remove(best_model_path)
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in core_indicies]
    print("Core set label distribution:", Counter(core_labels))

    # Create a subset using selected indices
    subset_data = Subset(dst_train, list(core_indicies))
    core_loader = DataLoader(subset_data, batch_size=128, shuffle=True)
    select_end = time.time()
    selection_time = select_end - select_start
    print(f"selection time:{selection_time}")
    
    # Train the model using core set data
    train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(train_model_instance.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    train_model(train_model_instance, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path, patience=args.patience)
    train_model_instance.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(train_model_instance, test_loader, device)
    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    print(f"Total time: {time.time() - start_time:.2f}s")
    os.remove(best_model_path)
    end_time = time.time()

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
