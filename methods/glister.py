import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets  # 假设您有一个包含模型定义的模块
from torchvision import transforms
import gc
import time

# 设置 CUDA_LAUNCH_BLOCKING 为 1 以便更准确地定位错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def batch_select_data(selector, remaining_indices, batch_size):
    remaining_indices = np.array(remaining_indices)
    valid_indices = remaining_indices[remaining_indices < selector.train_gradients.size(0)]

    if len(valid_indices) <= batch_size:
        return valid_indices

    gains = selector.calc_gain(valid_indices)
    top_indices = np.argpartition(gains, -batch_size)[-batch_size:]
    top_indices = top_indices[np.argsort(-gains[top_indices])]
    return valid_indices[top_indices]


class FacilityLocation:
    def __init__(self, train_gradients, val_loss_gradient, device='cpu', batch_size=100):
        self.train_gradients = train_gradients  # 保留在 CPU
        self.val_loss_gradient = val_loss_gradient  # 验证集损失梯度
        self.device = device
        self.batch_size = batch_size  # 分批处理大小

    def calc_gain(self, idx_gain):
        idx_gain = np.array(idx_gain)
        valid_indices = idx_gain[idx_gain < self.train_gradients.size(0)]  # 检查索引范围

        # 分批计算增益
        gains = []
        for i in range(0, len(valid_indices), self.batch_size):
            batch_indices = valid_indices[i:i + self.batch_size]
            train_subset_gradient = self.train_gradients[batch_indices].to(self.device)
            batch_gain = torch.matmul(train_subset_gradient, self.val_loss_gradient)
            gains.append(batch_gain.cpu().numpy())

            # 清理显存
            del train_subset_gradient, batch_gain
            torch.cuda.empty_cache()

        return np.concatenate(gains)

    def update_val_gradient(self, selected_indices):
        # 更新验证集梯度，分批处理
        subset_gradient = []
        for i in range(0, len(selected_indices), self.batch_size):
            batch_indices = selected_indices[i:i + self.batch_size]
            batch_gradient = self.train_gradients[batch_indices].mean(dim=0)
            subset_gradient.append(batch_gradient.cpu())

        subset_gradient = torch.stack(subset_gradient, dim=0).mean(dim=0).to(self.device)
        self.val_loss_gradient -= subset_gradient

        # 清理显存
        del subset_gradient
        torch.cuda.empty_cache()

def calc_gradient(data_loader, model, criterion, device):
    model.eval()
    all_gradients = []
    total_samples = len(data_loader.dataset)

    # 提取需要梯度的参数列表
    params = [param for param in model.parameters() if param.requires_grad]
    total_param_size = sum(p.numel() for p in params)

    for inputs, targets in tqdm(data_loader, desc="Calculating Gradients"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        batch_size = inputs.size(0)

        for i in range(batch_size):
            x = inputs[i].unsqueeze(0)  # [1, C, H, W]
            y = targets[i].unsqueeze(0)  # [1]
            outputs = model(x)
            loss = criterion(outputs, y)
            grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
            flat_grads = torch.cat([g.view(-1) for g in grads], dim=0)  # [total_param_size]
            all_gradients.append(flat_grads.cpu())

            # 清理显存
            del x, y, outputs, loss, grads, flat_grads
            torch.cuda.empty_cache()

    gradients = torch.stack(all_gradients, dim=0)  # [total_samples, total_param_size]

    # 检查是否缺失样本
    if gradients.size(0) != total_samples:
        missing_samples = total_samples - gradients.size(0)
        print(f"Warning: Missing {missing_samples} samples in gradient calculation.")
        print(f"Gradients calculated: {gradients.size(0)}; Total samples: {total_samples}")
        # 用零填充缺失样本
        gradients = torch.cat([gradients, torch.zeros(missing_samples, gradients.size(1))], dim=0)

    assert gradients.size(0) == total_samples, (
        f"Gradient size {gradients.size(0)} does not match dataset size {total_samples}"
    )
    return gradients
    
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


def glister_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
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

    select_start = time.time()
    # 初始化代理模型
    proxy_model = nets.__dict__[args.proxy_model](channel=channel, num_classes=num_classes, im_size=im_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(proxy_model.parameters(), lr=args.selection_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.proxy_epochs)

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset=dst_train,
        batch_size=128,
        shuffle=False,  # 确保顺序不变
        drop_last=False,  # 保证不会丢弃样本
        num_workers=1
    )
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False)

    # 训练代理模型
    train_model(
        proxy_model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.proxy_epochs,
        device=device,
        val_loader=val_loader,
        best_model_path= best_model_path
    )
    os.remove(best_model_path)

    # 计算梯度
    train_gradients = calc_gradient(train_loader, proxy_model, criterion, device)
    val_gradients = calc_gradient(val_loader, proxy_model, criterion, device).mean(dim=0).to(device)

    # 初始化选择器
    selector = FacilityLocation(train_gradients=train_gradients, val_loss_gradient=val_gradients, device=device, batch_size=100)

    # 子集选择（修改后的逻辑）
    budget = int(len(dst_train) * args.selection_fraction)
    remaining_indices = np.arange(len(dst_train))
    gains = selector.calc_gain(remaining_indices)
    top_budget_indices = remaining_indices[np.argsort(-gains)][:budget]
    selected_indices = top_budget_indices.tolist()

    select_end = time.time()
    selection_time = select_end - select_start
    print(f"selection time:{selection_time}")

    print(f"Selected {len(selected_indices)} samples out of {len(dst_train)}")
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))

    # 使用子集训练主模型
    subset_data = Subset(dst_train, selected_indices)
    core_loader = DataLoader(subset_data, batch_size=128, shuffle=True)
    main_model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size).to(device)
    optimizer = optim.Adam(main_model.parameters(), lr=args.selection_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 使用你的 train_model 函数训练主模型
    train_model(
        main_model,
        core_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        device=device,
        val_loader=val_loader,
        best_model_path= best_model_path,
        patience=args.patience
    )

    # 评估主模型
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False)
    acc, precision, recall, f1 = evaluate_model(main_model, test_loader, device)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
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