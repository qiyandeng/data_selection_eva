import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import Counter
import nets
import logging
import time

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class DataValueEstimator(nn.Module):
    def __init__(self, input_dim):
        super(DataValueEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.model(x).squeeze()


def calc_loss(y, t):
    y = torch.clamp(y, min=-10, max=10)  # 防止溢出
    y = torch.nn.functional.log_softmax(y, dim=1)
    return torch.nn.functional.nll_loss(y, t, reduction='mean')


def train_dvrl(data_value_estimator, predictor_model, train_loader, val_loader, args, device):
    optimizer_dve = optim.Adam(data_value_estimator.parameters(), lr=args.dve_lr)
    optimizer_predictor = optim.SGD(predictor_model.parameters(), lr=args.selection_lr,
                                    momentum=args.selection_momentum, weight_decay=args.selection_weight_decay)
    scheduler_predictor = lr_scheduler.CosineAnnealingLR(optimizer_predictor, T_max=args.epochs)

    moving_avg_loss = 0.0
    T = 20  # 滑动窗口大小
    inner_iteration = 50  # 内部迭代次数
    outer_iteration = 10  # 外部迭代次数
    rl_select_size = int(args.selection_fraction * len(train_loader.dataset))

    for epoch in tqdm(range(outer_iteration), desc="DVRL Training"):
        # **第一步：选择样本**
        data_value_estimator.eval()
        selected_indices = []
        current_index = 0  # 用于跟踪样本的全局索引

        # print("Step 1: Selecting samples")
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                selection_probs = data_value_estimator(inputs)  # 获取样本的选择概率
            selection_probs = selection_probs.cpu().numpy().squeeze()

            # 根据选择概率对每个样本进行采样
            selected_mask = np.random.binomial(1, selection_probs)
            batch_indices = np.arange(current_index, current_index + len(inputs))
            selected_batch_indices = batch_indices[selected_mask.astype(bool)]
            selected_indices.extend(selected_batch_indices)
            current_index += len(inputs)

        if len(selected_indices) == 0:
            print("No samples selected, skipping epoch.")
            continue

        # 创建一个新的 DataLoader，仅包含被选择的样本
        if len(selected_indices) < rl_select_size:
            selected_indices = selected_indices
        else:
            selected_indices = np.random.choice(selected_indices, rl_select_size, replace=False)

        selected_dataset = torch.utils.data.Subset(train_loader.dataset, selected_indices)
        selected_train_loader = torch.utils.data.DataLoader(
            selected_dataset, batch_size=train_loader.batch_size, shuffle=True)

        # **第二步：训练预测器**
        predictor_model.train()
        for _ in range(inner_iteration):  # 内部迭代 NI 次
            for inputs, labels in selected_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_predictor.zero_grad()
                outputs = predictor_model(inputs)
                loss = calc_loss(outputs, labels)
                loss.backward()
                optimizer_predictor.step()

        scheduler_predictor.step()

        # **第三步：计算奖励信号**
        predictor_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = predictor_model(val_inputs)
                val_loss += calc_loss(val_outputs, val_labels).item()

        val_loss /= len(val_loader)
        reward = moving_avg_loss - val_loss  # 奖励函数
        moving_avg_loss = (1 - 1 / T) * moving_avg_loss + (1 / T) * val_loss

        # **第四步：更新数据选择器**
        data_value_estimator.train()
        optimizer_dve.zero_grad()

        # 逐批处理，以避免一次性将所有数据加载到 GPU 中
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            selection_probs = data_value_estimator(inputs)
            outputs = predictor_model(inputs)

            # 计算每个样本的损失
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(outputs, labels)

            # 计算加权损失
            weighted_losses = selection_probs.squeeze() * losses * reward
            weighted_loss = weighted_losses.mean()
            weighted_loss.backward()
            optimizer_dve.step()


def select_core_set_top(data_value_estimator, data_loader, target_number, device):
    data_value_estimator.eval()
    scores = []
    indices = []

    with torch.no_grad():
        for idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            values = data_value_estimator(inputs)
            scores.extend(values.cpu().numpy())
            # 这里需要记录全局的索引，而不是每个batch的局部索引
            indices.extend(range(idx * data_loader.batch_size, idx * data_loader.batch_size + inputs.size(0)))

    scores = np.array(scores)
    top_indices = np.argsort(scores)[-target_number:]  # 选择概率最高的 target_number 个索引
    return top_indices

def select_core_set_balance(data_value_estimator, data_loader, target_number, device, num_classes):
    data_value_estimator.eval()
    scores = []
    labels_list = []
    indices = []

    # 使用 IndexedDataset 包装原始数据集
    dataset = data_loader.dataset
    indexed_dataset = IndexedDataset(dataset)
    indexed_data_loader = DataLoader(
        indexed_dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,  # 不打乱数据，以确保索引对应
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory
    )

    with torch.no_grad():
        for inputs, labels, batch_indices in indexed_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            values = data_value_estimator(inputs)
            scores.extend(values.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            indices.extend(batch_indices.numpy())

    scores = np.array(scores)
    labels_list = np.array(labels_list)
    indices = np.array(indices)

    # 获取训练集中每个类别的样本数量
    class_counts = Counter(labels_list)
    total_samples = len(labels_list)

    # 计算每个类别在核心集中应有的样本数量，保持与训练集相同的比例
    core_set_indices = []
    for c in range(num_classes):
        class_indices = indices[labels_list == c]
        class_scores = scores[labels_list == c]
        num_samples_c = int(target_number * class_counts[c] / total_samples)
        print(f"Class {c}: {num_samples_c} samples")
        if len(class_indices) == 0 or num_samples_c == 0:
            continue
        # 选择得分最高的 num_samples_c 个样本
        sorted_indices = np.argsort(class_scores)
        top_class_indices = class_indices[sorted_indices][-num_samples_c:]
        core_set_indices.extend(top_class_indices)

    return core_set_indices

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
    

def DVRL_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    from collections import Counter
    import os
    start_time = time.time()
    
    # log_path = args.save_log_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target) + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    log_path = args.save_log_path + args.dataset + '/' + args.model + '/' +  str(args.selection_fraction)  +'/' + args.method + '.log'
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
        print("###############################Start  Selection...###############################")
        print("")
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

        select_start = time.time()

        model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
        input_dim = int(np.prod(im_size)) * int(channel) if isinstance(im_size, (tuple, list)) else int(im_size) * int(channel)
        data_value_estimator = DataValueEstimator(input_dim=input_dim).to(device)

        train_loader = DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)
        # val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        val_set = dst_val

        val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        dvrl_train_loader = DataLoader(dst_train, batch_size=128, shuffle=True)

        train_dvrl(data_value_estimator, model, dvrl_train_loader, test_loader, args, device)
        
        target_number = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_number}")

        balance = args.balance_target

        if balance:
            core_set_indices = select_core_set_balance(data_value_estimator, train_loader, target_number, device, num_classes)
        else:
            core_set_indices = select_core_set_top(data_value_estimator, train_loader, target_number, device)

        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in core_set_indices]
        print("Core set label distribution:", Counter(core_labels))

        select_end = time.time()
        selection_time = select_end - select_start

        core_loader = DataLoader(Subset(dst_train, core_set_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        # 使用核心集重新训练目标模型
        model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss().to(device)
        
        train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)

        model.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
        print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        os.remove(best_model_path)
        end_time = time.time()
        print(f"Selection time elapsed: {select_end - select_start:.2f} seconds.")
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

            
    except Exception as e:
        # 记录异常信息到日志
        logging.error(f"An error occurred in Arnoldi_selection: {str(e)}")
        logging.error("Stack trace: ", exc_info=True)  # 记录完整的错误堆栈信息

        