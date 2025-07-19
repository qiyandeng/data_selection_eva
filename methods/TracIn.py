import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets
from functorch import vmap, grad
import time

def calc_loss(y, t):
    y = torch.nn.functional.log_softmax(y, dim=1)
    loss = torch.nn.functional.nll_loss(y, t, reduction='mean')
    return loss

def trained_model_with_caching(model, data_loader, optimizer, num_epochs, device):
    """
    预训练模型并缓存每个批次的梯度和模型参数
    """
    batch_gradients_cache = []  # 存储每个检查点的批次梯度
    checkpoint_cache = []  # 存储模型状态的缓存
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    for epoch in tqdm(range(num_epochs), desc="Training pretrain model"):
        model.train()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 缓存梯度和模型参数
        model.eval()
        batch_gradients = []
        for inputs, labels in tqdm(data_loader, desc=f"Caching batch gradients for epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 将每个参数的梯度展平并连接起来，形成一个一维向量
            batch_grad = []
            for param in model.parameters():
                if param.grad is not None:
                    batch_grad.append(param.grad.view(-1))
            batch_grad = torch.cat(batch_grad).detach().cpu()
            batch_gradients.append(batch_grad)

        batch_gradients_cache.append(batch_gradients)
        checkpoint_cache.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

    return batch_gradients_cache, checkpoint_cache

def compute_tracin_scores_fast(model,num_train_samples, batch_gradients_cache, checkpoint_cache, target_loader, optimizer, device):
    """
    手动实现 TracIn 快速分数计算
    """
    model.eval()
    # num_train_samples = len(target_loader.dataset)  # 实际训练样本数量

    # 初始化影响分数数组
    influence_scores = np.zeros(num_train_samples)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    # 对于每个检查点
    for epoch_idx, checkpoint in enumerate(checkpoint_cache):
        # 加载模型状态
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        # 获取训练集的批量梯度
        train_batch_gradients = batch_gradients_cache[epoch_idx]

        # 计算目标数据的批量梯度
        target_batch_gradients = []
        for inputs, labels in tqdm(target_loader, desc=f"Computing target batch gradients for epoch {epoch_idx}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 将每个参数的梯度展平并连接起来，形成一个一维向量
            batch_grad = []
            for param in model.parameters():
                if param.grad is not None:
                    batch_grad.append(param.grad.view(-1))
            batch_grad = torch.cat(batch_grad).detach().cpu()
            target_batch_gradients.append(batch_grad)

        # 计算训练批次梯度和目标批次梯度的内积
        idx = 0  # 用于跟踪样本索引
        for train_grad in train_batch_gradients:
            for target_grad in target_batch_gradients:
                influence = torch.dot(train_grad, target_grad).item()
                if idx < num_train_samples:
                    influence_scores[idx] += influence
                    idx += 1
                else:
                    break
            if idx >= num_train_samples:
                break

    return influence_scores


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

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, device, val_loader, best_model_path, patience=10,best_acc=0.0):
    trigger_times = 0

    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        for inputs, labels_batch in data_loader:
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 数值稳定性处理：防止输出过大或过小
            outputs = torch.clamp(outputs, min=-1000, max=1000)

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
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return best_acc

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    core_indices = set()
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    accumulated_core_indices = set()
    batch_gradients_cache = []
    checkpoint_cache = []


    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        # 预训练模型并缓存梯度（随机选择部分样本）

        new_batch_gradients_cache, new_checkpoint_cache = trained_model_with_caching(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            num_epochs=1,
            device=device
        )
        batch_gradients_cache.extend(new_batch_gradients_cache)
        checkpoint_cache.extend(new_checkpoint_cache)

        # 初始化核心数据集索引集合
        target_core_set_size = int(len(dst_train) * args.selection_fraction)
        print(f"Target core set size: {target_core_set_size}")
        
        num_train_samples = target_core_set_size

        # 计算 TracIn 分数
        target_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
        # 计算 TracIn 分数
        tracin_scores = compute_tracin_scores_fast(model,num_train_samples, batch_gradients_cache, checkpoint_cache, target_loader, optimizer, device)

        # 获取 TracIn 分数最高的样本索引
        sorted_indices = np.argsort(tracin_scores)[::-1]
        core_indices = sorted_indices[:target_core_set_size]
        print(f"Selected {len(core_indices)} samples for core set.")
        
        accumulated_core_indices.update(core_indices)

        # 使用核心数据集进行训练
        core_loader = DataLoader(Subset(dst_train, core_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        criterion_core = nn.CrossEntropyLoss().to(device)
        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, core_loader, criterion_core, optimizer, scheduler, epochs_per_selection, device, test_loader, best_model_path, patience=epochs_per_selection,best_acc=best_acc)
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")
    
def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    select_start = time.time()
    core_indices = set()
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    target_core_set_size = int(len(dst_train) * args.selection_fraction)

    batch_gradients_cache = []
    checkpoint_cache = []

    # best_model_path = args.save_best_model_path
    # 预训练模型并缓存梯度（随机选择部分样本）

    new_batch_gradients_cache, new_checkpoint_cache = trained_model_with_caching(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        num_epochs=1,
        device=device
    )
    batch_gradients_cache.extend(new_batch_gradients_cache)
    checkpoint_cache.extend(new_checkpoint_cache)

    # 初始化核心数据集索引集合
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    print(f"Target core set size: {target_core_set_size}")
    
    num_train_samples = target_core_set_size

    # 计算 TracIn 分数
    target_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    # 计算 TracIn 分数
    tracin_scores = compute_tracin_scores_fast(model,num_train_samples, batch_gradients_cache, checkpoint_cache, target_loader, optimizer, device)

    # 获取 TracIn 分数最高的样本索引
    sorted_indices = np.argsort(tracin_scores)[::-1]
    core_indices = sorted_indices[:target_core_set_size]
    print(f"Selected {len(core_indices)} samples for core set.")
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in core_indices]
    print("Core set label distribution:", Counter(core_labels))

    # 使用核心数据集进行训练
    core_loader = DataLoader(Subset(dst_train, core_indices), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    criterion_core = nn.CrossEntropyLoss().to(device)
    select_end = time.time()
    selection_time = select_end - select_start
    print(f"selection time:{selection_time}")
    train_model(model, core_loader, criterion_core, optimizer, scheduler, args.epochs , device, test_loader, best_model_path,patience=args.patience)

    return selection_time

def TracIn_selection(
    args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, k=10, interval_epoch=3
):
    import logging
    from collections import Counter
    import os

    start_time = time.time()
    selection_time = 0.0  # 默认设置为 0.0，如果是 interval 模式不会覆盖

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

        model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=args.selection_lr,
            momentum=args.selection_momentum,
            weight_decay=args.selection_weight_decay,
            nesterov=True
        )
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss().to(device)

        if args.selection_framework == 'interval':
            interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
        elif args.selection_framework == 'once':
            selection_time = once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)

        
        model.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True), device)
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


            
    except Exception as e:
        # 记录异常信息到日志
        logging.error(f"An error occurred in Arnoldi_selection: {str(e)}")
        logging.error("Stack trace: ", exc_info=True)  # 记录完整的错误堆栈信息


