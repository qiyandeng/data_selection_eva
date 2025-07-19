import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets
import gc

def calc_loss(y, t):
    y = torch.nn.functional.log_softmax(y, dim=1)
    loss = torch.nn.functional.nll_loss(y, t, reduction='mean')
    return loss


def grad_z(z, t, model, criterion, gpu=-1):
    model.eval()
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y = model(z)
    loss = criterion(y, t)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    return grads


def hvp(loss, params, v):
    grad1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    elem_prod = sum(torch.sum(g1 * v1) for g1, v1 in zip(grad1, v))
    hv = torch.autograd.grad(elem_prod, params, retain_graph=True)
    return hv


def lissa(model, z_loader, grad_z_test, criterion, gpu=-1, recursion_depth=100, damping=0.01, scale=25.0, tolerance=1e-4):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    h_estimate = [torch.zeros_like(p) for p in params]

    for i in tqdm(range(recursion_depth), desc="LiSSA iterations"):
        prev_h = [h.clone() for h in h_estimate]

        for z, t in z_loader:
            if gpu >= 0:
                z, t = z.cuda(non_blocking=True), t.cuda(non_blocking=True)
            y = model(z)
            loss = criterion(y, t)
            hv = hvp(loss, params, h_estimate)

            hv = [h / len(z) for h in hv]
            h_estimate = [
                _v + (1 - damping) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(grad_z_test, h_estimate, hv)
            ]
            h_estimate = [h.detach().clone() for h in h_estimate]
            break  # 仅采样一个批次

        if all(
            (torch.norm(_h - _prev_h) < tolerance).item()
            for _h, _prev_h in zip(h_estimate, prev_h)
        ):
            print(f"在迭代 {i} 时收敛")
            break

    return h_estimate

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

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, device , dst_val , best_model_path, patience=10,best_acc=0.0):
    trigger_times = 0
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

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
                break
    return best_acc

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion,optimizer, scheduler, device, best_model_path):
    core_indices = set()
    s_test_samples = []
    num_s_test_samples = 1
    recursion_depth = 200  # 减少递归深度
    test_loader = DataLoader(dst_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    accumulated_core_indices = set()

    # best_model_path = args.save_best_model_path
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection),desc="Epoch group"):
        try:
            model.load_state_dict(torch.load(best_model_path))
        except:
            pass

        # 重新计算 s_test
        s_test_samples = []
        for i, (z_test, t_test) in enumerate(test_loader):
            if i >= num_s_test_samples:
                break
            z_test, t_test = z_test.to(device), t_test.to(device)
            s_test_sample = lissa(
                model,
                DataLoader(dst_train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True),
                grad_z(z_test, t_test, model, criterion, gpu=0 if device.type == 'cuda' else -1),
                criterion,
                gpu=0 if device.type == 'cuda' else -1,
                recursion_depth=recursion_depth,
                damping=0.01,
                scale=25.0
            )
            s_test_samples.append(s_test_sample)

        all_indices = np.arange(len(dst_train))
        remaining_indices = all_indices
        batch_size = 100
        remaining_indices_batches = [remaining_indices[i:i + batch_size] for i in range(0, len(remaining_indices), batch_size)]

        influence_scores = []
        for batch_indices in tqdm(remaining_indices_batches, desc="calculating influence"):
            batch_z = []
            batch_t = []
            for idx in batch_indices:
                z, t = dst_train[idx]

                # 确保 z 是 float32 类型的 torch 张量
                if isinstance(z, np.ndarray):
                    z = torch.from_numpy(z)
                if z.dtype != torch.float32:
                    z = z.float()
                batch_z.append(z.unsqueeze(0))

                # 确保 t 是 torch 张量
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                elif not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                # 确保 t 是 1D 张量
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                batch_t.append(t)

            batch_z = torch.cat(batch_z).to(device)
            batch_t = torch.cat(batch_t).to(device)

            grad_z_vecs = grad_z(batch_z, batch_t, model, criterion, gpu=0 if device.type == 'cuda' else -1)

            for idx, grad_z_vec in zip(batch_indices, zip(*grad_z_vecs)):
                influence = 0.0
                for s_test_vec in s_test_samples:
                    # 确保 s_test_vec 与 grad_z_vec 的形状匹配
                    if isinstance(s_test_vec, torch.Tensor):
                        s_test_vec = s_test_vec.to(device)
                    influence += -sum([
                        torch.sum(k * j).item() for k, j in zip(grad_z_vec, s_test_vec)
                    ]) / len(dst_train)
                # influence_scores.append((idx, influence))
                for idx in batch_indices:
                    influence_scores.append((idx, influence))

        influence_scores.sort(key=lambda x: -x[1])
        selected_indices = [idx for idx, score in influence_scores[:target_core_set_size]]

        # core_indices清空
        core_indices.clear()

        # core_indices更新
        core_indices.update(selected_indices)
        accumulated_core_indices.update(core_indices)
        print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, epochs_per_selection, device, dst_val, best_model_path, patience= epochs_per_selection, best_acc=best_acc)

    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")
    
def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)

    s_test_samples = []
    num_s_test_samples = 1
    recursion_depth = 100
    test_loader = DataLoader(dst_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    for i, (z_test, t_test) in enumerate(test_loader):
        if i >= num_s_test_samples:
            break
        z_test, t_test = z_test.to(device), t_test.to(device)
        s_test_sample = lissa(
            model,
            DataLoader(dst_train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True),
            grad_z(z_test, t_test, model, criterion, gpu=0 if device.type == 'cuda' else -1),
            criterion,
            gpu=0 if device.type == 'cuda' else -1,
            recursion_depth=recursion_depth,
            damping=0.01,
            scale=25.0
        )
        s_test_samples.append(s_test_sample)

    while len(core_indices) < target_core_set_size:
        print(f"current coreset size: {len(core_indices)} / {target_core_set_size}")

        all_indices = np.arange(len(dst_train))
        remaining_indices = all_indices

        batch_size = 100
        remaining_indices_batches = [
            remaining_indices[i:i + batch_size] for i in range(0, len(remaining_indices), batch_size)
        ]

        influence_scores = []
        for batch_indices in tqdm(remaining_indices_batches, desc="calculating influence"):
            batch_z = []
            batch_t = []
            for idx in batch_indices:
                z, t = dst_train[idx]

                # 确保 z 是 float32 类型的 torch 张量
                if isinstance(z, np.ndarray):
                    z = torch.from_numpy(z)
                if z.dtype != torch.float32:
                    z = z.float()
                batch_z.append(z.unsqueeze(0))

                # 确保 t 是 torch 张量
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t)
                elif not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                # 确保 t 是 1D 张量
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                batch_t.append(t)

            batch_z = torch.cat(batch_z).to(device)
            batch_t = torch.cat(batch_t).to(device)

            grad_z_vecs = grad_z(batch_z, batch_t, model, criterion, gpu=0 if device.type == 'cuda' else -1)

            for idx, grad_z_vec in zip(batch_indices, zip(*grad_z_vecs)):
                influence = 0.0
                for s_test_vec in s_test_samples:
                    # 确保 s_test_vec 与 grad_z_vec 的形状匹配
                    if isinstance(s_test_vec, torch.Tensor):
                        s_test_vec = s_test_vec.to(device)
                    influence += -sum([
                        torch.sum(k * j).item() for k, j in zip(grad_z_vec, s_test_vec)
                    ]) / len(dst_train)
                # influence_scores.append((idx, influence))
                for idx in batch_indices:
                    influence_scores.append((idx, influence))

        # 根据影响分数排序并选择前 b 个样本
        influence_scores.sort(key=lambda x: -x[1])
        selected_indices = [idx for idx, score in influence_scores[:target_core_set_size]]
        core_indices.update(selected_indices)
        
        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"select {len(selected_indices)} samples")

        core_loader = DataLoader(
            Subset(dst_train, list(core_indices)),
            batch_size=128,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

        train_model(model, core_loader, criterion, optimizer, scheduler, args.epochs, device, dst_val, best_model_path)


def LiSSA_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, k=10):
    import logging
    from collections import Counter
    import os
    import time

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
    print(f"device: {device}")

    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def pretrained_model(model, data_loader, criterion, optimizer, num_epochs):
        for epoch in tqdm(range(num_epochs), desc="pretraining model"):
            model.train()
            for inputs, labels_batch in data_loader:
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

    random_size = int(len(dst_train) * args.selection_fraction)
    random_indices = torch.randperm(len(dst_train))[:random_size]
    random_loader = DataLoader(Subset(dst_train, random_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    core_loader = random_loader
    
    pretrained_model(model, core_loader, criterion, optimizer, num_epochs=3)
    
    try:
        del core_loader
        torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

    if args.selection_framework == "once":
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == "interval":
        interval_selection_and_train(args=args, model=model, 
                                    dst_train=dst_train, dst_val=dst_val, dst_test=dst_test,
                                    criterion=criterion, optimizer=optimizer, scheduler=scheduler,device=device , best_model_path=best_model_path)
        

    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True), device)
    print(f"final evaluation: {acc:.4f}, acc: {precision:.4f}, recall: {recall:.4f}, F1 score: {f1:.4f}")
    os.remove(best_model_path)
    end_time = time.time()
    print(f"Time cost: {end_time - start_time:.2f}s")

    # 记录最终评估结果到csv文件中
    import os
    import pandas as pd
    import time

    elapsed_time = end_time - start_time
    csv_path = os.path.join(args.save_log_path, args.dataset, args.model + '.csv')
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    selection_ratio = str(args.selection_fraction)
    metrics = ['precision', 'recall', 'f1', 'time']
    columns = pd.MultiIndex.from_product([[selection_ratio], metrics])

    # 转为字符串以支持追加
    precision_str = f"{precision:.4f}"
    recall_str = f"{recall:.4f}"
    f1_str = f"{f1:.4f}"
    time_str = f"{elapsed_time:.2f}"

    new_data = pd.DataFrame([[precision_str, recall_str, f1_str, time_str]], index=[args.method], columns=columns)

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