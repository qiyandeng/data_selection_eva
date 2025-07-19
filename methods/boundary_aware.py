import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets
import time
from collections import Counter

# def compute_deepfool_perturbations(args, model, data_loader, device):
#     # Acknowledgement to
#     # https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/adversarial_deepfool.py

#     model.to(device)
#     model.eval()
    
#     # 禁用模型参数的梯度计算
#     for param in model.parameters():
#         param.requires_grad = False

#     distances = []
#     max_iter = 50  # DeepFool最大迭代次数，您可根据需要调整
    
#     for inputs, _ in tqdm(data_loader, desc="Computing adversarial distances"):
#         inputs = inputs.to(device)
#         sample_size = inputs.shape[0]
        
#         # 初始化扰动记录数组 r_tot 和 boolean_mask
#         r_tot = np.zeros([sample_size, np.prod(inputs.shape[1:])], dtype=np.float32)
#         boolean_mask = np.ones(sample_size, dtype=bool)   # True表示该样本仍需计算
#         all_idx = np.arange(sample_size)
        
#         # 设置输入需要梯度
#         cur_inputs = inputs.detach().clone().requires_grad_(True)
        
#         with torch.no_grad():
#             outputs = model(cur_inputs)
#             preds = outputs.argmax(dim=1)

#         # 保存初始预测，用于判断何时扰动成功
#         first_preds = preds.clone()

#         for _ in range(max_iter):
#             if boolean_mask.sum() == 0:
#                 # 所有样本都已被扰动成功或处理完毕
#                 break
            
#             # 对当前仍未成功扰动的样本进行计算
#             sample_size_current = cur_inputs.shape[0]
#             outputs = model(cur_inputs)
#             f_all = outputs
            
#             num_classes = f_all.shape[1]
#             gradients = []
#             # 计算每个类的梯度
#             for c in range(num_classes):
#                 grad = torch.autograd.grad(f_all[:, c].sum(), cur_inputs, retain_graph=True if c < num_classes - 1 else False)[0]
#                 gradients.append(grad)
#             w_k = torch.stack(gradients, dim=1)  # [batch_size_current, num_classes, C, H, W]
            
#             grad_np = w_k[range(sample_size_current), preds].clone()
#             w_k = w_k - grad_np.unsqueeze(1)  # [batch_size_current, num_classes, C, H, W]
#             w_k = w_k.view(sample_size_current, num_classes, -1)
#             w_k_norm = torch.norm(w_k, dim=2)  # [batch_size_current, num_classes]
            
#             f_all = f_all.detach()
#             f_p = f_all[range(sample_size_current), preds].unsqueeze(1)
#             f_diff = (f_all - f_p).abs()
            
#             l_all = f_diff / w_k_norm  # [batch_size_current, num_classes]
#             l_all[range(sample_size_current), preds] = float('inf')
            
#             min_indices = l_all.argmin(dim=1)
#             min_values = l_all[range(sample_size_current), min_indices]
            
#             ri = (min_values / w_k_norm[range(sample_size_current), min_indices]).unsqueeze(1) * w_k[range(sample_size_current), min_indices]
#             ri = ri.view(sample_size_current, *inputs.shape[1:])
            
#             # 更新r_tot，只对boolean_mask为True的样本进行更新
#             # boolean_mask为True的样本对应cur_inputs中的样本顺序一一对应
#             r_tot[boolean_mask] += ri.detach().cpu().numpy().reshape(sample_size_current, -1)
#             cur_inputs = cur_inputs + ri.detach()
            
#             with torch.no_grad():
#                 new_outputs = model(cur_inputs)
#                 new_preds = new_outputs.argmax(dim=1)
            
#             # 判断哪些样本已成功扰动（预测类别改变）
#             changed_mask = (new_preds != preds)  # 在当前未完成的样本集上
#             if changed_mask.any():
#                 # 对已成功扰动的样本，在boolean_mask中对应位置设False
#                 finished_global_indices = all_idx[boolean_mask][changed_mask.cpu().numpy()]
#                 boolean_mask[finished_global_indices] = False
            
#             # 更新未成功扰动的样本集合
#             index_unfinished = (new_preds == preds)
#             if not index_unfinished.any():
#                 # 所有剩余的样本都被扰动成功
#                 break

#             # 保留未完成的样本
#             cur_inputs = cur_inputs[index_unfinished]
#             preds = preds[index_unfinished]

#         # 计算扰动距离（L2范数）
#         distances_batch = np.linalg.norm(r_tot, axis=1)
#         distances.append(distances_batch)

#         # 释放显存
#         del inputs, cur_inputs, outputs, new_outputs, preds, new_preds, ri, f_all, w_k, grad_np, f_diff, l_all, min_indices, min_values
#         torch.cuda.empty_cache()

#     return np.concatenate(distances)

def compute_deepfool_perturbations(args, model, data_loader, device):
    # Acknowledgement to
    # https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/adversarial_deepfool.py

    model.to(device)
    model.eval()
    
    # 禁用模型参数的梯度计算
    for param in model.parameters():
        param.requires_grad = False

    distances = []
    max_iter = 50  # DeepFool最大迭代次数，可根据需要调整
    alpha = 0.5    # 扰动步长缩放因子

    for inputs, _ in tqdm(data_loader, desc="Computing adversarial distances"):
        inputs = inputs.to(device)
        sample_size = inputs.shape[0]
        
        # 保存原始输入，用于后续clamp
        original_inputs = inputs.detach().clone()
        
        # 初始化扰动记录数组 r_tot 和 boolean_mask
        r_tot = np.zeros([sample_size, np.prod(inputs.shape[1:])], dtype=np.float32)
        boolean_mask = np.ones(sample_size, dtype=bool)   # True 表示该样本仍需计算
        all_idx = np.arange(sample_size)
        
        # 设置输入需要梯度
        cur_inputs = inputs.detach().clone().requires_grad_(True)
        
        with torch.no_grad():
            outputs = model(cur_inputs)
            preds = outputs.argmax(dim=1)

        # 保存初始预测，用于判断何时扰动成功
        first_preds = preds.clone()

        for _ in range(max_iter):
            if boolean_mask.sum() == 0:
                break
            
            sample_size_current = cur_inputs.shape[0]
            outputs = model(cur_inputs)
            f_all = outputs
            
            num_classes = f_all.shape[1]
            gradients = []
            # 计算每个类别的梯度
            for c in range(num_classes):
                grad = torch.autograd.grad(f_all[:, c].sum(), cur_inputs,
                                           retain_graph=True if c < num_classes - 1 else False)[0]
                gradients.append(grad)
            w_k = torch.stack(gradients, dim=1)  # [batch_size_current, num_classes, ...]
            
            grad_np = w_k[range(sample_size_current), preds].clone()
            w_k = w_k - grad_np.unsqueeze(1)
            w_k = w_k.view(sample_size_current, num_classes, -1)
            w_k_norm = torch.norm(w_k, dim=2)  # [batch_size_current, num_classes]
            
            f_all = f_all.detach()
            f_p = f_all[range(sample_size_current), preds].unsqueeze(1)
            f_diff = (f_all - f_p).abs()
            
            l_all = f_diff / w_k_norm  # [batch_size_current, num_classes]
            l_all[range(sample_size_current), preds] = float('inf')
            
            min_indices = l_all.argmin(dim=1)
            min_values = l_all[range(sample_size_current), min_indices]
            
            ri = (min_values / w_k_norm[range(sample_size_current), min_indices]).unsqueeze(1) * \
                 w_k[range(sample_size_current), min_indices]
            ri = ri.view(sample_size_current, *inputs.shape[1:])
            
            # 对扰动步长进行缩放
            ri = alpha * ri
            
            # 更新 r_tot（仅对 boolean_mask 为 True 的样本进行更新）
            r_tot[boolean_mask] += ri.detach().cpu().numpy().reshape(sample_size_current, -1)
            
            # 更新 cur_inputs，并将其 clmap 到原始输入范围内
            cur_inputs = cur_inputs + ri.detach()
            cur_inputs = torch.clamp(cur_inputs, original_inputs.min(), original_inputs.max())
            
            with torch.no_grad():
                new_outputs = model(cur_inputs)
                new_preds = new_outputs.argmax(dim=1)
            
            # 判断哪些样本已成功扰动（预测类别改变）
            changed_mask = (new_preds != preds)
            if changed_mask.any():
                finished_global_indices = all_idx[boolean_mask][changed_mask.cpu().numpy()]
                boolean_mask[finished_global_indices] = False
            
            # 更新未成功扰动的样本集合
            index_unfinished = (new_preds == preds)
            if not index_unfinished.any():
                break

            cur_inputs = cur_inputs[index_unfinished]
            preds = preds[index_unfinished]

        # 计算扰动距离（L2范数）
        distances_batch = np.linalg.norm(r_tot, axis=1)
        distances.append(distances_batch)

        # 清理显存
        del inputs, cur_inputs, outputs, new_outputs, preds, new_preds, ri, f_all, w_k, grad_np, f_diff, l_all, min_indices, min_values
        torch.cuda.empty_cache()

    return np.concatenate(distances)


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


def active_learning_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device , best_model_path):
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    target_core_set_size_per_selection = int(target_core_set_size * args.active_fraction_ratio)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    target_core_set_size_per_selection = max(50, target_core_set_size_per_selection)  # 最小选择 50 个样本

    best_acc = 0.0
    trained_epochs = 0
    # 进行多轮挑选
    while len(core_indices) < target_core_set_size:
        print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

        # 提取未选择样本
        remaining_indices = np.setdiff1d(np.arange(len(dst_train)), list(core_indices))
        remaining_loader = DataLoader(Subset(dst_train, remaining_indices), batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

        # 计算DeepFool扰动
        deepfool_distances = compute_deepfool_perturbations(args, model, remaining_loader, device)

        print(f"DeepFool distances len: {len(deepfool_distances)}")
        print(f"DeepFool distances max = {deepfool_distances.max()}, min = {deepfool_distances.min()}")

        # 将样本按扰动距离分组
        distance_bins = np.linspace(deepfool_distances.min(), deepfool_distances.max(), num=10)  # 将距离分为 10 组
        grouped_indices = {i: [] for i in range(len(distance_bins) - 1)}
        for idx, distance in zip(remaining_indices, deepfool_distances):
            for bin_idx in range(len(distance_bins) - 1):
                if distance_bins[bin_idx] <= distance < distance_bins[bin_idx + 1]:
                    grouped_indices[bin_idx].append(idx)
                    break

        # 按组分配样本选择预算
        samples_per_group = target_core_set_size_per_selection // len(grouped_indices)
        for group, indices in grouped_indices.items():
            if len(indices) > 0:
                selected_indices = np.random.choice(indices, min(samples_per_group, len(indices)), replace=False)
                core_indices.update(selected_indices)

        # 检查是否有样本被选中
        if len(core_indices) == 0:
            print("Error: No samples selected for the core set. Ensure DeepFool logic is correct.")
            return

        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")
        
        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        # 在核心数据集上继续训练模型
        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        # 在训练之前，确保模型参数需要梯度
        for param in model.parameters():
            param.requires_grad = True

        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=1, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)
        trained_epochs += 1
    
    print(f"len(core_indices): {len(core_indices)}")
    core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    train_epochs = args.epochs - trained_epochs
    train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=train_epochs, device=device, val_loader=val_loader, best_model_path=best_model_path,patience=args.patience, best_acc=best_acc)

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    # 初始化核心数据集索引集合
    core_indices = set()
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    target_core_set_size_per_selection = int(target_core_set_size * args.active_fraction_ratio)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    target_core_set_size_per_selection = target_core_set_size

    selection_start_time = time.time()

    best_acc = 0.0
    trained_epochs = 0
    # 进行多轮挑选
    print(f"Current core set size: {len(core_indices)} / {target_core_set_size}")

    # 提取未选择样本
    remaining_indices = np.setdiff1d(np.arange(len(dst_train)), list(core_indices))
    remaining_loader = DataLoader(Subset(dst_train, remaining_indices), batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

    # 计算DeepFool扰动
    deepfool_distances = compute_deepfool_perturbations(args, model, remaining_loader, device)

    print(f"DeepFool distances len: {len(deepfool_distances)}")
    print(f"DeepFool distances max = {deepfool_distances.max()}, min = {deepfool_distances.min()}")

    # 将样本按扰动距离分组
    distance_bins = np.linspace(deepfool_distances.min(), deepfool_distances.max(), num=10)  # 将距离分为 10 组
    grouped_indices = {i: [] for i in range(len(distance_bins) - 1)}
    for idx, distance in zip(remaining_indices, deepfool_distances):
        for bin_idx in range(len(distance_bins) - 1):
            if distance_bins[bin_idx] <= distance < distance_bins[bin_idx + 1]:
                grouped_indices[bin_idx].append(idx)
                break

    # 按组分配样本选择预算
    samples_per_group = target_core_set_size_per_selection // len(grouped_indices)
    for group, indices in grouped_indices.items():
        if len(indices) > 0:
            selected_indices = np.random.choice(indices, min(samples_per_group, len(indices)), replace=False)
            core_indices.update(selected_indices)

    # 检查是否有样本被选中
    if len(core_indices) == 0:
        print("Error: No samples selected for the core set. Ensure DeepFool logic is correct.")
        return

    if len(core_indices) > target_core_set_size:
        core_indices = set(list(core_indices)[:target_core_set_size])

    print(f"Selected {len(core_indices)} samples for core set.")
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in core_indices]
    print("Core set label distribution:", Counter(core_labels))

    # 在核心数据集上继续训练模型
    core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

    # 在训练之前，确保模型参数需要梯度
    for param in model.parameters():
        param.requires_grad = True
    
    selection_end_time = time.time()
    selection_time = selection_end_time - selection_start_time
    print(f"Selection time: {selection_time:.2f} seconds")

    train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)

    return selection_time

def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion,optimizer, scheduler, device, best_model_path):
    # 初始化核心数据集索引集合
    test_loader = DataLoader(dst_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    # 初始化核心数据集索引集合，全局维护
    accumulated_core_indices = set()

    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    best_acc = 0.0
    for epoch_group in tqdm(range(args.epochs // epochs_per_selection),desc="Epoch group"):
        try:
            model.load_state_dict(torch.load(best_model_path))
        except:
            pass
        
        core_indices = set()
        # 提取未选择样本
        remaining_indices = np.setdiff1d(np.arange(len(dst_train)), list(core_indices))
        remaining_loader = DataLoader(Subset(dst_train, remaining_indices), batch_size=256, shuffle=False, num_workers=1, pin_memory=True)

        # 计算DeepFool扰动
        deepfool_distances = compute_deepfool_perturbations(args, model, remaining_loader, device)

        print(f"DeepFool distances len: {len(deepfool_distances)}")
        print(f"DeepFool distances max = {deepfool_distances.max()}, min = {deepfool_distances.min()}")

        # 将样本按扰动距离分组
        distance_bins = np.linspace(deepfool_distances.min(), deepfool_distances.max(), num=10)  # 将距离分为 10 组
        grouped_indices = {i: [] for i in range(len(distance_bins) - 1)}
        for idx, distance in zip(remaining_indices, deepfool_distances):
            for bin_idx in range(len(distance_bins) - 1):
                if distance_bins[bin_idx] <= distance < distance_bins[bin_idx + 1]:
                    grouped_indices[bin_idx].append(idx)
                    break

        # 按组分配样本选择预算
        samples_per_group = target_core_set_size // len(grouped_indices)
        for group, indices in grouped_indices.items():
            if len(indices) > 0:
                selected_indices = np.random.choice(indices, min(samples_per_group, len(indices)), replace=False)
                core_indices.update(selected_indices)

        # 检查是否有样本被选中
        if len(core_indices) == 0:
            print("Error: No samples selected for the core set. Ensure DeepFool logic is correct.")
            return

        if len(core_indices) > target_core_set_size:
            core_indices = set(list(core_indices)[:target_core_set_size])

        print(f"Selected {len(core_indices)} samples for core set.")

        # 将本次选择的样本加入全局集合
        accumulated_core_indices.update(core_indices)

        # 在核心数据集上继续训练模型
        core_loader = DataLoader(Subset(dst_train, list(core_indices)), batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

        # 在训练之前，确保模型参数需要梯度
        for param in model.parameters():
            param.requires_grad = True

        if best_acc > 0:
                try:
                    model.load_state_dict(torch.load(best_model_path))
                except:
                    pass
                
        best_acc = train_model(model, core_loader, criterion, optimizer, scheduler, num_epochs=epochs_per_selection, device=device, val_loader=val_loader, best_model_path=best_model_path, best_acc=best_acc)

    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in accumulated_core_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")

def boundary_aware_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, k=10):
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
    try:
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

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        model_name = args.model
        # best_model_path = args.save_best_model_path
        val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        # 初始化模型（只初始化一次）
        if args.model == 'ResNet50':
            print("Using ResNet50 as the training model.")
            model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
        else:
            model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
        
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        def pretrained_model(model, data_loader, criterion, optimizer, num_epochs):
            for epoch in tqdm(range(num_epochs), desc="Training pretrain model"):
                model.train()
                running_loss = 0.0
                for inputs, labels_batch in data_loader:
                    # print(f"Pretrain - Input shape: {inputs.shape}")  # 调试信息
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

        random_size = int(len(dst_train) * args.selection_fraction)
        ramndom_indices = np.random.choice(len(dst_train), random_size, replace=False)
        pretrained_model(model, DataLoader(Subset(dst_train, ramndom_indices), batch_size=256, shuffle=True, num_workers=1, pin_memory=True), criterion, optimizer, 3)
        # pretrained_model(model, DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=1, pin_memory=True), criterion, optimizer, 1)

        if args.selection_framework == 'once':
            selection_time = once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
        elif args.selection_framework == 'active_learning':
            active_learning_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
        elif args.selection_framework == 'interval':
            interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)

        # 最终评估模型
        model.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
        print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        os.remove(best_model_path)
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")

        # 记录最终评估结果到csv文件中
        import os
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