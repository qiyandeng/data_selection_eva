import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import nets
import torch.optim.lr_scheduler as lr_scheduler
from functorch import grad, vmap
from collections import Counter
import math
import time
import os
import logging

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

# 自定义的 IndexedSubset 类，用于返回样本的原始索引
class IndexedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.indices = subset.indices  # 确保 indices 是一个列表

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        if len(self.subset) == 0:
            raise IndexError("IndexedSubset is empty.")
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            if len(idx) > 0:
                idx = idx[0]
            else:
                raise IndexError("Received empty index list in IndexedSubset.__getitem__.")
        idx = int(idx)  
        data, labels = self.subset[idx]
        original_idx = self.indices[idx]
        # 确保 labels 是标量
        if isinstance(labels, torch.Tensor):
            labels = labels.item()
        else:
            labels = int(labels)
        return data, labels, original_idx

# 引入论文中的 Adahessian 实现
class Adahessian(optim.Optimizer):
    """Approximates local gradients and 2nd order information as per Adahessian.
    """
    def __init__(
        self,
        params,
        lr=0.15,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=0,
        hessian_power=1,
        single_gpu=True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        self.single_gpu = single_gpu
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        # Compute Hessian trace using Hutchinson's method
        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]
        hvs = torch.autograd.grad(
            grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )
        reduced_grads = []
        hutchinson_trace = []
        for grad, hv, vi in zip(grads, hvs, v):
            tmp_output = hv * vi
            tmp_grad = grad
            hutchinson_trace.append(tmp_output.detach())
            reduced_grads.append(tmp_grad.detach().flatten())
        return hutchinson_trace, reduced_grads

    def step(self, momentum=False):
        """Performs a single approximation step.
        Arguments:
            momentum (bool, optional): enables the momentum technique (default: False)
        """
        params = []
        groups = []
        grads = []

        # Flatten parameter groups
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # Get Hessian diagonal
        hutchinson_trace_moment = []
        if momentum:
            hut_traces, reduced_grads = self.get_trace(params, grads)
            reduced_grads = []
            for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state["exp_avg"], state["exp_hessian_diag_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad.detach(), alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute denominator with Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    (math.sqrt(bias_correction2) ** k)
                ).add_(group['eps'])

                reduced_grads.append((exp_avg / bias_correction1).detach().flatten())
                hutchinson_trace_moment.append(denom.detach().flatten())

            hutchinson_trace_moment = torch.cat(hutchinson_trace_moment)
        else:
            reduced_grads = []
            for grad in grads:
                reduced_grads.append(grad.detach().flatten())
            reduced_grads = torch.cat(reduced_grads).detach()
            return reduced_grads, None, None

        hutchinson_trace = []
        for hut_trace in hut_traces:
            hutchinson_trace.append(hut_trace.detach().flatten())

        hutchinson_trace = torch.cat(hutchinson_trace).detach()
        reduced_grads = torch.cat(reduced_grads).detach()
        return reduced_grads, hutchinson_trace, hutchinson_trace_moment

# CREST 数据选择函数

def crest_select_optimized(model, data_loader, device, subset_size, class_indices):
    model.eval()
    sample_gradients = []
    sample_indices = []
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)  # 设置 reduction='none'

    def compute_loss(params, input, target):
        # 确保 input 和 target 没有批次维度
        output = model(input.unsqueeze(0))  # 增加批次维度
        loss = criterion(output, torch.tensor([target], device=device))
        return loss.sum()

    # 使用 vmap 和 grad 计算每个样本的梯度
    compute_grad = grad(compute_loss)

    print("Computing gradients for candidate selection...")
    # 计算每个样本的梯度和损失
    for batch_idx, (inputs, labels, original_idxs) in enumerate(tqdm(data_loader, desc="Computing gradients")):
        # inputs: [batch_size, feature_dim], labels: [batch_size]
        if inputs.size(0) == 0 or labels.size(0) == 0:
            print(f"Empty batch at batch_idx {batch_idx}, skipping.")
            continue
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = inputs.size(0)
        for i in range(batch_size):
            input_sample = inputs[i]  # [feature_dim]
            label_sample = labels[i].item()  # 标量
            try:
                sample_grad = compute_grad(list(model.parameters()), input_sample, label_sample)
                # 将每个参数的梯度展平并连接起来，得到每个样本的梯度向量
                flattened_grad = torch.cat([g.detach().view(-1) for g in sample_grad]).cpu().numpy()
                sample_gradients.append(flattened_grad)
                # 获取原始索引
                if isinstance(original_idxs[i], torch.Tensor):
                    original_idx = original_idxs[i].item()
                else:
                    original_idx = int(original_idxs[i])
                sample_indices.append(int(class_indices[original_idx]))
            except Exception as e:
                print(f"Error computing gradients for sample {i} in batch {batch_idx}: {e}")
                continue

    if sample_gradients:
        sample_gradients = np.vstack(sample_gradients).astype(np.float32)  # 形状为 (num_samples, grad_dim)
    else:
        print("No valid sample gradients were collected.")
        return [], np.array([])

    # 检查子集大小
    if len(sample_gradients) == 0:
        raise ValueError("Candidate gradients are empty. Check your data loader or input data.")
    subset_size = min(subset_size, len(sample_gradients))
    if len(sample_gradients) < subset_size:
        print(f"Warning: Not enough candidates. Adjusting subset size to {len(sample_gradients)}")
        subset_size = len(sample_gradients)

    selected_indices = []
    remaining_indices = list(range(len(sample_gradients)))
    
    print("Selecting core set...")

    # 使用向量化操作优化贪心选择核心集合
    for _ in tqdm(range(subset_size), desc="Selecting samples"):
        if len(selected_indices) == 0:
            # 初始选择第一个样本
            best_index = np.random.choice(remaining_indices)
        else:
            # 计算每个剩余样本与已选择样本的相似度
            selected_grads = sample_gradients[selected_indices]  # 形状为 (selected, grad_dim)
            remaining_grads = sample_gradients[remaining_indices]  # 形状为 (remaining, grad_dim)
            similarity = np.dot(remaining_grads, selected_grads.T)  # 形状为 (remaining, selected)
            max_sim = np.max(similarity, axis=1)  # 形状为 (remaining,)
            # 选择相似度最小的样本
            best_idx_in_remaining = np.argmin(max_sim)
            best_index = remaining_indices[best_idx_in_remaining]

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    # 确保返回的索引为整数类型
    selected_global_indices = [int(sample_indices[idx]) for idx in selected_indices]

    return selected_global_indices


def interval_selection_and_train(args, model , dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device,best_model_path):
    core_indices = set()
    train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    epochs_per_selection = args.selection_interval
    target_core_set_size = int(len(dst_train) * args.selection_fraction)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    accumulated_core_indices = set()
    best_acc = 0.0

    for epoch_group in tqdm(range(args.epochs // epochs_per_selection), desc="Epoch group"):
        balance = True
        selected_indices = []
        full_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

        try:
            num_classes = len(set(int(target.item()) for target in dst_train.targets))
            train_labels = dst_train.targets
        except:
            num_classes = len(set(int(target.item()) for target in dst_train.labels))
            train_labels = dst_train.labels

        if balance:
            for c in range(num_classes):
                class_indices = np.where(np.array(train_labels) == c)[0]
                # 若该类别无样本，跳过
                if len(class_indices) == 0:
                    continue

                class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))

                # 创建当前类别的数据加载器
                class_subset = Subset(dst_train, class_indices)
                class_loader = DataLoader(IndexedSubset(class_subset), batch_size=256, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
                # class_loader = DataLoader(class_subset, batch_size=256, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

                # 使用 crest 子集选择函数
                class_selected = crest_select_optimized(model, class_loader, device, class_subset_size, class_indices)
                # 将选择的样本索引映射回全数据集
                selected_class_indices = [class_indices[idx] for idx in class_selected]
                selected_indices.extend(selected_class_indices)
        else:
            selected_indices = crest_select_optimized(model, full_loader, device, target_core_set_size, list(range(len(dst_train))))

        print(f"Final number of selected samples: {len(selected_indices)}")
        
        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))
        
        accumulated_core_indices.update(selected_indices)
        subset_data = Subset(dst_train, selected_indices)
        subset_loader = DataLoader(subset_data, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
        if best_acc > 0:
            try:
                model.load_state_dict(torch.load(best_model_path))
            except:
                pass
        best_acc = train_model(model, subset_loader, criterion, optimizer, scheduler, epochs_per_selection, device, val_loader, best_model_path, patience=epochs_per_selection, best_acc=best_acc)

    print(f"Final number of selected samples: {len(accumulated_core_indices)}")
    final_coverage_fraction = len(accumulated_core_indices) / len(dst_train)
    print(f"Final coverage fraction: {final_coverage_fraction:.4f}")

def once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path):
    subset_size = int(len(dst_train) * args.selection_fraction)
    balance = True
    selected_indices = []
    full_loader = DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(dst_val, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 获取数据集类别数
    try:
        num_classes = len(set(int(target.item()) for target in dst_train.targets))
        train_labels = dst_train.targets
    except:
        num_classes = len(set(int(target.item()) for target in dst_train.labels))
        train_labels = dst_train.labels

    print(f"Num classes: {num_classes}")

    if balance:
        # if hasattr(dst_train, 'labels'):
        #     train_labels = dst_train.labels
        # else:
        #     train_labels = dst_train.targets

        for c in tqdm(range(num_classes),desc="Class"):
            # 获取当前类别的样本索引
            class_indices = np.where(np.array(train_labels) == c)[0]
            # 若该类别无样本，跳过
            if len(class_indices) == 0:
                continue

            class_subset_size = max(1, int(len(class_indices) * args.selection_fraction))

            # 创建当前类别的数据加载器
            class_subset = Subset(dst_train, class_indices)
            class_loader = DataLoader(IndexedSubset(class_subset), batch_size=256, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)
            # class_loader = DataLoader(class_subset, batch_size=256, shuffle=False, drop_last=False, num_workers=1, pin_memory=True)

            print("len(class_subset_size ):",class_subset_size)
            # 使用 craig 子集选择函数
            class_selected = crest_select_optimized(model, class_loader, device, class_subset_size, class_indices)
            # 将选择的样本索引映射回全数据集
            selected_class_indices = [class_indices[idx] for idx in class_selected]
            selected_indices.extend(selected_class_indices)
    else:
        selected_indices = crest_select_optimized(model, full_loader, device, subset_size, list(range(len(dst_train))))
    
    from collections import Counter
    core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
    print("Core set label distribution:", Counter(core_labels))
    
    print(f"Final number of selected samples: {len(selected_indices)}")
    subset_data = Subset(dst_train, selected_indices)
    subset_loader = DataLoader(subset_data, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
    train_model(model, subset_loader, criterion, optimizer, scheduler, args.epochs, device, val_loader, best_model_path)


# 主函数
def crest_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    start_time = time.time()
    # log_path = args.save_log_path + args.dataset + '/' +  args.method + '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +  '/'+ args.selection_framework + '/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
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

    # 替换全局 print 为日志输出
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
    print("CUDA is available:", torch.cuda.is_available())

    model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    test_loader = DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    # 根据选择框架运行相应的子集选择和训练流程
    if args.selection_framework == 'interval':
        interval_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)
    elif args.selection_framework == 'once':
        once_selection_and_train(args, model, dst_train, dst_val, dst_test, criterion, optimizer, scheduler, device, best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(model, test_loader, device)
    print(f"Final Evaluation - acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    os.remove(best_model_path)
    print(f"Total time: {time.time() - start_time:.4f}s")