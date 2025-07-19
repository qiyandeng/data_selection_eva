import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import mydatasets  # 请确保此模块存在并包含所需的数据集
import nets        # 请确保此模块存在并包含所需的网络模型
import time

class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # 替换最后的全连接层

    def forward(self, x):
        x = self.model(x)
        return x

def extract_embeddings(extractor, data_loader, device):
    """
    使用提取器从数据加载器中提取嵌入表示。
    """
    extractor.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            batch_embeddings = extractor(images)
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

def extract_embeddings_flatten(data_loader):
    """
    当嵌入提取失败时，直接将图像展平成一维向量作为嵌入表示。
    """
    embeddings, labels = [], []
    for images, targets in tqdm(data_loader, desc="Building dataset"):
        embeddings.append(images.view(images.size(0), -1).numpy())
        labels.append(targets.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

def build_quadtree(data, max_depth=15, min_points=10):
    """
    高效地构建四叉树，以递归方式划分数据空间。
    """
    class QuadTreeNode:
        def __init__(self, point_indices, bounds, depth=0):
            self.point_indices = point_indices
            self.bounds = bounds  # [(min_dim0, max_dim0), (min_dim1, max_dim1), ...]
            self.children = []
            self.depth = depth

        def subdivide(self, data, max_depth, min_points=10):
            if self.depth >= max_depth or len(self.point_indices) <= min_points:
                return

            # 选择方差最大的维度进行划分
            variances = np.var(data[self.point_indices], axis=0)
            split_dim = np.argmax(variances)
            split_value = np.median(data[self.point_indices, split_dim])

            # 划分点集
            left_mask = data[self.point_indices, split_dim] <= split_value
            right_mask = data[self.point_indices, split_dim] > split_value
            left_indices = self.point_indices[left_mask]
            right_indices = self.point_indices[right_mask]

            # 确保子节点分布均匀，避免单侧分裂
            if len(left_indices) < min_points or len(right_indices) < min_points:
                return

            # 定义子节点的边界
            left_bounds = self.bounds.copy()
            left_bounds[split_dim] = (self.bounds[split_dim][0], split_value)
            right_bounds = self.bounds.copy()
            right_bounds[split_dim] = (split_value, self.bounds[split_dim][1])

            # 创建子节点
            left_child = QuadTreeNode(left_indices, left_bounds, self.depth + 1)
            right_child = QuadTreeNode(right_indices, right_bounds, self.depth + 1)
            self.children = [left_child, right_child]

    initial_bounds = [(np.min(data[:, dim]), np.max(data[:, dim])) for dim in range(data.shape[1])]
    root = QuadTreeNode(np.array(range(data.shape[0])), initial_bounds, depth=0)
    stack = [root]

    while stack:
        node = stack.pop()
        node.subdivide(data, max_depth, min_points)
        if node.children:
            stack.extend(node.children)
    return root

def select_representatives_quadtree(root, data, core_set_size, epsilon):
    """
    基于四叉树选择代表点，并调整权重以构建 Coreset。
    """
    representative_indices = []
    weights = np.ones(data.shape[0])  # 初始权重分配

    def traverse(node):
        nonlocal representative_indices, core_set_size, epsilon, weights
        if len(representative_indices) >= core_set_size:
            return
        if not node.children:
            if len(node.point_indices) == 0:
                return
            # 选择与均值距离最小的点作为代表点
            points = data[node.point_indices]
            mean = np.mean(points, axis=0)
            distances = np.linalg.norm(points - mean, axis=1)
            sorted_indices = np.argsort(distances)
            num_points_to_select = min(len(sorted_indices), core_set_size - len(representative_indices))
            if num_points_to_select > 0:
                selected_points = sorted_indices[:num_points_to_select]
                selected_indices = node.point_indices[selected_points]
                representative_indices.extend(selected_indices.tolist())
                # 调整权重
                weights[selected_indices] *= (1 + epsilon)
        else:
            for child in node.children:
                traverse(child)
                if len(representative_indices) >= core_set_size:
                    break

    traverse(root)
    return representative_indices, weights

class WeightedCrossEntropyLoss(nn.Module):
    """
    自定义加权交叉熵损失函数。
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, labels, weights):
        loss = self.loss_fn(outputs, labels)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

def evaluate_model(model, data_loader, device):
    """
    评估模型在测试集上的性能。
    """
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

class CoresetDataset(Dataset):
    """
    自定义数据集，包含样本数据、标签和权重。
    """
    def __init__(self, subset, weights):
        self.subset = subset
        self.weights = weights

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]
        weight = self.weights[idx]
        return data, label, weight

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, patience, device, best_model_path, dst_val):
    """
    训练模型，并在验证集上进行早停。
    """
    best_acc = 0.0
    trigger_times = 0
    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels_batch, weights_batch) in enumerate(data_loader):
            inputs, labels_batch, weights_batch = inputs.to(device), labels_batch.to(device), weights_batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch, weights_batch)  # 使用 WeightedCrossEntropyLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 添加调试信息
            if epoch == 0 and batch_idx == 0:
                print("Sample outputs:", outputs[0].detach().cpu().numpy())
                print("Sample label:", labels_batch[0].cpu().item())
                print("Sample weight:", weights_batch[0].cpu().item())

        scheduler.step()

        # 评估模型
        acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_val, batch_size=256, shuffle=False, num_workers=1, pin_memory=True), device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Acc: {best_acc:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
    return best_acc


def lowdimention_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    """
    主函数：进行低维度 Coreset 选择、模型训练和评估。
    """

    import logging
    from collections import Counter
    import os


    start_time = time.time()
    
    # log_path = args.save_log_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target) + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  args.method +'/' + str(args.selection_fraction) + '/' + str(args.balance_target)  + '.pth'
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' +  str(args.selection_fraction)  +'/' + args.method + '.log'
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

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        bool_del = 0  # 标记是否需要删除模型以释放显存

        select_start = time.time()

        # 初始化模型并提取嵌入
        try:
            print("Initializing model for embedding extraction...")
            model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
            extractor = EmbeddingExtractor(model).to(device)
            extractor.eval()

            # 使用提取器获取数据嵌入
            train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
            data, labels = extract_embeddings(extractor, train_loader, device)
            bool_del = 1  # 标记需要删除模型以释放显存
            print(f"Extracted embeddings shape: {data.shape}")

        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            # 创建数据加载器
            train_loader = DataLoader(dst_train, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
            data, labels = extract_embeddings_flatten(train_loader)
            print(f"Flattened data shape: {data.shape}")

        n, d = data.shape
        core_set_size = int(n * args.selection_fraction)
        epsilon = 0.1  # 根据论文，epsilon 可以根据需要调整

        print(f"Total data points: {n}, Data dimension: {d}")
        print(f"Coreset size: {core_set_size}, Epsilon: {epsilon}")

        # 初始权重分配
        weights = np.ones(n) / n

        # 构建四叉树
        print("Building QuadTree...")
        root = build_quadtree(data, max_depth=15, min_points=10)
        print("QuadTree built.")

        # 选择代表点并调整权重
        print("Selecting representative points...")
        representative_indices, weights = select_representatives_quadtree(root, data, core_set_size, epsilon)
        representative_indices = np.array(representative_indices)
        print(f"Selected {len(representative_indices)} representative points.")

        # 检查代表点是否为空
        if len(representative_indices) == 0:
            raise ValueError("No representative points were selected. Check your QuadTree and point selection logic.")

        # 误差控制并调整权重（不归一化）
        weights[representative_indices] *= (1 + epsilon)
        # weights /= weights.sum()  # 移除归一化
        
        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in representative_indices]
        print("Core set label distribution:", Counter(core_labels))

        select_end = time.time()

        # 创建核心集数据加载器，包含权重
        subset = Subset(dst_train, representative_indices)
        coreset_weights = weights[representative_indices]
        coreset_dataset = CoresetDataset(subset, coreset_weights)
        core_loader = DataLoader(coreset_dataset, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

        # 自定义加权损失函数
        criterion = WeightedCrossEntropyLoss().to(device)
        # 或者使用标准交叉熵损失
        # criterion = nn.CrossEntropyLoss().to(device)

        # 删除提取器和模型以释放显存
        if bool_del == 1:
            del extractor, model
            torch.cuda.empty_cache()

        # 初始化新的模型用于训练
        print("Initializing model for training...")
        if args.model == 'ResNet50':
            train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
        else:
            train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
            
        train_model_instance.train()

        # 定义优化器和学习率调度器
        optimizer = optim.SGD(train_model_instance.parameters(), lr=args.selection_lr, momentum=args.selection_momentum,
                            weight_decay=args.selection_weight_decay, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 训练与评估
        print("Starting training...")
        train_model(train_model_instance, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, 
                patience=args.patience, device=device, best_model_path=best_model_path, dst_val=dst_val)

        # 加载最佳模型并评估
        if os.path.exists(best_model_path):
            print("Loading best model for evaluation...")
            train_model_instance.load_state_dict(torch.load(best_model_path))
            acc, precision, recall, f1 = evaluate_model(train_model_instance, 
                                                        DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=1, pin_memory=True),
                                                        device)
            print(f"Evaluation Results - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        else:
            print(f"Model file {best_model_path} does not exist.")
        
        os.remove(best_model_path)
        end_time = time.time()
        selection_time = select_end-select_start
        print(f"Selection time elapsed: {select_end - select_start:.2f} seconds.")
        print(f"Total time: {end_time - start_time:.2f} seconds")

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
