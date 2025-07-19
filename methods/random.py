import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import mydatasets  # Ensure this module exists and contains the required datasets
import nets        # Ensure this module exists and contains the required network models
import argparse


class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # 替换最后的全连接层

    def forward(self, x):
        x = self.model(x)
        return x
        


def select_representatives_random(data, core_set_size, epsilon, random_seed=None):
    """
    Randomly select representative points and adjust weights to build a coreset.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = data.shape[0]
    representative_indices = np.random.choice(n, size=core_set_size, replace=False)
    weights = np.ones(n)
    weights[representative_indices] *= (1 + epsilon)
    return representative_indices.tolist(), weights

class WeightedCrossEntropyLoss(nn.Module):
    """
    Custom weighted cross-entropy loss function.
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
    Evaluate the model on the test set.
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
    Custom dataset that includes data samples, labels, and weights.
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

def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, patience, device, best_model_path, dst_test,best_acc=0.0):
    """
    Train the model with early stopping on the validation set.
    """
    trigger_times = 0
    for epoch in tqdm(range(num_epochs), desc="Training model"):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels_batch, weights_batch) in enumerate(data_loader):
            inputs, labels_batch, weights_batch = inputs.to(device), labels_batch.to(device), weights_batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch, weights_batch)  # Use WeightedCrossEntropyLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Debug information
            if epoch == 0 and batch_idx == 0:
                print("Sample outputs:", outputs[0].detach().cpu().numpy())
                print("Sample label:", labels_batch[0].cpu().item())
                print("Sample weight:", weights_batch[0].cpu().item())

        scheduler.step()

        # Evaluate the model
        acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4, pin_memory=True), device)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"Best model saved with Acc: {best_acc:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
    return best_acc

def random_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test):
    """
    Random selection method: Randomly select coreset samples, adjust weights, train and evaluate the model.
    """
    import logging
    from collections import Counter
    import os
    import time
    start_time = time.time()
    
    log_path = args.save_log_path + args.dataset + '/' + args.model + '/' +  args.method +'/' + str(args.selection_fraction) + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  args.model + '/' + args.method + '/' +  str(args.selection_fraction)  + '.pth'
    
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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    bool_del = 0  # Flag to indicate whether to delete the model to free memory

    # Initialize model and extract embeddings
    # 初始化模型并提取嵌入
    try:
        model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
        extractor = EmbeddingExtractor(model).to(device)
        extractor.eval()

        # 使用提取器获取数据嵌入
        train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
        data, labels = [], []
        with torch.no_grad():
            for images, targets in tqdm(train_loader, desc="Extracting embeddings"):
                images = images.to(device)
                embeddings = extractor(images)
                data.append(embeddings.cpu().numpy())
                labels.append(targets.numpy())
        data = np.concatenate(data)
        labels = np.concatenate(labels)
        bool_del = 1
        
    except:
        # 创建数据加载器
        train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
        data, labels = [], []

        # 收集所有数据点和标签
        for images, targets in tqdm(train_loader, desc="Building dataset"):
            data.append(images.view(images.size(0), -1).numpy())
            labels.append(targets.numpy())
        data = np.concatenate(data)
        labels = np.concatenate(labels)

    n, d = data.shape
    core_set_size = int(n * args.selection_fraction)
    epsilon = 0.1  # According to the paper, epsilon can be adjusted as needed

    print(f"Total data points: {n}, Data dimension: {d}")
    print(f"Coreset size: {core_set_size}, Epsilon: {epsilon}")

    select_start = time.time()
    # Initial weight allocation
    weights = np.ones(n) / n

    # Randomly select representative points and adjust weights
    print("Selecting representative points randomly...")
    representative_indices, weights = select_representatives_random(data, core_set_size, epsilon, random_seed=42)
    representative_indices = np.array(representative_indices)
    print(f"Selected {len(representative_indices)} representative points.")

    # Check if representative points are empty
    if len(representative_indices) == 0:
        raise ValueError("No representative points were selected. Check your selection logic.")

    # Error control and adjust weights (without normalization)
    weights[representative_indices] *= (1 + epsilon)
    # weights /= weights.sum()  # Remove normalization

    # Create coreset data loader with weights
    subset = Subset(dst_train, representative_indices)
    coreset_weights = weights[representative_indices]
    coreset_dataset = CoresetDataset(subset, coreset_weights)
    core_loader = DataLoader(coreset_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # Custom weighted loss function
    criterion = WeightedCrossEntropyLoss().to(device)
    # Or use standard cross entropy loss
    # criterion = nn.CrossEntropyLoss().to(device)

    # Delete extractor and model to free memory
    if bool_del == 1:
        del extractor, model
        torch.cuda.empty_cache()

    # Initialize new model for training
    print("Initializing model for training...")
    train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
    train_model_instance.train()

    # Define optimizer and learning rate scheduler
    optimizer = optim.SGD(train_model_instance.parameters(), lr=args.selection_lr, momentum=args.selection_momentum,
                          weight_decay=args.selection_weight_decay, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    select_end = time.time()
    selection_time = select_end-select_start
    print(f"selection time:{selection_time}")
    # Training and evaluation
    print("Starting training...")
    train_model(train_model_instance, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, 
               patience=10, device=device, best_model_path=best_model_path, dst_test=dst_test)

    # Load best model and evaluate
    if os.path.exists(best_model_path):
        print("Loading best model for evaluation...")
        train_model_instance.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(train_model_instance, 
                                                    DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=4, pin_memory=True),
                                                    device)
        print(f"Evaluation Results - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    else:
        print(f"Model file {best_model_path} does not exist.")

    os.remove(best_model_path)  # Remove the best model file
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # 记录最终评估结果到csv文件中
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
