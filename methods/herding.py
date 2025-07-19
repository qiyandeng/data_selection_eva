import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import nets
from methods.methods_utils import euclidean_dist
from torchvision import transforms
from copy import deepcopy
import time

class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # Replace the final FC layer

    def forward(self, x):
        x = self.model(x)
        return x

def herding_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, balance=False):
    import logging
    from collections import Counter
    import os


    start_time = time.time()
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
    best_model_path = args.save_best_model_path + args.dataset + '/' +  args.model + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
    index_log_path = args.save_log_path + 'index/'+ args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(index_log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

    global print
    print = lambda *args: logging.info(" ".join(map(str, args)))

    try:
        print(f"Log path: {log_path}")
        print(f"method: {args.method}")
        print(f"network: {args.model}")
        print(f"dataset: {args.dataset}")
        print(f"selection_fraction: {args.selection_fraction}")
        if args.proxy_model != 'none':
            print(f"proxy_model: {args.proxy_model}")

        print(f"Dataset info: channels={channel}, image size={im_size}, num_classes={num_classes}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

        print("Train label distribution:", Counter([int(dst_train[idx][1]) for idx in range(len(dst_train))]))
        print("Val label distribution:", Counter([int(dst_val[idx][1]) for idx in range(len(dst_val))]))
        print("Test label distribution:", Counter([int(dst_test[idx][1]) for idx in range(len(dst_test))]))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        bool_del = 0
        select_start = time.time()

        try:
            model = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
            extractor = EmbeddingExtractor(model).to(device)
            extractor.eval()

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
            train_loader = DataLoader(dst_train, batch_size=128, shuffle=False)
            data, labels = [], []
            for images, targets in tqdm(train_loader, desc="Building dataset"):
                data.append(images.view(images.size(0), -1).numpy())
                labels.append(targets.numpy())
            data = np.concatenate(data)
            labels = np.concatenate(labels)

        n = data.shape[0]
        core_set_size = int(n * args.selection_fraction)
        print(f"Core set size: {core_set_size}")

        balance = True
        if balance:
            print("Performing balanced Herding selection per class.")
            selected_indices = []
            for c in range(num_classes):
                class_indices = np.where(labels == c)[0]
                class_data = torch.tensor(data[class_indices], dtype=torch.float32, device=device)
                class_mu = torch.mean(class_data, dim=0)
                selected_class_embeddings = torch.zeros_like(class_mu, device=device)
                selected_class_indices = np.zeros(len(class_indices), dtype=bool)
                class_core_set_size = max(1, int(core_set_size * (np.sum(labels == c) / n)))

                for i in tqdm(range(class_core_set_size), desc=f"Herding selection for class {c}"):
                    target = (i + 1) * class_mu - selected_class_embeddings
                    unselected_mask = ~selected_class_indices
                    dist = euclidean_dist(target.view(1, -1), class_data[unselected_mask])
                    if dist.numel() == 0:
                        break
                    p = torch.argmax(dist).item()
                    unselected_indices = np.where(unselected_mask)[0]
                    actual_index = class_indices[unselected_indices[p]]
                    selected_indices.append(actual_index)
                    selected_class_indices[unselected_indices[p]] = True
                    selected_class_embeddings += class_data[unselected_indices[p]]

            selected_indices = np.array(selected_indices)
        else:
            print("Performing global Herding selection.")
            matrix = torch.tensor(data, dtype=torch.float32, device=device)
            mu = torch.mean(matrix, dim=0)
            selected_embeddings = torch.zeros_like(mu, device=device)
            representative_indices = np.zeros(n, dtype=bool)

            for i in tqdm(range(core_set_size), desc="Herding selection"):
                target = (i + 1) * mu - selected_embeddings
                dist = euclidean_dist(target.view(1, -1), matrix[~representative_indices])
                if dist.numel() == 0:
                    break
                p = torch.argmax(dist).item()
                actual_index = np.where(~representative_indices)[0][p]
                representative_indices[actual_index] = True
                selected_embeddings += matrix[actual_index]

            selected_indices = np.where(representative_indices)[0]

        print(f"Selected {len(selected_indices)} samples.")
        core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        # Write selected indices to log file
        with open(index_log_path, 'w') as f:
            for idx in selected_indices:
                f.write(f"{idx}\n")

        select_end = time.time()

        subset_data = Subset(dst_train, selected_indices)
        core_loader = DataLoader(subset_data, batch_size=128, shuffle=True)

        if bool_del:
            del model
            del extractor
            torch.cuda.empty_cache()

        if args.model == 'ResNet50':
            print("Using ResNet50 model.")
            train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
        else:
            train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
        train_model_instance.train()

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(train_model_instance.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        def evaluate_model(model, data_loader):
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

        def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, patience=10):
            best_acc = 0.0
            trigger_times = 0

            for epoch in tqdm(range(num_epochs), desc="Training model"):
                model.train()
                for inputs, labels_batch in data_loader:
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    outputs = model(inputs)
                    outputs = torch.clamp(outputs, min=-10, max=10)
                    loss = criterion(outputs, labels_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()

                acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_val, batch_size=128, shuffle=False))
                print(f"Epoch {epoch+1}/{num_epochs}, acc={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, F1 score={f1:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    trigger_times = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print("Early stopping triggered.")
                        break

        train_model(train_model_instance, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, patience=args.patience)

        train_model_instance.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(train_model_instance, DataLoader(dst_test, batch_size=128, shuffle=False))
        print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        os.remove(best_model_path)
        end_time = time.time()
        selection_time = select_end-select_start
        print(f"Selection time elapsed: {select_end - select_start:.2f} seconds.")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds.")


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
        logging.error(f"An error occurred in herding_selection: {str(e)}")
        logging.error("Stack trace: ", exc_info=True)
