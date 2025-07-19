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

def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None, print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            # if i % print_freq == 0:
            #     print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]


class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # 替换最后的全连接层

    def forward(self, x):
        x = self.model(x)
        return x

def kcentergreedy_selection(args, channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test, balance=False):

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
    # print(f"Log path: {log_path}")

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
        print("CUDA is available:", torch.cuda.is_available())

        select_start = time.time()

        use_extracor = False
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
            use_extracor = True
            
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

        # 参数设置
        n = data.shape[0]
        core_set_size = int(n * args.selection_fraction)
        print(f"Core set size: {core_set_size}")

        balance = True
        # 如果需要按类别平衡选择
        if balance:
            print("Performing balanced k-Center Greedy selection per class.")
            selected_indices = []
            for c in range(num_classes):
                class_indices = np.where(labels == c)[0]
                class_data = torch.tensor(data[class_indices], dtype=torch.float32, device=device)
                class_core_set_size = max(1, int(core_set_size * (np.sum(labels == c) / n)))  # 确保至少选择一个样本

                # 使用 k-Center Greedy 选择样本
                selected_class_indices = k_center_greedy(
                    class_data,
                    budget=class_core_set_size,
                    metric=euclidean_dist,
                    device=device,
                    random_seed=args.random_seed,
                    index=class_indices,
                    already_selected=[]
                )
                selected_indices.extend(selected_class_indices)

            selected_indices = np.array(selected_indices)
        else:
            print("Performing global k-Center Greedy selection.")
            # 构建核心集并进行 k-Center Greedy 选择
            matrix = torch.tensor(data, dtype=torch.float32, device=device)
            selected_indices = k_center_greedy(
                matrix,
                budget=core_set_size,
                metric=euclidean_dist,
                device=device,
                random_seed=args.random_seed,
                already_selected=[]
            )

        print(f"Selected {len(selected_indices)} samples.")
        
        from collections import Counter
        core_labels = [int(dst_train[idx][1]) for idx in selected_indices]
        print("Core set label distribution:", Counter(core_labels))

        select_end = time.time()

        # 创建核心集数据加载器
        subset_data = Subset(dst_train, selected_indices)
        core_loader = DataLoader(subset_data, batch_size=128, shuffle=True)

        if use_extracor:
            # 分离用于训练的模型
            del extractor
            del model
            torch.cuda.empty_cache()

        # 初始化新的模型用于训练
        if args.model == 'ResNet50':
            train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=True).to(device)
        else:
            train_model_instance = nets.__dict__[args.model](channel=channel, num_classes=num_classes, im_size=im_size, pretrained=False).to(device)
            
        train_model_instance.train()

        # 定义损失函数、优化器和学习率调度器
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(train_model_instance.parameters(), lr=args.selection_lr, momentum=args.selection_momentum, weight_decay=args.selection_weight_decay, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 定义验证函数
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

        # 定义训练函数
        def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs, patience=10):
            best_acc = 0.0
            trigger_times = 0

            for epoch in tqdm(range(num_epochs), desc="Training model"):
                model.train()
                running_loss = 0.0
                for i, (inputs, labels_batch) in enumerate(data_loader):
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                scheduler.step()  # 更新学习率

                # 每个 epoch 结束后验证模型
                acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_val, batch_size=128, shuffle=False))
                print(f"Epoch {epoch+1}/{num_epochs}, acc={acc:.4f}, precession={precision:.4f}, recall={recall:.4f}, F1 score={f1:.4f}")
                
                # 早停机制
                if acc > best_acc:
                    best_acc = acc
                    trigger_times = 0
                    # 保存最佳模型权重（可选）
                    torch.save(model.state_dict(), best_model_path)
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print("Early stopping triggered.")
                        break

        # 训练模型
        train_model(train_model_instance, core_loader, criterion, optimizer, scheduler, num_epochs=args.epochs,patience=args.patience)

        # 加载最佳模型并评估
        train_model_instance.load_state_dict(torch.load(best_model_path))
        acc, precision, recall, f1 = evaluate_model(train_model_instance, DataLoader(dst_test, batch_size=128, shuffle=False))
        print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        os.remove(best_model_path)  # 删除最佳模型权重文件
        end_time = time.time()
        selection_time = select_end-select_start
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