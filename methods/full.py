import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # 用于进度条显示
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import mydatasets
import nets
import torch.optim.lr_scheduler as lr_scheduler


class EmbeddingExtractor(nn.Module):
    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.model.fc = nn.Identity()  # 替换最后的全连接层

    def forward(self, x):
        x = self.model(x)
        return x

def full_selection(args,channel,im_size,num_classes,class_names,mean,std, dst_train,dst_val,dst_test):

    import logging
    from collections import Counter
    import os
    import time
    start_time = time.time()
    
    # log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' + str(args.selection_fraction)  +'/' + args.method + '.log'
    # best_model_path = args.save_best_model_path + args.dataset + '/' +  str(args.selection_fraction)  +'/' + args.method + '.pth'
    log_path = args.save_log_path + args.dataset + '/' +  args.model + '/' +  args.method + '.log'
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

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("CUDA is available:", torch.cuda.is_available())
    # best_model_path = args.save_best_model_path + args.dataset + '_' +  args.method +'_' + str(args.selection_fraction) + '_' + str(args.balance_target)  + '.pth'

    # 初始化模型用于提取嵌入
    global bool_del
    bool_del = 0

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

    
    # 初始化新的模型用于训练
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
                

                # 数值稳定性处理：防止输出过大或过小
                outputs = torch.clamp(outputs, min=-10, max=10)

                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                running_loss += loss.item()

            scheduler.step()  # 更新学习率

            # 每个 epoch 结束后验证模型
            acc, precision, recall, f1 = evaluate_model(model, DataLoader(dst_val, batch_size=128, shuffle=False))
            print(f"Epoch {epoch + 1}/{num_epochs}, loss: {running_loss:.4f}, acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

            
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
    train_model(train_model_instance, train_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, patience=args.patience)

    # 加载最佳模型并评估
    train_model_instance.load_state_dict(torch.load(best_model_path))
    acc, precision, recall, f1 = evaluate_model(train_model_instance, DataLoader(dst_test, batch_size=128, shuffle=False))
    print(f"acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    os.remove(best_model_path)
    end_time = time.time()
    print(f"Time cost: {end_time - start_time:.2f}s")
