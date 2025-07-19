# model_utils.py

import os
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)

def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    task_type: str,
    epochs: int,
    patience: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    best_model_path: str
) -> torch.nn.Module:
    """
    训练 model 并根据 val_loader 上的指标做早停，保存最优权重到 best_model_path，
    最后加载并返回最优权重的 model。
    """
    model = model.to(device)
    # 优化器与调度器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 损失函数
    if task_type == "classification":
        criterion = torch.nn.CrossEntropyLoss().to(device)
        best_metric = 0.0
    else:
        criterion = torch.nn.MSELoss().to(device)
        best_metric = float('inf')

    trigger_times = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            if task_type == "classification":
                loss = criterion(out, y)
            else:
                loss = criterion(out.squeeze(), y.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            y_preds, y_trues = [], []
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                if task_type == "classification":
                    _, preds = out_val.max(1)
                    y_preds.extend(preds.cpu().numpy())
                    y_trues.extend(y_val.cpu().numpy())
                else:
                    vals = out_val.squeeze()
                    y_preds.extend(vals.cpu().numpy())
                    y_trues.extend(y_val.cpu().numpy())

        # 计算指标
        if task_type == "classification":
            acc = accuracy_score(y_trues, y_preds)
            metric = acc
        else:
            mse = mean_squared_error(y_trues, y_preds)
            metric = mse

        # 早停判断
        improved = (
            (task_type == "classification" and metric > best_metric) or
            (task_type == "regression" and metric < best_metric)
        )
        if improved:
            best_metric = metric
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 加载最优权重
    model.load_state_dict(torch.load(best_model_path))
    return model


def evaluate_model(
    model: torch.nn.Module,
    best_model_path: str,
    test_loader,
    device: torch.device,
    task_type: str
):
    """
    加载 best_model_path 中的权重，然后在 test_loader 上评估。
    返回分类任务时 (acc, precision, recall, f1)；
    回归任务时 (mse, mae)。
    """
    model = model.to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_preds, y_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if task_type == "classification":
                _, preds = out.max(1)
                y_preds.extend(preds.cpu().numpy())
                y_trues.extend(y.cpu().numpy())
            else:
                vals = out.squeeze()
                y_preds.extend(vals.cpu().numpy())
                y_trues.extend(y.cpu().numpy())

    if task_type == "classification":
        acc = accuracy_score(y_trues, y_preds)
        precision = precision_score(y_trues, y_preds, average='weighted', zero_division=0)
        recall = recall_score(y_trues, y_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_trues, y_preds, average='weighted', zero_division=0)
        print(f"Test → Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return acc, precision, recall, f1
    else:
        mse = mean_squared_error(y_trues, y_preds)
        mae = mean_absolute_error(y_trues, y_preds)
        print(f"Test → MSE={mse:.4f}, MAE={mae:.4f}")
        return mse, mae
