# main.py
import torch
from torch.utils.data import DataLoader

import nets
from mydatasets import adult
from methods.kmeans import KMeansSelection
from model_utils import train_model, evaluate_model  # 新的训练/评估函数

########################################################
####### 如果你有新加入的模型或方法，请确保在这里导入它们 #######
########################################################
# 例如， from methods.some_other_method import SomeOtherSelection

# 基本参数设置
data_path = "mydatasets"
balance_target = True
method = 'kmeans'
task_type = 'classification'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_path = f"best_models/{method}.pth"

epochs = 100
batch_size = 128
patience = 10

selection_fraction = 0.1
model_name = 'MLP'
selection_lr = 0.001
selection_momentum = 0.9
selection_weight_decay = 5e-4

# 数据处理
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = adult(data_path, balance_target)

# 初始化一个 nn.Module 实例（用于后面选择和正式训练）
base_model = nets.__dict__[model_name](
    channel=channel,
    num_classes=num_classes,
    im_size=im_size,
    pretrained=False
)

# 核心集选择
selector = KMeansSelection()
core_loader = selector.select(
    model=base_model,
    selection_fraction=selection_fraction,
    k=num_classes,
    train_set=dst_train,
    task_type=task_type,
    device=device
)

# 正式训练（会保存并加载 best_model_path）
trained_model = train_model(
    model=base_model,              # nn.Module 实例
    train_loader=core_loader,      # 选出的核心集 DataLoader
    val_loader=DataLoader(
        dst_val, batch_size=batch_size, shuffle=False
    ),                              # 验证集 DataLoader
    device=device,
    task_type=task_type,
    epochs=epochs,
    patience=patience,
    lr=selection_lr,
    momentum=selection_momentum,
    weight_decay=selection_weight_decay,
    best_model_path=best_model_path
)

# 在测试集上评估
evaluate_model(
    model=trained_model,
    best_model_path=best_model_path,
    test_loader=DataLoader(
        dst_test, batch_size=batch_size, shuffle=False
    ),
    device=device,
    task_type=task_type
)
