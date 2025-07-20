# main.py
import torch
from torch.utils.data import DataLoader

import nets
from mydatasets import adult
from selection.kmeans import KMeansSelection
from model_utils import train_model, evaluate_model 

################################################################
####### if you have new methods, please import them here #######
################################################################

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

# process data
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = adult(data_path, balance_target)

# initialize a nn.Module instance (for selection and training)
base_model = nets.__dict__[model_name](
    channel=channel,
    num_classes=num_classes,
    im_size=im_size,
    pretrained=False
)

# core set selection
selector = KMeansSelection()
core_loader = selector.select(
    model=base_model,
    selection_fraction=selection_fraction,
    k=num_classes,
    train_set=dst_train,
    task_type=task_type,
    device=device
)

# training
trained_model = train_model(
    model=base_model,              # nn.Module instance
    train_loader=core_loader,      # core set DataLoader
    val_loader=DataLoader(
        dst_val, batch_size=batch_size, shuffle=False
    ),                              # validation set DataLoader
    device=device,
    task_type=task_type,
    epochs=epochs,
    patience=patience,
    lr=selection_lr,
    momentum=selection_momentum,
    weight_decay=selection_weight_decay,
    best_model_path=best_model_path
)

# evaluate on test set
evaluate_model(
    model=trained_model,
    best_model_path=best_model_path,
    test_loader=DataLoader(
        dst_test, batch_size=batch_size, shuffle=False
    ),
    device=device,
    task_type=task_type
)
