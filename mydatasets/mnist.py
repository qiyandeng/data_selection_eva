from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch import tensor, long
import numpy as np

def MNIST(data_path, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    # 数据转换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # 如果需要打乱像素顺序，进行像素置换
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))]
        )

    # 加载完整的训练和测试数据集
    full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # 将完整的训练集拆分为训练集和验证集（80/20 拆分）
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    dst_train, dst_val = random_split(full_train_dataset, [train_size, val_size])

    # 定义类别名称
    class_names = [str(c) for c in range(num_classes)]

    # 将标签转换为 `tensor`
    dst_train.targets = tensor([full_train_dataset.targets[i] for i in dst_train.indices], dtype=long)
    dst_val.targets = tensor([full_train_dataset.targets[i] for i in dst_val.indices], dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test

def permutedMNIST(data_path, permutation_seed=None):
    return MNIST(data_path, True, permutation_seed)