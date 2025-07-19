# from torchvision import datasets, transforms
# from torch import tensor, long
# from torch.utils.data import random_split
# import numpy as np
# from torch.utils.data import Subset 

# def Number(data_path, permuted=False, permutation_seed=None):
#     channel = 3  # 扩展为 3 通道
#     im_size = (32, 32)  # 调整尺寸到 32x32
#     num_classes = 10
#     mnist_mean = [0.1307, 0.1307, 0.1307]  # 复制灰度均值到三个通道
#     mnist_std = [0.3081, 0.3081, 0.3081]  # 复制灰度标准差到三个通道
#     svhn_mean = [0.4377, 0.4438, 0.4728]
#     svhn_std = [0.1980, 0.2010, 0.1970]

#     # 数据转换：ToTensor + Normalize + Resize + Repeat(扩展通道)
#     mnist_transform = transforms.Compose([
#         transforms.Resize(im_size),  # 调整尺寸到 32x32
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道扩展为三通道
#         transforms.Normalize(mean=mnist_mean, std=mnist_std)
#     ])
#     svhn_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=svhn_mean, std=svhn_std)])

#     # 加载完整的训练和测试数据集
#     full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=mnist_transform)
#     # dst_test = datasets.MNIST(data_path, train=False, download=True, transform=mnist_transform)
#     svhn_path = data_path + '/SVHN'
#     full_test_dataset = datasets.SVHN(svhn_path, split='test', download=True, transform=svhn_transform)

#     # 将完整的test set拆分为训练集和验证集（80/20 拆分）
#     val_idx = []
#     test_idx = []
    
#     for c in range(num_classes):
#         idx = np.where(np.array(full_test_dataset.labels) == c)[0]
#         len_c = len(idx)
#         c_val_size = int(0.5 * len_c)
#         c_test_size = len_c - c_val_size
#         c_val_idx, c_test_idx = random_split(idx, [c_val_size, c_test_size])
#         val_idx.extend(c_val_idx)
#         test_idx.extend(c_test_idx)
        
#     dst_val = Subset(full_test_dataset, val_idx)
#     dst_test = Subset(full_test_dataset, test_idx)

#     dst_train = full_train_dataset

#     # 定义类别名称
#     class_names = [str(c) for c in range(num_classes)]

#     # Function to set targets and labels
#     def set_targets_to_labels(dataset, full_dataset):
#         if isinstance(dataset, Subset):
#             indices = dataset.indices
#             dataset.targets = tensor([full_dataset.targets[i] for i in indices], dtype=long)
#             dataset.labels = [full_dataset.targets[i] for i in indices]
#         else:
#             dataset.targets = tensor(full_dataset.targets, dtype=long)
#             dataset.labels = full_dataset.targets
    
#     def set_labels_to_targets(dataset, full_dataset):
#         if isinstance(dataset, Subset):
#             indices = dataset.indices
#             dataset.labels = [full_dataset.labels[i] for i in indices]
#             dataset.targets = tensor([full_dataset.labels[i] for i in indices], dtype=long)
#         else:
#             dataset.labels = full_dataset.targets
#             dataset.targets = tensor(full_dataset.targets, dtype=long)

#     # Set targets and labels for training, validation, and test datasets
#     set_targets_to_labels(dst_train, full_train_dataset)
#     set_labels_to_targets(dst_val, full_test_dataset)
#     set_labels_to_targets(dst_test, full_test_dataset)

#     return channel, im_size, num_classes, class_names, mnist_mean, mnist_std, dst_train, dst_val, dst_test


from torchvision import datasets, transforms
from torch import tensor, long
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import Subset 

def Number(data_path, permuted=False, permutation_seed=None):
    channel = 3  # 扩展为 3 通道
    im_size = (32, 32)  # 调整尺寸到 32x32
    num_classes = 10
    mnist_mean = [0.1307, 0.1307, 0.1307]  # 复制灰度均值到三个通道
    mnist_std = [0.3081, 0.3081, 0.3081]  # 复制灰度标准差到三个通道
    svhn_mean = [0.4377, 0.4438, 0.4728]
    svhn_std = [0.1980, 0.2010, 0.1970]

    # 数据转换：ToTensor + Normalize + Resize + Repeat(扩展通道)
    mnist_transform = transforms.Compose([
        transforms.Resize(im_size),  # 调整尺寸到 32x32
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道扩展为三通道
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])
    svhn_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=svhn_mean, std=svhn_std)])

    # 加载完整的训练和测试数据集
    full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=mnist_transform)
    svhn_path = data_path + '/SVHN'
    full_test_dataset = datasets.SVHN(svhn_path, split='test', download=True, transform=svhn_transform)

    # 将完整的train set拆分为训练集和验证集（80/20 拆分）
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    dst_train = train_dataset
    dst_val = val_dataset
    dst_test = full_test_dataset

    # 定义类别名称
    class_names = [str(c) for c in range(num_classes)]

    # Function to set targets and labels
    # def set_targets_to_labels(dataset, full_dataset):
    #     if isinstance(dataset, Subset):
    #         indices = dataset.indices
    #         try:
    #             dataset.targets = tensor([full_dataset.targets[i] for i in indices], dtype=long)
    #         except:
    #             dataset.targets = tensor([full_dataset.labels[i] for i in indices], dtype=long)
    #         dataset.labels = [full_dataset.targets[i] for i in indices]
    #     else:
    #         dataset.targets = tensor(full_dataset.targets, dtype=long)
    #         dataset.labels = full_dataset.targets
    
    # def set_labels_to_targets(dataset, full_dataset):
    #     if isinstance(dataset, Subset):
    #         indices = dataset.indices
    #         # dataset.labels = [full_dataset.targets[i] for i in indices]  # 修复在这里
    #         try:
    #             dataset.labels = [full_dataset.targets[i] for i in indices]  # 修复在这里
    #         except:
    #             dataset.labels = [full_dataset.labels[i] for i in indices]
    #         dataset.targets = tensor([full_dataset.targets[i] for i in indices], dtype=long)
    #     else:
    #         dataset.labels = full_dataset.targets
    #         dataset.targets = tensor(full_dataset.targets, dtype=long)
    def set_targets_to_labels(dataset, full_dataset):
        if hasattr(full_dataset, 'targets'):
            full_labels = full_dataset.targets
        else:
            full_labels = full_dataset.labels

        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset.targets = tensor([full_labels[i] for i in indices], dtype=long)
            dataset.labels = [full_labels[i] for i in indices]
        else:
            dataset.targets = tensor(full_labels, dtype=long)
            dataset.labels = full_labels

    def set_labels_to_targets(dataset, full_dataset):
        if hasattr(full_dataset, 'targets'):
            full_labels = full_dataset.targets
        else:
            full_labels = full_dataset.labels

        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset.labels = [full_labels[i] for i in indices]
            dataset.targets = tensor([full_labels[i] for i in indices], dtype=long)
        else:
            dataset.labels = full_labels
            dataset.targets = tensor(full_labels, dtype=long)


    # Set targets and labels for training, validation, and test datasets
    set_targets_to_labels(dst_train, full_train_dataset)
    set_labels_to_targets(dst_val, full_train_dataset)  # 使用训练集的标签
    set_labels_to_targets(dst_test, full_test_dataset)

    return channel, im_size, num_classes, class_names, mnist_mean, mnist_std, dst_train, dst_val, dst_test



