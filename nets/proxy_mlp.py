import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
import numpy as np

class ProxyMLP(nn.Module):
    def __init__(self, channel, num_classes, im_size, no_grad: bool = False, pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("Pretrained models are not available for ProxyMLP.")
        super(ProxyMLP, self).__init__()
        
        # 计算输入特征数
        if isinstance(im_size, (tuple, list)):
            input_features = channel * np.prod(im_size)
        else:
            input_features = im_size * channel if channel != 1 else im_size

        # 简化模型，仅使用一层隐藏层
        self.fc_1 = nn.Linear(input_features, 64)
        self.fc_out = nn.Linear(64, num_classes)
        self.no_grad = no_grad

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = torch.flatten(x, 1)  # 展平输入
            x = F.relu(self.fc_1(x))
            x = self.fc_out(x)
        return x