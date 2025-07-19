import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from nets.nets_utils import EmbeddingRecorder
import numpy as np


class REMLP(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False, round_output: bool = False):
        if pretrained:
            raise NotImplementedError("torchvision pretrained models not available.")
        super(REMLP, self).__init__()
        
        # 计算输入特征数
        if isinstance(im_size, (tuple, list)):
            input_features = channel * np.prod(im_size)
        else:
            input_features = im_size * channel if channel != 1 else im_size

        # 定义全连接层
        self.fc_1 = nn.Linear(input_features, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 1)  # 输出为 1 个连续值

        # 添加 `fc` 作为别名
        self.fc = self.fc_3

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad
        self.round_output = round_output  # 是否进行输出离散化

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            # 展平输入
            out = torch.flatten(x, 1)  # 从第1维开始展平
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            out = self.embedding_recorder(out)
            out = self.fc_3(out)  # 输出连续值
            
            if self.round_output:
                out = torch.round(out)  # 可选：将连续值离散化为整数
            
        return out

    def feature_extractor(self, x):
        """特征提取器：返回最后的特征嵌入（fc_2 输出后的结果）"""
        with set_grad_enabled(not self.no_grad):
            # 展平输入
            out = torch.flatten(x, 1)  # [batch_size, input_features]
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            return self.embedding_recorder(out)  # 返回嵌入结果