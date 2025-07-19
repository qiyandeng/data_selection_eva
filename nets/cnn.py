import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from nets.nets_utils import EmbeddingRecorder

''' 1D CNN 模型 适用于表格数据 '''

class CNN(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("预训练模型不可用。")
        super(CNN, self).__init__()

        # 对于表格数据，im_size 表示特征数
        if isinstance(im_size, int):
            feature_dim = im_size
        elif isinstance(im_size, (tuple, list)):
            if len(im_size) == 1:
                feature_dim = im_size[0]
            else:
                raise ValueError("对于表格数据，im_size 应为单个数值或单元素元组。")
        else:
            raise TypeError("im_size 必须为 int 或 单元素 tuple/list。")
        
        # 假设输入数据形状为 (batch_size, 1, feature_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算池化后的特征数
        # 这里假设经过两次池化操作，特征数缩减为 feature_dim // 4
        conv_output_features = 64 * (feature_dim // 4)

        self.fc_1 = nn.Linear(conv_output_features, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)
        self.fc = self.fc_3

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            # 假设输入 x 的形状为 (batch_size, feature_dim)
            # 先将数据 reshape 成 (batch_size, 1, feature_dim)
            x = x.unsqueeze(1)
            out = F.relu(self.conv1(x))
            out = self.pool(out)
            out = F.relu(self.conv2(out))
            out = self.pool(out)
            out = out.flatten(1)
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            out = self.embedding_recorder(out)
            out = self.fc_3(out)
        return out

    def feature_extractor(self, x):
        with set_grad_enabled(not self.no_grad):
            x = x.unsqueeze(1)
            out = F.relu(self.conv1(x))
            out = self.pool(out)
            out = F.relu(self.conv2(out))
            out = self.pool(out)
            out = out.flatten(1)
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            return self.embedding_recorder(out)