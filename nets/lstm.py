import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from nets.nets_utils import EmbeddingRecorder
import numpy as np

''' LSTM 模型 '''

class LSTM(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("预训练模型不可用。")
        super(LSTM, self).__init__()

        # 假设 im_size 是 BERT 的嵌入维度（例如，768 或 512）
        input_features = im_size  # 即 BERT 输出的维度（例如 768）

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_features, 128, batch_first=True)  # 输出维度为 128
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

        # 添加 `fc` 作为别名
        self.fc = self.fc_3

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            # LSTM 层处理时，确保输入的形状为 (batch_size, seq_length, input_features)
            out, (hn, cn) = self.lstm(x)  # 获取 LSTM 的输出

            # 如果 `out` 是三维的，取最后一个时间步的输出
            if out.dim() == 3:
                out = out[:, -1, :]  # 只取最后一个时间步的输出（适用于分类任务）
            else:
                # 如果 `out` 只有二维（batch_size, hidden_size），直接使用它
                out = out

            # 全连接层部分
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            out = self.embedding_recorder(out)
            out = self.fc_3(out)
        return out

    def feature_extractor(self, x):
        with set_grad_enabled(not self.no_grad):
            # LSTM 部分
            out, (hn, cn) = self.lstm(x)
            out = out[:, -1, :]  # 取最后一个时间步的输出
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            return self.embedding_recorder(out)