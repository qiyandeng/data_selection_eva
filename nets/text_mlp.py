# import torch  
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import set_grad_enabled
# from nets.nets_utils import EmbeddingRecorder  # 确保这个模块存在且正确导入
# import numpy as np

# ''' Text MLP for Text Training '''

# class TextMLP(nn.Module):
#     def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
#                  pretrained: bool = False):
#         if pretrained:
#             raise NotImplementedError("Pretrained models are not available for TextMLP.")
#         super(TextMLP, self).__init__()
        
#         # 计算输入特征数
#         if isinstance(im_size, (tuple, list)):
#             input_features = channel * np.prod(im_size)
#         else:
#             input_features = im_size * channel if channel != 1 else im_size

#         # 定义多个隐藏层
#         self.fc1 = nn.Linear(input_features, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.dropout1 = nn.Dropout(0.5)
        
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(0.5)
        
#         self.fc3 = nn.Linear(256, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.dropout3 = nn.Dropout(0.5)
        
#         self.fc4 = nn.Linear(128, num_classes)

#         self.fc = self.fc4  # 将 `fc` 指向 `fc4`

#         # 记录嵌入（如果需要）
#         self.embedding_recorder = EmbeddingRecorder(record_embedding)

#         self.no_grad = no_grad

#     def get_last_layer(self):
#         return self.fc4

#     def forward(self, x):
#         with set_grad_enabled(not self.no_grad):
#             # 展平输入
#             out = torch.flatten(x, 1)  # 从第1维开始展平，形状：[batch_size, input_features]
            
#             # 第一隐藏层
#             out = self.fc1(out)          # [batch_size, 512]
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.dropout1(out)
            
#             # 第二隐藏层
#             out = self.fc2(out)          # [batch_size, 256]
#             out = self.bn2(out)
#             out = F.relu(out)
#             out = self.dropout2(out)
            
#             # 第三隐藏层
#             out = self.fc3(out)          # [batch_size, 128]
#             out = self.bn3(out)
#             out = F.relu(out)
#             out = self.dropout3(out)
            
#             # 记录嵌入
#             out = self.embedding_recorder(out)  # [batch_size, 128] 或自定义的形状
            
#             # 输出层
#             out = self.fc4(out)          # [batch_size, num_classes]
#         return out

#     def feature_extractor(self, x):
#         """返回特征提取结果（输出到最后一层之前的特征）"""
#         with set_grad_enabled(not self.no_grad):
#             # 展平输入
#             out = torch.flatten(x, 1)  # [batch_size, input_features]
            
#             # 第一隐藏层
#             out = self.fc1(out)
#             out = self.bn1(out)
#             out = F.relu(out)
#             out = self.dropout1(out)
            
#             # 第二隐藏层
#             out = self.fc2(out)
#             out = self.bn2(out)
#             out = F.relu(out)
#             out = self.dropout2(out)
            
#             # 第三隐藏层
#             out = self.fc3(out)
#             out = self.bn3(out)
#             out = F.relu(out)
#             out = self.dropout3(out)
            
#             return out
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from nets.nets_utils import EmbeddingRecorder
import numpy as np

''' Text MLP '''

class TextMLP(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("Pretrained models are not available for TextMLP.")
        super(TextMLP, self).__init__()
        
        # 计算输入特征数
        if isinstance(im_size, (tuple, list)):
            input_features = channel * np.prod(im_size)
        else:
            input_features = im_size * channel if channel != 1 else im_size

        self.fc_1 = nn.Linear(input_features, 128)
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
            # 展平输入
            out = torch.flatten(x, 1)  # 从第1维开始展平
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            out = self.embedding_recorder(out)  # 记录嵌入
            out = self.fc_3(out)  # 输出层
        return out

    def feature_extractor(self, x):
        """特征提取器：返回最后的特征嵌入（fc_2 输出后的结果）"""
        with set_grad_enabled(not self.no_grad):
            # 展平输入
            out = torch.flatten(x, 1)  # [batch_size, input_features]
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            return self.embedding_recorder(out)  # 返回嵌入结果