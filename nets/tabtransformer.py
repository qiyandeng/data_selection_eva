from tab_transformer_pytorch import TabTransformer
import torch
import torch.nn as nn
import numpy as np
from nets.nets_utils import EmbeddingRecorder
from torch import set_grad_enabled

# class TabTransformerModel(nn.Module):
#     def __init__(self, channel, num_classes, im_size, pretrained=False, record_embedding: bool = False, no_grad: bool = False):
#         super(TabTransformerModel, self).__init__()
        
#         # 计算总特征数
#         if isinstance(im_size, (tuple, list)):
#             total_features = int(np.prod(im_size))
#         else:
#             total_features = im_size
        
#         # 假设前6列为类别特征，其余为连续特征
#         num_categ = 6
#         num_continuous = total_features - num_categ
        
#         # 默认每个类别特征有10个取值（根据实际情况修改）
#         categories = (10,) * num_categ

#         # 定义各模块默认参数
#         dim = 128          # 嵌入维度
#         dim_out = num_classes  # 输出维度，即类别数
#         depth = 4          # Transformer 堆叠深度
#         heads = 4          # 多头注意力头数
#         attn_dropout = 0.1 # 注意力 dropout 比例
#         ff_dropout = 0.1   # 前馈网络 dropout 比例
#         mlp_hidden_mults = (4, 2)  # MLP 隐藏层倍数
#         mlp_act = nn.ReLU()        # 激活函数

#         # 定义 TabTransformer 模型
#         self.tab_transformer = TabTransformer(
#             categories=categories,      # 类别特征信息
#             num_continuous=num_continuous,  # 连续特征数量
#             dim=dim,                    # 嵌入维度
#             dim_out=dim_out,            # 输出维度
#             depth=depth,                # 堆叠深度
#             heads=heads,                # 注意力头数
#             attn_dropout=attn_dropout,  # 注意力 dropout
#             ff_dropout=ff_dropout,      # 前馈 dropout
#             mlp_hidden_mults=mlp_hidden_mults,  # MLP 隐藏层倍数
#             mlp_act=mlp_act,            # 激活函数
#         )

#         # 嵌入记录器（如需要记录嵌入则可使用）
#         self.embedding_recorder = EmbeddingRecorder(record_embedding)
#         self.no_grad = no_grad

#     def forward(self, x):
#         """
#         前向传播：假设 x 的形状为 (batch_size, total_features)
#         其中前6列为类别数据，其他为连续数据。
#         """
#         with set_grad_enabled(not self.no_grad):
#             # 分离类别和连续数据
#             x_categ = x[:, :6].long()      # 前6列作为类别数据，转为 long 类型（嵌入需要）
#             x_cont = x[:, 6:]              # 剩余部分作为连续数据

#             # 将数据传入 TabTransformer
#             out = self.tab_transformer(x_categ, x_cont)
#             out = self.embedding_recorder(out)
#             return out

#     def feature_extractor(self, x):
#         """
#         特征提取器：返回 TabTransformer 提取的特征
#         """
#         with set_grad_enabled(not self.no_grad):
#             x_categ = x[:, :6].long()
#             x_cont = x[:, 6:]
#             out = self.tab_transformer(x_categ, x_cont)
#             return self.embedding_recorder(out)

class TabTransformerModel(nn.Module):
    def __init__(self, channel, num_classes, im_size, pretrained=False, record_embedding: bool = False, no_grad: bool = False):
        super(TabTransformerModel, self).__init__()
        
        # 计算总特征数
        if isinstance(im_size, (tuple, list)):
            total_features = int(np.prod(im_size))
        else:
            total_features = im_size
        
        # 假设前6列为类别特征，其余为连续特征
        num_categ = 6
        num_continuous = total_features - num_categ
        
        # 默认每个类别特征有10个取值（根据实际情况修改）
        categories = (10,) * num_categ

        # 定义各模块默认参数
        dim = 128          # 嵌入维度
        depth = 4          # Transformer 堆叠深度
        heads = 4          # 多头注意力头数
        attn_dropout = 0.1 # 注意力 dropout 比例
        ff_dropout = 0.1   # 前馈网络 dropout 比例
        mlp_hidden_mults = (4, 2)  # MLP 隐藏层倍数
        mlp_act = nn.ReLU()        # 激活函数

        # 定义 TabTransformer 模型
        self.tab_transformer = TabTransformer(
            categories=categories,          # 类别特征信息
            num_continuous=num_continuous,  # 连续特征数量
            dim=dim,                        # 嵌入维度
            dim_out=dim,                    # 输出嵌入维度（注意这里输出维度设置为 dim，而不是直接 num_classes）
            depth=depth,                    # 堆叠深度
            heads=heads,                    # 注意力头数
            attn_dropout=attn_dropout,      # 注意力 dropout
            ff_dropout=ff_dropout,          # 前馈 dropout
            mlp_hidden_mults=mlp_hidden_mults,  # MLP 隐藏层倍数
            mlp_act=mlp_act,                # 激活函数
        )

        # 添加全连接层，将提取的特征映射到类别数上
        self.fc = nn.Linear(dim, num_classes)
        # 嵌入记录器（如需要记录嵌入则可使用）
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def forward(self, x):
        """
        前向传播：假设 x 的形状为 (batch_size, total_features)
        其中前6列为类别数据，其他为连续数据。
        """
        with torch.set_grad_enabled(not self.no_grad):
            # 分离类别和连续数据
            x_categ = x[:, :6].long()      # 前6列作为类别数据，转为 long 类型（嵌入需要）
            x_cont = x[:, 6:]              # 剩余部分作为连续数据

            # 将数据传入 TabTransformer
            out = self.tab_transformer(x_categ, x_cont)
            # 记录或处理嵌入
            features = self.embedding_recorder(out)
            # 通过全连接层得到 logits
            logits = self.fc(features)
            return logits

    def feature_extractor(self, x):
        """
        特征提取器：返回 TabTransformer 提取的特征
        """
        with torch.set_grad_enabled(not self.no_grad):
            x_categ = x[:, :6].long()
            x_cont = x[:, 6:]
            out = self.tab_transformer(x_categ, x_cont)
            features = self.embedding_recorder(out)
            return features
