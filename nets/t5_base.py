import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nets.nets_utils import EmbeddingRecorder

from transformers import T5EncoderModel

class T5(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = True, fine_tune: bool = True):
        super(T5, self).__init__()

        # 检查通道数
        if channel != 1:
            raise ValueError(f"T5 模型仅支持 1 通道输入，当前输入通道数为 {channel}。")

        # 使用 T5 Encoder 而不是 T5ForConditionalGeneration
        self.t5 = T5EncoderModel.from_pretrained("/data/dengqiyan/EVA/t5-base")

        # 添加分类头
        self.fc = nn.Linear(self.t5.config.d_model, num_classes)

        # 嵌入记录器
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

        # 训练配置
        if pretrained:
            for param in self.t5.parameters():
                param.requires_grad = False

            if fine_tune:
                for param in self.t5.parameters():
                    param.requires_grad = True
            else:
                for param in self.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        if len(x.shape) == 2:  # 如果 x 形状是 (batch_size, hidden_size)
            x = x.unsqueeze(1)  # 变成 (batch_size, seq_length=1, hidden_size)
        
        # print(f"Fixed input shape: {x.shape}")  # Debug

        with torch.set_grad_enabled(not self.no_grad):
            outputs = self.t5(inputs_embeds=x)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            x = self.fc(pooled_output)
            x = self.embedding_recorder(x)
            return x

