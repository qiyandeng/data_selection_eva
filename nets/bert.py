import torch
import torch.nn as nn
from transformers import T5EncoderModel

class T5(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = True, fine_tune: bool = True):
        super(T5, self).__init__()

        if channel != 1:
            raise ValueError(f"T5 模型仅支持 1 通道输入，当前输入通道数为 {channel}。")

        self.t5 = T5EncoderModel.from_pretrained("/data/dengqiyan/EVA/t5-small")

        # 降维层：把 768 维度的 BERT embedding 转换为 T5 需要的 512 维
        self.embedding_transform = nn.Linear(768, self.t5.config.d_model)  

        self.fc = nn.Linear(self.t5.config.d_model, num_classes)
        self.no_grad = no_grad

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
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, hidden_size) → (batch_size, 1, hidden_size)
        
        # print(f"Before transform: {x.shape}")  # Debugging

        x = self.embedding_transform(x)  # 降维到 512
        # print(f"After transform: {x.shape}")  # Debugging

        with torch.set_grad_enabled(not self.no_grad):
            outputs = self.t5(inputs_embeds=x)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
            x = self.fc(pooled_output)
            return x
