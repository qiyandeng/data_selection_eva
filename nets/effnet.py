import torch
import torch.nn as nn
from torch import set_grad_enabled
from nets.nets_utils import EmbeddingRecorder
import timm

class EffNetL2(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding=False, no_grad=False,
                 pretrained=True, fine_tune=True):
        super(EffNetL2, self).__init__()
        
        # 初始化预训练模型
        self.backbone = timm.create_model(
            'tf_efficientnet_l2.ns_jft_in1k_475',
            pretrained=False,
            num_classes=0,  # 移除原始分类头
            global_pool='avg'  # 保持特征提取方式
        )
        
        # 加载本地预训练权重
        if pretrained:
            checkpoint_path = "/data/dengqiyan/EVA/tf_efficientnet_l2.ns_jft_in1k_475/pytorch_model.bin"
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {checkpoint_path}")

        # 添加输入适配层（关键修改）
        self.input_adapt = nn.Sequential(
            nn.Conv2d(channel, 3, kernel_size=1),  # 将任意输入通道转换为3
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # 自定义分类头
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # 嵌入记录器
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

        # 参数冻结设置
        if pretrained and not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.input_adapt.parameters():
                param.requires_grad = True

    def get_last_layer(self):
        return self.fc[-1]

    def forward_features(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.input_adapt(x)
            return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        out = self.fc(features)
        out = self.embedding_recorder(out)
        return out

    def feature_extractor(self, x):
        return self.forward_features(x)