import torch
import torch.nn as nn
import timm
from nets.nets_utils import EmbeddingRecorder

class ViT(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = True, fine_tune: bool = True):  # 新增 fine_tune 参数用于全微调
        super(ViT, self).__init__()
        
        # 检查输入图像的通道数是否与 ViT 兼容（ViT 默认输入为 3 通道）
        if channel != 3:
            raise ValueError(f"ViT 模型仅支持 3 通道输入，当前输入通道数为 {channel}。")
        
        # 使用 timm 提供的 Tiny ViT 模型
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
        
        # 如果使用预训练权重，则加载本地权重
        if pretrained:
            print("Loading pretrained weights...")
            # 加载本地预训练权重
            pretrained_weights_path = "/data/dengqiyan/EVA/vit-tiny-patch16-224/pytorch_model.bin"  # 修改为你的权重文件路径
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')  # 确保加载到 CPU 上
            self.vit.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {pretrained_weights_path}")
        
        # 添加 `fc` 作为别名
        self.fc = self.vit.head

        # 嵌入记录器
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

        # 上采样层，将输入图像从 32x32 调整为 224x224
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # 训练配置：控制是否进行全微调
        if pretrained:
            for param in self.vit.parameters():
                param.requires_grad = False  # 默认冻结所有参数
            
            if fine_tune:  # 如果是全微调，则解冻所有层
                for param in self.vit.parameters():
                    param.requires_grad = True
            else:
                # 只训练最后的分类头
                for param in self.vit.head.parameters():
                    param.requires_grad = True

    def get_last_layer(self):
        """返回最后一层（分类头）"""
        return self.vit.head

    def forward(self, x):
        """前向传播"""
        with torch.set_grad_enabled(not self.no_grad):
            # 上采样输入图像到 224x224
            x = self.upsample(x)
            # ViT 的输入需要归一化到 [0, 1]
            x = self.vit(x)
            x = self.embedding_recorder(x)  # 记录嵌入
            return x

    def feature_extractor(self, x):
        """特征提取器：返回最后的特征嵌入（ViT 的输出，不包括分类头）"""
        with torch.set_grad_enabled(not self.no_grad):
            # 上采样输入图像到 224x224
            x = self.upsample(x)
            # 提取 ViT 的特征
            x = self.vit.forward_features(x)  # 提取特征
            return self.embedding_recorder(x)  # 返回嵌入结果