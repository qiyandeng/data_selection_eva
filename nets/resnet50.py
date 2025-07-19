import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
from torchvision.models import resnet
import torchvision.models as models
import torch
from torchvision.models import ResNet18_Weights, ResNet50_Weights

def ResNet50(channel, num_classes, im_size, pretrained=False):
    # 根据 pretrained 参数加载预训练权重（使用权重枚举方式）
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None
    model = models.resnet50(weights=weights)
    
    # 修改第一个卷积层：将7x7卷积替换为3x3，stride=1，padding=1
    model.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 移除第一个最大池化层
    model.maxpool = nn.Identity()
    
    # 定义一个函数，用于对下采样块使用 ResNet-D 的方式
    def modify_downsample_block(block):
        if block.downsample is not None:
            # 获取下采样分支中卷积层的输入和输出通道数
            in_channels = block.downsample[0].in_channels
            out_channels = block.downsample[0].out_channels
            # 用平均池化 + 1x1 卷积（stride=1）+ BN 来替换原来的下采样分支
            block.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    # 对 layer2[0] 和 layer3[0] 采用 ResNet-D 的下采样方式
    modify_downsample_block(model.layer2[0])
    modify_downsample_block(model.layer3[0])
    
    # 修改最后一个 block（layer4[0]）：将其下采样 stride 改为1
    block = model.layer4[0]
    # 原始 Bottleneck 中，conv2 的 stride 用于下采样；这里将其置为 (1,1)
    block.conv2.stride = (1, 1)
    if block.downsample is not None:
        # 对于 layer4，由于不再下采样，仅需要 1x1 卷积（stride=1）和 BN 来匹配通道数
        in_channels = block.downsample[0].in_channels
        out_channels = block.downsample[0].out_channels
        block.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    # 提取特征部分：将除最后全连接层之外的所有层构成 feature_extractor
    model.feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
    
    # 将所有 ReLU 替换为 Softplus，以支持高阶导数（需保证 replace_relu_with_softplus 定义）
    replace_relu_with_softplus(model)
    
    # 修改最后一个全连接层以适应类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def replace_relu_with_softplus(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.Softplus())
        else:
            replace_relu_with_softplus(module)
