import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
from torchvision.models import resnet
import torchvision.models as models
import torch
from torchvision.models import ResNet18_Weights, ResNet50_Weights


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_32x32(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, record_embedding: bool = False,
                 no_grad: bool = False):
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(channel, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            #print(out.shape)  # 添加此行来查看输出形状
            out = self.embedding_recorder(out)
            out = self.linear(out)
        return out


class ResNet_224x224(resnet.ResNet):
    def __init__(self, block, layers, channel: int, num_classes: int, record_embedding: bool = False,
                 no_grad: bool = False, **kwargs):
        super().__init__(block, layers, **kwargs)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if channel != 3:
            self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        with set_grad_enabled(not self.no_grad):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = flatten(x, 1)
            x = self.embedding_recorder(x)
            x = self.fc(x)

        return x


def ResNet(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False):
    arch = arch.lower()
    if pretrained:
        if arch == "resnet18":
            net = ResNet_224x224(resnet.BasicBlock, [2, 2, 2, 2], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(resnet.BasicBlock, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 6, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 23, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(resnet.Bottleneck, [3, 8, 36, 3], channel=3, num_classes=1000,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(resnet.model_urls[arch], progress=True)
        net.load_state_dict(state_dict)

        if channel != 3:
            net.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif im_size[0] == 224 and im_size[1] == 224:
        if arch == "resnet18":
            net = ResNet_224x224(resnet.BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_224x224(resnet.BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_224x224(resnet.Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_224x224(resnet.Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                                 record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "resnet18":
            net = ResNet_32x32(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet34":
            net = ResNet_32x32(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet50":
            net = ResNet_32x32(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet101":
            net = ResNet_32x32(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "resnet152":
            net = ResNet_32x32(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes,
                               record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


# def ResNet18(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
#              pretrained: bool = False):
#     return ResNet("resnet18", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
# def ResNet18(channel, num_classes, im_size, pretrained=False):
#     model = models.resnet18(pretrained=pretrained)
#     if channel != 3:
#         # 修改第一个卷积层以适应不同的通道数
#         model.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     else:
#         # 对于 CIFAR-10 等小尺寸图像，调整第一个卷积层
#         model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     # 移除最大池化层
#     model.maxpool = nn.Identity()
#     # 修改最后一个全连接层以适应类别数
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model
# def ResNet18(channel, num_classes, im_size, pretrained=False):
#     model = models.resnet18(pretrained=pretrained)

#     if pretrained:
#         weights = ResNet18_Weights.DEFAULT  # 使用预训练权重
#     else:
#         weights = None  # 不使用预训练权重
#     model = models.resnet18(weights=weights)
    
#     # 修改第一个卷积层以适应不同的通道数
#     if channel != 3:
#         model.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     else:
#         model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
#     # 对于 CIFAR-10 等小尺寸图像，移除最大池化层
#     model.maxpool = nn.Identity()
#     # 提取特征部分 (去掉全连接层)
#     model.feature_extractor = nn.Sequential(
#         *(list(model.children())[:-1])  # 去掉最后的全连接层
#     )
#     # 修改最后一个全连接层以适应类别数
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model
def ResNet18(channel, num_classes, im_size, pretrained=False):
    model = models.resnet18(pretrained=pretrained)

    if pretrained:
        weights = ResNet18_Weights.DEFAULT  # 使用预训练权重
    else:
        weights = None  # 不使用预训练权重
    model = models.resnet18(weights=weights)
    
    # 修改第一个卷积层以适应不同的通道数
    if channel != 3:
        model.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 移除最大池化层
    model.maxpool = nn.Identity()
    # 提取特征部分 (去掉全连接层)
    model.feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
    
    # 将所有 ReLU 替换为 Softplus，以支持高阶导数
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

def ResNet34(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet34", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


# def ResNet50(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
#              pretrained: bool = False):
#     return ResNet("resnet50", channel, num_classes, im_size, record_embedding, no_grad, pretrained)

def ResNet50(channel, num_classes, im_size, pretrained=False):
    model = models.resnet50(pretrained=pretrained)

    if pretrained:
        weights = ResNet50_Weights.DEFAULT  # 使用预训练权重
    else:
        weights = None  # 不使用预训练权重
    model = models.resnet50(weights=weights)
    
    # 修改第一个卷积层以适应不同的通道数
    if channel != 3:
        model.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 移除最大池化层
    model.maxpool = nn.Identity()
    # 提取特征部分 (去掉全连接层)
    model.feature_extractor = nn.Sequential(
        *(list(model.children())[:-1])  # 去掉最后的全连接层
    )
    
    # 将所有 ReLU 替换为 Softplus，以支持高阶导数
    replace_relu_with_softplus(model)
    
    # 修改最后一个全连接层以适应类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model



def ResNet101(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet101", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def ResNet152(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet152", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
