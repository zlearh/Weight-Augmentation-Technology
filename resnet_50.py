import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import random

def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)


class BottleNeck(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        # 1x1 卷积用于通道数的降维和升维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.param = self.parameters_k(2)
        # 3x3 卷积用于特征提取
        self.stride = stride
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积用于通道数的升维
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 残差连接的 1x1 卷积
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn4 = nn.BatchNorm2d(out_channels)
    def parameters_k(self, m):
        parameter = torch.nn.Parameter(torch.rand(m))
        return parameter
    def _qu_bias(self, conv):
        biases = conv.bias
        return biases

    def _qu_weight(self, conv,radom):
       weights = conv.weight
       if radom == 1:
           transform = transforms.Compose([
               # transforms.ToTensor(),  # 将张量转换为Tensor对象
               # transforms.RandomHorizontalFlip(),  # 随机水平翻转
               # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
               # transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
               transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
               # transforms.RandomResizedCrop((3, 3), scale=(0.8, 1.0))1
               # transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
           ])
       # elif radom == 2:
       #     transform = transforms.Compose([
       # 		# transforms.ToTensor(),  # 将张量转换为Tensor对象
       # 		#transforms.RandomHorizontalFlip(),  # 随机水平翻转
       # 		#transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
       # 		#transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
       #      transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
       #      # transforms.RandomResizedCrop((3, 3), scale=(0.8, 1.0))
       # 		#transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
       # 	])
       else:
            transform = transforms.Compose([
                # transforms.ToTensor(),  # 将张量转换为Tensor对象
                #transforms.RandomHorizontalFlip(),  # 随机水平翻转
                #transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                #transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
                # transforms.RandomAffine(degrees=0, scale=(3,3.5)),
                transforms.RandomResizedCrop((3, 3), scale=(0.8, 1.0))
                #transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
            ])
       weights = transform(weights)
       return weights


    def ture_conv(self, input, conv, param,stride,radom):
        x = F.conv2d(input, param * self._qu_weight(conv,radom), bias=self._qu_bias(conv), stride=stride,padding=1)  # 第一次卷积操作
        return x

    def forward(self, x):
        (x, deploy) = x
        # 残差分支
        residual = self.downsample(x)
        residual = self.bn4(residual)

        # 主分支
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if deploy == "1":
            x = self.ture_conv(x,self.conv2,self.param[0],self.stride,radom=random.choice([1,2]))
        elif deploy == "2":
            x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # 主分支和残差分支相加
        x += residual
        x = self.relu(x)
        x = (x, deploy)
        return x

class ResNet_50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet_50, self).__init__()
        self.deploys = "1"
        self.block = BottleNeck
        """Constructs a ResNet-18 model."""
        # self.layers = [2, 2, 2, 2]
        """Constructs a ResNet-34 model."""
        self.layers = [3, 4, 6, 3]
        self.in_channels = 64
        self.conv1 = conv7x7(3,64 , 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)


        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x,deploy="0"):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x =(x, deploy)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        (x, deploy) = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output



