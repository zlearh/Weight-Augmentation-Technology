"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as transforms

import random
class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True))

        self.conv = nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t)

        self.residual_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.param = self.parameters_k(2)
        self.in_channels = in_channels
        self.out_channels = out_channels
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


    def ture_conv(self, inputs, conv, param,stride,radom):
        x = F.conv2d(inputs, param * self._qu_weight(conv,radom), bias=self._qu_bias(conv), stride=stride,padding=1,groups=inputs.size(1))  # 第一次卷积操作
        return x

    def forward(self, x):
        (x,deploy) = x
        residual = self.residual(x)
        if deploy == "1":
            residual = self.ture_conv(residual,self.conv,self.param[0],self.stride,radom=random.choice([1,2]))
        elif deploy == "2":
            residual = self.conv(residual)
        else:
            residual = self.ture_conv(residual,self.conv,self.param[0],self.stride,radom=random.choice([1,2]))

        residual = self.residual_1(residual)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        residual = (residual, deploy)
        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1, padding=0)

    def forward(self, x,deploy="0"):
        x = self.pre(x)
        x = (x,deploy)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        (x, deploy) = x
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)