import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import random

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv3x3(in_channels, out_channels, self.stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.param = self.parameters_k(4)
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                conv3x3(in_channels, self.expansion * out_channels, stride),
            nn.BatchNorm2d(self.expansion*out_channels))

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

    def forward(self, x_tuple):
        x, deploy = x_tuple
        if deploy == "1":
            out = self.relu(self.bn1(self.ture_conv(x,self.conv1,self.param[0],self.stride,radom=random.choice([1,2]))))
        elif deploy == "2":
            out = self.relu(self.bn1(self.conv1(x)))
        else:
            out = self.relu(self.bn1(self.ture_conv(x,self.conv1,self.param[0],self.stride,radom=random.choice([1,2]))))

        if deploy == "1":
            out =self.bn2(self.ture_conv(out,self.conv2,self.param[1],1,radom=random.choice([1,2])))
        elif deploy == "2":
            out =self.bn2(self.conv2(out))
        else:
            out = self.bn2(self.ture_conv(out,self.conv2,self.param[1],1,radom=random.choice([1,2])))
        out += self.shortcut(x)
        out = self.relu(out)
        x_tuple = (out, deploy)
        return x_tuple

class ResNet_2d(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_2d, self).__init__()
        self.deploys = "1"
        self.block = ResidualBlock
        """Constructs a ResNet-18 model."""
        # self.layers = [2, 2, 2, 2]
        """Constructs a ResNet-34 model."""
        self.layers = [3, 4, 6, 3]
        self.in_channels = 64
        self.conv = conv3x3(3,64 , 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.fc = nn.Linear(512*self.block.expansion, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,deploy="0"):
        out = self.relu(self.bn(self.conv(x)))
        x_tuple = (out, deploy)
        out = self.layer1(x_tuple)
        out = self.layer2(out)
        out = self.layer3(out)
        out_1 = self.layer4(out)
        out, deploy = out_1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



