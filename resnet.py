import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride, deploy, downsample=None):
        super(ResidualBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.param = self.parameters_k(4)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.bn5 = nn.BatchNorm2d(out_channels)


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

    def _qu_weight(self, conv,tra=0.3):
        weights = conv.weight

        transform = transforms.Compose([
            # transforms.ToTensor(),  # 将张量转换为Tensor对象
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            # transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
            transforms.RandomAffine(degrees=0, translate=(tra, tra)),
            # transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
        ])
        weights = transform(weights)

        return weights

    def ture_conv(self, input, conv, param):
        x = F.conv2d(input, param * self._qu_weight(conv,0.3), bias=self._qu_bias(conv), stride=1,padding=1)  # 第一次卷积操作
        return x

    def ture_conv_result(self, input, conv, param):
        weights = conv.weight
        transform = transforms.Compose([
            # transforms.ToTensor(),  # 将张量转换为Tensor对象
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            # transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
            transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
            # transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
        ])
        weights_1 =  transform(weights)
        weights_all = weights

        x = F.conv2d(input, weights_all, bias=self._qu_bias(conv), stride=1,padding=1)  # 第一次卷积操作
        return x

    def forward(self, x):
        if self.deploy == False:
            out = self.relu(self.bn1(self.conv1(x)))
            if(self.in_channels == self.out_channels):
                out = self.relu(self.bn3(self.ture_conv(out,self.conv1, self.param[0])))
                out = self.relu(self.bn4(self.ture_conv(out,self.conv1, self.param[1])))
        else:
            if(self.in_channels == self.out_channels):
                out = self.relu(self.bn1(self.conv1(x)))
                #out = self.relu(self.bn3(self.ture_conv(x, self.conv1, self.param[0])))
            else:
                out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10,deploy= False):
        super(ResNet, self).__init__()
        self.deploy = deploy
        self.block = block
        self.layers = layers
        self.in_channels = 64
        self.conv = conv3x3(3, 64, 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,self.deploy))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet(deploy=False):
    return ResNet(ResidualBlock, [2, 2, 2, 2],deploy=deploy)


