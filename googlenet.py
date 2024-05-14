import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
class Inception(nn.Module):
    def __init__(self, in_planes, n1, n3_reduce, n3, n5_reduce, n5, pool_proj):
        super(Inception, self).__init__()
        self.param = self.parameters_k(8)
        self.stride = 1
        self.block1 = nn.Sequential(
						nn.Conv2d(in_planes, n1, kernel_size=1),
						nn.BatchNorm2d(n1),
						nn.ReLU(True),
				 )

        self.block2 = nn.Sequential(
						nn.Conv2d(in_planes, n3_reduce, kernel_size=1),
						nn.BatchNorm2d(n3_reduce),
						nn.ReLU(True),

						nn.Conv2d(n3_reduce, n3, kernel_size=3, padding=1),
						nn.BatchNorm2d(n3),
						nn.ReLU(True),
				 )

        self.block3 = nn.Sequential(
						nn.Conv2d(in_planes, n5_reduce, kernel_size=1),
						nn.BatchNorm2d(n5_reduce),
						nn.ReLU(True),

						nn.Conv2d(n5_reduce, n5, kernel_size=5, padding=2),
						nn.BatchNorm2d(n5),
						nn.ReLU(True),
				 )

        self.block4 = nn.Sequential(
						nn.MaxPool2d(3, stride=1, padding=1),
						nn.Conv2d(in_planes, pool_proj, kernel_size=1),
						nn.BatchNorm2d(pool_proj),
						nn.ReLU(True),
				 )
    def parameters_k(self, m):
        parameter = torch.nn.Parameter(torch.rand(m))
        return parameter
    def _qu_bias(self, conv):
        biases = conv.bias
        return biases

    def _qu_weight(self, conv, radom):
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
        else:
            transform = transforms.Compose([
		# transforms.ToTensor(),  # 将张量转换为Tensor对象
		# transforms.RandomHorizontalFlip(),  # 随机水平翻转
		# transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
		# transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
		# transforms.RandomAffine(degrees=0, scale=(3,3.5)),
		transforms.RandomResizedCrop((weights.shape[-1], weights.shape[-1]), scale=(0.6, 0.8))
		# transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
		])
        weights = transform(weights)
        return weights

    def ture_conv(self, input, conv, param, stride, radom):
        padd = (conv.weight.shape[-1] - 1) // 2
        x = F.conv2d(input, param * self._qu_weight(conv, radom), bias=self._qu_bias(conv), stride=stride,padding=padd)  # 第一次卷积操作
        return x

    def forward(self, x, deploy):

        y1 = self.block1(x)

        if deploy == "1":
            y2 = self.block2[:3](x)
            y2 = self.block2[4:6](
				self.ture_conv(y2, self.block2[3], self.param[0], self.stride, radom=random.choice([1,2])))
        else:
            y2 = self.block2(x)

        if deploy == "1":
            y3 = self.block3[:3](x)
            y3 = self.block3[4:6](
                    self.ture_conv(y3, self.block3[3], self.param[1], self.stride, radom=random.choice([1, 2])))
        else:
            y3 = self.block3(x)

        y4 = self.block4(x)
        return torch.cat([y1, y2, y3, y4], 1)



class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.pre_layers = nn.Sequential(
						nn.Conv2d(3, 192, kernel_size=3, padding=1),
						nn.BatchNorm2d(192),
						nn.ReLU(True),
				 )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
		
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 100)


    def forward(self, x, deploy="0"):
        out = self.pre_layers(x)
        out = self.a3(out,deploy)
        out = self.b3(out,deploy)
        out = self.maxpool(out)
        out = self.a4(out,deploy)
        out = self.b4(out,deploy)
        out = self.c4(out,deploy)
        out = self.d4(out,deploy)
        out = self.e4(out,deploy)
        out = self.maxpool(out)
        out = self.a5(out,deploy)
        out = self.b5(out,deploy)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out