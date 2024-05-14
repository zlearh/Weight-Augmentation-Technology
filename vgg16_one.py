import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

n =10
pool_label = [1,4]
class VGG_3d(nn.Module):
	def __init__(self,in_channels=3, num_classes=10):
		super(VGG_3d, self).__init__()
		self.in_channels = in_channels
		self.conv = self._make_layer(vgg16)
		self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		self.param = self.parameters_k(n)
		self.fc = nn.Sequential(
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(512, num_classes)
			)
		self.ReLU = nn.ReLU()
		self.BN = nn.BatchNorm2d(512)

	def forward(self, x):
		x = self.conv[:3](x)
		x = self.conv[3:6](x)
		x = self.ture_conv(x, self.conv[3:6], self.param[0])
		# x = self.ture_conv(x, self.conv[3:6], self.param[1])
		x = self.max_pool(x)

		x = self.conv[7:10](x)
		x = self.conv[10:13](x)
		x = self.ture_conv(x, self.conv[10:13], self.param[2])
		# x = self.ture_conv(x, self.conv[10:13], self.param[3])
		x = self.max_pool(x)


		x = self.conv[14:17](x)
		x = self.conv[17:20](x)
		x = self.ture_conv(x, self.conv[17:20], self.param[4])
		# x = self.ture_conv(x, self.conv[17:20], self.param[5])
		x = self.conv[20:23](x)
		x = self.ture_conv(x, self.conv[20:23], self.param[6])
		# x = self.ture_conv(x, self.conv[20:23], self.param[7])
		x = self.max_pool(x)

		x = self.conv[24:27](x)
		x = self.conv[27:30](x)
		x = self.ture_conv(x, self.conv[27:30], self.param[8])
		# x = self.ture_conv(x, self.conv[27:30], self.param[9])
		x = self.conv[30:33](x)
		x = self.ture_conv(x, self.conv[30:33], self.param[10])
		# x = self.ture_conv(x, self.conv[30:33], self.param[11])
		x = self.max_pool(x)


		x = self.conv[34:37](x)
		x = self.ture_conv(x, self.conv[34:37], self.param[12])
		# x = self.ture_conv(x, self.conv[34:37], self.param[13])
		x = self.conv[37:40](x)
		x = self.ture_conv(x, self.conv[37:40], self.param[14])
		# x = self.ture_conv(x, self.conv[37:40], self.param[15])
		x = self.conv[40:43](x)
		x = self.ture_conv(x, self.conv[40:43], self.param[16])
		#x = self.ture_conv(x, self.conv[40:43], self.param[17])
		x = self.max_pool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)
		return x

	def _make_layer(self, architecture):
		layers = []
		in_channels = self.in_channels

		for x in architecture:
			if type(x) == int:
					out_channels = x
					layers += [
						nn.Conv2d(
							in_channels=in_channels,
							out_channels=out_channels,
							kernel_size=(3, 3),
							stride=(1, 1),
							padding=(1, 1),
						),
						nn.BatchNorm2d(x),
						nn.ReLU(),
					]
					in_channels = x
			elif x == "M":
					layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

		return nn.Sequential(*layers)


	def _Sum_start(self,m):
		num = sum(range(1, m))
		return num
	def parameters_k(self,m):
		num = self._Sum_start(m)
		parameter = torch.nn.Parameter(torch.rand(num))
		return parameter

	def _qu_weight(self,conv):
		weights = conv[0].weight
		import torchvision.transforms as transforms

		transform = transforms.Compose([
			# transforms.ToTensor(),  # 将张量转换为Tensor对象
			#transforms.RandomHorizontalFlip(),  # 随机水平翻转
			#transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
			#transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
			transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
			#transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
		])
		weights = transform(weights)

		return weights

	def _qu_bias(self,conv):
		biases = conv[0].bias
		return biases

	def ture_conv(self,input,conv,param):
		x = self.ReLU(F.conv2d(input, param * self._qu_weight(conv), bias=self._qu_bias(conv), stride=1, padding=1)) # 第一次卷积操作
		return x







