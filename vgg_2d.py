import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
n =10
pool_label = [1,4]
class VGG_2d(nn.Module):
	def __init__(self,in_channels=3, num_classes=100):
		super(VGG_2d, self).__init__()
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

	def forward(self, x,deploy = "0"):
		x = self.conv[:3](x)
		if deploy == "1":
			x = self.conv[4:6](self.ture_conv(x, self.conv[3:6], self.param[1],random.choice([1])))
		elif deploy == "2":
			x = self.conv[3:6](x)
		else:
			x = self.conv[4:6](self.ture_conv(x, self.conv[3:6], self.param[1],random.choice([1])))

		# x = self.ture_conv(x, self.conv[3:6], self.param[1])
		x = self.max_pool(x)

		x = self.conv[7:10](x)
		if deploy == "1":
			x = self.conv[11:13](self.ture_conv(x, self.conv[10:13], self.param[3],random.choice([1])))
		elif deploy == "2":
			x = self.conv[10:13](x)
		else:
			x = self.conv[11:13](self.ture_conv(x, self.conv[10:13], self.param[3],random.choice([1])))
		# x = self.ture_conv(x, self.conv[10:13], self.param[3])
		x = self.max_pool(x)

		x = self.conv[14:17](x)
		if deploy == "1":
			x = self.conv[18:20](self.ture_conv(x, self.conv[17:20], self.param[5],random.choice([1])))
		elif deploy == "2":
			x = self.conv[17:20](x)
		else:
			x = self.conv[18:20](self.ture_conv(x, self.conv[17:20], self.param[5],random.choice([1])))
		# # x = self.ture_conv(x, self.conv[17:20], self.param[5])
		if deploy == "1":
			x =self.conv[21:23](self.ture_conv(x, self.conv[20:23], self.param[7],random.choice([1])))
		elif deploy == "2":
			x = self.conv[20:23](x)
		else:
			x = self.conv[21:23](self.ture_conv(x, self.conv[20:23], self.param[7],random.choice([1])))
		# # x = self.ture_conv(x, self.conv[20:23], self.param[7])
		x = self.max_pool(x)

		x = self.conv[24:27](x)
		if deploy == "1":
			x = self.conv[28:30](self.ture_conv(x, self.conv[27:30], self.param[9],random.choice([1])))
		elif deploy == "2":
			x = self.conv[27:30](x)
		else:
			x = self.conv[28:30](self.ture_conv(x, self.conv[27:30], self.param[9],random.choice([1])))
		if deploy == "1":
			x = self.conv[31:33](self.ture_conv(x, self.conv[30:33], self.param[11],random.choice([1])))
		elif deploy == "2":
			x = self.conv[30:33](x)
		else:
			x = self.conv[31:33](self.ture_conv(x, self.conv[30:33], self.param[11],random.choice([1])))
		x = self.max_pool(x)

		if deploy == "1":
			x = self.conv[35:37](self.ture_conv(x, self.conv[34:37], self.param[13],random.choice([1])))
		elif deploy == "2":
			x = self.conv[34:37](x)
		else:
			x = self.conv[35:37](self.ture_conv(x, self.conv[34:37], self.param[13],random.choice([1])))
		# x = self.ture_conv(x, self.conv[34:37], self.param[13])
		if deploy == "1":
			x = self.conv[38:40](self.ture_conv(x, self.conv[37:40], self.param[15],random.choice([1])))
		elif deploy == "2":
			x = self.conv[37:40](x)
		else:
			x = self.conv[38:40](self.ture_conv(x, self.conv[37:40], self.param[15],random.choice([1])))
		# x = self.ture_conv(x, self.conv[37:40], self.param[15])
		if deploy == "1":
			x = self.conv[41:43](self.ture_conv(x, self.conv[40:43], self.param[17],random.choice([1])))
		elif deploy == "2":
			x = self.conv[40:43](x)
		else:
			x = self.conv[41:43](self.ture_conv(x, self.conv[40:43], self.param[17],random.choice([1])))
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
							dilation=1,
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

	def _qu_weight(self,conv,radom):
		weights = conv[0].weight
		tensor_var = radom

		if tensor_var == 1:
			transform = transforms.Compose([
				# transforms.ToTensor(),  # 将张量转换为Tensor对象
				#transforms.RandomHorizontalFlip(),  # 随机水平翻转
				#transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
				#transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
				transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
				# transforms.RandomResizedCrop((3, 3), scale=(0.8, 1.0))1
				#transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
			])
		# elif tensor_var == 2:
		# 	transform = transforms.Compose([
		# 		# transforms.ToTensor(),  # 将张量转换为Tensor对象
		# 		#transforms.RandomHorizontalFlip(),  # 随机水平翻转
		# 		#transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
		# 		#transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
		# 		transforms.RandomAffine(degrees=0, scale=(3,3.5)),
		# 		#transforms.RandomResizedCrop((3, 3), scale=(0.8, 1.0))
		# 		#transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
		# 	])
		# else:
		# 	transform = transforms.Compose([
		# 		# transforms.ToTensor(),  # 将张量转换为Tensor对象
		# 		#transforms.RandomHorizontalFlip(),  # 随机水平翻转
		# 		#transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
		# 		#transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
		# 		# transforms.RandomAffine(degrees=0, scale=(3,3.5)),
		# 		transforms.RandomResizedCrop((3, 3), scale=(0.8, 1.0))
		# 		#transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
		# 	])

		tra_weight = transform(weights)
		return tra_weight

	def _qu_bias(self,conv):
		biases = conv[0].bias
		return biases

	def ture_conv(self,input,conv,param,radom):
		x = F.conv2d(input, param * self._qu_weight(conv,radom), bias=self._qu_bias(conv), stride=1, padding=1) # 第一次卷积操作
		return x





