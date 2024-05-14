import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from models import *
import numpy as np
import visdom

device = torch.device('cuda')

print('Using device:', device)
#print('GPU:', torch.cuda.get_device_name(0))

# net = ResNet_2d().to(device)
# net = MobileNetV2().to(device)
# net = build_efficientnet_lite().to(device)
#net = LeNet().to(device)
# net = VGG().to(device)
# net = GoogLeNet().to(device)
net = VGG_1d().to(device)
# from torchsummary import summary
# #
# summary(net, input_size=(3, 32, 32))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
# trainset = torchvision.datasets.CIFAR100(root='D:\\RepVGG-main\\RepVGG-main\\imagenet',split='train', transform=transform)
# trainloader = torch.utils.data.CIFAR100(trainset, batch_size=128, shuffle=True)
#
# testset = torchvision.datasets.ImageNet(root='D:\\RepVGG-main\\RepVGG-main\\imagenet', split='val', transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

torch.manual_seed("123")


num_epochs = 500
losses = []
batches = len(trainloader)
#python3 -m visdom.server

#  instantiate the window class （实例化监听窗口类）
viz = visdom.Visdom(server='http://localhost',port=8097)
#  create a window and initialize it （创建监听窗口）
viz.line([[0.]], [0], win='train', opts=dict(title='loss', legend=['loss']))
viz.line([[0.]], [0], win='acc', opts=dict(title='acc', legend=['acc']))

def test():
   total_correct = 0
   total_images = 0
   confusion_matrix = np.zeros([100,100], int)
   with torch.no_grad():
       for inputs, labels in testloader:
           inputs, labels = inputs.to(device), labels.to(device)
           # loaded_model = VGG_1d().to(device)
           # loaded_model.load_state_dict(torch.load('sample_model.pt'))
           # transform = transforms.Compose([
           #     # transforms.ToTensor(),  # 将张量转换为Tensor对象
           #     # transforms.RandomHorizontalFlip(),  # 随机水平翻转
           #     # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
           #     # transforms.ColorJitter(brightness=0.1),  # 随机颜色抖动
           #     transforms.RandomAffine(degrees=(0,45)),
           #     # transforms.RandomAffine(degrees=0,translate=(0.4, 0.4)),
           #     # transforms.RandomResizedCrop((32, 32), scale=(0.2,1.0))
           #     # transforms.RandomRotation(90),  # 随机旋转（-10度到+10度之间）
           # ])
           # inputs = transform(inputs)
           # outputs = loaded_model(inputs,"2")
           outputs = net(inputs,"1")
           _, predicted = torch.max(outputs.data, 1)
           total_images += labels.size(0)
           total_correct += (predicted == labels).sum().item()
           for i, l in enumerate(labels):
               confusion_matrix[l.item(), predicted[i].item()] += 1
       #if hasattr(loaded_model, 'conv2'):


   model_accuracy = total_correct / total_images * 100
   print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))
   return model_accuracy

def test_two():
   total_correct = 0
   total_images = 0
   confusion_matrix = np.zeros([100,100], int)
   with torch.no_grad():
       for inputs, labels in testloader:
           inputs, labels = inputs.to(device), labels.to(device)
           # loaded_model = VGG_2d().to(device)
           # loaded_model.load_state_dict(torch.load('sample_model.pt'))
           # outputs = loaded_model(inputs)
           outputs = net(inputs,"2")
           _, predicted = torch.max(outputs.data, 1)
           total_images += labels.size(0)
           total_correct += (predicted == labels).sum().item()
           for i, l in enumerate(labels):
               confusion_matrix[l.item(), predicted[i].item()] += 1
       #if hasattr(loaded_model, 'conv2'):


   model_accuracy = total_correct / total_images * 100
   print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))
   return model_accuracy

#python -m visdom.server
def train():
   print("TRAINING")
   acc_best= 0
   for epoch in range(num_epochs):
      progress = tqdm(enumerate(trainloader), desc="Loss: ", total=batches)
      total_loss = 0
      for i, (inputs, labels) in progress:
         inputs, labels = inputs.to(device), labels.to(device)
         optimizer.zero_grad()
         output= net(inputs,"1")
         loss = criterion(output, labels)
         loss.backward()
         optimizer.step()
         current_loss = loss.item()
         total_loss += current_loss
         progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
      losses.append(total_loss/batches)
      # 显示图形
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/batches}")
      acc = test()
      acc_1 = test_two()
      acc_best = max(acc_best,acc_1)
      print("acc_best= ",acc_best)
      torch.save(net.state_dict(), 'sample_model.pt')
      #  update window image （传递数据到监听窗口进行画图）
      viz.line([[total_loss/batches]], [epoch], win='train', update='append')
      viz.line([[acc]], [epoch], win='acc', update='append')




   #torch.save(net, './save')



#net = torch.load("save")
# test()
train()
