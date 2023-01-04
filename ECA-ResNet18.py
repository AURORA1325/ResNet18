import torch.nn as nn
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.nn.parameter import Parameter

#ECA模块

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        
        #self.sigmoid = nn.ReLU()
        #self.sigmoid = nn.Tanh()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def write_file(res_train_loss,res_train_erro,res_test_loss,res_test_erro,file_name):
    train_loss=['train_loss']
    train_erro=['train_erro']
    test_loss=['test_loss']
    test_erro=['test_erro']
    
    for i in res_train_loss:
        train_loss.append(i)

    for i in res_train_erro:
        train_erro.append(i)
    
    for i in res_test_loss:
        test_loss.append(i)
    
    for i in res_test_erro:
        test_erro.append(i)
    
    res_list=[train_loss,train_erro,test_loss,test_erro]
    #column=range(1,51) # 列表对应每列的列名
    
    test=pd.DataFrame(data=res_list)
    test.to_csv('./process/'+file_name+'.csv')

    
# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 50 #50轮
batch_size = 50 #50步长
learning_rate = 0.01 #学习率0.01



# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Resnet 的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.eca = eca_layer(out_channels, 3)#3是Kernelsize

        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.eca(out)
        
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, inter_layer=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = conv3x3(3,64)

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        #self.eca = eca_layer(self.in_channels, 3)#3是Kernelsize

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[2], 2)#4

        #self.eca = eca_layer(self.in_channels, 3)#3是Kernelsize
        
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        #out = self.eca(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        #out = self.eca(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return torch.div(correct, total)

# 更新学习率
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(model,device,train_loader,optimizer,epoch,criterion,curr_lr):
    model.train()
    train_loss=0
    train_acc=0
    acc1=0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += get_acc(outputs,labels).item()
        
        if (i+1) % 200 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    train_loss = train_loss / len(train_loader)
    train_acc = train_acc * 100 / len(train_loader)             
    print("Train Acc {:.4f}%, Train Loss {:.4f}".format(train_acc/(epoch+1),train_loss/(epoch+1)))

    # 延迟学习率
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
    
    return train_loss,100-train_acc


def test(model,devive,test_loader,criterion):
# 测试网络模型
    erro=[]
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
     
        test_loss=0
        test_acc=0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
           
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs,labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_loss += loss.item()
            test_acc += get_acc(outputs,labels).item()

        erro.append(100 *(1- correct / total))
        test_loss = test_loss / len(test_loader)
        test_acc = test_acc * 100 / len(test_loader)

        print('Accuracy of the model on the test images: {} %, test_loss: {}'.format(100 * correct / total,test_loss))
        print('test_acc of the model on the test images: {} %'.format(test_acc))

        return test_loss,100-test_acc
    

def draw(train_data,test_data,x_data,x_lable,y_lable,title):
     #设置图像属性
    y1 = train_data
    y2 = test_data
    #绘制图像
    plt.plot(x_data,y1,'r--')
    plt.plot(x_data,y2,'bo-')
    #设置图表属性
    plt.legend(['Train','Test'])#图例
    plt.xlabel(x_lable)#设置x,y轴标记
    plt.ylabel(y_lable)
    plt.title(title)#设置图像标题

import time #统计训练时间

if __name__ == "__main__":
    # 图像预处理
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

# CIFAR-100 数据集下载
    train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True, 
                                             transform=transform,
                                             download=False)

    test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False, 
                                            transform=transforms.ToTensor())

# 数据载入
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
    
    
    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # 训练数据集
    total_step = len(train_loader)
    curr_lr = learning_rate
    
    #记录数据
    train_loss=[]
    train_erro=[]

    test_loss=[]
    test_erro=[]

    starttime = time.time()
    time.sleep(0.1) #延时0.1s

    for epoch in range(num_epochs):
        loss1,erro1=train(model,device,train_loader,optimizer,epoch,criterion,curr_lr)
        loss2,erro2=test(model,device,test_loader,criterion)

        train_loss.append(loss1)
        train_erro.append(erro1)
        test_loss.append(loss2)
        test_erro.append(erro2)
    
    endtime = time.time()
    dtime = endtime - starttime
    print("Training cost time %.8s s" % dtime)

    x=range(1,num_epochs+1)

    draw(train_erro,test_erro,x,"epoch","erro_rate %","ResNet18")
    write_file(train_loss,train_erro,test_loss,test_erro,'Eca')