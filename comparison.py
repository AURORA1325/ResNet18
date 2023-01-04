import pandas as pd
import numpy as np

def read_data(file,row):
    data = pd.read_csv(file)
    order1=[]
    if row=='train_loss':
        order1 = np.array(data.iloc[0])
    elif row=='train_erro':
        order1 = np.array(data.iloc[1])
    elif row=='test_loss':
        order1 = np.array(data.iloc[2])
    elif row=='test_erro':
        order1 = np.array(data.iloc[3])
    else:
        order1=['','','NONE']
    order1=order1[2:]
    return order1
#读取第一行
#print(data[0:1])




import matplotlib.pyplot as plt

re_train=read_data('./process/ResNet18.csv','train_erro')
se_train=read_data('./process/se.csv','train_erro')
cbam_train=read_data('./process/cbam.csv','train_erro')
eca_train=read_data('./process/eca.csv','train_erro')

y1 = re_train
y2 = se_train
y3=cbam_train
y4=eca_train
#x=len(y1)
x=range(1,len(y1)+1)
    #绘制图像
plt.plot(x,y1,x,y2,x,y3,x,y4)
#plt.plot(x_data,y2,'bo-')
    #设置图表属性
plt.legend(['resnet','SE','CBAM','ECA'])#图例
plt.xlabel('EPOCH')#设置x,y轴标记
plt.ylabel('train erro %')
plt.title('Train')#设置图像标题
