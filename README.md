# ResNet18
在ResNet18中嵌入视觉注意力机制
# 实验环境

实验环境配置

| **相关设置** |                                                |
| ------------ | ---------------------------------------------- |
| 实验数据集   | CIFAR-100                                      |
| 实验平台     | Pytorch 1.12.0                                 |
| 神经网络     | ResNet18                                       |
| 损失函数     | CrossEntropyLoss                               |
| 优化器       | Adam                                           |
| 学习率       | 0.001，每迭代20次更新                          |
| BATCH_SIZE   | 50                                             |
| Epochs       | 20，50                                         |
| Python 版本  | Python 3.8                                     |
| 代码编辑器   | Jupyter lab                                    |
| 操作系统     | Win10                                          |
| CPU          | IntelI CoreI i7-10750H CPU @ 2.60GHz  2.59 GHz |
| CUDA版本     | 4.8.3                                          |



需要的包：

- ` torch`，`torchvision`，
- `matplotlib`，
- `numpy`，`pandas`，`time`
- `thop`

安装方式：

- 依赖包安装方式为：打开Anaconda Prompt，在命令行输入

```
pip install + 包名 
```

- Pytorch安装方式

在Anaconda中，安装pytorch 环境，在Anaconda Promp中，输入以下指令

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```



# 数据集下载

本次实验所使用的数据集为CIFAR100，CIFAR100数据集有100个类。每个类有600张大小为32 × 32 32的彩色图像，其中500张作为训练集，100张作为测试集。对于每一张图像，它有fine_labels和coarse_labels两个标签，分别代表图像的细粒度和粗粒度标签。

数据集下载方式：

方式一：下载链接http://www.cs.toronto.edu/~kriz/cifar.html

方式二：运行下列代码时，将download 置为 ” True ”，即可自动下载。

```python
  train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                             train=True, 
                                             transform=transform,
                                             download=True)

    test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                            train=False, 
                                            transform=transforms.ToTensor())
```



# 运行方式

模型训练：

模型文件一共包含了ResNet18.py，SE-ResNet18.py，ECA-ResNet18.py，CBAM-ResNet18.py，my_attention.py 五个文件，分别代表了ResNet18，嵌入SE模型。嵌入ECA模型，嵌入CBAM模型，以及修改后的SE模型的文件。每个文件均可独立运行，在安装完对应的包之后，分别运行四个文件，即可得到四个模型的结果，并将结果存入 `./process` 目录下的csv文件中。

ResNet18中，具有param函数，可以用来计算模型的计算量和参数量。

模型对比：

对比文件为comparison.py，运行该文件，通过文件中的 `read_data()`函数读取文件中的数据，即可对各个模型的结果数据做对比



# 实验结果

三种视觉注意力机制对比

|                | **ResNet18** | **SE-ResNet** | **ECA-ResNet** | **CBAM-ResNet** |
| -------------- | ------------ | ------------- | -------------- | --------------- |
| 嵌入位置       | ——           | 残差结构      | 残差结构       | 卷积层间        |
| 激活函数       | ReLU         | Sigmoid       | Sigmoid        | Sigmoid         |
| 迭代次数       | 50           | 50            | 50             | 50              |
| 50次迭代准确率 | 66.49%       | 67.12%        | 67.08%         | 65.43%          |
| 最大准确率     | 67.49%       | 68.03%        | 67.57%         | 66.08%          |
| 参数量         | 12.596 M     | 12.639M       | 12.596 M       | 12.613M         |
| FLOAPs         | 1216.51M     | 1217.09M      | 1217.02M       | 1216.93M        |



CBAM 消融实验

| **网络**    | **激活函数**   | **epoch** | **准确率** |
| ----------- | -------------- | --------- | ---------- |
| ResNet18    | Relu           | 50        | 66.49%     |
| CBAM: CA    | Sigmod         | 50        | 67.51%     |
| CBAM: SA    | Sigmod         | 50        | 66.44%     |
| CBAM: SA    | ReLU           | 50        | 63.99%     |
| CBAM-ResNet | Sigmod  Sigmod | 50        | 65.43%     |

SE模型修改后得到Conv-SE，测试结果对比如下表

|                | **SE-ResNet** | **Conv-SE** |
| -------------- | ------------- | ----------- |
| 嵌入位置       | 残差结构      | 残差结构    |
| 激活函数       | Sigmoid       | Sigmoid     |
| 迭代次数       | 50            | 50          |
| 50次迭代准确率 | 67.12%        | 67.27%      |
| 最大准确率     | 68.03%        | 67.82%      |
| 参数量         | 12.639M       | 12.705 M    |
| FLOAPs         | 1217.09M      | 1217.28M    |
