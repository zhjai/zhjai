import torch
import numpy as np
import random

#生成数据集
feature = 2
examples = 1000
w = torch.tensor([2, -3.4])
b = 4.2
features = torch.randn((examples, feature), dtype=torch.float32)
labels = features.matmul(w.T) + b
labels = labels.reshape(examples, 1)
labels += torch.tensor(np.random.normal(0, 0.01, (examples, 1)), dtype=torch.float32)

#读取数据
def data_iter(features, labels, batch_size):
    features_num = len(features)
    indices = list(range(features_num))
    random.shuffle(indices)
    for i in range(0, features_num, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size, features_num)])
        yield features.index_select(0, j), labels.index_select(0, j)

#初始化参数
w = torch.tensor(np.random.normal(0, 0.01, (feature, 1)), dtype=torch.float32)
b = torch.tensor([0], dtype=torch.float32)
w.requires_grad = True
b.requires_grad = True

#定义模型
def net(x, w, b):
    return torch.mm(x, w) + b

#定义损失函数
def loss(y_pre, y):
    y.reshape(y_pre.shape)
    return (y_pre - y)**2 / 2

#定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

#训练模型
lr = 0.03
num_epochs = 100
batch_size = 10
for each_epochs in range(num_epochs):
    for x, y in data_iter(features, labels, batch_size):
        l = loss(net(x, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    with torch.no_grad():
        train = loss(net(features, w, b), labels)
        print("epoch {}, loss {}".format(each_epochs + 1, train.mean().item()))
print(w)
print(b)