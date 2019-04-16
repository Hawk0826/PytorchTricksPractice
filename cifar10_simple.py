# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import time

########################################################################
# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 4
LR = 0.001
MOMENTUM = 0.9

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # (x-mean)/std = [-0.5, 0.5]/0.5

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,  # !!!!!!!!!调整策略
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, kernel=5):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # x.size = [4, 3, 32, 32] = [BATCH_SIZE, inC, inH, inW]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(BATCH_SIZE, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def write_in_log(log):
    with open("log.txt", 'a') as f:
        f.write(log + '\n')
        print(log)
    pass


########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)  # !!!!!!!!!调整策略

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimizer.

for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    tic = time.time()

    #   enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    #   同时列出数据和数据下标，一般用在 for 循环当中。
    for batch, data in enumerate(trainloader, 0):  # trainloader已被分为50000/4=12500份
        # get the inputs
        inputs, labels_train = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels_train)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if batch % 2500 == 2499:    # print every 2500 mini-batches(10000 samples)
            toc = time.time()
            dif = toc - tic
            loss_log = 'Epoch:{:2d} ' \
                       '| {:5d}/{} ' \
                       '| loss: {:.3f} | {:.2f}s'.format(epoch + 1,
                                               (batch + 1) * BATCH_SIZE,
                                               len(trainloader.dataset),
                                               running_loss / 2500,
                                               dif)
            write_in_log(loss_log)
            running_loss = 0.0
            tic = time.time()

    correct = 0
    total = 0
    with torch.no_grad():  # 不反传
        for data in testloader:
            images, labels_test = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # type_tensor.data仍然是type_tensor,等价
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()
            # correct += (predicted == labels_test)

    accuracy = 'Epoch:{:2d} | Accuracy of the network on ' \
               'the 10000 test images: {:.2f}%\n'.format(epoch + 1, (100 * correct / total))
    write_in_log(accuracy)

print('Finished Training')