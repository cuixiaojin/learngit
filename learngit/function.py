import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models


'''
预测训练集的准确率
'''
def accuracy_test(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = Variable(images), Variable(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('the accuracy is {:.4f}'.format(correct / total))


# 创建深度学习功能
'''
将训练集数据进行深度学习
'''


def deep_learning(model, trainloader, epochs,  criterion, optimizer):
    epochs = epochs

    steps = 0

    for e in range(epochs):
        running_loss=0
        correct = 0
        total = 0

        for ii, (images, labels) in enumerate(trainloader):
            steps += 1
            images, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()

            # forward and backward
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if steps % 40 == 0 or (steps) == len(trainloader):
                # 测试准确性
                print('EPOCHS : {}/{}'.format(e + 1, epochs),
                      'Loss : {:.4f}'.format(running_loss/steps))
                print('the accuracy is {:.4f}'.format(100 * correct / total))




'''
检查加载函数
'''


# def checkpoint loading function
def network_loading(model, ckp_path):
    state_dict = torch.load(ckp_path)
    model.load_state_dict(state_dict,strict=False)
    print('The Network is Loaded')


def network_saving():
    torch.save(fmodel.state_dict(), 't.pth')

    print('The Network is Saved')

