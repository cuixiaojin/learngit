# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from cui import deep_learning
from collections import OrderedDict
from cui import accuracy_test
from cui import network_loading
from cui import network_saving



def main():

    # 数据库输入
    train_dir = r'C:\Users\Administrator\Desktop\train'
    test_dir = r'C:\Users\Administrator\Desktop\test'

    # TODO: 定义培训集、验证集和测试集的转换
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # TODO: 使用ImageFolder加载数据集
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # TODO: 使用图像数据集和训练表，定义数据加载器
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1)



    caa = models.vgg.vgg16(pretrained=True)



    # TODO: 构建并培训您的网络
    fmodel = caa
    # 冻结要素中的参数
    for param in fmodel.parameters():
        param.require_grad = False

    # 创建新的分类器并替换旧的分类器

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(1000, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    fmodel.classifier = classifier
    # 是否加载检查点


    network_loading(fmodel, 't.pth')
    # 定制标准和优化器

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(fmodel.classifier.parameters(), lr=0.001)

    deep_learning(fmodel, trainloader, 5, criterion, optimizer)
    accuracy_test(fmodel,testloader)



    network_saving()


if __name__ == "__main__":
    main()