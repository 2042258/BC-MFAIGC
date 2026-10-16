import numpy as np
import gzip
import os
import platform
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# # 定义数据加载函数
# def load_datasets():
#     # 加载 CIFAR-10 数据集
#     transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),  # 增加随机旋转
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加颜色抖动
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
#     testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
#
#     # 将训练集划分为训练集和验证集
#     train_size = int(0.8 * len(trainset))
#     val_size = len(trainset) - train_size
#     trainset, valset = random_split(trainset, [train_size, val_size])
#
#     # 创建数据加载器
#     BATCH_SIZE = 64  # 你可以根据需要调整批量大小
#     trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#     valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
#     testloader = DataLoader(testset, batch_size=BATCH_SIZE)
#
#     return trainloader, valloader, testloader
#
# # 测试数据加载函数
# if __name__ == "__main__":
#     trainloader, valloader, testloader = load_datasets()
#     print("Trainloader size:", len(trainloader.dataset))
#     print("Valloader size:", len(valloader.dataset))
#     print("Testloader size:", len(testloader.dataset))

import torchvision
import torchvision.transforms as transforms

# Define transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batchsize'], shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args['batchsize'], shuffle=False, num_workers=2)