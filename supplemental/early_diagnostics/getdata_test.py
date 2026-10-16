import sys
import numpy as np
import gzip
import os
import platform
import pickle

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'cifar10':
            self.cifar10DataSetConstruct(isIID)
        else:
            pass

    def cifar10DataSetConstruct(self, isIID):
        # 加载 CIFAR-10 数据集
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 增加随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 增加颜色抖动
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

        # 将训练集划分为训练集和验证集
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])

        # 创建数据加载器
        BATCH_SIZE = 64  # 你可以根据需要调整批量大小
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)

        # 将数据加载器转换为 numpy 数组
        self.train_data = np.concatenate([x.numpy() for x, _ in trainloader], axis=0)
        self.train_label = np.concatenate([y.numpy() for _, y in trainloader], axis=0)
        self.train_data_size = len(self.train_data)

        self.test_data = np.concatenate([x.numpy() for x, _ in testloader], axis=0)
        self.test_label = np.concatenate([y.numpy() for _, y in testloader], axis=0)
        self.test_data_size = len(self.test_data)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]
        else:
            labels = np.argmax(self.train_label, axis=1)
            order = np.argsort(labels)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__ == "__main__":
    'test data set'
    cifar10DataSet = GetDataSet('cifar10', True)  # test NON-IID
    if type(cifar10DataSet.train_data) is np.ndarray and type(cifar10DataSet.test_data) is np.ndarray and \
            type(cifar10DataSet.train_label) is np.ndarray and type(cifar10DataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(cifar10DataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(cifar10DataSet.test_data.shape))
    print(cifar10DataSet.train_label[0:100], cifar10DataSet.train_label[11000:11100])


# add Gaussian Noise to dataset
# URL_REQUIRED_2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)