import numpy as np
import gzip
import os
import pickle
import torch


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
        data_dir = '../data/cifar-10-batches-py'
        train_images_path = os.path.join(data_dir, 'data_batch_1')
        test_images_path = os.path.join(data_dir, 'test_batch')

        train_images, train_labels = self.load_cifar10_batch(train_images_path)
        test_images, test_labels = self.load_cifar10_batch(test_images_path)



        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        train_images = train_images.reshape(train_images.shape[0], 3, 32, 32)
        test_images = test_images.reshape(test_images.shape[0], 3, 32, 32)

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

    def load_cifar10_batch(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            images = data_dict[b'data']
            labels = np.array(data_dict[b'labels'])
            labels = np.eye(10)[labels]  # One-hot encoding
        return images, labels


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
	mnistDataSet = GetDataSet('cifar10', True)  # test NON-IID
	if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
			type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
		print('the type of data is numpy ndarray')
	else:
		print('the type of data is not numpy ndarray')
	print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
	print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
	print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
	# 创建 CIFAR-10 数据集实例
	data_loader = GetDataSet('cifar10', isIID=True)  # 根据需要设置 isIID 参数

	# 检查数据集的形状
	print(f"Train data shape: {data_loader.train_data.shape}")
	print(f"Train label shape: {data_loader.train_label.shape}")
	print(f"Test data shape: {data_loader.test_data.shape}")
	print(f"Test label shape: {data_loader.test_label.shape}")


# add Gussian Noise to dataset
# URL_REQUIRED_2
class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean

	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


