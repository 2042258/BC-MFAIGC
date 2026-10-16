"""Read CIFAR-10 with the original normalization and IID/non-IID ordering.

The relative data path and serialized array attributes are retained for existing
experiment callers. Dataset selection is a compatibility keyword; only CIFAR-10
is implemented, as in the original loader.
"""

import os
import pickle

import numpy as np


class Cifar10Dataset:
    def __init__(self, is_iid, *, dataset_name="cifar10"):
        self.name = dataset_name
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self._index_in_train_epoch = 0

        if self.name == "cifar10":
            self.load_cifar10(is_iid)
        else:
            pass

    def load_cifar10(self, is_iid):
        data_dir = "../data/cifar-10-batches-py"
        train_batches = [
            self.read_batch(os.path.join(data_dir, f"data_batch_{batch_id}"))
            for batch_id in range(1, 6)
        ]
        test_images_path = os.path.join(data_dir, "test_batch")
        train_images = np.concatenate([images for images, _ in train_batches], axis=0)
        train_labels = np.concatenate([labels for _, labels in train_batches], axis=0)
        test_images, test_labels = self.read_batch(test_images_path)

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

        if is_iid:
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

    def read_batch(self, file_path):
        with open(file_path, "rb") as handle:
            data_dict = pickle.load(handle, encoding="bytes")
            images = data_dict[b"data"]
            labels = np.array(data_dict[b"labels"])
            labels = np.eye(10)[labels]
        return images, labels
