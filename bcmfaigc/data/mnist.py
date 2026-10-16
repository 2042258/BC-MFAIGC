"""Retained MNIST IDX readers, separate from CIFAR-10 loading."""

import gzip

import numpy as np


def read_uint32(bytestream):
    dtype = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dtype)[0]


def read_mnist_images(filename):
    """Read the original [index, y, x, depth] uint8 image representation."""
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        magic = read_uint32(bytestream)
        if magic != 2051:
            raise ValueError(
                "Invalid magic number %d in MNIST image file: %s" % (magic, filename)
            )
        num_images = read_uint32(bytestream)
        rows = read_uint32(bytestream)
        cols = read_uint32(bytestream)
        buffer = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def one_hot_encode(labels_dense, num_classes=10):
    """Convert scalar class labels to one-hot vectors without changing dtype."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_mnist_labels(filename):
    """Read one-hot labels from a gzip-compressed MNIST IDX file."""
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        magic = read_uint32(bytestream)
        if magic != 2049:
            raise ValueError(
                "Invalid magic number %d in MNIST label file: %s" % (magic, filename)
            )
        num_items = read_uint32(bytestream)
        buffer = bytestream.read(num_items)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return one_hot_encode(labels)
