"""Legacy dataset imports; reading and transforms live in bcmfaigc.data."""

from bcmfaigc.data.compat import DatasetLoad
from bcmfaigc.data.transforms import AddGaussianNoise
from bcmfaigc.data.mnist import (
    read_uint32 as _read32,
    read_mnist_images as extract_images,
    one_hot_encode as dense_to_one_hot,
    read_mnist_labels as extract_labels,
)

__all__ = [
    "DatasetLoad",
    "AddGaussianNoise",
    "extract_images",
    "dense_to_one_hot",
    "extract_labels",
]
