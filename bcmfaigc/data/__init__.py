"""Dataset readers; partitioning remains in the existing network until migrated."""

from .cifar10 import Cifar10Dataset

__all__ = ["Cifar10Dataset"]
