"""Trusted network snapshots with explicit historical class-name resolution.

Pickle remains unsafe for untrusted input. This resolver only preserves old
experiment import paths; it is not a security filter or a snapshot migration.
"""

import pickle

from bcmfaigc.models.ddpm import DDPM, UNet, TimeEmbedding
from bcmfaigc.data.compat import DatasetLoad
from bcmfaigc.data.transforms import AddGaussianNoise
from bcmfaigc.blockchain import Block, Blockchain
from .compat import (
    ClassicDevice,
    ClassicNetwork,
    ExtendedDevice,
    ExtendedNetwork,
    KrumDevice,
    KrumNetwork,
)


class LegacyNetworkUnpickler(pickle.Unpickler):
    def __init__(self, file, variant):
        super().__init__(file)
        device, network = {
            "classic": (ClassicDevice, ClassicNetwork),
            "extended": (ExtendedDevice, ExtendedNetwork),
            "krum": (KrumDevice, KrumNetwork),
        }[variant.name]
        self.aliases = {
            ("models.Models", "DDPM"): DDPM,
            ("models.Models", "UNet"): UNet,
            ("models.Models", "TimeEmbedding"): TimeEmbedding,
            ("Device", "Device"): device,
            ("Device", "DevicesInNetwork"): network,
            ("DatasetLoad", "DatasetLoad"): DatasetLoad,
            ("DatasetLoad", "AddGaussianNoise"): AddGaussianNoise,
            ("Block", "Block"): Block,
            ("Blockchain", "Blockchain"): Blockchain,
        }

    def find_class(self, module, name):
        if (module, name) in self.aliases:
            return self.aliases[(module, name)]
        return super().find_class(module, name)


def load_network_snapshot(path, variant):
    with open(path, "rb") as file:
        return LegacyNetworkUnpickler(file, variant).load()
