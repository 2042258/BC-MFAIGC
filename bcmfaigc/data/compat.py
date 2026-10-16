"""Legacy dataset constructor/method names used by existing experiment callers.

Only argument names are adapted here; data reading and transformations have one
implementation. Keep this adapter until old callers and pickle paths migrate.
"""

from .cifar10 import Cifar10Dataset


class DatasetLoad(Cifar10Dataset):
    def __init__(self, dataSetName, isIID):
        super().__init__(is_iid=isIID, dataset_name=dataSetName)

    def load_cifar10(self, is_iid):
        # Keep existing subclass hooks on the original method names active.
        return self.cifar10DataSetConstruct(is_iid)

    def cifar10DataSetConstruct(self, isIID):
        return super().load_cifar10(is_iid=isIID)

    def read_batch(self, file_path):
        return self.load_cifar10_batch(file_path)

    def load_cifar10_batch(self, file_path):
        return super().read_batch(file_path)
