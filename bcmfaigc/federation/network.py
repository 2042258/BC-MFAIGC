"""Network construction and existing dataset-to-node allocation."""

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from bcmfaigc.data.compat import DatasetLoad
from bcmfaigc.variants import EXTENDED
from .node import FederatedNode


class FederatedNetwork(object):  # 对标ClientsGroup
    variant = EXTENDED
    node_type = FederatedNode

    def __init__(
        self,
        data_set_name,
        is_iid,
        batch_size,
        learning_rate,
        loss_func,
        opti,
        num_devices,
        network_stability,
        net,
        dev,
        knock_out_rounds,
        lazy_worker_knock_out_rounds,
        shard_test_data,
        miner_acception_wait_time,
        miner_accepted_transactions_size_limit,
        validator_threshold,
        pow_difficulty,
        even_link_speed_strength,
        base_data_transmission_speed,
        even_computation_power,
        malicious_updates_discount,
        num_malicious,
        noise_variance,
        check_signature,
        not_resync_chain,
        validation_seed,
        test_batch_size=256,
        defense_on=1,
        attack_type="gaussian_weight_noise",
    ):
        self.data_set_name = data_set_name
        self.is_iid = is_iid
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opti = opti
        self.num_devices = num_devices
        self.net = net
        self.dev = dev
        self.devices_set = {}
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        # self.test_data_loader = None
        self.default_network_stability = network_stability
        self.shard_test_data = shard_test_data
        self.even_link_speed_strength = even_link_speed_strength
        self.base_data_transmission_speed = base_data_transmission_speed
        self.even_computation_power = even_computation_power
        self.validation_seed = validation_seed
        self.test_batch_size = test_batch_size
        self.defense_on = bool(defense_on)
        if self.variant.attack_dispatch:
            self.attack_type = attack_type
        self.num_malicious = num_malicious
        self.malicious_updates_discount = malicious_updates_discount
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        # distribute dataset
        """ validator """
        self.validator_threshold = validator_threshold
        """ miner """
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = (
            miner_accepted_transactions_size_limit
        )
        self.pow_difficulty = pow_difficulty
        """ shard """
        self.data_set_balanced_allocation()

    def data_set_balanced_allocation(self):
        # read dataset
        cifar10_dataset = DatasetLoad("cifar10", self.is_iid)

        # perpare training data
        train_data = cifar10_dataset.train_data
        train_label = cifar10_dataset.train_label
        # shard dataset and distribute among devices
        # shard train
        shard_size_train = cifar10_dataset.train_data_size // self.num_devices // 2
        shards_id_train = np.random.permutation(
            cifar10_dataset.train_data_size // shard_size_train
        )

        # perpare test data
        if not self.shard_test_data:
            test_data = torch.tensor(cifar10_dataset.test_data)
            test_label = torch.argmax(torch.tensor(cifar10_dataset.test_label), dim=1)
            test_data_loader = DataLoader(
                TensorDataset(test_data, test_label),
                batch_size=self.test_batch_size,
                shuffle=False,
            )
        else:
            test_data = cifar10_dataset.test_data
            test_label = cifar10_dataset.test_label
            # shard test
            shard_size_test = cifar10_dataset.test_data_size // self.num_devices // 2
            shards_id_test = np.random.permutation(
                cifar10_dataset.test_data_size // shard_size_test
            )

        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(
                range(self.num_devices), self.num_malicious
            )

        for i in range(self.num_devices):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[
                shards_id_train1 * shard_size_train : shards_id_train1
                * shard_size_train
                + shard_size_train
            ]
            data_shards2 = train_data[
                shards_id_train2 * shard_size_train : shards_id_train2
                * shard_size_train
                + shard_size_train
            ]
            label_shards1 = train_label[
                shards_id_train1 * shard_size_train : shards_id_train1
                * shard_size_train
                + shard_size_train
            ]
            label_shards2 = train_label[
                shards_id_train2 * shard_size_train : shards_id_train2
                * shard_size_train
                + shard_size_train
            ]
            local_train_data, local_train_label = (
                np.vstack((data_shards1, data_shards2)),
                np.vstack((label_shards1, label_shards2)),
            )
            local_train_label = np.argmax(local_train_label, axis=1)
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[
                    shards_id_test1 * shard_size_test : shards_id_test1
                    * shard_size_test
                    + shard_size_test
                ]
                data_shards2 = test_data[
                    shards_id_test2 * shard_size_test : shards_id_test2
                    * shard_size_test
                    + shard_size_test
                ]
                label_shards1 = test_label[
                    shards_id_test1 * shard_size_test : shards_id_test1
                    * shard_size_test
                    + shard_size_test
                ]
                label_shards2 = test_label[
                    shards_id_test2 * shard_size_test : shards_id_test2
                    * shard_size_test
                    + shard_size_test
                ]
                local_test_data, local_test_label = (
                    np.vstack((data_shards1, data_shards2)),
                    np.vstack((label_shards1, label_shards2)),
                )
                local_test_label = torch.argmax(torch.tensor(local_test_label), dim=1)
                test_data_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(local_test_data), torch.tensor(local_test_label)
                    ),
                    batch_size=self.test_batch_size,
                    shuffle=False,
                )
            # assign data to a device and put in the devices set
            if i in malicious_nodes_set:
                is_malicious = True
                # add Gussian Noise

            device_idx = f"device_{i + 1}"
            a_device = self.node_type(
                device_idx,
                TensorDataset(
                    torch.tensor(local_train_data), torch.tensor(local_train_label)
                ),
                test_data_loader,
                self.batch_size,
                self.learning_rate,
                self.loss_func,
                self.opti,
                self.default_network_stability,
                self.net,
                self.dev,
                self.miner_acception_wait_time,
                self.miner_accepted_transactions_size_limit,
                self.validator_threshold,
                self.pow_difficulty,
                self.even_link_speed_strength,
                self.base_data_transmission_speed,
                self.even_computation_power,
                is_malicious,
                self.noise_variance,
                self.check_signature,
                self.not_resync_chain,
                self.malicious_updates_discount,
                self.knock_out_rounds,
                self.lazy_worker_knock_out_rounds,
                self.validation_seed,
                defense_on=self.defense_on,
                **(
                    {"attack_type": self.attack_type}
                    if self.variant.attack_dispatch
                    else {}
                ),
            )
            # device index starts from 1
            self.devices_set[device_idx] = a_device
            print(f"Sharding dataset to {device_idx} done.")
        print(f"Sharding dataset done!")
