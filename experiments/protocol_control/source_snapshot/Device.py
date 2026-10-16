import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from DatasetLoad import DatasetLoad
from torch import optim
import torch.nn as nn
import random
import copy
import json
import time
# URL_REQUIRED_3
from Crypto.PublicKey import RSA
from Blockchain import Blockchain
from stable_hash import payload_size_bytes, stable_digest
import os
import torch.nn.functional as F
from trust_aggregation import combine_validator_weights, compute_trust_metrics

class Device:  # 对标client
    def __init__(self, idx, assigned_train_ds, assigned_test_dl, local_batch_size, learning_rate, loss_func, opti, network_stability, net, dev, miner_acception_wait_time, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, is_malicious, noise_variance, check_signature, not_resync_chain, malicious_updates_discount, knock_out_rounds, lazy_worker_knock_out_rounds, validation_seed, defense_on=1, attack_type="gaussian_weight_noise"):
        self.idx = idx
        # deep learning variables
        self.train_ds = assigned_train_ds
        self.test_dl = assigned_test_dl
        self.local_batch_size = local_batch_size
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.optimizer_name = opti
        self.network_stability = network_stability
        self.net = copy.deepcopy(net)
        self.opti = self._make_optimizer(self.net)
        self.dev = dev
        self.validation_seed = validation_seed
        self.defense_on = bool(defense_on)
        self.attack_type = attack_type
        # Keep the original hard-filter behavior as the default.  The
        # trust-weighted mode is enabled explicitly by the experiment config.
        self.aggregation_method = "hard_filter"
        self.trust_center = 0.0
        self.trust_scale = 1.0
        self.trust_soft_threshold = 0.0
        self.trust_beta = 1.0
        self.trust_min_weight = 0.05
        self.trust_hard_reject_score = 5.0
        # in real system, new data can come in, so train_dl should get reassigned before training when that happens
        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.local_batch_size, shuffle=True)
        self.local_train_parameters = None
        self.initial_net_parameters = None
        self.global_parameters = None
        self.global_version = 0
        self.last_applied_round = 0
        # blockchain variables
        self.role = None
        self.pow_difficulty = pow_difficulty
        if even_link_speed_strength:
            self.link_speed = base_data_transmission_speed
        else:
            self.link_speed = random.random() * base_data_transmission_speed
        self.devices_dict = None
        self.aio = False
        ''' simulating hardware equipment strength, such as good processors and RAM capacity. Following recorded times will be shrunk by this value of times
        # for workers, its update time
        # for miners, its PoW time
        # for validators, its validation time
        # might be able to simulate molopoly on computation power when there are block size limit, as faster devices' transactions will be accepted and verified first
        '''
        if even_computation_power:
            self.computation_power = 1
        else:
            self.computation_power = random.randint(0, 4)
        self.peer_list = set()
        # used in cross_verification and in the PoS
        self.online = True
        self.rewards = 0
        self.blockchain = Blockchain()
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        # black_list stores device index rather than the object
        self.black_list = set()
        self.knock_out_rounds = knock_out_rounds
        self.lazy_worker_knock_out_rounds = lazy_worker_knock_out_rounds
        self.worker_accuracy_accross_records = {}
        self.has_added_block = False
        self.the_added_block = None
        self.is_malicious = is_malicious
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        self.malicious_updates_discount = malicious_updates_discount
        # used to identify slow or lazy workers
        self.active_worker_record_by_round = {}
        self.untrustworthy_workers_record_by_comm_round = {}
        self.untrustworthy_validators_record_by_comm_round = {}
        # for picking PoS legitimate blockd;bs
        # self.stake_tracker = {} # used some tricks in main.py for ease of programming
        # used to determine the slowest device round end time to compare PoW with PoS round end time. If simulate under computation_power = 0, this may end up equaling infinity
        self.round_end_time = 0
        ''' For workers '''
        self.local_updates_rewards_per_transaction = 0
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_training_time = None
        self.validator_training_time = None
        self.validator_evaluation_time = None
        self.local_total_epoch = 0
        ''' For validators '''
        self.validator_associated_worker_set = set()
        self.validation_rewards_this_round = 0
        self.accuracies_this_round = {}
        self.validator_associated_miner = None
        # when validator directly accepts workers' updates
        self.unordered_arrival_time_accepted_worker_transactions = {}
        self.validator_accepted_broadcasted_worker_transactions = None or []
        self.final_transactions_queue_to_validate = {}
        self.post_validation_transactions_queue = None or []
        self.validator_threshold = validator_threshold
        self.validator_local_accuracy = None
        ''' For miners '''
        self.miner_associated_worker_set = set()
        self.miner_associated_validator_set = set()
        # dict cannot be added to set()
        self.unconfirmmed_transactions = None or []
        self.broadcasted_transactions = None or []
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        # when miner directly accepts validators' updates
        self.unordered_arrival_time_accepted_validator_transactions = {}
        self.miner_accepted_broadcasted_validator_transactions = None or []
        self.final_candidate_transactions_queue_to_mine = {}
        self.block_generation_time_point = None
        # pure simulation queue and does not exist in real distributed system
        self.unordered_propagated_block_processing_queue = {}
        ''' For malicious node '''
        self.variance_of_noises = None or []

    ''' Common Methods '''

    ''' setters '''

    def _make_optimizer(self, model):
        if self.optimizer_name == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate)
        if self.optimizer_name == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def set_devices_dict_and_aio(self, devices_dict, aio):
        self.devices_dict = devices_dict
        self.aio = aio

    def set_aggregation_config(self, aggregation_method="hard_filter", multikrum_f=1, trust_center=0.0,
                               trust_scale=1.0, trust_soft_threshold=0.0,
                               trust_beta=1.0, trust_min_weight=0.05,
                               trust_hard_reject_score=5.0):
        """Configure validation-guided aggregation without changing DDPM training."""
        if aggregation_method not in {"hard_filter", "trust_weighted", "multikrum"}:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        self.aggregation_method = aggregation_method
        self.multikrum_f = max(int(multikrum_f), 0)
        self.trust_center = float(trust_center)
        self.trust_scale = float(trust_scale)
        self.trust_soft_threshold = float(trust_soft_threshold)
        self.trust_beta = float(trust_beta)
        self.trust_min_weight = float(trust_min_weight)
        self.trust_hard_reject_score = float(trust_hard_reject_score)

    def _compute_aggregation_metrics(self, raw_score):
        if self.aggregation_method == "hard_filter":
            return float(raw_score), 1.0
        return compute_trust_metrics(
            raw_score,
            center=self.trust_center,
            scale=self.trust_scale,
            soft_threshold=self.trust_soft_threshold,
            beta=self.trust_beta,
            min_weight=self.trust_min_weight,
        )

    @staticmethod
    def _transaction_aggregation_weight(transaction):
        try:
            weight = float(transaction.get("aggregation_weight", 1.0))
        except (AttributeError, TypeError, ValueError):
            weight = 1.0
        return max(weight, 0.0)

    def _aggregation_eligibility(self, normalized_score, update_direction):
        """Separate security voting from soft aggregation eligibility."""
        if self.aggregation_method == "trust_weighted":
            try:
                return float(normalized_score) <= self.trust_hard_reject_score
            except (TypeError, ValueError):
                return False
        return bool(update_direction)

    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e

    def init_global_parameters(self):
        self.initial_net_parameters = copy.deepcopy(self.net.state_dict())
        self.global_parameters = copy.deepcopy(self.net.state_dict())
        self.global_version = 0
        self.last_applied_round = 0

    def assign_role(self):
        # equal probability
        role_choice = random.randint(0, 2)
        if role_choice == 0:
            self.role = "worker"
        elif role_choice == 1:
            self.role = "miner"
        else:
            self.role = "validator"

    # used for hard_assign
    def assign_miner_role(self):
        self.role = "miner"

    def assign_worker_role(self):
        self.role = "worker"

    def assign_validator_role(self):
        self.role = "validator"

    ''' getters '''

    def return_idx(self):
        return self.idx

    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}

    def return_peers(self):
        return self.peer_list

    def return_role(self):
        return self.role

    def is_online(self):
        return self.online

    def return_is_malicious(self):
        return self.is_malicious

    @staticmethod
    def _ddpm_core(model):
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    @staticmethod
    def _normalize_ddpm_batch(data):
        return data.mul(2.0).sub(1.0)

    def _ddpm_training_loss(self, model, data):
        core = self._ddpm_core(model)
        x_start = self._normalize_ddpm_batch(data.to(self.dev))
        timesteps = torch.randint(
            0,
            core.timesteps,
            (x_start.shape[0],),
            device=self.dev,
        )
        noise = torch.randn_like(x_start)
        noisy_data = core.q_sample(x_start, timesteps, noise)
        predicted_noise = model(noisy_data, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def _run_local_epoch(self, model, optimizer, accumulation_steps=4):
        """Run one epoch with identical accumulation for workers and validators."""
        model.train()
        optimizer.zero_grad()
        pending_steps = 0
        for data, _ in self.train_dl:
            loss = self._ddpm_training_loss(model, data)
            (loss / accumulation_steps).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad()
                pending_steps = 0
        if pending_steps:
            optimizer.step()
            optimizer.zero_grad()

    def _evaluate_ddpm_loss(self, model):
        core = self._ddpm_core(model)
        was_training = model.training
        model.eval()
        generator = torch.Generator(device=self.dev.type)
        generator.manual_seed(self.validation_seed)
        total_loss = 0.0
        num_elements = 0
        try:
            with torch.no_grad():
                for data, _ in self.test_dl:
                    x_start = self._normalize_ddpm_batch(data.to(self.dev))
                    timesteps = torch.randint(
                        0,
                        core.timesteps,
                        (x_start.shape[0],),
                        device=self.dev,
                        generator=generator,
                    )
                    noise = torch.randn(
                        x_start.shape,
                        device=self.dev,
                        generator=generator,
                    )
                    noisy_data = core.q_sample(x_start, timesteps, noise)
                    predicted_noise = model(noisy_data, timesteps)
                    total_loss += F.mse_loss(
                        predicted_noise, noise, reduction="sum"
                    ).item()
                    num_elements += x_start.numel()
        finally:
            model.train(was_training)
        return total_loss / num_elements if num_elements else float("nan")

    def return_black_list(self):
        return self.black_list

    def return_blockchain_object(self):
        return self.blockchain

    def return_stake(self):
        return self.rewards

    def return_computation_power(self):
        return self.computation_power

    def return_the_added_block(self):
        return self.the_added_block

    def return_round_end_time(self):
        return self.round_end_time

    ''' functions '''

    def sign_msg(self, msg):
        hash = int.from_bytes(stable_digest(msg), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def add_peers(self, new_peers):
        if isinstance(new_peers, Device):
            self.peer_list.add(new_peers)
        else:
            self.peer_list.update(new_peers)

    def remove_peers(self, peers_to_remove):
        if isinstance(peers_to_remove, Device):
            self.peer_list.discard(peers_to_remove)
        else:
            self.peer_list.difference_update(peers_to_remove)

    def synchronize_global_parameters_from_peer(self):
        """Recover the latest peer model without requiring transaction history."""
        online_peers = [peer for peer in self.peer_list if peer.is_online()]
        if not online_peers or self.global_parameters is None:
            return False
        source = max(
            online_peers,
            key=lambda peer: (
                peer.global_version,
                peer.return_blockchain_object().return_chain_length(),
                peer.return_idx(),
            ),
        )
        if source.global_version <= self.global_version:
            return False
        self.global_parameters = copy.deepcopy(source.global_parameters)
        self.global_version = source.global_version
        self.last_applied_round = source.last_applied_round
        self.net.load_state_dict(self.global_parameters, strict=True)
        print(f"{self.idx} recovered global parameters from online peer {source.return_idx()}.")
        return True

    def online_switcher(self):
        old_status = self.online
        online_indicator = random.random()
        if online_indicator < self.network_stability:
            self.online = True
            if old_status is False:
                print(f"{self.idx} goes back online.")
                self.update_peer_list()
                self.synchronize_global_parameters_from_peer()
        else:
            self.online = False
            print(f"{self.idx} goes offline.")
        return self.online

    def update_peer_list(self):
        print(f"\n{self.idx} - {self.role} is updating peer list...")
        old_peer_list = copy.copy(self.peer_list)
        online_peers = set()
        for peer in self.peer_list:
            if peer.is_online():
                online_peers.add(peer)
        # for online peers, suck in their peer list
        for online_peer in online_peers:
            self.add_peers(online_peer.return_peers())
        # remove itself from the peer_list if there is
        self.remove_peers(self)
        # remove malicious peers
        removed_peers = []
        potential_malicious_peer_set = set()
        for peer in self.peer_list:
            if peer.return_idx() in self.black_list:
                potential_malicious_peer_set.add(peer)
        self.remove_peers(potential_malicious_peer_set)
        removed_peers.extend(potential_malicious_peer_set)
        # print updated peer result
        if old_peer_list == self.peer_list:
            print("Peer list NOT changed.")
        else:
            print("Peer list has been changed.")
            added_peers = self.peer_list.difference(old_peer_list)
            if potential_malicious_peer_set:
                print("These malicious peers are removed")
                for peer in removed_peers:
                    print(
                        f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            if added_peers:
                print("These peers are added")
                for peer in added_peers:
                    print(
                        f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
                print()
            print("Final peer list:")
            for peer in self.peer_list:
                print(
                    f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}", end=', ')
            print()
        # WILL ALWAYS RETURN TRUE AS OFFLINE PEERS WON'T BE REMOVED ANY MORE, UNLESS ALL PEERS ARE MALICIOUS...but then it should not register with any other peer. Original purpose - if peer_list ends up empty, randomly register with another device
        return False if not self.peer_list else True

    def check_pow_proof(self, block_to_check):
        # remove its block hash(compute_hash() by default) to verify pow_proof as block hash was set after pow
        pow_proof = block_to_check.return_pow_proof()
        # print("pow_proof", pow_proof)
        # print("compute_hash", block_to_check.compute_hash())
        return pow_proof.startswith('0' * self.pow_difficulty) and pow_proof == block_to_check.compute_hash()

    def check_chain_validity(self, chain_to_check):
        chain_len = chain_to_check.return_chain_length()
        if chain_len == 0 or chain_len == 1:
            pass
        else:
            chain_to_check = chain_to_check.return_chain_structure()
            for block in chain_to_check[1:]:
                if self.check_pow_proof(block) and block.return_previous_block_hash() == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_entire_block=True):
                    pass
                else:
                    return False
        return True

    def accumulate_chain_stake(self, chain_to_accumulate):
        accumulated_stake = 0
        chain_to_accumulate = chain_to_accumulate.return_chain_structure()
        for block in chain_to_accumulate:
            accumulated_stake += self.devices_dict[block.return_mined_by()
                                                   ].return_stake()
        return accumulated_stake

    def recover_global_state_from_best_peer(self):
        """Catch up a lagging node without replaying pruned transactions.

        Blocks are pruned when ``-dtx 1`` is used, so historical replay is
        impossible. A node whose chain height (or model version) is behind the
        best online peer instead adopts that peer's chain height and current
        global parameters, which is the state the chain represents.
        """
        online_peers = [peer for peer in self.peer_list if peer.is_online()]
        if not online_peers:
            return False
        best_peer = max(
            online_peers,
            key=lambda peer: (
                peer.return_blockchain_object().return_chain_length(),
                peer.global_version,
                peer.return_idx(),
            ),
        )
        own_height = self.return_blockchain_object().return_chain_length()
        peer_height = best_peer.return_blockchain_object().return_chain_length()
        recovered = False
        if peer_height > own_height:
            self.return_blockchain_object().replace_chain(
                best_peer.return_blockchain_object().return_chain_structure()
            )
            recovered = True
        if best_peer.global_version > self.global_version:
            self.global_parameters = copy.deepcopy(best_peer.global_parameters)
            self.global_version = best_peer.global_version
            self.last_applied_round = best_peer.last_applied_round
            self.net.load_state_dict(self.global_parameters, strict=True)
            recovered = True
        if recovered:
            print(
                f"{self.role} {self.idx} recovered chain/model state from peer "
                f"{best_peer.return_idx()} (height {peer_height}, version {best_peer.global_version})"
            )
        return recovered

    def resync_chain(self, mining_consensus):
        if self.not_resync_chain:
            # Transactions were pruned to save memory, so replay is disabled;
            # lagging nodes recover the latest peer model/chain height instead.
            return self.recover_global_state_from_best_peer()
        if mining_consensus == 'PoW':
            return self.pow_resync_chain()
        return self.pos_resync_chain()

    def pos_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a chain with the highest accumulated miner's stake in the network...")
        highest_stake_chain = None
        updated_from_peer = None
        curr_chain_stake = self.accumulate_chain_stake(
            self.return_blockchain_object())
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                peer_chain_stake = self.accumulate_chain_stake(peer_chain)
                if peer_chain_stake > curr_chain_stake:
                    if self.check_chain_validity(peer_chain):
                        print(
                            f"A chain from {peer.return_idx()} with total stake {peer_chain_stake} has been found (> currently compared chain stake {curr_chain_stake}) and verified.")
                        # Higher stake valid chain found!
                        curr_chain_stake = peer_chain_stake
                        highest_stake_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(
                            f"A chain from {peer.return_idx()} with higher stake has been found BUT NOT verified. Skipped this chain for syncing.")
        if highest_stake_chain:
            # compare chain difference
            highest_stake_chain_structure = highest_stake_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(highest_stake_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            # return block_iter
            return True
        print("Chain not resynced.")
        return False

    # 在区块链网络中查找最长的合法链，并将当前节点的链与之同步。如果同步成功，函数返回True；否则返回False。
    def pow_resync_chain(self):
        print(f"{self.role} {self.idx} is looking for a longer chain in the network...")
        longest_chain = None
        updated_from_peer = None
        curr_chain_len = self.return_blockchain_object().return_chain_length()
        for peer in self.peer_list:
            if peer.is_online():
                peer_chain = peer.return_blockchain_object()
                if peer_chain.return_chain_length() > curr_chain_len:
                    if self.check_chain_validity(peer_chain):
                        print(
                            f"A longer chain from {peer.return_idx()} with chain length {peer_chain.return_chain_length()} has been found (> currently compared chain length {curr_chain_len}) and verified.")
                        # Longer valid chain found!
                        curr_chain_len = peer_chain.return_chain_length()
                        longest_chain = peer_chain
                        updated_from_peer = peer.return_idx()
                    else:
                        print(
                            f"A longer chain from {peer.return_idx()} has been found BUT NOT verified. Skipped this chain for syncing.")
        if longest_chain:
            # compare chain difference
            longest_chain_structure = longest_chain.return_chain_structure()
            # need more efficient machenism which is to reverse updates by # of blocks
            self.return_blockchain_object().replace_chain(longest_chain_structure)
            print(f"{self.idx} chain resynced from peer {updated_from_peer}.")
            # return block_iter
            return True
        print("Chain not resynced.")
        return False

    # TODO DPOS

    def receive_rewards(self, rewards):
        self.rewards += rewards

    def verify_miner_transaction_by_signature(self, transaction_to_verify, miner_device_idx):
        if miner_device_idx in self.black_list:
            print(
                f"{miner_device_idx} is in miner's blacklist. Trasaction won't get verified.")
            return False
        if self.check_signature:
            transaction_before_signed = copy.deepcopy(transaction_to_verify)
            del transaction_before_signed["miner_signature"]
            modulus = transaction_to_verify['miner_rsa_pub_key']["modulus"]
            pub_key = transaction_to_verify['miner_rsa_pub_key']["pub_key"]
            signature = transaction_to_verify["miner_signature"]
            # verify
            hash = int.from_bytes(
                stable_digest(sorted(transaction_before_signed.items())), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                print(
                    f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
                return True
            else:
                print(
                    f"Signature invalid. Transaction recorded by {miner_device_idx} is NOT verified.")
                return False
        else:
            print(
                f"A transaction recorded by miner {miner_device_idx} in the block is verified!")
            return True

    def verify_block(self, block_to_verify, sending_miner):
        if not self.online_switcher():
            print(f"{self.idx} goes offline when verifying a block")
            return False, False
        verification_time = time.time()
        mined_by = block_to_verify.return_mined_by()
        if sending_miner in self.black_list:
            print(
                f"The miner propagating/sending this block {sending_miner} is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        if mined_by in self.black_list:
            print(
                f"The miner {mined_by} mined this block is in {self.idx}'s black list. Block will not be verified.")
            return False, False
        # check if the proof is valid(verify _block_hash).
        if not self.check_pow_proof(block_to_verify):
            print(
                f"PoW proof of the block from miner {self.idx} is not verified.")
            return False, False
        # # check if miner's signature is valid
        if self.check_signature:
            signature_dict = block_to_verify.return_miner_rsa_pub_key()
            modulus = signature_dict["modulus"]
            pub_key = signature_dict["pub_key"]
            signature = block_to_verify.return_signature()
            # verify signature
            block_to_verify_before_sign = copy.deepcopy(block_to_verify)
            block_to_verify_before_sign.remove_signature_for_verification()
            hash = int.from_bytes(
                stable_digest(block_to_verify_before_sign.__dict__), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash != hashFromSignature:
                print(
                    f"Signature of the block sent by miner {sending_miner} mined by miner {mined_by} is not verified by {self.role} {self.idx}.")
                return False, False
            # check previous hash based on own chain
            last_block = self.return_blockchain_object().return_last_block()
            if last_block is not None:
                # check if the previous_hash referred in the block and the hash of latest block in the chain match.
                last_block_hash = last_block.compute_hash(
                    hash_entire_block=True)
                if block_to_verify.return_previous_block_hash() != last_block_hash:
                    print(f"Block sent by miner {sending_miner} mined by miner {mined_by} has the previous hash recorded as {block_to_verify.return_previous_block_hash()}, but the last block's hash in chain is {last_block_hash}. This is possibly due to a forking event from last round. Block not verified and won't be added. Device needs to resync chain next round.")
                    return False, False
        # All verifications done.
        print(
            f"Block accepted from miner {sending_miner} mined by {mined_by} has been verified by {self.idx}!")
        verification_time = (
            time.time() - verification_time)/self.computation_power
        return block_to_verify, verification_time

    def add_block(self, block_to_add):
        self.return_blockchain_object().append_block(block_to_add)
        print(
            f"d_{self.idx.split('_')[-1]} - {self.role[0]} has appened a block to its chain. Chain length now - {self.return_blockchain_object().return_chain_length()}")
        # TODO delete has_added_block
        # self.has_added_block = True
        self.the_added_block = block_to_add
        return True

    # also accumulate rewards here
    def process_block(self, block_to_process, log_files_folder_path, conn, conn_cursor, when_resync=False):
        # collect usable updated params, malicious nodes identification, get rewards and do local udpates
        processing_time = time.time()
        if not self.online_switcher():
            # may need to set up a flag indicating if a block has been processed
            print(f"{self.role} {self.idx} goes offline when processing the added block. Model not updated and rewards information not upgraded. Outdated information may be obtained by this node if it never resyncs to a different chain.")
        if block_to_process:
            mined_by = block_to_process.return_mined_by()
            if mined_by in self.black_list:
                # in this system black list is also consistent across devices as it is calculated based on the information on chain, but individual device can decide its own validation/verification mechanisms and has its own
                print(
                    f"The added block is mined by miner {block_to_process.return_mined_by()}, which is in this device's black list. Block will not be processed.")
            else:
                # process validator sig valid transactions
                # used to count positive and negative transactions worker by worker, select the transaction to do global update and identify potential malicious worker
                self_rewards_accumulator = 0
                valid_transactions_records_by_worker = {}
                valid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()[
                    'valid_validator_sig_transacitons']
                comm_round = block_to_process.return_block_idx()
                self.active_worker_record_by_round[comm_round] = set()
                for valid_validator_sig_worker_transaciton in valid_validator_sig_worker_transacitons_in_block:
                    # verify miner's signature(miner does not get reward for receiving and aggregating)
                    if self.verify_miner_transaction_by_signature(valid_validator_sig_worker_transaciton, mined_by):
                        worker_device_idx = valid_validator_sig_worker_transaciton['worker_device_idx']
                        self.active_worker_record_by_round[comm_round].add(
                            worker_device_idx)
                        if not worker_device_idx in valid_transactions_records_by_worker.keys():
                            valid_transactions_records_by_worker[worker_device_idx] = {
                            }
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'] = set(
                            )
                            valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'] = set(
                            )
                            valid_transactions_records_by_worker[worker_device_idx]['positive_votes'] = 0
                            valid_transactions_records_by_worker[worker_device_idx]['negative_votes'] = 0
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'] = set(
                            )
                            valid_transactions_records_by_worker[worker_device_idx]['finally_used_params'] = None
                            valid_transactions_records_by_worker[worker_device_idx]['finally_used_weight'] = 1.0
                        # epoch of this worker's local update
                        local_epoch_seq = valid_validator_sig_worker_transaciton[
                            'local_total_accumulated_epochs_this_round']
                        positive_direction_validators = valid_validator_sig_worker_transaciton[
                            'positive_direction_validators']
                        negative_direction_validators = valid_validator_sig_worker_transaciton[
                            'negative_direction_validators']
                        aggregation_validators = valid_validator_sig_worker_transaciton.get(
                            'aggregation_validators',
                            positive_direction_validators,
                        )
                        if self.aggregation_method == "trust_weighted" and aggregation_validators:
                            # Soft aggregation is independent of the security
                            # vote.  The latter is still counted for detection
                            # and blacklisting, while eligible updates receive
                            # a bounded validation-guided contribution.
                            valid_transactions_records_by_worker[worker_device_idx]['positive_votes'] += len(positive_direction_validators)
                            valid_transactions_records_by_worker[worker_device_idx]['negative_votes'] += len(negative_direction_validators)
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(
                                local_epoch_seq)
                            if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                valid_transactions_records_by_worker[worker_device_idx][
                                    'finally_used_params'] = valid_validator_sig_worker_transaciton['local_updates_params']
                                valid_transactions_records_by_worker[worker_device_idx][
                                    'finally_used_weight'] = combine_validator_weights(
                                        validator_record.get('aggregation_weight', 1.0)
                                        for validator_record in aggregation_validators
                                    )
                            if self.idx == worker_device_idx:
                                self_rewards_accumulator += valid_validator_sig_worker_transaciton[
                                    'local_updates_rewards']
                        elif len(positive_direction_validators) >= len(negative_direction_validators):
                            # worker transaction can be used
                            valid_transactions_records_by_worker[worker_device_idx]['positive_votes'] += len(positive_direction_validators)
                            valid_transactions_records_by_worker[worker_device_idx]['negative_votes'] += len(negative_direction_validators)
                            valid_transactions_records_by_worker[worker_device_idx]['positive_epochs'].add(
                                local_epoch_seq)
                            valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(
                                local_epoch_seq)
                            # see if this is the latest epoch from this worker
                            if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                valid_transactions_records_by_worker[worker_device_idx][
                                    'finally_used_params'] = valid_validator_sig_worker_transaciton['local_updates_params']
                                valid_transactions_records_by_worker[worker_device_idx][
                                    'finally_used_weight'] = combine_validator_weights(
                                        validator_record.get('aggregation_weight', 1.0)
                                        for validator_record in positive_direction_validators
                                    )
                            # give rewards to this worker
                            if self.idx == worker_device_idx:
                                self_rewards_accumulator += valid_validator_sig_worker_transaciton[
                                    'local_updates_rewards']
                        else:
                            valid_transactions_records_by_worker[worker_device_idx]['positive_votes'] += len(positive_direction_validators)
                            valid_transactions_records_by_worker[worker_device_idx]['negative_votes'] += len(negative_direction_validators)
                            if self.malicious_updates_discount:
                                # worker transaction voted negative and has to be applied for a discount
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(
                                    local_epoch_seq)
                                valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs'].add(
                                    local_epoch_seq)
                                # see if this is the latest epoch from this worker
                                if local_epoch_seq == max(valid_transactions_records_by_worker[worker_device_idx]['all_valid_epochs']):
                                    # apply discount
                                    discounted_valid_validator_sig_worker_transaciton_local_updates_params = copy.deepcopy(
                                        valid_validator_sig_worker_transaciton['local_updates_params'])
                                    for var in discounted_valid_validator_sig_worker_transaciton_local_updates_params:
                                        discounted_valid_validator_sig_worker_transaciton_local_updates_params[
                                            var] *= self.malicious_updates_discount
                                    valid_transactions_records_by_worker[worker_device_idx][
                                        'finally_used_params'] = discounted_valid_validator_sig_worker_transaciton_local_updates_params
                                    valid_transactions_records_by_worker[worker_device_idx][
                                        'finally_used_weight'] = self._transaction_aggregation_weight(
                                            valid_validator_sig_worker_transaciton
                                        ) * self.malicious_updates_discount
                                # worker receive discounted rewards for negative update
                                if self.idx == worker_device_idx:
                                    self_rewards_accumulator += valid_validator_sig_worker_transaciton[
                                        'local_updates_rewards'] * self.malicious_updates_discount
                            else:
                                # discount specified as 0, worker transaction voted negative and cannot be used
                                valid_transactions_records_by_worker[worker_device_idx]['negative_epochs'].add(
                                    local_epoch_seq)
                                # worker does not receive rewards for negative update
                        # give rewards to validators and the miner in this transaction
                        for validator_record in positive_direction_validators + negative_direction_validators:
                            if self.idx == validator_record['validator']:
                                self_rewards_accumulator += validator_record['validation_rewards']
                            if self.idx == validator_record['miner_device_idx']:
                                self_rewards_accumulator += validator_record['miner_rewards_for_this_tx']
                    else:
                        print(
                            f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return

                # Resolve the final worker decision by the actual validator
                # vote counts. The previous implementation compared sets of
                # epoch IDs, so mixed votes for the same epoch collapsed to a
                # tie and were always accepted.
                for record in valid_transactions_records_by_worker.values():
                    if (
                        record['negative_votes'] > record['positive_votes']
                        and not self.malicious_updates_discount
                        and self.aggregation_method != "trust_weighted"
                    ):
                        record['finally_used_params'] = None
                        record['finally_used_weight'] = 0.0

                self.untrustworthy_workers_record_by_comm_round[comm_round] = set()
                for worker_idx, local_updates_direction_records in valid_transactions_records_by_worker.items():
                    if local_updates_direction_records['negative_votes'] > local_updates_direction_records['positive_votes']:
                        self.untrustworthy_workers_record_by_comm_round[comm_round].add(worker_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_workers_record_by_comm_round.keys():
                                if worker_idx in self.untrustworthy_workers_record_by_comm_round[comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(worker_idx)
                            # is it right?
                            if when_resync:
                                msg_end = " when resyncing!\n"
                            else:
                                msg_end = "!\n"
                            if self.devices_dict[worker_idx].return_is_malicious():
                                msg = f"{self.idx} has successfully identified a malicious worker device {worker_idx} in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (
                                    worker_idx, 1, self.idx, "", comm_round, when_resync))
                                conn.commit()
                            else:
                                msg = f"WARNING: {self.idx} has mistakenly regard {worker_idx} as a malicious worker device in comm_round {comm_round}{msg_end}"
                                with open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'a') as file:
                                    file.write(msg)
                                conn_cursor.execute("INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)", (
                                    worker_idx, 0, "", self.idx, comm_round, when_resync))
                                conn.commit()
                            print(msg)

                            # cont = print("Press ENTER to continue")

                # identify potentially compromised validator
                self.untrustworthy_validators_record_by_comm_round[comm_round] = set()
                invalid_validator_sig_worker_transacitons_in_block = block_to_process.return_transactions()[
                    'invalid_validator_sig_transacitons']
                for invalid_validator_sig_worker_transaciton in invalid_validator_sig_worker_transacitons_in_block:
                    if self.verify_miner_transaction_by_signature(invalid_validator_sig_worker_transaciton, mined_by):
                        validator_device_idx = invalid_validator_sig_worker_transaciton['validator']
                        self.untrustworthy_validators_record_by_comm_round[comm_round].add(
                            validator_device_idx)
                        kick_out_accumulator = 1
                        # check previous rounds
                        for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                            if comm_round_to_check in self.untrustworthy_validators_record_by_comm_round.keys():
                                if validator_device_idx in self.untrustworthy_validators_record_by_comm_round[
                                    comm_round_to_check]:
                                    kick_out_accumulator += 1
                        if kick_out_accumulator == self.knock_out_rounds:
                            # kick out
                            self.black_list.add(validator_device_idx)
                            print(
                                f"{validator_device_idx} has been regarded as a compromised validator by {self.idx} in {comm_round}.")
                            # actually, we did not let validator do malicious thing if is_malicious=1 is set to this device. In the submission of 2020/10, we only focus on catching malicious worker
                            # is it right?
                            # if when_resync:
                            #	 msg_end = " when resyncing!\n"
                            # else:
                            #	 msg_end = "!\n"
                            # if self.devices_dict[validator_device_idx].return_is_malicious():
                            #	 msg = f"{self.idx} has successfully identified a compromised validator device {validator_device_idx} in comm_round {comm_round}{msg_end}"
                            #	 with open(f"{log_files_folder_path}/correctly_kicked_validators.txt", 'a') as file:
                            #		 file.write(msg)
                            # else:
                            #	 msg = f"WARNING: {self.idx} has mistakenly regard {validator_device_idx} as a compromised validator device in comm_round {comm_round}{msg_end}"
                            #	 with open(f"{log_files_folder_path}/mistakenly_kicked_validators.txt", 'a') as file:
                            #		 file.write(msg)
                            # print(msg)
                            # cont = print("Press ENTER to continue")
                    else:
                        print(
                            f"one validator transaction miner sig found invalid in this block. {self.idx} will drop this block and roll back rewards information")
                        return
                    # give rewards to the miner in this transaction
                    if self.idx == invalid_validator_sig_worker_transaciton['miner_device_idx']:
                        self_rewards_accumulator += invalid_validator_sig_worker_transaciton[
                            'miner_rewards_for_this_tx']
                # miner gets mining rewards
                if self.idx == mined_by:
                    self_rewards_accumulator += block_to_process.return_mining_rewards()
                # set received rewards this round based on info from this block
                self.receive_rewards(self_rewards_accumulator)
                print(
                    f"{self.role} {self.idx} has received total {self_rewards_accumulator} rewards for this comm round.")
                # collect usable worker updates and do global updates
                finally_used_local_params = []
                # record True Positive, False Positive, True Negative and False Negative for identified workers
                TP, FP, TN, FN = 0, 0, 0, 0
                for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                    is_worker_malicious = self.devices_dict[worker_device_idx].return_is_malicious()
                    if local_params_record['finally_used_params']:
                        # Accepted update: benign -> TN, malicious -> FN.
                        finally_used_local_params.append(
                            (worker_device_idx,
                             local_params_record['finally_used_params'],
                             local_params_record['finally_used_weight']))
                        if is_worker_malicious:
                            FN += 1
                        else:
                            TN += 1
                    else:
                        # Rejected update: malicious -> TP, benign -> FP.
                        if is_worker_malicious:
                            TP += 1
                        else:
                            FP += 1

                with open(f"{log_files_folder_path}/aggregation_decisions.jsonl", "a", encoding="utf-8") as file:
                    for worker_device_idx, local_params_record in valid_transactions_records_by_worker.items():
                        file.write(json.dumps({
                            "round": comm_round,
                            "observer_id": self.return_idx(),
                            "worker_id": worker_device_idx,
                            "worker_malicious": self.devices_dict[worker_device_idx].return_is_malicious(),
                            "positive_votes": local_params_record["positive_votes"],
                            "negative_votes": local_params_record["negative_votes"],
                            "filter_accepted": bool(local_params_record["finally_used_params"]),
                            "accepted_for_aggregation": bool(local_params_record["finally_used_params"]),
                            "aggregation_weight": float(local_params_record["finally_used_weight"]) if local_params_record["finally_used_params"] else 0.0,
                            "aggregation_method": self.aggregation_method,
                        }, sort_keys=True) + "\n")

                applied_global_update = False
                if self.online_switcher():
                    applied_global_update = self.global_update(finally_used_local_params, update_round=comm_round)
                else:
                    print(
                        f"Unfortunately, {self.role} {self.idx} goes offline when it's doing global_updates.")
                applied_worker_ids = {
                    worker_idx for worker_idx, _, _ in finally_used_local_params
                    if worker_idx not in self.black_list
                }
                with open(f"{log_files_folder_path}/aggregation_applications.jsonl", "a", encoding="utf-8") as file:
                    for worker_idx, record in valid_transactions_records_by_worker.items():
                        filter_accepted = bool(record['finally_used_params'])
                        file.write(json.dumps({
                            "round": comm_round,
                            "observer_id": self.return_idx(),
                            "worker_id": worker_idx,
                            "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                            "filter_accepted": filter_accepted,
                            "aggregation_applied": bool(applied_global_update and worker_idx in applied_worker_ids),
                            "skip_reason": None if (applied_global_update and worker_idx in applied_worker_ids) else "offline_or_blacklist_or_empty_update",
                            "aggregation_mode": "blockchain",
                            "aggregation_weight": float(record['finally_used_weight']) if filter_accepted else 0.0,
                            "aggregation_method": self.aggregation_method,
                        }, sort_keys=True) + "\n")

                malicious_worker_validation_log_path = f"{log_files_folder_path}/comm_{comm_round}/malicious_worker_validation_log.txt"
                if not os.path.exists(malicious_worker_validation_log_path):
                    with open(malicious_worker_validation_log_path, 'w') as file:
                        accuracy = (TP + TN) / (TP + TN + FP + FN)
                        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
                        file.write(
                            f"In comm_{comm_round} of validating workers, TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}. \
                                \nAccuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 Score = {f1}")
                processing_time = (time.time() - processing_time) / \
                                  self.computation_power
                return processing_time

    def apply_direct_aggregation(self, validated_transactions, log_files_folder_path, comm_round, conn, conn_cursor):
        """Aggregate filtered validator decisions without blockchain transport.

        This mirrors the counting and blacklist logic of ``process_block`` but
        takes the validated transactions directly from the validators, so the
        same MSE filter and aggregation rule are applied while consensus,
        mining and block propagation are removed.
        """
        per_epoch_votes = {}
        for transaction in validated_transactions:
            if not transaction or 'worker_device_idx' not in transaction:
                continue
            worker_idx = transaction['worker_device_idx']
            epoch_seq = transaction['local_total_accumulated_epochs_this_round']
            key = (worker_idx, epoch_seq)
            votes = per_epoch_votes.setdefault(
                key,
                {
                    'positive': 0,
                    'negative': 0,
                    'positive_weights': [],
                    'negative_weights': [],
                    'aggregation_transactions': [],
                    'transaction': transaction,
                },
            )
            if transaction.get('update_direction') is True:
                votes['positive'] += 1
                votes['positive_weights'].append(
                    self._transaction_aggregation_weight(transaction)
                )
            else:
                votes['negative'] += 1
                votes['negative_weights'].append(
                    self._transaction_aggregation_weight(transaction)
                )
            if transaction.get(
                'aggregation_eligible',
                transaction.get('update_direction') is True,
            ):
                votes['aggregation_transactions'].append(transaction)

        positive_epochs = {}
        negative_epochs = {}
        finally_used_params = {}
        finally_used_weights = {}
        for (worker_idx, epoch_seq), votes in per_epoch_votes.items():
            if self.aggregation_method == "trust_weighted" and votes['aggregation_transactions']:
                aggregation_transaction = votes['aggregation_transactions'][0]
                positive_epochs.setdefault(worker_idx, set()).add(epoch_seq)
                if epoch_seq == max(positive_epochs[worker_idx]):
                    finally_used_params[worker_idx] = aggregation_transaction['local_updates_params']
                    finally_used_weights[worker_idx] = combine_validator_weights(
                        self._transaction_aggregation_weight(item)
                        for item in votes['aggregation_transactions']
                    )
            elif votes['positive'] >= votes['negative']:
                positive_epochs.setdefault(worker_idx, set()).add(epoch_seq)
                if epoch_seq == max(positive_epochs[worker_idx]):
                    finally_used_params[worker_idx] = votes['transaction']['local_updates_params']
                    finally_used_weights[worker_idx] = combine_validator_weights(
                        votes['positive_weights']
                    )
            else:
                negative_epochs.setdefault(worker_idx, set()).add(epoch_seq)

        # Resolve mixed votes by counts, matching the blockchain path.
        for worker_idx in {key[0] for key in per_epoch_votes}:
            positive_count = sum(
                votes['positive'] for (candidate, _), votes in per_epoch_votes.items()
                if candidate == worker_idx
            )
            negative_count = sum(
                votes['negative'] for (candidate, _), votes in per_epoch_votes.items()
                if candidate == worker_idx
            )
            if negative_count > positive_count and self.aggregation_method != "trust_weighted":
                finally_used_params.pop(worker_idx, None)
                finally_used_weights.pop(worker_idx, None)

        # Same progressive blacklist rule as process_block.
        self.untrustworthy_workers_record_by_comm_round[comm_round] = set()
        for worker_idx in {key[0] for key in per_epoch_votes}:
            positive_count = sum(
                votes['positive'] for (candidate, _), votes in per_epoch_votes.items()
                if candidate == worker_idx
            )
            negative_count = sum(
                votes['negative'] for (candidate, _), votes in per_epoch_votes.items()
                if candidate == worker_idx
            )
            if negative_count > positive_count:
                self.untrustworthy_workers_record_by_comm_round[comm_round].add(worker_idx)
                kick_out_accumulator = 1
                for comm_round_to_check in range(comm_round - self.knock_out_rounds + 1, comm_round):
                    if comm_round_to_check in self.untrustworthy_workers_record_by_comm_round:
                        if worker_idx in self.untrustworthy_workers_record_by_comm_round[comm_round_to_check]:
                            kick_out_accumulator += 1
                if kick_out_accumulator == self.knock_out_rounds:
                    self.black_list.add(worker_idx)
                    is_malicious_worker = self.devices_dict[worker_idx].return_is_malicious()
                    if is_malicious_worker:
                        msg = f"{self.idx} has successfully identified a malicious worker device {worker_idx} in comm_round {comm_round}!\n"
                        with open(f"{log_files_folder_path}/correctly_kicked_workers.txt", 'a') as file:
                            file.write(msg)
                        conn_cursor.execute(
                            "INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)",
                            (worker_idx, 1, self.idx, "", comm_round, False),
                        )
                    else:
                        msg = f"WARNING: {self.idx} has mistakenly regard {worker_idx} as a malicious worker device in comm_round {comm_round}!\n"
                        with open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", 'a') as file:
                            file.write(msg)
                        conn_cursor.execute(
                            "INSERT INTO malicious_workers_log VALUES (?, ?, ?, ?, ?, ?)",
                            (worker_idx, 0, "", self.idx, comm_round, False),
                        )
                    conn.commit()

        with open(f"{log_files_folder_path}/aggregation_decisions.jsonl", "a", encoding="utf-8") as file:
            for (worker_idx, epoch_seq), votes in sorted(per_epoch_votes.items()):
                file.write(json.dumps({
                    "round": comm_round,
                    "observer_id": self.return_idx(),
                    "worker_id": worker_idx,
                    "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                    "positive_votes": votes['positive'],
                    "negative_votes": votes['negative'],
                    "filter_accepted": worker_idx in finally_used_params,
                    "accepted_for_aggregation": worker_idx in finally_used_params,
                    "aggregation_weight": float(finally_used_weights.get(worker_idx, 0.0)),
                    "aggregation_method": self.aggregation_method,
                    "aggregation_mode": "direct_no_chain",
                }, sort_keys=True) + "\n")

        aggregation_start = time.time()
        applied_global_update = False
        if self.is_online():
            applied_global_update = self.global_update(
                [
                    (worker_idx, params, finally_used_weights[worker_idx])
                    for worker_idx, params in finally_used_params.items()
                ],
                update_round=comm_round,
            )
        else:
            print(
                f"Unfortunately, {self.role} {self.idx} is offline during direct global updates."
            )
        applied_worker_ids = {
            worker_idx for worker_idx in finally_used_params
            if worker_idx not in self.black_list
        }
        with open(f"{log_files_folder_path}/aggregation_applications.jsonl", "a", encoding="utf-8") as file:
            for worker_idx in {key[0] for key in per_epoch_votes}:
                filter_accepted = worker_idx in finally_used_params
                applied = bool(applied_global_update and worker_idx in applied_worker_ids)
                file.write(json.dumps({
                    "round": comm_round,
                    "observer_id": self.return_idx(),
                    "worker_id": worker_idx,
                    "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                    "filter_accepted": filter_accepted,
                    "aggregation_applied": applied,
                    "skip_reason": None if applied else "offline_or_blacklist_or_empty_update",
                    "aggregation_mode": "direct_no_chain",
                    "aggregation_weight": float(finally_used_weights.get(worker_idx, 0.0)),
                    "aggregation_method": self.aggregation_method,
                }, sort_keys=True) + "\n")
        processing_time = time.time() - aggregation_start
        self.add_to_round_end_time(processing_time)
        return processing_time

    def apply_plain_fedavg(self, local_updates, log_files_folder_path, comm_round):
        """Apply every current-round worker update without validator or chain filtering."""
        applied_updates = []
        for worker_idx, params in local_updates:
            applied_updates.append(worker_idx)
        applied_global_update = False
        if self.is_online() and local_updates:
            applied_global_update = self.global_update(
                local_updates, respect_blacklist=False, update_round=comm_round
            )
        with open(f"{log_files_folder_path}/aggregation_decisions.jsonl", "a", encoding="utf-8") as file:
            for worker_idx in applied_updates:
                with open(f"{log_files_folder_path}/aggregation_applications.jsonl", "a", encoding="utf-8") as application_file:
                    application_file.write(json.dumps({
                        "round": comm_round,
                        "observer_id": self.return_idx(),
                        "worker_id": worker_idx,
                        "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                        "filter_accepted": True,
                        "aggregation_applied": applied_global_update,
                        "skip_reason": None if applied_global_update else "offline_or_empty_update",
                        "aggregation_mode": "plain_fedavg",
                    }, sort_keys=True) + "\n")
                file.write(json.dumps({
                    "round": comm_round,
                    "observer_id": self.return_idx(),
                    "worker_id": worker_idx,
                    "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                    "positive_votes": 1,
                    "negative_votes": 0,
                    "filter_accepted": True,
                    "accepted_for_aggregation": True,
                    "aggregation_mode": "plain_fedavg",
                }, sort_keys=True) + "\n")

    def apply_multikrum(self, local_updates, log_files_folder_path, comm_round):
        """Apply a deterministic Multi-Krum baseline to current worker updates."""
        usable_updates = [
            (worker_idx, params)
            for worker_idx, params in local_updates
            if params is not None
        ]
        if not usable_updates:
            return
        num_updates = len(usable_updates)
        configured_f = max(int(getattr(self, "multikrum_f", 1)), 0)
        # Multi-Krum requires n >= 2f + 3. Reduce the effective assumption for
        # small role assignments and record it in the decision log.
        effective_f = min(configured_f, max(0, (num_updates - 3) // 2))
        neighbor_count = num_updates - effective_f - 2
        fallback_reason = None
        selected_positions = list(range(num_updates))
        scores = [None] * num_updates
        if num_updates < 3 or neighbor_count < 1:
            fallback_reason = "insufficient_updates_for_multikrum"
        else:
            ordered_keys = sorted(usable_updates[0][1].keys())
            vectors = []
            for _, params in usable_updates:
                parts = [
                    params[key].detach().reshape(-1).float().cpu()
                    for key in ordered_keys
                    if torch.is_floating_point(params[key])
                ]
                vectors.append(torch.cat(parts) if parts else torch.zeros(1))
            vector_matrix = torch.stack(vectors)
            distance_matrix = torch.cdist(vector_matrix, vector_matrix, p=2).pow(2)
            for position in range(num_updates):
                distances = [
                    float(distance_matrix[position, other])
                    for other in range(num_updates) if other != position
                ]
                scores[position] = sum(sorted(distances)[:neighbor_count])
            selected_positions = sorted(
                range(num_updates),
                key=lambda position: (scores[position], usable_updates[position][0]),
            )[:max(1, num_updates - effective_f - 2)]

        selected_ids = {usable_updates[position][0] for position in selected_positions}
        selected_weight = 1.0 / len(selected_positions)
        aggregation_records = [
            (worker_idx, params, selected_weight)
            for worker_idx, params in usable_updates
            if worker_idx in selected_ids
        ]
        applied_global_update = bool(
            self.is_online()
            and self.global_update(aggregation_records, respect_blacklist=False, update_round=comm_round)
        )
        with open(f"{log_files_folder_path}/aggregation_decisions.jsonl", "a", encoding="utf-8") as file:
            for position, (worker_idx, _) in enumerate(usable_updates):
                file.write(json.dumps({
                    "round": comm_round,
                    "observer_id": self.return_idx(),
                    "worker_id": worker_idx,
                    "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                    "filter_accepted": worker_idx in selected_ids,
                    "accepted_for_aggregation": worker_idx in selected_ids,
                    "aggregation_weight": selected_weight if worker_idx in selected_ids else 0.0,
                    "aggregation_method": "multikrum",
                    "multikrum_configured_f": configured_f,
                    "multikrum_effective_f": effective_f,
                    "multikrum_score": scores[position],
                    "multikrum_fallback_reason": fallback_reason,
                }, sort_keys=True) + "\n")
        with open(f"{log_files_folder_path}/aggregation_applications.jsonl", "a", encoding="utf-8") as file:
            for worker_idx, _ in usable_updates:
                applied = bool(applied_global_update and worker_idx in selected_ids)
                file.write(json.dumps({
                    "round": comm_round,
                    "observer_id": self.return_idx(),
                    "worker_id": worker_idx,
                    "worker_malicious": self.devices_dict[worker_idx].return_is_malicious(),
                    "filter_accepted": worker_idx in selected_ids,
                    "aggregation_applied": applied,
                    "skip_reason": None if applied else "not_selected_or_offline_or_empty_update",
                    "aggregation_mode": "multikrum_baseline",
                    "aggregation_weight": selected_weight if worker_idx in selected_ids else 0.0,
                    "aggregation_method": "multikrum",
                }, sort_keys=True) + "\n")

    def add_to_round_end_time(self, time_to_add):
        self.round_end_time += time_to_add

    def other_tasks_at_the_end_of_comm_round(self, this_comm_round, log_files_folder_path):
        self.kick_out_slow_or_lazy_workers(
            this_comm_round, log_files_folder_path)

    def kick_out_slow_or_lazy_workers(self, this_comm_round, log_files_folder_path):
        for device in self.peer_list:
            if device.return_role() == 'worker':
                if this_comm_round in self.active_worker_record_by_round.keys():
                    if not device.return_idx() in self.active_worker_record_by_round[this_comm_round]:
                        not_active_accumulator = 1
                        # check if not active for the past (lazy_worker_knock_out_rounds - 1) rounds
                        for comm_round_to_check in range(this_comm_round - self.lazy_worker_knock_out_rounds + 1, this_comm_round):
                            if comm_round_to_check in self.active_worker_record_by_round.keys():
                                if not device.return_idx() in self.active_worker_record_by_round[comm_round_to_check]:
                                    not_active_accumulator += 1
                        if not_active_accumulator == self.lazy_worker_knock_out_rounds:
                            # kick out
                            self.black_list.add(device.return_idx())
                            msg = f"worker {device.return_idx()} has been regarded as a lazy worker by {self.idx} in comm_round {this_comm_round}.\n"
                            with open(f"{log_files_folder_path}/kicked_lazy_workers.txt", 'a') as file:
                                file.write(msg)
                else:
                    # this may happen when a device is put into black list by every worker in a certain comm round
                    pass

    def update_model_after_chain_resync(self, log_files_folder_path, conn, conn_cursor):
        if self.not_resync_chain:
            # -dtx 1 prunes transactions from blocks, so the chain cannot be
            # replayed here. resync_chain() has already adopted the best peer's
            # chain height and global parameters in that case; replaying would
            # reset the model to the initial weights and crash on pruned blocks.
            return
        # reset global params to the initial weights of the net
        self.global_parameters = copy.deepcopy(self.initial_net_parameters)
        self.global_version = 0
        self.last_applied_round = 0
        # in future version, develop efficient updating algorithm based on chain difference
        for block in self.return_blockchain_object().return_chain_structure():
            self.process_block(block, log_files_folder_path,
                               conn, conn_cursor, when_resync=True)

    def return_pow_difficulty(self):
        return self.pow_difficulty

    def register_in_the_network(self, check_online=False):
        if self.aio:
            self.add_peers(set(self.devices_dict.values()))
        else:
            potential_registrars = set(self.devices_dict.values())
            # it cannot register with itself
            potential_registrars.discard(self)
            # pick a registrar
            registrar = random.sample(potential_registrars, 1)[0]
            if check_online:
                if not registrar.is_online():
                    online_registrars = set()
                    for registrar in potential_registrars:
                        if registrar.is_online():
                            online_registrars.add(registrar)
                    if not online_registrars:
                        return False
                    registrar = random.sample(online_registrars, 1)[0]
            # registrant add registrar to its peer list
            self.add_peers(registrar)
            # this device sucks in registrar's peer list
            self.add_peers(registrar.return_peers())
            # registrar adds registrant(must in this order, or registrant will add itself from registrar's peer list)
            registrar.add_peers(self)
            return True

    ''' Worker '''

    def malicious_worker_add_noise_to_weights(self, m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                noise = self.noise_variance * torch.randn_like(m.weight)
                variance_of_noise = torch.var(noise)
                m.weight.add_(noise)

                self.variance_of_noises.append(float(variance_of_noise))

    def malicious_worker_apply_attack(self):
        attack_type = getattr(self, "attack_type", "gaussian_weight_noise")
        if attack_type == "gaussian_weight_noise":
            self.net.apply(self.malicious_worker_add_noise_to_weights)
            return
        if attack_type == "sign_flip":
            with torch.no_grad():
                for parameter in self.net.parameters():
                    if torch.is_floating_point(parameter):
                        parameter.mul_(-1.0)
            return
        raise ValueError(f"Unsupported attack type: {attack_type}")


    # TODO change to computation power
    def worker_local_update(self, rewards, log_files_folder_path_comm_round, comm_round, local_epochs=1):
        print(
            f"Worker {self.idx} is doing local_update with computation power {self.computation_power} and link speed {round(self.link_speed, 3)} bytes/s")
        # Reset optimizer state at every local update so worker and validator
        # use the same fresh local-optimizer protocol.
        self.opti = self._make_optimizer(self.net)
        self.net.load_state_dict(self.global_parameters, strict=True)
        self.local_train_parameters = None
        self.local_update_time = time.time()
        # 本地工作节点更新指定轮数
        # 通常，如果验证器接受时间已指定，local_epochs 应为 1
        # 记录恶意节点
        is_malicious_node = "M" if self.return_is_malicious() else "B"
        self.local_updates_rewards_per_transaction = 0
        # Train the epsilon predictor with the standard DDPM objective.
        training_start = time.time()
        for epoch in range(local_epochs):
            self._run_local_epoch(self.net, self.opti)

            # 记录准确率以找到好的 -vh
            with open(
                    f"{log_files_folder_path_comm_round}/worker_{self.idx}_{is_malicious_node}_local_updating_accuracies_comm_{comm_round}.txt",
                    "a") as file:
                file.write(
                    f"{self.return_idx()} epoch_{epoch + 1} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")
            self.local_total_epoch += 1
        try:
            self.local_training_time = (time.time() - training_start) / self.computation_power
        except ZeroDivisionError:
            self.local_training_time = float('inf')

        # 本地更新完成
        try:
            self.local_update_time = (time.time() - self.local_update_time) / self.computation_power
        except:
            self.local_update_time = float('inf')

        if self.is_malicious:
            self.malicious_worker_apply_attack()
            attack_type = getattr(self, "attack_type", "gaussian_weight_noise")
            print(f"恶意工作节点 {self.idx} 在传输前执行攻击: {attack_type}")
            with open(f"{log_files_folder_path_comm_round}/comm_{comm_round}_variance_of_noises.txt", "a") as file:
                file.write(
                    f"{self.return_idx()} {self.return_role()} {is_malicious_node} attack_type: {attack_type}; 噪声方差: {self.variance_of_noises}\n")

        # 记录准确率以找到好的 -vh
        with open(f"{log_files_folder_path_comm_round}/worker_final_local_accuracies_comm_{comm_round}.txt",
                  "a") as file:
            file.write(
                f"{self.return_idx()} {self.return_role()} {is_malicious_node}: {self.validate_model_weights(self.net.state_dict())}\n")

        print(f"完成 {local_epochs} 轮更新，总共 {self.local_total_epoch} 轮")
        self.local_train_parameters = self.net.state_dict()
        return self.local_update_time

    # used to simulate time waste when worker goes offline during transmission to validator
    def waste_one_epoch_local_update_time(self, opti):
        if self.computation_power == 0:
            return float('inf'), None
        else:
            validation_net = copy.deepcopy(self.net)
            currently_used_lr = 0.01
            for param_group in self.opti.param_groups:
                currently_used_lr = param_group['lr']
            # by default use SGD. Did not implement others
            if opti == 'SGD':
                validation_opti = optim.SGD(
                    validation_net.parameters(), lr=currently_used_lr)
            elif opti == "Adam":
                validation_opti = optim.Adam(
                    validation_net.parameters(), lr=currently_used_lr, weight_decay=1e-5
                )
            local_update_time = time.time()
            validation_net.train()
            for data, _ in self.train_dl:
                loss = self._ddpm_training_loss(validation_net, data)
                loss.backward()
                validation_opti.step()
                validation_opti.zero_grad()
            return (time.time() - local_update_time)/self.computation_power, validation_net.state_dict()

    def set_accuracy_this_round(self, accuracy):
        self.accuracy_this_round = accuracy

    def return_accuracy_this_round(self):
        return self.accuracy_this_round

    def return_link_speed(self):
        return self.link_speed

    def return_local_updates_and_signature(self, comm_round):
        # local_total_accumulated_epochs_this_round also stands for the lastest_epoch_seq for this transaction(local params are calculated after this amount of local epochs in this round)
        # last_local_iteration(s)_spent_time may be recorded to determine calculating time? But what if nodes do not wish to disclose its computation power
        local_updates_dict = {'worker_device_idx': self.idx, 'in_round_number': comm_round, "local_updates_params": copy.deepcopy(
            self.local_train_parameters), "local_updates_rewards": self.local_updates_rewards_per_transaction, "local_iteration(s)_spent_time": self.local_update_time, "local_training_time": self.local_training_time, "local_total_accumulated_epochs_this_round": self.local_total_epoch, "worker_rsa_pub_key": self.return_rsa_pub_key()}
        local_updates_dict["payload_bytes"] = payload_size_bytes(local_updates_dict)
        local_updates_dict["worker_signature"] = self.sign_msg(
            sorted(local_updates_dict.items()))
        return local_updates_dict

    def worker_reset_vars_for_new_round(self):
        self.received_block_from_miner = None
        self.accuracy_this_round = float('-inf')
        self.local_updates_rewards_per_transaction = 0
        self.has_added_block = False
        self.the_added_block = None
        self.worker_associated_validator = None
        self.worker_associated_miner = None
        self.local_update_time = None
        self.local_training_time = None
        self.local_total_epoch = 0
        self.local_train_parameters = None
        self.variance_of_noises.clear()
        self.round_end_time = 0

    def receive_block_from_miner(self, received_block, source_miner):
        if not (received_block.return_mined_by() in self.black_list or source_miner in self.black_list):
            self.received_block_from_miner = copy.deepcopy(received_block)
            print(f"{self.role} {self.idx} has received a new block from {source_miner} mined by {received_block.return_mined_by()}.")
        else:
            print(
                f"Either the block sending miner {source_miner} or the miner {received_block.return_mined_by()} mined this block is in worker {self.idx}'s black list. Block is not accepted.")

    def toss_received_block(self):
        self.received_block_from_miner = None

    def return_received_block_from_miner(self):
        return self.received_block_from_miner

    def validate_model_weights(self, weights_to_eval=None):
        original_parameters = copy.deepcopy(self.net.state_dict())
        was_training = self.net.training
        try:
            if weights_to_eval is None:
                self.net.load_state_dict(self.global_parameters, strict=True)
            else:
                self.net.load_state_dict(weights_to_eval, strict=True)
            return self._evaluate_ddpm_loss(self.net)
        finally:
            self.net.load_state_dict(original_parameters, strict=True)
            self.net.train(was_training)

    def global_update(self, local_update_params_potentially_to_be_used, respect_blacklist=True, update_round=None):
        # Accept the original (worker, params) records and the new optional
        # (worker, params, trust_weight) records.  This keeps plain FedAvg
        # behavior unchanged while allowing validation-guided weighting.
        local_params_by_benign_workers = []
        for update_record in local_update_params_potentially_to_be_used:
            if len(update_record) == 2:
                worker_device_idx, local_params = update_record
                aggregation_weight = 1.0
            elif len(update_record) == 3:
                worker_device_idx, local_params, aggregation_weight = update_record
                try:
                    aggregation_weight = max(float(aggregation_weight), 0.0)
                except (TypeError, ValueError):
                    aggregation_weight = 0.0
            else:
                raise ValueError("Each aggregation record must contain 2 or 3 values")
            if not respect_blacklist or worker_device_idx not in self.black_list:
                if local_params is not None and aggregation_weight > 0.0:
                    local_params_by_benign_workers.append((local_params, aggregation_weight))
            else:
                print(
                    f"global update skipped for a worker {worker_device_idx} in {self.idx}'s black list")
        if local_params_by_benign_workers:
            if update_round is not None and update_round <= self.last_applied_round:
                return False
            # avg the gradients
            sum_parameters = None
            total_weight = sum(weight for _, weight in local_params_by_benign_workers)
            for local_updates_params, aggregation_weight in local_params_by_benign_workers:
                if sum_parameters is None:
                    sum_parameters = copy.deepcopy(local_updates_params)
                    for var in sum_parameters:
                        if torch.is_floating_point(sum_parameters[var]):
                            sum_parameters[var] *= aggregation_weight
                else:
                    for var in sum_parameters:
                        if torch.is_floating_point(sum_parameters[var]):
                            sum_parameters[var] += local_updates_params[var] * aggregation_weight
            for var in self.global_parameters:
                if torch.is_floating_point(sum_parameters[var]):
                    self.global_parameters[var] = sum_parameters[var] / total_weight
                else:
                    # Integer/bool buffers are not part of the DDPM update,
                    # but retaining the first value avoids invalid division.
                    self.global_parameters[var] = sum_parameters[var]
            if update_round is None:
                self.global_version += 1
                self.last_applied_round = self.global_version
            else:
                self.global_version = max(self.global_version, update_round)
                self.last_applied_round = update_round
            print(f"global updates done by {self.idx} (global_version={self.global_version})")
            return True
        else:
            print(
                f"There are no available local params for {self.idx} to perform global updates in this comm round.")
            return False

    ''' miner '''

    def request_to_download(self, block_to_download, requesting_time_point):
        print(
            f"miner {self.idx} is requesting its associated devices to download the block it just added to its chain")
        devices_in_association = self.miner_associated_validator_set.union(
            self.miner_associated_worker_set)
        for device in devices_in_association:
            # theoratically, one device is associated to a specific miner, so we don't have a miner_block_arrival_queue here
            if self.online_switcher() and device.online_switcher():
                miner_link_speed = self.return_link_speed()
                device_link_speed = device.return_link_speed()
                lower_link_speed = device_link_speed if device_link_speed < miner_link_speed else miner_link_speed
                transmission_delay = payload_size_bytes(
                    block_to_download.__dict__)/lower_link_speed
                verified_block, verification_time = device.verify_block(
                    block_to_download, block_to_download.return_mined_by())
                if verified_block:
                    # forgot to check for maliciousness of the block miner
                    device.add_block(verified_block)
                device.add_to_round_end_time(
                    requesting_time_point + transmission_delay + verification_time)
            else:
                print(
                    f"Unfortunately, either miner {self.idx} or {device.return_idx()} goes offline while processing this request-to-download block.")

    def propagated_the_block(self, propagating_time_point, block_to_propagate):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"{self.role} {self.idx} is propagating its mined block to {peer.return_role()} {peer.return_idx()}.")
                        if peer.online_switcher():
                            peer.accept_the_propagated_block(
                                self, self.block_generation_time_point, block_to_propagate)
                    else:
                        print(
                            f"Destination miner {peer.return_idx()} is in {self.role} {self.idx}'s black_list. Propagating skipped for this dest miner.")

    def accept_the_propagated_block(self, source_miner, source_miner_propagating_time_point, propagated_block):
        if not source_miner.return_idx() in self.black_list:
            source_miner_link_speed = source_miner.return_link_speed()
            this_miner_link_speed = self.link_speed
            lower_link_speed = this_miner_link_speed if this_miner_link_speed < source_miner_link_speed else source_miner_link_speed
            transmission_delay = payload_size_bytes(
                propagated_block.__dict__)/lower_link_speed
            self.unordered_propagated_block_processing_queue[
                source_miner_propagating_time_point + transmission_delay] = propagated_block
            print(
                f"{self.role} {self.idx} has accepted accepted a propagated block from miner {source_miner.return_idx()}")
        else:
            print(
                f"Source miner {source_miner.return_role()} {source_miner.return_idx()} is in {self.role} {self.idx}'s black list. Propagated block not accepted.")

    def add_propagated_block_to_processing_queue(self, arrival_time, propagated_block):
        self.unordered_propagated_block_processing_queue[arrival_time] = propagated_block

    def return_unordered_propagated_block_processing_queue(self):
        return self.unordered_propagated_block_processing_queue

    def return_associated_validators(self):
        return self.miner_associated_validator_set

    def return_miner_acception_wait_time(self):
        return self.miner_acception_wait_time

    def return_miner_accepted_transactions_size_limit(self):
        return self.miner_accepted_transactions_size_limit

    def return_miners_eligible_to_continue(self):
        miners_set = set()
        for peer in self.peer_list:
            if peer.return_role() == 'miner':
                miners_set.add(peer)
        miners_set.add(self)
        return miners_set

    def return_accepted_broadcasted_transactions(self):
        return self.broadcasted_transactions

    def verify_validator_transaction(self, transaction_to_verify):
        if self.computation_power == 0:
            print(
                f"miner {self.idx} has computation power 0 and will not be able to verify this transaction in time")
            return False, None
        else:
            transaction_validator_idx = transaction_to_verify['validation_done_by']
            if transaction_validator_idx in self.black_list:
                print(
                    f"{transaction_validator_idx} is in miner's blacklist. Trasaction won't get verified.")
                return False, None
            verification_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(
                    transaction_to_verify)
                del transaction_before_signed["validator_signature"]
                modulus = transaction_to_verify['validator_rsa_pub_key']["modulus"]
                pub_key = transaction_to_verify['validator_rsa_pub_key']["pub_key"]
                signature = transaction_to_verify["validator_signature"]
                # begin verification
                hash = int.from_bytes(
                    stable_digest(sorted(transaction_before_signed.items())), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(
                        f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                    verification_time = (
                        time.time() - verification_time)/self.computation_power
                    return verification_time, True
                else:
                    print(
                        f"Signature invalid. Transaction from validator {transaction_validator_idx} is NOT verified.")
                    return (time.time() - verification_time)/self.computation_power, False
            else:
                print(
                    f"Signature of transaction from validator {transaction_validator_idx} is verified by {self.role} {self.idx}!")
                verification_time = (
                    time.time() - verification_time)/self.computation_power
                return verification_time, True

    def sign_candidate_transaction(self, candidate_transaction):
        signing_time = time.time()
        candidate_transaction['miner_rsa_pub_key'] = self.return_rsa_pub_key()
        if 'miner_signature' in candidate_transaction.keys():
            del candidate_transaction['miner_signature']
        candidate_transaction["miner_signature"] = self.sign_msg(
            sorted(candidate_transaction.items()))
        signing_time = (time.time() - signing_time)/self.computation_power
        return signing_time

    def mine_block(self, candidate_block, rewards, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        pow_mined_block = self.proof_of_work(candidate_block)
        # pow_mined_block.set_mined_by(self.idx)
        pow_mined_block.set_mining_rewards(rewards)
        return pow_mined_block

    def proof_of_work(self, candidate_block, starting_nonce=0):
        candidate_block.set_mined_by(self.idx)
        ''' Brute Force the nonce '''
        candidate_block.set_nonce(starting_nonce)
        current_hash = candidate_block.compute_hash()
        # candidate_block.set_pow_difficulty(self.pow_difficulty)
        while not current_hash.startswith('0' * self.pow_difficulty):
            candidate_block.nonce_increment()
            current_hash = candidate_block.compute_hash()
        # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
        # also set its hash as well. block_hash is the same as pow proof
        candidate_block.set_pow_proof(current_hash)
        return candidate_block

    def set_block_generation_time_point(self, block_generation_time_point):
        self.block_generation_time_point = block_generation_time_point

    def return_block_generation_time_point(self):
        return self.block_generation_time_point

    def receive_propagated_block(self, received_propagated_block):
        if not received_propagated_block.return_mined_by() in self.black_list:
            self.received_propagated_block = copy.deepcopy(
                received_propagated_block)
            print(
                f"Miner {self.idx} has received a propagated block from {received_propagated_block.return_mined_by()}.")
        else:
            print(
                f"Propagated block miner {received_propagated_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def receive_propagated_validator_block(self, received_propagated_validator_block):
        if not received_propagated_validator_block.return_mined_by() in self.black_list:
            self.received_propagated_validator_block = copy.deepcopy(
                received_propagated_validator_block)
            print(
                f"Miner {self.idx} has received a propagated validator block from {received_propagated_validator_block.return_mined_by()}.")
        else:
            print(
                f"Propagated validator block miner {received_propagated_validator_block.return_mined_by()} is in miner {self.idx}'s blacklist. Block not accepted.")

    def return_propagated_block(self):
        return self.received_propagated_block

    def return_propagated_validator_block(self):
        return self.received_propagated_validator_block

    def toss_propagated_block(self):
        self.received_propagated_block = None

    def toss_ropagated_validator_block(self):
        self.received_propagated_validator_block = None

    def miner_reset_vars_for_new_round(self):
        self.miner_associated_worker_set.clear()
        self.miner_associated_validator_set.clear()
        self.unconfirmmed_transactions.clear()
        self.broadcasted_transactions.clear()
        # self.unconfirmmed_validator_transactions.clear()
        # self.validator_accepted_broadcasted_worker_transactions.clear()
        self.mined_block = None
        self.received_propagated_block = None
        self.received_propagated_validator_block = None
        self.has_added_block = False
        self.the_added_block = None
        self.unordered_arrival_time_accepted_validator_transactions.clear()
        self.miner_accepted_broadcasted_validator_transactions.clear()
        self.block_generation_time_point = None
        # self.block_to_add = None
        self.unordered_propagated_block_processing_queue.clear()
        self.round_end_time = 0

    def set_unordered_arrival_time_accepted_validator_transactions(self, unordered_arrival_time_accepted_validator_transactions):
        self.unordered_arrival_time_accepted_validator_transactions = unordered_arrival_time_accepted_validator_transactions

    def return_unordered_arrival_time_accepted_validator_transactions(self):
        return self.unordered_arrival_time_accepted_validator_transactions

    def miner_broadcast_validator_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "miner":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"miner {self.idx} is broadcasting received validator transactions to miner {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner = copy.copy(
                            self.unordered_arrival_time_accepted_validator_transactions)
                        # offline situation similar in validator_broadcast_worker_transactions()
                        for arrival_time, tx in self.unordered_arrival_time_accepted_validator_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner[
                                    arrival_time]
                        peer.accept_miner_broadcasted_validator_transactions(
                            self, final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)
                        print(f"miner {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_validator_transactions_for_dest_miner)} validator transactions to miner {peer.return_idx()}.")
                    else:
                        print(
                            f"Destination miner {peer.return_idx()} is in miner {self.idx}'s black_list. broadcasting skipped for this dest miner.")

    def accept_miner_broadcasted_validator_transactions(self, source_device, unordered_transaction_arrival_queue_from_source_miner):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.miner_accepted_broadcasted_validator_transactions.append({'source_device_link_speed': source_device.return_link_speed(
            ), 'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_miner)})
            print(f"{self.role} {self.idx} has accepted validator transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(
                f"Source miner {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_validator_transactions(self):
        return self.miner_accepted_broadcasted_validator_transactions

    def set_candidate_transactions_for_final_mining_queue(self, final_transactions_arrival_queue):
        self.final_candidate_transactions_queue_to_mine = final_transactions_arrival_queue

    def return_final_candidate_transactions_mining_queue(self):
        return self.final_candidate_transactions_queue_to_mine

    ''' validator '''

    def validator_reset_vars_for_new_round(self):
        self.validation_rewards_this_round = 0
        # self.accuracies_this_round = {}
        self.has_added_block = False
        self.the_added_block = None
        self.validator_associated_miner = None
        self.validator_local_accuracy = None
        self.validator_associated_worker_set.clear()
        # self.post_validation_transactions.clear()
        # self.broadcasted_post_validation_transactions.clear()
        self.unordered_arrival_time_accepted_worker_transactions.clear()
        self.final_transactions_queue_to_validate.clear()
        self.validator_accepted_broadcasted_worker_transactions.clear()
        self.post_validation_transactions_queue.clear()
        self.round_end_time = 0

    def add_post_validation_transaction_to_queue(self, transaction_to_add):
        self.post_validation_transactions_queue.append(transaction_to_add)

    def return_post_validation_transactions_queue(self):
        return self.post_validation_transactions_queue

    def return_online_workers(self):
        online_workers_in_peer_list = set()
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "worker":
                    online_workers_in_peer_list.add(peer)
        return online_workers_in_peer_list

    def return_validations_and_signature(self, comm_round):
        validation_transaction_dict = {'validator_device_idx': self.idx, 'round_number': comm_round, 'accuracies_this_round': copy.deepcopy(
            self.accuracies_this_round), 'validation_effort_rewards': self.validation_rewards_this_round, "rsa_pub_key": self.return_rsa_pub_key()}
        validation_transaction_dict["signature"] = self.sign_msg(
            sorted(validation_transaction_dict.items()))
        return validation_transaction_dict

    def add_worker_to_association(self, worker_device):
        if not worker_device.return_idx() in self.black_list:
            self.associated_worker_set.add(worker_device)
        else:
            print(
                f"WARNING: {worker_device.return_idx()} in validator {self.idx}'s black list. Not added by the validator.")

    def associate_with_miner(self):
        miners_in_peer_list = set()
        for peer in self.peer_list:
            if peer.return_role() == "miner":
                if not peer.return_idx() in self.black_list:
                    miners_in_peer_list.add(peer)
        if not miners_in_peer_list:
            return False
        self.validator_associated_miner = random.sample(
            miners_in_peer_list, 1)[0]
        return self.validator_associated_miner

    ''' miner and validator '''

    def add_device_to_association(self, to_add_device):
        if not to_add_device.return_idx() in self.black_list:
            vars(self)[f'{self.role}_associated_{to_add_device.return_role()}_set'].add(
                to_add_device)
        else:
            print(
                f"WARNING: {to_add_device.return_idx()} in {self.role} {self.idx}'s black list. Not added by the {self.role}.")

    def return_associated_workers(self):
        return vars(self)[f'{self.role}_associated_worker_set']

    def sign_block(self, block_to_sign):
        block_to_sign.set_signature(self.sign_msg(block_to_sign.__dict__))

    def add_unconfirmmed_transaction(self, unconfirmmed_transaction, souce_device_idx):
        if not souce_device_idx in self.black_list:
            self.unconfirmmed_transactions.append(
                copy.deepcopy(unconfirmmed_transaction))
            print(
                f"{souce_device_idx}'s transaction has been recorded by {self.role} {self.idx}")
        else:
            print(
                f"Source device {souce_device_idx} is in the black list of {self.role} {self.idx}. Transaction has not been recorded.")

    def return_unconfirmmed_transactions(self):
        return self.unconfirmmed_transactions

    def broadcast_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == self.role:
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"{self.role} {self.idx} is broadcasting transactions to {peer.return_role()} {peer.return_idx()}.")
                        peer.accept_broadcasted_transactions(
                            self, self.unconfirmmed_transactions)
                    else:
                        print(
                            f"Destination {peer.return_role()} {peer.return_idx()} is in {self.role} {self.idx}'s black_list. broadcasting skipped.")

    def accept_broadcasted_transactions(self, source_device, broadcasted_transactions):
        # discard malicious node
        if not source_device.return_idx() in self.black_list:
            self.broadcasted_transactions.append(
                copy.deepcopy(broadcasted_transactions))
            print(f"{self.role} {self.idx} has accepted transactions from {source_device.return_role()} {source_device.return_idx()}")
        else:
            print(f"Source {source_device.return_role()} {source_device.return_idx()} is in {self.role} {self.idx}'s black list. Transaction not accepted.")

    ''' worker and validator '''

    def set_mined_block(self, mined_block):
        self.mined_block = mined_block

    def return_mined_block(self):
        return self.mined_block

    def associate_with_device(self, to_associate_device_role):
        to_associate_device = vars(
            self)[f'{self.role}_associated_{to_associate_device_role}']
        shuffled_peer_list = list(self.peer_list)
        random.shuffle(shuffled_peer_list)
        for peer in shuffled_peer_list:
            # select the first found eligible device from a shuffled order
            if peer.return_role() == to_associate_device_role and peer.is_online():
                if not peer.return_idx() in self.black_list:
                    to_associate_device = peer
        if not to_associate_device:
            # there is no device matching the required associated role in this device's peer list
            return False
        print(f"{self.role} {self.idx} associated with {to_associate_device.return_role()} {to_associate_device.return_idx()}")
        return to_associate_device

    ''' validator '''

    def set_unordered_arrival_time_accepted_worker_transactions(self, unordered_transaction_arrival_queue):
        self.unordered_arrival_time_accepted_worker_transactions = unordered_transaction_arrival_queue

    def return_unordered_arrival_time_accepted_worker_transactions(self):
        return self.unordered_arrival_time_accepted_worker_transactions

    def validator_broadcast_worker_transactions(self):
        for peer in self.peer_list:
            if peer.is_online():
                if peer.return_role() == "validator":
                    if not peer.return_idx() in self.black_list:
                        print(
                            f"validator {self.idx} is broadcasting received validator transactions to validator {peer.return_idx()}.")
                        final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator = copy.copy(
                            self.unordered_arrival_time_accepted_worker_transactions)
                        # if offline, it's like the broadcasted transaction was not received, so skip a transaction
                        for arrival_time, tx in self.unordered_arrival_time_accepted_worker_transactions.items():
                            if not (self.online_switcher() and peer.online_switcher()):
                                del final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator[
                                    arrival_time]
                        # in the real distributed system, it should be broadcasting transaction one by one. Here we send the all received transactions(while online) and later calculate the order for the individual broadcasting transaction's arrival time mixed with the transactions itself received
                        peer.accept_validator_broadcasted_worker_transactions(
                            self, final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)
                        print(
                            f"validator {self.idx} has broadcasted {len(final_broadcasting_unordered_arrival_time_accepted_worker_transactions_for_dest_validator)} worker transactions to validator {peer.return_idx()}.")
                    else:
                        print(
                            f"Destination validator {peer.return_idx()} is in this validator {self.idx}'s black_list. broadcasting skipped for this dest validator.")

    def accept_validator_broadcasted_worker_transactions(self, source_validator, unordered_transaction_arrival_queue_from_source_validator):
        if not source_validator.return_idx() in self.black_list:
            self.validator_accepted_broadcasted_worker_transactions.append({'source_validator_link_speed': source_validator.return_link_speed(
            ), 'broadcasted_transactions': copy.deepcopy(unordered_transaction_arrival_queue_from_source_validator)})
            print(
                f"validator {self.idx} has accepted worker transactions from validator {source_validator.return_idx()}")
        else:
            print(
                f"Source validator {source_validator.return_idx()} is in validator {self.idx}'s black list. Broadcasted transactions not accepted.")

    def return_accepted_broadcasted_worker_transactions(self):
        return self.validator_accepted_broadcasted_worker_transactions

    def set_transaction_for_final_validating_queue(self, final_transactions_arrival_queue):
        self.final_transactions_queue_to_validate = final_transactions_arrival_queue

    def return_final_transactions_validating_queue(self):
        return self.final_transactions_queue_to_validate

    def validator_update_model_by_one_epoch_and_validate_local_accuracy(self, opti, local_epochs=1):
        # Return the simulated time spent on the validator reference update.
        print(
            f"validator {self.idx} is performing {local_epochs} local epoch(s) "
            "of update and validation"
        )
        if self.computation_power == 0:
            print(f"validator {self.idx} has computation power 0 and will not be able to complete this validation")
            return float('inf')

        updated_net = copy.deepcopy(self.net)
        updated_net.load_state_dict(self.global_parameters, strict=True)
        currently_used_lr = self.learning_rate
        if opti == 'SGD':
            validation_opti = optim.SGD(updated_net.parameters(), lr=currently_used_lr)
        elif opti == 'Adam':
            validation_opti = optim.Adam(
                updated_net.parameters(), lr=currently_used_lr, weight_decay=1e-5
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opti}")

        local_validation_time = time.time()
        # Use the same update rule as workers so that the score is not dominated
        # by a difference in optimizer steps.
        training_start = time.time()
        for _ in range(local_epochs):
            self._run_local_epoch(updated_net, validation_opti)
        try:
            self.validator_training_time = (time.time() - training_start) / self.computation_power
        except ZeroDivisionError:
            self.validator_training_time = float('inf')

        evaluation_start = time.time()
        self.validator_local_accuracy = self._evaluate_ddpm_loss(updated_net)
        try:
            self.validator_evaluation_time = (time.time() - evaluation_start) / self.computation_power
        except ZeroDivisionError:
            self.validator_evaluation_time = float('inf')
        print(
            f"validator {self.idx} locally updated model has epsilon-prediction loss "
            f"{self.validator_local_accuracy} on its local test set"
        )
        return (time.time() - local_validation_time) / self.computation_power

    # TODO validator_threshold
    def validate_worker_transaction(self, transaction_to_validate, rewards, log_files_folder_path, comm_round, malicious_validator_on):
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if self.computation_power == 0:
            print(
                f"validator {self.idx} has computation power 0 and will not be able to validate this transaction in time")
            return False, False
        else:
            worker_transaction_device_idx = transaction_to_validate['worker_device_idx']
            if worker_transaction_device_idx in self.black_list:
                print(
                    f"{worker_transaction_device_idx} is in validator's blacklist. Trasaction won't get validated.")
                return False, False
            validation_time = time.time()
            if self.check_signature:
                transaction_before_signed = copy.deepcopy(
                    transaction_to_validate)
                del transaction_before_signed["worker_signature"]
                modulus = transaction_to_validate['worker_rsa_pub_key']["modulus"]
                pub_key = transaction_to_validate['worker_rsa_pub_key']["pub_key"]
                signature = transaction_to_validate["worker_signature"]
                # begin validation
                # 1 - verify signature
                hash = int.from_bytes(
                    stable_digest(sorted(transaction_before_signed.items())), byteorder='big')
                hashFromSignature = pow(signature, pub_key, modulus)
                if hash == hashFromSignature:
                    print(
                        f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                    transaction_to_validate['worker_signature_valid'] = True
                else:
                    print(
                        f"Signature invalid. Transaction from worker {worker_transaction_device_idx} does NOT pass verification.")
                    # will also add sig not verified transaction due to the validator's verification effort and its rewards needs to be recorded in the block
                    transaction_to_validate['worker_signature_valid'] = False
            else:
                print(
                    f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                transaction_to_validate['worker_signature_valid'] = True
            # 2 - validate worker's local_updates_params if worker's signature is valid
            if transaction_to_validate['worker_signature_valid']:
                # accuracy validated by worker's update
                score_evaluation_start = time.time()
                accuracy_by_worker_update_using_own_data = self.validate_model_weights(
                    transaction_to_validate["local_updates_params"])
                try:
                    score_evaluation_time = (time.time() - score_evaluation_start) / self.computation_power
                except ZeroDivisionError:
                    score_evaluation_time = float('inf')
                transaction_to_validate['score_evaluation_time'] = score_evaluation_time
                # if worker's accuracy larger, or lower but the difference falls within the validator threshold value, meaning worker's updated model favors validator's dataset, so their updates are in the same direction - True, otherwise False. We do not consider the accuracy gap so far, meaning if worker's update is way too good, it is still fine
                print(
                    f'validator updated model MSE - {self.validator_local_accuracy}')
                print(
                    f"After applying worker's update, model MSE becomes - {accuracy_by_worker_update_using_own_data}")
                # Record the signed score used by the threshold rule.
                validation_score = accuracy_by_worker_update_using_own_data - self.validator_local_accuracy
                normalized_score, aggregation_weight = self._compute_aggregation_metrics(validation_score)
                transaction_to_validate['normalized_score'] = normalized_score
                transaction_to_validate['aggregation_weight'] = aggregation_weight
                is_malicious_validator = "M" if self.is_malicious else "B"
                with open(f"{log_files_folder_path_comm_round}/validator_{self.idx}_{is_malicious_validator}_validation_records_comm_{comm_round}.txt", "a") as file:
                    is_malicious_node = "M" if self.devices_dict[worker_transaction_device_idx].return_is_malicious(
                    ) else "B"
                    file.write(f"{accuracy_by_worker_update_using_own_data - self.validator_local_accuracy}: validator {self.return_idx()} {is_malicious_validator} in round {comm_round} evluating worker {worker_transaction_device_idx}, diff = v_acc:{self.validator_local_accuracy} - w_acc:{accuracy_by_worker_update_using_own_data} {worker_transaction_device_idx}_maliciousness: {is_malicious_node}\n")

                #  区块链联邦学习的恶意节点检测
                if self.defense_on and validation_score > self.validator_threshold:
                # 没有区块链验证的联邦学习对恶意节点的恶意检测
                # cycle = 1
                # if cycle < 0:

                    transaction_to_validate['update_direction'] = False
                    print(
                        f"NOTE: worker {worker_transaction_device_idx}'s updates is deemed as suspiciously malicious by validator {self.idx}")
                    # is it right?
                    if not self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(
                            f"Warning - {worker_transaction_device_idx} is benign and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(
                        f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {self.idx}")
                    # is it right?
                    if self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(
                            f"Warning - {worker_transaction_device_idx} is malicious and this validation is wrong.")
                        # for experiments
                        with open(f"{log_files_folder_path}/false_positive_malious_nodes_inside_slipped.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_positive_good_nodes_inside_correct.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")

                if self.is_malicious and malicious_validator_on:
                    old_voting = transaction_to_validate['update_direction']
                    transaction_to_validate['update_direction'] = not transaction_to_validate['update_direction']
                    with open(f"{log_files_folder_path_comm_round}/malicious_validator_log.txt", 'a') as file:
                        file.write(
                            f"malicious validator {self.idx} has flipped the voting of worker {worker_transaction_device_idx} from {old_voting} to {transaction_to_validate['update_direction']} in round {comm_round}\n")
                transaction_to_validate['aggregation_eligible'] = self._aggregation_eligibility(
                    normalized_score,
                    transaction_to_validate['update_direction'],
                )
                with open(f"{log_files_folder_path}/validation_records.jsonl", "a", encoding="utf-8") as file:
                    file.write(json.dumps({
                        "round": comm_round,
                        "validator_id": self.return_idx(),
                        "validator_malicious": self.is_malicious,
                        "worker_id": worker_transaction_device_idx,
                        "worker_malicious": self.devices_dict[worker_transaction_device_idx].return_is_malicious(),
                        "validator_loss": float(self.validator_local_accuracy),
                        "worker_loss": float(accuracy_by_worker_update_using_own_data),
                        "score": float(validation_score),
                        "normalized_score": float(normalized_score),
                        "aggregation_weight": float(aggregation_weight),
                        "aggregation_method": self.aggregation_method,
                        "aggregation_eligible": bool(transaction_to_validate['aggregation_eligible']),
                        "threshold": float(self.validator_threshold),
                        "rejected_by_threshold": bool(validation_score > self.validator_threshold),
                        "accepted_for_aggregation": bool(transaction_to_validate["update_direction"]),
                    }, sort_keys=True) + "\n")
                transaction_to_validate['validation_rewards'] = rewards
            else:
                transaction_to_validate['update_direction'] = 'N/A'
                transaction_to_validate['validation_rewards'] = 0
            transaction_to_validate['validation_done_by'] = self.idx
            validation_time = (time.time() - validation_time) / \
                self.computation_power
            transaction_to_validate['validation_time'] = validation_time
            transaction_to_validate['validator_rsa_pub_key'] = self.return_rsa_pub_key(
            )
            # assume signing done in negligible time
            transaction_to_validate["validator_signature"] = self.sign_msg(
                sorted(transaction_to_validate.items()))
            return validation_time, transaction_to_validate


class DevicesInNetwork(object):  # 对标ClientsGroup
    def __init__(self, data_set_name, is_iid, batch_size, learning_rate, loss_func, opti, num_devices, network_stability, net, dev, knock_out_rounds, lazy_worker_knock_out_rounds, shard_test_data, miner_acception_wait_time, miner_accepted_transactions_size_limit, validator_threshold, pow_difficulty, even_link_speed_strength, base_data_transmission_speed, even_computation_power, malicious_updates_discount, num_malicious, noise_variance, check_signature, not_resync_chain, validation_seed, test_batch_size=256, defense_on=1, attack_type="gaussian_weight_noise"):
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
        self.attack_type = attack_type
        self.num_malicious = num_malicious
        self.malicious_updates_discount = malicious_updates_discount
        self.noise_variance = noise_variance
        self.check_signature = check_signature
        self.not_resync_chain = not_resync_chain
        # distribute dataset
        ''' validator '''
        self.validator_threshold = validator_threshold
        ''' miner '''
        self.miner_acception_wait_time = miner_acception_wait_time
        self.miner_accepted_transactions_size_limit = miner_accepted_transactions_size_limit
        self.pow_difficulty = pow_difficulty
        ''' shard '''
        self.data_set_balanced_allocation()

    def data_set_balanced_allocation(self):
        # read dataset
        cifar10_dataset = DatasetLoad('cifar10', self.is_iid)

        # perpare training data
        train_data = cifar10_dataset.train_data
        train_label = cifar10_dataset.train_label
        # shard dataset and distribute among devices
        # shard train
        shard_size_train = cifar10_dataset.train_data_size // self.num_devices // 2
        shards_id_train = np.random.permutation(
            cifar10_dataset.train_data_size // shard_size_train)

        # perpare test data
        if not self.shard_test_data:
            test_data = torch.tensor(cifar10_dataset.test_data)
            test_label = torch.argmax(torch.tensor(
                cifar10_dataset.test_label), dim=1)
            test_data_loader = DataLoader(TensorDataset(
                test_data, test_label), batch_size=self.test_batch_size, shuffle=False)
        else:
            test_data = cifar10_dataset.test_data
            test_label = cifar10_dataset.test_label
            # shard test
            shard_size_test = cifar10_dataset.test_data_size // self.num_devices // 2
            shards_id_test = np.random.permutation(
                cifar10_dataset.test_data_size // shard_size_test)

        malicious_nodes_set = []
        if self.num_malicious:
            malicious_nodes_set = random.sample(
                range(self.num_devices), self.num_malicious)

        for i in range(self.num_devices):
            is_malicious = False
            # make it more random by introducing two shards
            shards_id_train1 = shards_id_train[i * 2]
            shards_id_train2 = shards_id_train[i * 2 + 1]
            # distribute training data
            data_shards1 = train_data[shards_id_train1 *
                                      shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            data_shards2 = train_data[shards_id_train2 *
                                      shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            label_shards1 = train_label[shards_id_train1 *
                                        shard_size_train: shards_id_train1 * shard_size_train + shard_size_train]
            label_shards2 = train_label[shards_id_train2 *
                                        shard_size_train: shards_id_train2 * shard_size_train + shard_size_train]
            local_train_data, local_train_label = np.vstack(
                (data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_train_label = np.argmax(local_train_label, axis=1)
            # distribute test data
            if self.shard_test_data:
                shards_id_test1 = shards_id_test[i * 2]
                shards_id_test2 = shards_id_test[i * 2 + 1]
                data_shards1 = test_data[shards_id_test1 *
                                         shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                data_shards2 = test_data[shards_id_test2 *
                                         shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                label_shards1 = test_label[shards_id_test1 *
                                           shard_size_test: shards_id_test1 * shard_size_test + shard_size_test]
                label_shards2 = test_label[shards_id_test2 *
                                           shard_size_test: shards_id_test2 * shard_size_test + shard_size_test]
                local_test_data, local_test_label = np.vstack(
                    (data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
                local_test_label = torch.argmax(
                    torch.tensor(local_test_label), dim=1)
                test_data_loader = DataLoader(TensorDataset(torch.tensor(
                    local_test_data), torch.tensor(local_test_label)),
                    batch_size=self.test_batch_size, shuffle=False)
            # assign data to a device and put in the devices set
            if i in malicious_nodes_set:
                is_malicious = True
                # add Gussian Noise

            device_idx = f'device_{i + 1}'
            a_device = Device(device_idx,
                              TensorDataset(torch.tensor(local_train_data), torch.tensor(local_train_label)),
                              test_data_loader, self.batch_size, self.learning_rate, self.loss_func, self.opti,
                              self.default_network_stability, self.net, self.dev, self.miner_acception_wait_time,
                              self.miner_accepted_transactions_size_limit,
                              self.validator_threshold, self.pow_difficulty, self.even_link_speed_strength,
                              self.base_data_transmission_speed, self.even_computation_power, is_malicious,
                              self.noise_variance, self.check_signature, self.not_resync_chain,
							  self.malicious_updates_discount, self.knock_out_rounds,
							  self.lazy_worker_knock_out_rounds, self.validation_seed,
                              defense_on=self.defense_on,
                              attack_type=self.attack_type)
            # device index starts from 1
            self.devices_set[device_idx] = a_device
            print(f"Sharding dataset to {device_idx} done.")
        print(f"Sharding dataset done!")
