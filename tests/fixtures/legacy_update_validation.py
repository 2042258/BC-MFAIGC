"""Frozen original update-validation methods; characterization only, never runtime."""
import copy
import json
import time
from torch import optim
from bcmfaigc.blockchain.serialization import stable_digest
from bcmfaigc.aggregation.trust_weighting import compute_validation_trust as compute_trust_metrics


class LegacyUpdateValidation:
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

    def _aggregation_eligibility(self, normalized_score, update_direction):
        """Separate security voting from soft aggregation eligibility."""
        if self.aggregation_method == "trust_weighted":
            try:
                return float(normalized_score) <= self.trust_hard_reject_score
            except (TypeError, ValueError):
                return False
        return bool(update_direction)

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

    def validator_update_model_by_one_epoch_and_validate_local_accuracy(self, opti, local_epochs=1):
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
                    transaction_to_validate['worker_signature_valid'] = False
            else:
                print(
                    f"Signature of transaction from worker {worker_transaction_device_idx} is verified by validator {self.idx}!")
                transaction_to_validate['worker_signature_valid'] = True
            if transaction_to_validate['worker_signature_valid']:
                score_evaluation_start = time.time()
                accuracy_by_worker_update_using_own_data = self.validate_model_weights(
                    transaction_to_validate["local_updates_params"])
                try:
                    score_evaluation_time = (time.time() - score_evaluation_start) / self.computation_power
                except ZeroDivisionError:
                    score_evaluation_time = float('inf')
                transaction_to_validate['score_evaluation_time'] = score_evaluation_time
                print(
                    f'validator updated model MSE - {self.validator_local_accuracy}')
                print(
                    f"After applying worker's update, model MSE becomes - {accuracy_by_worker_update_using_own_data}")
                validation_score = accuracy_by_worker_update_using_own_data - self.validator_local_accuracy
                normalized_score, aggregation_weight = self._compute_aggregation_metrics(validation_score)
                transaction_to_validate['normalized_score'] = normalized_score
                transaction_to_validate['aggregation_weight'] = aggregation_weight
                is_malicious_validator = "M" if self.is_malicious else "B"
                with open(f"{log_files_folder_path_comm_round}/validator_{self.idx}_{is_malicious_validator}_validation_records_comm_{comm_round}.txt", "a") as file:
                    is_malicious_node = "M" if self.devices_dict[worker_transaction_device_idx].return_is_malicious(
                    ) else "B"
                    file.write(f"{accuracy_by_worker_update_using_own_data - self.validator_local_accuracy}: validator {self.return_idx()} {is_malicious_validator} in round {comm_round} evluating worker {worker_transaction_device_idx}, diff = v_acc:{self.validator_local_accuracy} - w_acc:{accuracy_by_worker_update_using_own_data} {worker_transaction_device_idx}_maliciousness: {is_malicious_node}\n")

                if self.defense_on and validation_score > self.validator_threshold:

                    transaction_to_validate['update_direction'] = False
                    print(
                        f"NOTE: worker {worker_transaction_device_idx}'s updates is deemed as suspiciously malicious by validator {self.idx}")
                    if not self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(
                            f"Warning - {worker_transaction_device_idx} is benign and this validation is wrong.")
                        with open(f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                    else:
                        with open(f"{log_files_folder_path}/true_negative_malicious_nodes_inside_caught.txt", 'a') as file:
                            file.write(f"{self.validator_local_accuracy - accuracy_by_worker_update_using_own_data} = current_validator_accuracy {self.validator_local_accuracy} - accuracy_by_worker_update_using_own_data {accuracy_by_worker_update_using_own_data} , by validator {self.idx} on worker {worker_transaction_device_idx} in round {comm_round}\n")
                else:
                    transaction_to_validate['update_direction'] = True
                    print(
                        f"worker {worker_transaction_device_idx}'s' updates is deemed as GOOD by validator {self.idx}")
                    if self.devices_dict[worker_transaction_device_idx].return_is_malicious():
                        print(
                            f"Warning - {worker_transaction_device_idx} is malicious and this validation is wrong.")
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
            transaction_to_validate["validator_signature"] = self.sign_msg(
                sorted(transaction_to_validate.items()))
            return validation_time, transaction_to_validate
