"""Shared round scheduling; loop-carried legacy names retain one function scope."""

import os
import json
import random
import time
import copy
import pickle
import shutil
from pathlib import Path
import torch
from bcmfaigc.blockchain import Block
from bcmfaigc.blockchain.serialization import stable_digest, payload_size_bytes
from .reports import (
    write_protocol_event,
    write_global_checkpoint,
    save_network_snapshot,
)


def run_rounds(
    args,
    dev,
    devices_in_network,
    devices_list,
    latest_round_num,
    log_files_folder_path,
    network_snapshot_save_path,
    rewards,
    workers_needed,
    miners_needed,
    validators_needed,
    mining_consensus,
    conn,
    conn_cursor,
    variant,
):
    protocol_replay_history = []
    for comm_round in range(latest_round_num + 1, args["max_num_comm"] + 1):
        random.seed(args["seed"] + 104729 * comm_round)
        devices_list.sort(key=lambda device: device.return_idx())
        log_files_folder_path_comm_round = f"{log_files_folder_path}/comm_{comm_round}"
        if os.path.exists(log_files_folder_path_comm_round):
            print(f"Deleting {log_files_folder_path_comm_round} and create a new one.")
            shutil.rmtree(log_files_folder_path_comm_round)
        os.mkdir(log_files_folder_path_comm_round)
        if dev == torch.device("cuda"):
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
        print(f"\nCommunication round {comm_round}")
        comm_round_start_time = time.time()
        protocol_attack_injected = False
        protocol_attack_enabled = (
            args["protocol_attack"] != "none"
            and comm_round >= args["protocol_attack_start_round"]
            and (
                (comm_round - args["protocol_attack_start_round"])
                % max(args["protocol_attack_every"], 1)
                == 0
            )
        )
        protocol_attack_transaction = None
        workers_to_assign = workers_needed
        miners_to_assign = miners_needed
        validators_to_assign = validators_needed
        workers_this_round = []
        miners_this_round = []
        validators_this_round = []
        random.shuffle(devices_list)
        for device in devices_list:
            if workers_to_assign:
                device.assign_worker_role()
                workers_to_assign -= 1
            elif miners_to_assign:
                device.assign_miner_role()
                miners_to_assign -= 1
            elif validators_to_assign:
                device.assign_validator_role()
                validators_to_assign -= 1
            else:
                device.assign_role()
            if device.return_role() == "worker":
                workers_this_round.append(device)
            elif device.return_role() == "miner":
                miners_this_round.append(device)
            else:
                validators_this_round.append(device)
            device.online_switcher()
        with open(
            f"{log_files_folder_path}/round_metadata.jsonl", "a", encoding="utf-8"
        ) as metadata_file:
            for device in devices_list:
                metadata_file.write(
                    json.dumps(
                        {
                            "round": comm_round,
                            "device_id": device.return_idx(),
                            "role": device.return_role(),
                            "online_at_round_start": device.is_online(),
                            "global_version_at_round_start": device.global_version,
                            "malicious": device.return_is_malicious(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        " DEBUGGING CODE "
        if args["verbose"]:
            for device in devices_list:
                if device.is_online():
                    print(
                        f"{device.return_idx()} {device.return_role()} online - ",
                        end="",
                    )
                else:
                    print(
                        f"{device.return_idx()} {device.return_role()} offline - ",
                        end="",
                    )
                print(
                    f"chain length {device.return_blockchain_object().return_chain_length()}"
                )
            print(
                f"\nThere are {len(workers_this_round)} workers, {len(miners_this_round)} miners and {len(validators_this_round)} validators in this round."
            )
            print("\nworkers this round are")
            for worker in workers_this_round:
                print(
                    f"d_{worker.return_idx().split('_')[-1]} online - {worker.is_online()} with chain len {worker.return_blockchain_object().return_chain_length()}"
                )
            print("\nminers this round are")
            for miner in miners_this_round:
                print(
                    f"d_{miner.return_idx().split('_')[-1]} online - {miner.is_online()} with chain len {miner.return_blockchain_object().return_chain_length()}"
                )
            print("\nvalidators this round are")
            for validator in validators_this_round:
                print(
                    f"d_{validator.return_idx().split('_')[-1]} online - {validator.is_online()} with chain len {validator.return_blockchain_object().return_chain_length()}"
                )
            print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
            for device_seq, device in devices_in_network.devices_set.items():
                peers = device.return_peers()
                print(
                    f"d_{device_seq.split('_')[-1]} - {device.return_role()[0]} has peer list ",
                    end="",
                )
                for peer in peers:
                    print(
                        f"d_{peer.return_idx().split('_')[-1]} - {peer.return_role()[0]}",
                        end=", ",
                    )
                print()
            print(f"+++++++++ Round {comm_round} Beginning Peer Lists +++++++++")
        " DEBUGGING CODE ENDS "
        for miner in miners_this_round:
            miner.miner_reset_vars_for_new_round()
        for worker in workers_this_round:
            worker.worker_reset_vars_for_new_round()
        for validator in validators_this_round:
            validator.validator_reset_vars_for_new_round()
        random.shuffle(workers_this_round)
        random.shuffle(miners_this_round)
        random.shuffle(validators_this_round)
        " workers, validators and miners take turns to perform jobs "
        " 工人、验证者和矿工轮流执行工作 "
        plain_fedavg = not args["blockchain_on"] and (not args["defense_on"])
        print(
            " Step 1 - workers assign associated miner and validator (and do local updates, but it is implemented in code block of step 2) \n"
        )
        " Step 1 - worker分配关联的miner和validator（并进行本地更新，但它是在step 2的代码块中实现的） \n"
        for worker_iter in range(len(workers_this_round)):
            worker = workers_this_round[worker_iter]
            if worker.resync_chain(mining_consensus):
                worker.update_model_after_chain_resync(
                    log_files_folder_path_comm_round, conn, conn_cursor
                )
            print(
                f"{worker.return_idx()} - worker {worker_iter + 1}/{len(workers_this_round)} will associate with a validator and a miner, if online..."
            )
            if not plain_fedavg and worker.online_switcher():
                associated_miner = worker.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(worker)
                else:
                    print(
                        f"Cannot find a qualified miner in {worker.return_idx()} peer list."
                    )
            if not plain_fedavg and worker.online_switcher():
                associated_validator = worker.associate_with_device("validator")
                if associated_validator:
                    associated_validator.add_device_to_association(worker)
                else:
                    print(
                        f"Cannot find a qualified validator in {worker.return_idx()} peer list."
                    )
        print(
            " Step 2 - validators accept local updates and broadcast to other validators in their respective peer lists (workers local_updates() are called in this step.\n"
        )
        " Step 2 - validators接受本地更新和广播给其他同等节点列表中的validators（worker的local_updates()在这个步骤中被调用 "
        if plain_fedavg:
            for worker in workers_this_round:
                if worker.is_online():
                    worker.worker_local_update(
                        rewards,
                        log_files_folder_path_comm_round,
                        comm_round,
                        local_epochs=args["default_local_epochs"],
                    )
        for validator_iter in (
            range(len(validators_this_round)) if not plain_fedavg else []
        ):
            validator = validators_this_round[validator_iter]
            if validator.resync_chain(mining_consensus):
                validator.update_model_after_chain_resync(
                    log_files_folder_path, conn, conn_cursor
                )
            if validator.online_switcher():
                associated_miner = validator.associate_with_device("miner")
                if associated_miner:
                    associated_miner.add_device_to_association(validator)
                else:
                    print(
                        f"Cannot find a qualified miner in validator {validator.return_idx()} peer list."
                    )
            associated_workers = list(validator.return_associated_workers())
            if not associated_workers:
                print(
                    f"No workers are associated with validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} for this communication round."
                )
                continue
            validator_link_speed = validator.return_link_speed()
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is accepting workers' updates with link speed {validator_link_speed} bytes/s, if online..."
            )
            records_dict = dict.fromkeys(associated_workers, None)
            for worker, _ in records_dict.items():
                records_dict[worker] = {}
            transaction_arrival_queue = {}
            if args["miner_acception_wait_time"]:
                print(
                    f"miner wait time is specified as {args['miner_acception_wait_time']} seconds. let each worker do local_updates till time limit"
                )
                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if not worker.return_idx() in validator.return_black_list():
                        print(
                            f"worker {worker_iter + 1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates"
                        )
                        total_time_tracker = 0
                        update_iter = 1
                        worker_link_speed = worker.return_link_speed()
                        lower_link_speed = (
                            validator_link_speed
                            if validator_link_speed < worker_link_speed
                            else worker_link_speed
                        )
                        while (
                            total_time_tracker
                            < validator.return_miner_acception_wait_time()
                        ):
                            if worker.online_switcher():
                                local_update_spent_time = worker.worker_local_update(
                                    rewards,
                                    log_files_folder_path_comm_round,
                                    comm_round,
                                )
                                unverified_transaction = (
                                    worker.return_local_updates_and_signature(
                                        comm_round
                                    )
                                )
                                unverified_transactions_size = payload_size_bytes(
                                    unverified_transaction
                                )
                                transmission_delay = (
                                    unverified_transactions_size / lower_link_speed
                                )
                                if (
                                    local_update_spent_time + transmission_delay
                                    > validator.return_miner_acception_wait_time()
                                ):
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter][
                                    "local_update_time"
                                ] = local_update_spent_time
                                records_dict[worker][update_iter][
                                    "transmission_delay"
                                ] = transmission_delay
                                records_dict[worker][update_iter][
                                    "local_update_unverified_transaction"
                                ] = unverified_transaction
                                records_dict[worker][update_iter][
                                    "local_update_unverified_transaction_size"
                                ] = unverified_transactions_size
                                if update_iter == 1:
                                    total_time_tracker = (
                                        local_update_spent_time + transmission_delay
                                    )
                                else:
                                    total_time_tracker = (
                                        total_time_tracker
                                        - records_dict[worker][update_iter - 1][
                                            "transmission_delay"
                                        ]
                                        + local_update_spent_time
                                        + transmission_delay
                                    )
                                records_dict[worker][update_iter]["arrival_time"] = (
                                    total_time_tracker
                                )
                                if validator.online_switcher():
                                    print(
                                        f"validator {validator.return_idx()} has accepted this transaction."
                                    )
                                    transaction_arrival_queue[total_time_tracker] = (
                                        unverified_transaction
                                    )
                                else:
                                    print(
                                        f"validator {validator.return_idx()} offline and unable to accept this transaction"
                                    )
                            else:
                                wasted_update_time, wasted_update_params = (
                                    worker.waste_one_epoch_local_update_time(
                                        args["optimizer"]
                                    )
                                )
                                wasted_update_params_size = payload_size_bytes(
                                    wasted_update_params
                                )
                                wasted_transmission_delay = (
                                    wasted_update_params_size / lower_link_speed
                                )
                                if (
                                    wasted_update_time + wasted_transmission_delay
                                    > validator.return_miner_acception_wait_time()
                                ):
                                    break
                                records_dict[worker][update_iter] = {}
                                records_dict[worker][update_iter][
                                    "transmission_delay"
                                ] = transmission_delay
                                if update_iter == 1:
                                    total_time_tracker = (
                                        wasted_update_time + wasted_transmission_delay
                                    )
                                    print(
                                        f"worker goes offline and wasted {total_time_tracker} seconds for a transaction"
                                    )
                                else:
                                    total_time_tracker = (
                                        total_time_tracker
                                        - records_dict[worker][update_iter - 1][
                                            "transmission_delay"
                                        ]
                                        + wasted_update_time
                                        + wasted_transmission_delay
                                    )
                            update_iter += 1
            else:
                for worker_iter in range(len(associated_workers)):
                    worker = associated_workers[worker_iter]
                    if not worker.return_idx() in validator.return_black_list():
                        print(
                            f"worker {worker_iter + 1}/{len(associated_workers)} of validator {validator.return_idx()} is doing local updates"
                        )
                        if worker.online_switcher():
                            local_update_spent_time = worker.worker_local_update(
                                rewards,
                                log_files_folder_path_comm_round,
                                comm_round,
                                local_epochs=args["default_local_epochs"],
                            )
                            worker_link_speed = worker.return_link_speed()
                            lower_link_speed = (
                                validator_link_speed
                                if validator_link_speed < worker_link_speed
                                else worker_link_speed
                            )
                            unverified_transaction = (
                                worker.return_local_updates_and_signature(comm_round)
                            )
                            unverified_transactions_size = payload_size_bytes(
                                unverified_transaction
                            )
                            transmission_delay = (
                                unverified_transactions_size / lower_link_speed
                            )
                            if validator.online_switcher():
                                transaction_arrival_queue[
                                    local_update_spent_time + transmission_delay
                                ] = unverified_transaction
                                print(
                                    f"validator {validator.return_idx()} has accepted this transaction."
                                )
                            else:
                                print(
                                    f"validator {validator.return_idx()} offline and unable to accept this transaction"
                                )
                        else:
                            print(
                                f"worker {worker.return_idx()} offline and unable do local updates"
                            )
                    else:
                        print(
                            f"worker {worker.return_idx()} in validator {validator.return_idx()}'s black list. This worker's transactions won't be accpeted."
                        )
            validator.set_unordered_arrival_time_accepted_worker_transactions(
                transaction_arrival_queue
            )
            validator.set_transaction_for_final_validating_queue(
                sorted(transaction_arrival_queue.items())
            )
            if transaction_arrival_queue:
                validator.validator_broadcast_worker_transactions()
            else:
                print(
                    "No transactions have been received by this validator, probably due to workers and/or validators offline or timeout while doing local updates or transmitting updates, or all workers are in validator's black list."
                )
        plain_fedavg = not args["blockchain_on"] and (not args["defense_on"])
        if plain_fedavg:
            local_updates = [
                (worker.return_idx(), worker.local_train_parameters)
                for worker in workers_this_round
                if worker.local_train_parameters is not None and worker.is_online()
            ]
            for device in devices_list:
                if variant.multikrum and args["aggregation_method"] in (
                    "multikrum",
                    "krum",
                ):
                    device.apply_multikrum(
                        local_updates, log_files_folder_path, comm_round
                    )
                else:
                    device.apply_plain_fedavg(
                        local_updates, log_files_folder_path, comm_round
                    )
        print(
            " Step 2.5 - with the broadcasted workers transactions, validators decide the final transaction arrival order \n"
        )
        " Step 2.5 - 通过广播的worker交易，validators 决定最终的交易到达顺序 "
        for validator_iter in (
            range(len(validators_this_round)) if not plain_fedavg else []
        ):
            validator = validators_this_round[validator_iter]
            accepted_broadcasted_validator_transactions = (
                validator.return_accepted_broadcasted_worker_transactions()
            )
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions..."
            )
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_validator_transactions:
                self_validator_link_speed = validator.return_link_speed()
                for (
                    broadcasting_validator_record
                ) in accepted_broadcasted_validator_transactions:
                    broadcasting_validator_link_speed = broadcasting_validator_record[
                        "source_validator_link_speed"
                    ]
                    lower_link_speed = (
                        self_validator_link_speed
                        if self_validator_link_speed < broadcasting_validator_link_speed
                        else broadcasting_validator_link_speed
                    )
                    for (
                        arrival_time_at_broadcasting_validator,
                        broadcasted_transaction,
                    ) in broadcasting_validator_record[
                        "broadcasted_transactions"
                    ].items():
                        transmission_delay = (
                            payload_size_bytes(broadcasted_transaction)
                            / lower_link_speed
                        )
                        accepted_broadcasted_transactions_arrival_queue[
                            transmission_delay + arrival_time_at_broadcasting_validator
                        ] = broadcasted_transaction
            else:
                print(
                    f"validator {validator.return_idx()} {validator_iter + 1}/{len(validators_this_round)} did not receive any broadcasted worker transaction this round."
                )
            final_transactions_arrival_queue = sorted(
                {
                    **validator.return_unordered_arrival_time_accepted_worker_transactions(),
                    **accepted_broadcasted_transactions_arrival_queue,
                }.items()
            )
            validator.set_transaction_for_final_validating_queue(
                final_transactions_arrival_queue
            )
            print(
                f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions."
            )
        print(
            " Step 3 - validators do self and cross-validation(validate local updates from workers) by the order of transaction arrival time.\n"
        )
        " validators按交易到达时间的顺序进行自我和交叉验证（验证来自workers的本地更新）。 "
        validator_reference_update_times = {}
        for validator_iter in (
            range(len(validators_this_round)) if not plain_fedavg else []
        ):
            validator = validators_this_round[validator_iter]
            final_transactions_arrival_queue = (
                validator.return_final_transactions_validating_queue()
            )
            if final_transactions_arrival_queue:
                local_validation_time = validator.validator_update_model_by_one_epoch_and_validate_local_accuracy(
                    args["optimizer"], args["default_local_epochs"]
                )
                validator_reference_update_times[validator.return_idx()] = (
                    local_validation_time
                )
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} is validating received worker transactions..."
                )
                for (
                    arrival_time,
                    unconfirmmed_transaction,
                ) in final_transactions_arrival_queue:
                    if validator.online_switcher():
                        if arrival_time < local_validation_time:
                            arrival_time = local_validation_time
                        validation_time, post_validation_unconfirmmed_transaction = (
                            validator.validate_worker_transaction(
                                unconfirmmed_transaction,
                                rewards,
                                log_files_folder_path,
                                comm_round,
                                args["malicious_validator_on"],
                            )
                        )
                        if validation_time:
                            validator.add_post_validation_transaction_to_queue(
                                (
                                    arrival_time + validation_time,
                                    validator.return_link_speed(),
                                    post_validation_unconfirmmed_transaction,
                                )
                            )
                            print(
                                f"A validation process has been done for the transaction from worker {post_validation_unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()}"
                            )
                    else:
                        print(
                            f"A validation process is skipped for the transaction from worker {unconfirmmed_transaction['worker_device_idx']} by validator {validator.return_idx()} due to validator offline."
                        )
            else:
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(validators_this_round)} did not receive any transaction from worker or validator in this round."
                )
        validated_transaction_records_this_round = [
            (arrival_time, link_speed, transaction, validator.return_idx())
            for validator in validators_this_round
            for arrival_time, link_speed, transaction in validator.return_post_validation_transactions_queue()
            if isinstance(transaction, dict)
        ]
        validated_transactions_this_round = [
            transaction
            for _, _, transaction, _ in validated_transaction_records_this_round
        ]
        current_round_transaction_for_history = (
            copy.deepcopy(validated_transactions_this_round[0])
            if validated_transactions_this_round
            else None
        )
        if protocol_attack_enabled and (not protocol_attack_injected):
            if (
                args["protocol_attack"] == "validator_tamper"
                and validated_transactions_this_round
            ):
                attack_transaction = copy.deepcopy(validated_transactions_this_round[0])
                old_weight = float(attack_transaction.get("aggregation_weight", 1.0))
                attack_transaction["aggregation_weight"] = max(0.05, old_weight * 0.25)
                validated_transactions_this_round.append(attack_transaction)
                protocol_attack_transaction = attack_transaction
                write_protocol_event(
                    log_files_folder_path,
                    {
                        "attack": "validator_tamper",
                        "phase": "injected_after_validator_signature",
                        "round": comm_round,
                        "worker_id": attack_transaction.get("worker_device_idx"),
                        "validator_id": attack_transaction.get("validation_done_by"),
                        "original_weight": old_weight,
                        "tampered_weight": attack_transaction["aggregation_weight"],
                    },
                )
            elif args["protocol_attack"] == "replay" and protocol_replay_history:
                attack_transaction = copy.deepcopy(protocol_replay_history[-1])
                validated_transactions_this_round.append(attack_transaction)
                protocol_attack_transaction = attack_transaction
                write_protocol_event(
                    log_files_folder_path,
                    {
                        "attack": "replay",
                        "phase": "injected_previous_round_transaction",
                        "round": comm_round,
                        "original_round": attack_transaction.get("in_round_number"),
                        "worker_id": attack_transaction.get("worker_device_idx"),
                        "validator_id": attack_transaction.get("validation_done_by"),
                    },
                )
        if current_round_transaction_for_history is not None:
            protocol_replay_history.append(current_round_transaction_for_history)
            if len(protocol_replay_history) > 8:
                protocol_replay_history.pop(0)
        with open(
            f"{log_files_folder_path}/timing_records.jsonl", "a", encoding="utf-8"
        ) as timing_file:
            timing_file.write(
                json.dumps(
                    {
                        "round": comm_round,
                        "blockchain_on": int(args["blockchain_on"]),
                        "defense_on": int(args["defense_on"]),
                        "workers": [
                            worker.return_idx() for worker in workers_this_round
                        ],
                        "validators": [
                            validator.return_idx()
                            for validator in validators_this_round
                        ],
                        "miners": [miner.return_idx() for miner in miners_this_round],
                        "worker_local_update_times": {
                            worker.return_idx(): worker.local_update_time
                            for worker in workers_this_round
                        },
                        "worker_local_training_times": {
                            worker.return_idx(): worker.local_training_time
                            for worker in workers_this_round
                        },
                        "worker_local_epochs": {
                            worker.return_idx(): worker.local_total_epoch
                            for worker in workers_this_round
                        },
                        "worker_training_samples": {
                            worker.return_idx(): len(worker.train_ds)
                            * worker.local_total_epoch
                            for worker in workers_this_round
                        },
                        "validator_reference_update_times": validator_reference_update_times,
                        "validator_reference_training_times": {
                            validator.return_idx(): validator.validator_training_time
                            for validator in validators_this_round
                        },
                        "validator_reference_evaluation_times": {
                            validator.return_idx(): validator.validator_evaluation_time
                            for validator in validators_this_round
                        },
                        "validator_reference_losses": {
                            validator.return_idx(): validator.validator_local_accuracy
                            for validator in validators_this_round
                        },
                        "transactions": [
                            {
                                "worker_id": transaction.get("worker_device_idx"),
                                "validator_id": validator_id,
                                "validation_time": transaction.get("validation_time"),
                                "score_evaluation_time": transaction.get(
                                    "score_evaluation_time"
                                ),
                                "payload_bytes": transaction.get("payload_bytes"),
                                "modelled_arrival_time": arrival_time,
                                "link_speed_bytes_per_second": link_speed,
                                "modelled_payload_transmission_seconds": float(
                                    transaction.get("payload_bytes", 0.0)
                                )
                                / float(link_speed)
                                if link_speed
                                else None,
                            }
                            for arrival_time, link_speed, transaction, validator_id in validated_transaction_records_this_round
                        ],
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        if not args["blockchain_on"] and args["defense_on"]:
            aggregation_start = time.time()
            for device in devices_list:
                if device.is_online():
                    device.apply_direct_aggregation(
                        validated_transactions_this_round,
                        log_files_folder_path,
                        comm_round,
                        conn,
                        conn_cursor,
                    )
            print(
                f"direct aggregation without blockchain finished in {time.time() - aggregation_start} seconds"
            )
        print(
            " Step 4 - validators send post validation transactions to associated miner and miner broadcasts these to other miners in their respecitve peer lists\n"
        )
        " Step 4 - validators发送验证后交易给相关联的miner，并且miner广播这些交易给他同等节点的miner "
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not args["blockchain_on"]:
                continue
            if miner.resync_chain(mining_consensus):
                miner.update_model_after_chain_resync(
                    log_files_folder_path, conn, conn_cursor
                )
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} accepting validators' post-validation transactions..."
            )
            associated_validators = list(miner.return_associated_validators())
            if not associated_validators:
                print(
                    f"No validators are associated with miner {miner.return_idx()} for this communication round."
                )
                continue
            self_miner_link_speed = miner.return_link_speed()
            validator_transactions_arrival_queue = {}
            for validator_iter in range(len(associated_validators)):
                validator = associated_validators[validator_iter]
                print(
                    f"{validator.return_idx()} - validator {validator_iter + 1}/{len(associated_validators)} of miner {miner.return_idx()} is sending signature verified transaction..."
                )
                post_validation_transactions_by_validator = (
                    validator.return_post_validation_transactions_queue()
                )
                post_validation_unconfirmmed_transaction_iter = 1
                for (
                    validator_sending_time,
                    source_validator_link_spped,
                    post_validation_unconfirmmed_transaction,
                ) in post_validation_transactions_by_validator:
                    if validator.online_switcher() and miner.online_switcher():
                        lower_link_speed = (
                            self_miner_link_speed
                            if self_miner_link_speed < source_validator_link_spped
                            else source_validator_link_spped
                        )
                        transmission_delay = (
                            payload_size_bytes(post_validation_unconfirmmed_transaction)
                            / lower_link_speed
                        )
                        validator_transactions_arrival_queue[
                            validator_sending_time + transmission_delay
                        ] = post_validation_unconfirmmed_transaction
                        print(
                            f"miner {miner.return_idx()} has accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()}"
                        )
                    else:
                        print(
                            f"miner {miner.return_idx()} has not accepted {post_validation_unconfirmmed_transaction_iter}/{len(post_validation_transactions_by_validator)} post-validation transaction from validator {validator.return_idx()} due to one of devices or both offline."
                        )
                    post_validation_unconfirmmed_transaction_iter += 1
            if (
                protocol_attack_enabled
                and (not protocol_attack_injected)
                and (protocol_attack_transaction is not None)
                and (args["protocol_attack"] in {"validator_tamper", "replay"})
                and validator_transactions_arrival_queue
            ):
                latest_arrival_time = max(validator_transactions_arrival_queue)
                validator_transactions_arrival_queue[latest_arrival_time + 1e-06] = (
                    protocol_attack_transaction
                )
                protocol_attack_injected = True
            miner.set_unordered_arrival_time_accepted_validator_transactions(
                validator_transactions_arrival_queue
            )
            miner.miner_broadcast_validator_transactions()
        print(
            " Step 4.5 - with the broadcasted validator transactions, miners decide the final transaction arrival order\n "
        )
        " Step 4.5 - 随着广播validator交易，miners决定最终的到达顺序 "
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not args["blockchain_on"]:
                continue
            accepted_broadcasted_validator_transactions = (
                miner.return_accepted_broadcasted_validator_transactions()
            )
            self_miner_link_speed = miner.return_link_speed()
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} calculating the final transactions arrival order by combining the direct worker transactions received and received broadcasted transactions..."
            )
            accepted_broadcasted_transactions_arrival_queue = {}
            if accepted_broadcasted_validator_transactions:
                for (
                    broadcasting_miner_record
                ) in accepted_broadcasted_validator_transactions:
                    broadcasting_miner_link_speed = broadcasting_miner_record[
                        "source_device_link_speed"
                    ]
                    lower_link_speed = (
                        self_miner_link_speed
                        if self_miner_link_speed < broadcasting_miner_link_speed
                        else broadcasting_miner_link_speed
                    )
                    for (
                        arrival_time_at_broadcasting_miner,
                        broadcasted_transaction,
                    ) in broadcasting_miner_record["broadcasted_transactions"].items():
                        transmission_delay = (
                            payload_size_bytes(broadcasted_transaction)
                            / lower_link_speed
                        )
                        accepted_broadcasted_transactions_arrival_queue[
                            transmission_delay + arrival_time_at_broadcasting_miner
                        ] = broadcasted_transaction
            else:
                print(
                    f"miner {miner.return_idx()} {miner_iter + 1}/{len(miners_this_round)} did not receive any broadcasted validator transaction this round."
                )
            final_transactions_arrival_queue = sorted(
                {
                    **miner.return_unordered_arrival_time_accepted_validator_transactions(),
                    **accepted_broadcasted_transactions_arrival_queue,
                }.items()
            )
            miner.set_candidate_transactions_for_final_mining_queue(
                final_transactions_arrival_queue
            )
            print(
                f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} done calculating the ordered final transactions arrival order. Total {len(final_transactions_arrival_queue)} accepted transactions."
            )
        print(
            " Step 5 - miners do self and cross-verification (verify validators' signature) by the order of transaction arrival time and record the transactions in the candidate block according to the limit size. Also mine and propagate the block.\n"
        )
        " miners按照交易到达时间的顺序进行自我和交叉验证（验证validator的签名），并根据限制大小将交易记录在候选区块中，还挖掘并传播块。 "
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not args["blockchain_on"]:
                continue
            final_transactions_arrival_queue = (
                miner.return_final_candidate_transactions_mining_queue()
            )
            valid_validator_sig_candidate_transacitons = []
            invalid_validator_sig_candidate_transacitons = []
            begin_mining_time = 0
            new_begin_mining_time = begin_mining_time
            if final_transactions_arrival_queue:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is verifying received validator transactions..."
                )
                time_limit = miner.return_miner_acception_wait_time()
                size_limit = miner.return_miner_accepted_transactions_size_limit()
                for (
                    arrival_time,
                    unconfirmmed_transaction,
                ) in final_transactions_arrival_queue:
                    if miner.online_switcher():
                        if time_limit:
                            if arrival_time > time_limit:
                                break
                        if size_limit:
                            if (
                                payload_size_bytes(
                                    valid_validator_sig_candidate_transacitons
                                    + invalid_validator_sig_candidate_transacitons
                                )
                                > size_limit
                            ):
                                break
                        transaction_round = unconfirmmed_transaction.get(
                            "in_round_number"
                        )
                        if (
                            transaction_round is not None
                            and int(transaction_round) != comm_round
                        ):
                            print(
                                f"Stale validator transaction from round {transaction_round} rejected in round {comm_round}."
                            )
                            if (
                                protocol_attack_enabled
                                and args["protocol_attack"] == "replay"
                            ):
                                write_protocol_event(
                                    log_files_folder_path,
                                    {
                                        "attack": "replay",
                                        "phase": "rejected_stale_transaction_round",
                                        "round": comm_round,
                                        "original_round": transaction_round,
                                        "worker_id": unconfirmmed_transaction.get(
                                            "worker_device_idx"
                                        ),
                                        "validator_id": unconfirmmed_transaction.get(
                                            "validation_done_by"
                                        ),
                                    },
                                )
                            continue
                        verification_time, is_validator_sig_valid = (
                            miner.verify_validator_transaction(unconfirmmed_transaction)
                        )
                        if verification_time:
                            if is_validator_sig_valid:
                                validator_info_this_tx = {
                                    "validator": unconfirmmed_transaction[
                                        "validation_done_by"
                                    ],
                                    "validation_rewards": unconfirmmed_transaction[
                                        "validation_rewards"
                                    ],
                                    "validation_time": unconfirmmed_transaction[
                                        "validation_time"
                                    ],
                                    "validator_rsa_pub_key": unconfirmmed_transaction[
                                        "validator_rsa_pub_key"
                                    ],
                                    "validator_signature": unconfirmmed_transaction[
                                        "validator_signature"
                                    ],
                                    "update_direction": unconfirmmed_transaction[
                                        "update_direction"
                                    ],
                                    "normalized_score": unconfirmmed_transaction.get(
                                        "normalized_score"
                                    ),
                                    "aggregation_weight": unconfirmmed_transaction.get(
                                        "aggregation_weight", 1.0
                                    ),
                                    "aggregation_eligible": unconfirmmed_transaction.get(
                                        "aggregation_eligible",
                                        unconfirmmed_transaction.get(
                                            "update_direction", False
                                        ),
                                    ),
                                    "miner_device_idx": miner.return_idx(),
                                    "miner_verification_time": verification_time,
                                    "miner_rewards_for_this_tx": rewards,
                                }
                                found_same_worker_transaction = False
                                for (
                                    valid_validator_sig_candidate_transaciton
                                ) in valid_validator_sig_candidate_transacitons:
                                    if (
                                        valid_validator_sig_candidate_transaciton[
                                            "worker_signature"
                                        ]
                                        == unconfirmmed_transaction["worker_signature"]
                                    ):
                                        found_same_worker_transaction = True
                                        break
                                if not found_same_worker_transaction:
                                    valid_validator_sig_candidate_transaciton = (
                                        copy.deepcopy(unconfirmmed_transaction)
                                    )
                                    del valid_validator_sig_candidate_transaciton[
                                        "validation_done_by"
                                    ]
                                    del valid_validator_sig_candidate_transaciton[
                                        "validation_rewards"
                                    ]
                                    del valid_validator_sig_candidate_transaciton[
                                        "update_direction"
                                    ]
                                    del valid_validator_sig_candidate_transaciton[
                                        "validation_time"
                                    ]
                                    del valid_validator_sig_candidate_transaciton[
                                        "validator_rsa_pub_key"
                                    ]
                                    del valid_validator_sig_candidate_transaciton[
                                        "validator_signature"
                                    ]
                                    valid_validator_sig_candidate_transaciton[
                                        "positive_direction_validators"
                                    ] = []
                                    valid_validator_sig_candidate_transaciton[
                                        "negative_direction_validators"
                                    ] = []
                                    valid_validator_sig_candidate_transaciton[
                                        "aggregation_validators"
                                    ] = []
                                    valid_validator_sig_candidate_transacitons.append(
                                        valid_validator_sig_candidate_transaciton
                                    )
                                if unconfirmmed_transaction["update_direction"]:
                                    valid_validator_sig_candidate_transaciton[
                                        "positive_direction_validators"
                                    ].append(validator_info_this_tx)
                                else:
                                    valid_validator_sig_candidate_transaciton[
                                        "negative_direction_validators"
                                    ].append(validator_info_this_tx)
                                if validator_info_this_tx["aggregation_eligible"]:
                                    valid_validator_sig_candidate_transaciton[
                                        "aggregation_validators"
                                    ].append(validator_info_this_tx)
                                transaction_to_sign = (
                                    valid_validator_sig_candidate_transaciton
                                )
                            else:
                                if variant.rejection_event_logs and (
                                    protocol_attack_enabled
                                    and args["protocol_attack"] == "validator_tamper"
                                    and (protocol_attack_transaction is not None)
                                    and (
                                        unconfirmmed_transaction
                                        is protocol_attack_transaction
                                        or (
                                            unconfirmmed_transaction.get(
                                                "worker_device_idx"
                                            )
                                            == protocol_attack_transaction.get(
                                                "worker_device_idx"
                                            )
                                            and unconfirmmed_transaction.get(
                                                "validation_done_by"
                                            )
                                            == protocol_attack_transaction.get(
                                                "validation_done_by"
                                            )
                                            and (
                                                unconfirmmed_transaction.get(
                                                    "aggregation_weight"
                                                )
                                                == protocol_attack_transaction.get(
                                                    "aggregation_weight"
                                                )
                                            )
                                        )
                                    )
                                ):
                                    write_protocol_event(
                                        log_files_folder_path,
                                        {
                                            "attack": "validator_tamper",
                                            "phase": "rejected_invalid_validator_signature",
                                            "round": comm_round,
                                            "observer_miner_id": miner.return_idx(),
                                            "worker_id": unconfirmmed_transaction.get(
                                                "worker_device_idx"
                                            ),
                                            "validator_id": unconfirmmed_transaction.get(
                                                "validation_done_by"
                                            ),
                                            "reason": "validator_signature_verification_failed",
                                        },
                                    )
                                invalid_validator_sig_candidate_transaciton = (
                                    copy.deepcopy(unconfirmmed_transaction)
                                )
                                invalid_validator_sig_candidate_transaciton[
                                    "validator"
                                ] = unconfirmmed_transaction["validation_done_by"]
                                invalid_validator_sig_candidate_transaciton[
                                    "miner_device_idx"
                                ] = miner.return_idx()
                                invalid_validator_sig_candidate_transaciton[
                                    "miner_verification_time"
                                ] = verification_time
                                invalid_validator_sig_candidate_transaciton[
                                    "miner_rewards_for_this_tx"
                                ] = rewards
                                invalid_validator_sig_candidate_transacitons.append(
                                    invalid_validator_sig_candidate_transaciton
                                )
                                transaction_to_sign = (
                                    invalid_validator_sig_candidate_transaciton
                                )
                            signing_time = miner.sign_candidate_transaction(
                                transaction_to_sign
                            )
                            new_begining_mining_time = (
                                arrival_time + verification_time + signing_time
                            )
                    else:
                        print(
                            f"A verification process is skipped for the transaction from validator {unconfirmmed_transaction['validation_done_by']} by miner {miner.return_idx()} due to miner offline."
                        )
                        new_begining_mining_time = arrival_time
                    begin_mining_time = (
                        new_begining_mining_time
                        if new_begining_mining_time > begin_mining_time
                        else begin_mining_time
                    )
                transactions_to_record_in_block = {}
                transactions_to_record_in_block["valid_validator_sig_transacitons"] = (
                    valid_validator_sig_candidate_transacitons
                )
                transactions_to_record_in_block[
                    "invalid_validator_sig_transacitons"
                ] = invalid_validator_sig_candidate_transacitons
                start_time_point = time.time()
                candidate_block = Block(
                    idx=miner.return_blockchain_object().return_chain_length() + 1,
                    transactions=transactions_to_record_in_block,
                    miner_rsa_pub_key=miner.return_rsa_pub_key(),
                )
                miner_computation_power = miner.return_computation_power()
                if not miner_computation_power:
                    block_generation_time_spent = float("inf")
                    miner.set_block_generation_time_point(float("inf"))
                    print(
                        f"{miner.return_idx()} - miner mines a block in INFINITE time..."
                    )
                    continue
                recorded_transactions = candidate_block.return_transactions()
                if (
                    recorded_transactions["valid_validator_sig_transacitons"]
                    or recorded_transactions["valid_validator_sig_transacitons"]
                ):
                    print(
                        f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} mining the block..."
                    )
                    last_block = miner.return_blockchain_object().return_last_block()
                    if last_block is None:
                        candidate_block.set_previous_block_hash(None)
                    else:
                        candidate_block.set_previous_block_hash(
                            last_block.compute_hash(hash_entire_block=True)
                        )
                    mined_block = miner.mine_block(candidate_block, rewards)
                else:
                    print("No transaction to mine for this block.")
                    continue
                if miner.online_switcher():
                    miner.sign_block(mined_block)
                    if (
                        protocol_attack_enabled
                        and (not protocol_attack_injected)
                        and (args["protocol_attack"] == "block_tamper")
                    ):
                        old_previous_hash = mined_block.return_previous_block_hash()
                        mined_block.set_previous_block_hash(
                            f"tampered::{old_previous_hash}"
                        )
                        write_protocol_event(
                            log_files_folder_path,
                            {
                                "attack": "block_tamper",
                                "phase": "injected_after_block_signature",
                                "round": comm_round,
                                "miner_id": miner.return_idx(),
                                "original_previous_hash": old_previous_hash,
                                "tampered_previous_hash": mined_block.return_previous_block_hash(),
                            },
                        )
                        protocol_attack_injected = True
                    miner.set_mined_block(mined_block)
                    block_generation_time_spent = (
                        time.time() - start_time_point
                    ) / miner_computation_power
                    miner.set_block_generation_time_point(
                        begin_mining_time + block_generation_time_spent
                    )
                    print(
                        f"{miner.return_idx()} - miner mines a block in {block_generation_time_spent} seconds."
                    )
                    miner.propagated_the_block(
                        miner.return_block_generation_time_point(), mined_block
                    )
                else:
                    print(
                        f"Unfortunately, {miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} goes offline after, if successful, mining a block. This if-successful-mined block is not propagated."
                    )
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} did not receive any transaction from validator or miner in this round."
                )
        print(
            " Step 6 - miners decide if adding a propagated block or its own mined block as the legitimate block, and request its associated devices to download this block"
        )
        " miner决定是否添加传播区块或其自己的开采区块作为合法区块，并要求其相关设备下载此区块 "
        forking_happened = False
        comm_round_block_gen_time = []
        for miner_iter in range(len(miners_this_round)):
            miner = miners_this_round[miner_iter]
            if not args["blockchain_on"]:
                continue
            unordered_propagated_block_processing_queue = (
                miner.return_unordered_propagated_block_processing_queue()
            )
            this_miner_mined_block = miner.return_mined_block()
            if this_miner_mined_block:
                unordered_propagated_block_processing_queue[
                    miner.return_block_generation_time_point()
                ] = this_miner_mined_block
            ordered_all_blocks_processing_queue = sorted(
                unordered_propagated_block_processing_queue.items()
            )
            if ordered_all_blocks_processing_queue:
                if mining_consensus == "PoW":
                    print("\nselect winning block based on PoW")
                    print(
                        f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} is deciding if a valid propagated block arrived before it successfully mines its own block..."
                    )
                    for (
                        block_arrival_time,
                        block_to_verify,
                    ) in ordered_all_blocks_processing_queue:
                        verified_block, verification_time = miner.verify_block(
                            block_to_verify, block_to_verify.return_mined_by()
                        )
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(
                                    f"Miner {miner.return_idx()} is adding its own mined block."
                                )
                            else:
                                print(
                                    f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()}."
                                )
                            if miner.online_switcher():
                                miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain."
                                )
                            if miner.return_the_added_block():
                                miner.request_to_download(
                                    verified_block,
                                    block_arrival_time + verification_time,
                                )
                                break
                        elif variant.rejection_event_logs and (
                            protocol_attack_enabled
                            and protocol_attack_injected
                            and (args["protocol_attack"] == "block_tamper")
                        ):
                            write_protocol_event(
                                log_files_folder_path,
                                {
                                    "attack": "block_tamper",
                                    "phase": "rejected_tampered_block",
                                    "round": comm_round,
                                    "checker_id": miner.return_idx(),
                                    "miner_id": block_to_verify.return_mined_by(),
                                    "reason": "block_verification_failed",
                                },
                            )
                else:
                    candidate_PoS_blocks = {}
                    print("select winning block based on PoS")
                    for (
                        block_arrival_time,
                        block_to_verify,
                    ) in ordered_all_blocks_processing_queue:
                        if (
                            block_arrival_time
                            < args["miner_pos_propagated_block_wait_time"]
                        ):
                            candidate_PoS_blocks[
                                devices_in_network.devices_set[
                                    block_to_verify.return_mined_by()
                                ].return_stake()
                            ] = block_to_verify
                    high_to_low_stake_ordered_blocks = sorted(
                        candidate_PoS_blocks.items(), reverse=True
                    )
                    for stake, PoS_candidate_block in high_to_low_stake_ordered_blocks:
                        verified_block, verification_time = miner.verify_block(
                            PoS_candidate_block, PoS_candidate_block.return_mined_by()
                        )
                        if verified_block:
                            block_mined_by = verified_block.return_mined_by()
                            if block_mined_by == miner.return_idx():
                                print(
                                    f"Miner {miner.return_idx()} with stake {stake} is adding its own mined block."
                                )
                            else:
                                print(
                                    f"Miner {miner.return_idx()} will add a propagated block mined by miner {verified_block.return_mined_by()} with stake {stake}."
                                )
                            if miner.online_switcher():
                                miner.add_block(verified_block)
                            else:
                                print(
                                    f"Unfortunately, miner {miner.return_idx()} goes offline while adding this block to its chain."
                                )
                            if miner.return_the_added_block():
                                miner.request_to_download(
                                    verified_block,
                                    block_arrival_time + verification_time,
                                )
                                break
                        elif variant.rejection_event_logs and (
                            protocol_attack_enabled
                            and protocol_attack_injected
                            and (args["protocol_attack"] == "block_tamper")
                        ):
                            write_protocol_event(
                                log_files_folder_path,
                                {
                                    "attack": "block_tamper",
                                    "phase": "rejected_tampered_block",
                                    "round": comm_round,
                                    "checker_id": miner.return_idx(),
                                    "miner_id": PoS_candidate_block.return_mined_by(),
                                    "reason": "block_verification_failed",
                                },
                            )
                miner.add_to_round_end_time(block_arrival_time + verification_time)
            else:
                print(
                    f"{miner.return_idx()} - miner {miner_iter + 1}/{len(miners_this_round)} does not receive a propagated block and has not mined its own block yet."
                )
        added_blocks_miner_set = set()
        for device in devices_list:
            the_added_block = device.return_the_added_block()
            if the_added_block:
                print(
                    f"{device.return_role()} {device.return_idx()} has added a block mined by {the_added_block.return_mined_by()}"
                )
                added_blocks_miner_set.add(the_added_block.return_mined_by())
                block_generation_time_point = devices_in_network.devices_set[
                    the_added_block.return_mined_by()
                ].return_block_generation_time_point()
                comm_round_block_gen_time.append(block_generation_time_point)
        if len(added_blocks_miner_set) > 1:
            print("WARNING: a forking event just happened!")
            forking_happened = True
            with open(
                f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", "a"
            ) as file:
                file.write(f"Forking in round {comm_round}\n")
        else:
            print("No forking event happened.")
        print(
            " Step 6 last step - process the added block - 1.collect usable updated params\n 2.malicious nodes identification\n 3.get rewards\n 4.do local udpates\n This code block is skipped if no valid block was generated in this round"
        )
        all_devices_round_ends_time = []
        for device in devices_list:
            if device.return_the_added_block() and device.online_switcher():
                processing_time = device.process_block(
                    device.return_the_added_block(),
                    log_files_folder_path,
                    conn,
                    conn_cursor,
                )
                device.other_tasks_at_the_end_of_comm_round(
                    comm_round, log_files_folder_path
                )
                device.add_to_round_end_time(processing_time)
                all_devices_round_ends_time.append(device.return_round_end_time())
        print(" Logging Accuracies by Devices ")
        for device in devices_list:
            accuracy_this_round = device.validate_model_weights()
            with open(
                f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt",
                "a",
            ) as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {accuracy_this_round}\n"
                )
        comm_round_spent_time = time.time() - comm_round_start_time
        with open(
            f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt", "a"
        ) as file:
            try:
                comm_round_block_gen_time = max(comm_round_block_gen_time)
                file.write(f"comm_round_block_gen_time: {comm_round_block_gen_time}\n")
            except:
                no_block_msg = "No valid block has been generated this round."
                print(no_block_msg)
                file.write(f"comm_round_block_gen_time: {no_block_msg}\n")
                with open(
                    f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", "a"
                ) as file2:
                    file2.write(f"No valid block in round {comm_round}\n")
            try:
                if all_devices_round_ends_time:
                    slowest_round_ends_time = max(all_devices_round_ends_time)
                else:
                    slowest_round_ends_time = 0.0
                file.write(
                    f"slowest_device_round_ends_time: {slowest_round_ends_time}\n"
                )
            except:
                file.write(
                    "slowest_device_round_ends_time: No valid block has been generated this round.\n"
                )
                with open(
                    f"{log_files_folder_path}/forking_and_no_valid_block_log.txt", "a"
                ) as file2:
                    no_valid_block_msg = f"No valid block in round {comm_round}\n"
                    if file2.readlines()[-1] != no_valid_block_msg:
                        file2.write(no_valid_block_msg)
            file.write(
                f"mining_consensus: {mining_consensus} {args['pow_difficulty']}\n"
            )
            file.write(f"forking_happened: {forking_happened}\n")
            file.write(
                f"comm_round_spent_time_on_this_machine: {comm_round_spent_time}\n"
            )
        conn.commit()
        with open(
            f"{log_files_folder_path}/timing_records.jsonl", "a", encoding="utf-8"
        ) as timing_file:
            block_generation_seconds = (
                float(comm_round_block_gen_time)
                if isinstance(comm_round_block_gen_time, (int, float))
                else None
            )
            timing_file.write(
                json.dumps(
                    {
                        "round": comm_round,
                        "phase": "round_summary",
                        "blockchain_on": int(args["blockchain_on"]),
                        "defense_on": int(args["defense_on"]),
                        "wall_clock_seconds": comm_round_spent_time,
                        "block_generation_seconds": block_generation_seconds,
                        "slowest_device_round_end_seconds": locals().get(
                            "slowest_round_ends_time"
                        ),
                        "forking_happened": bool(forking_happened),
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        if not forking_happened:
            legitimate_block = None
            for device in devices_list:
                legitimate_block = device.return_the_added_block()
                if legitimate_block is not None:
                    break
            with open(
                f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt",
                "a",
            ) as file:
                if legitimate_block is None:
                    file.write("block_mined_by: no valid block generated this round\n")
                else:
                    block_mined_by = legitimate_block.return_mined_by()
                    is_malicious_node = (
                        "M"
                        if devices_in_network.devices_set[
                            block_mined_by
                        ].return_is_malicious()
                        else "B"
                    )
                    file.write(
                        f"block_mined_by: {block_mined_by} {is_malicious_node}\n"
                    )
        else:
            with open(
                f"{log_files_folder_path_comm_round}/accuracy_comm_{comm_round}.txt",
                "a",
            ) as file:
                file.write(f"block_mined_by: Forking happened\n")
        print(" Logging Stake by Devices ")
        for device in devices_list:
            with open(
                f"{log_files_folder_path_comm_round}/stake_comm_{comm_round}.txt", "a"
            ) as file:
                is_malicious_node = "M" if device.return_is_malicious() else "B"
                file.write(
                    f"{device.return_idx()} {device.return_role()} {is_malicious_node}: {device.return_stake()}\n"
                )
        if args["destroy_tx_in_block"]:
            for device in devices_list:
                last_block = device.return_blockchain_object().return_last_block()
                if last_block:
                    last_block.free_tx()
        write_global_checkpoint(args, devices_list, log_files_folder_path, comm_round)
        save_network_snapshot(
            args, devices_in_network, network_snapshot_save_path, comm_round
        )
