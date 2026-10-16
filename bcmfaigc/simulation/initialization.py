"""Fresh/resumed network initialization and explicitly owned run resources."""

import os
import sys
import json
import pickle
import sqlite3
from pathlib import Path
from hashlib import sha256
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from bcmfaigc.models import DDPM
from bcmfaigc.provenance import write_source_manifest
from bcmfaigc.federation import FederatedNetwork, FederatedNode
from bcmfaigc.federation.checkpoints import load_network_snapshot


@dataclass
class RunResources:
    devices_in_network: FederatedNetwork
    devices_list: list[FederatedNode]
    latest_round_num: int
    log_files_folder_path: str
    network_snapshot_save_path: str | None
    rewards: int
    workers_needed: int
    miners_needed: int
    validators_needed: int
    mining_consensus: str
    conn: sqlite3.Connection
    conn_cursor: sqlite3.Cursor


def initialize_run(
    args,
    date_time,
    log_files_folder_path,
    NETWORK_SNAPSHOTS_BASE_FOLDER,
    dev,
    entry_file,
    network_class,
    variant,
    command_line=None,
):
    network_snapshot_save_path = None
    latest_round_num = 0
    " If network_snapshot is specified, continue from left "
    if args["resume_path"]:
        if not args["save_network_snapshots"]:
            print(
                "NOTE: save_network_snapshots is set to 0. New network_snapshots won't be saved by conituing."
            )
        network_snapshot_save_path = (
            f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{args['resume_path']}"
        )
        latest_network_snapshot_file_name = sorted(
            [
                f
                for f in os.listdir(network_snapshot_save_path)
                if not f.startswith(".")
            ],
            key=lambda fn: int(fn.split("_")[-1]),
            reverse=True,
        )[0]
        print(
            f"Loading network snapshot from {args['resume_path']}/{latest_network_snapshot_file_name}"
        )
        print(
            "BE CAREFUL - loaded dev env must be the same as the current dev env, namely, cpu, gpu or gpu parallel"
        )
        latest_round_num = int(latest_network_snapshot_file_name.split("_")[-1])
        devices_in_network = load_network_snapshot(
            f"{network_snapshot_save_path}/{latest_network_snapshot_file_name}", variant
        )
        devices_list = list(devices_in_network.devices_set.values())
        log_files_folder_path = (
            f"../logs/blogs/{args['resume_path']}_{args['model_name']}"
        )
        with open(f"{log_files_folder_path}/time.txt", "w") as f:
            f.write("start time: " + date_time)
        args_used_file = f"{log_files_folder_path}/args_used.txt"
        file = open(args_used_file, "r")
        log_whole_text = file.read()
        lines_list = log_whole_text.split("\n")
        for line in lines_list:
            if line.startswith("--unit_reward"):
                rewards = int(line.split(" ")[-1])
            if line.startswith("--hard_assign"):
                roles_requirement = line.split(" ")[-1].split(",")
            if line.startswith("--pow_difficulty"):
                mining_consensus = "PoW" if int(line.split(" ")[-1]) else "PoS"
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1
    else:
        " SETTING UP FROM SCRATCH"
        " 从头开始设置"
        if not os.path.exists(log_files_folder_path):
            os.mkdir(log_files_folder_path)
        code_files = [
            Path(entry_file),
            Path(entry_file).with_name("Device.py"),
            Path(entry_file).with_name("DatasetLoad.py"),
            Path(entry_file).with_name("Block.py"),
            Path(entry_file).with_name("stable_hash.py"),
            Path(entry_file).with_name("trust_aggregation.py"),
            Path(entry_file).parent.parent / "models" / "Models.py",
        ]
        code_fingerprint = {
            path.name: sha256(path.read_bytes()).hexdigest() for path in code_files
        }
        with open(
            f"{log_files_folder_path}/code_fingerprint.json", "w", encoding="utf-8"
        ) as fingerprint_file:
            json.dump(code_fingerprint, fingerprint_file, indent=2, sort_keys=True)
        write_source_manifest(
            Path(log_files_folder_path) / "source_manifest.json",
            experiment_root=Path(entry_file).resolve().parent.parent,
        )
        with open(f"{log_files_folder_path}/args_used.txt", "w") as f:
            f.write("Command line arguments used -\n")
            f.write(" ".join(sys.argv[1:] if command_line is None else command_line))
            f.write("\n\nAll arguments used -\n")
            for arg_name, arg in args.items():
                f.write(f"\n--{arg_name} {arg}")
        if args["save_network_snapshots"]:
            network_snapshot_save_path = f"{NETWORK_SNAPSHOTS_BASE_FOLDER}/{date_time}"
            os.mkdir(network_snapshot_save_path)
        rewards = args["unit_reward"]
        roles_requirement = args["hard_assign"].split(",")
        try:
            workers_needed = int(roles_requirement[0])
        except:
            workers_needed = 1
        try:
            validators_needed = int(roles_requirement[1])
        except:
            validators_needed = 1
        try:
            miners_needed = int(roles_requirement[2])
        except:
            miners_needed = 1
        num_devices = args["num_devices"]
        num_malicious = args["num_malicious"]
        if num_devices < workers_needed + miners_needed + validators_needed:
            sys.exit(
                "ERROR: Roles assigned to the devices exceed the maximum number of allowed devices in the network."
            )
        if num_devices < 3:
            sys.exit(
                "ERROR: There are not enough devices in the network.\n The system needs at least one miner, one worker and/or one validator to start the operation.\nSystem aborted."
            )
        if num_malicious:
            if num_malicious > num_devices:
                sys.exit(
                    "ERROR: The number of malicious nodes cannot exceed the total number of devices set in this network"
                )
            else:
                print(
                    f"Malicious nodes vs total devices set to {num_malicious}/{num_devices} = {num_malicious / num_devices * 100:.2f}%"
                )
        net = None
        if args["model_name"] == "ddpm":
            net = DDPM()
        else:
            print("model name not found")
            exit()
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        print(f"{torch.cuda.device_count()} GPUs are available to use!")
        net = net.to(dev)
        loss_func = F.mse_loss
        devices_in_network = network_class(
            data_set_name="cifar10",
            is_iid=args["IID"],
            batch_size=args["batchsize"],
            learning_rate=args["learning_rate"],
            loss_func=loss_func,
            opti=args["optimizer"],
            num_devices=num_devices,
            network_stability=args["network_stability"],
            net=net,
            dev=dev,
            knock_out_rounds=args["knock_out_rounds"],
            lazy_worker_knock_out_rounds=args["lazy_worker_knock_out_rounds"],
            shard_test_data=args["shard_test_data"],
            miner_acception_wait_time=args["miner_acception_wait_time"],
            miner_accepted_transactions_size_limit=args[
                "miner_accepted_transactions_size_limit"
            ],
            validator_threshold=args["validator_threshold"],
            pow_difficulty=args["pow_difficulty"],
            even_link_speed_strength=args["even_link_speed_strength"],
            base_data_transmission_speed=args["base_data_transmission_speed"],
            even_computation_power=args["even_computation_power"],
            malicious_updates_discount=args["malicious_updates_discount"],
            num_malicious=num_malicious,
            noise_variance=args["noise_variance"],
            check_signature=args["check_signature"],
            not_resync_chain=args["destroy_tx_in_block"],
            validation_seed=args["validation_seed"],
            test_batch_size=args["test_batchsize"],
            defense_on=args["defense_on"],
            **{"attack_type": args["attack_type"]} if variant.attack_dispatch else {},
        )
        devices_list = list(devices_in_network.devices_set.values())
        del net
        for device in devices_list:
            device.init_global_parameters()
            device.set_devices_dict_and_aio(
                devices_in_network.devices_set, args["all_in_one"]
            )
            device.register_in_the_network()
        for device in devices_list:
            device.remove_peers(device)
        open(f"{log_files_folder_path}/correctly_kicked_workers.txt", "w").close()
        open(f"{log_files_folder_path}/mistakenly_kicked_workers.txt", "w").close()
        open(
            f"{log_files_folder_path}/false_positive_malicious_nodes_inside_slipped.txt",
            "w",
        ).close()
        open(
            f"{log_files_folder_path}/false_negative_good_nodes_inside_victims.txt", "w"
        ).close()
        open(f"{log_files_folder_path}/kicked_lazy_workers.txt", "w").close()
        mining_consensus = "PoW" if args["pow_difficulty"] else "PoS"
    for device in devices_list:
        if variant.attack_dispatch:
            device.attack_type = args["attack_type"]
        device.set_aggregation_config(
            aggregation_method=args["aggregation_method"],
            trust_center=args["trust_center"],
            trust_scale=args["trust_scale"],
            trust_soft_threshold=args["trust_soft_threshold"],
            trust_beta=args["trust_beta"],
            trust_min_weight=args["trust_min_weight"],
            trust_hard_reject_score=args["trust_hard_reject_score"],
            **{"multikrum_f": args["multikrum_f"]} if variant.multikrum else {},
        )
    conn = sqlite3.connect(
        f"{log_files_folder_path}/malicious_worker_identifying_log.db"
    )
    conn_cursor = conn.cursor()
    conn_cursor.execute(
        "CREATE TABLE if not exists  malicious_workers_log (\n\t\tdevice_seq text,\n\t\tif_malicious integer,\n\t\tcorrectly_identified_by text,\n\t\tincorrectly_identified_by text,\n\t\tin_round integer,\n\t\twhen_resyncing text\n\t\t)"
    )
    return RunResources(
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
    )
