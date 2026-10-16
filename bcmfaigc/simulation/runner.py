"""Single simulation entrypoint with separate CLI, initialization and round owners."""

import os
import random
from datetime import datetime
import numpy as np
import torch
from bcmfaigc.variants import EXTENDED
from bcmfaigc.federation.compat import ClassicNetwork, ExtendedNetwork, KrumNetwork
from .cli import build_parser, validate_arguments
from .initialization import initialize_run
from .rounds import run_rounds


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(variant=EXTENDED, argv=None, *, entry_file):
    print("Simulation run 1")
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files_folder_path = f"../logs/blogs/{date_time}"
    if not os.path.exists("../logs"):
        os.makedirs("../logs")
    if not os.path.exists("../logs/blogs"):
        os.makedirs("../logs/blogs")
    parser = build_parser(variant)
    args = vars(parser.parse_args(argv))
    validate_arguments(parser, args, variant)
    seed_everything(args["seed"])
    if args.get("run_tag"):
        log_files_folder_path = f"../logs/blogs/{date_time}_{args['run_tag']}"
        if not os.path.exists(log_files_folder_path):
            os.mkdir(log_files_folder_path)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    network_class = {
        "classic": ClassicNetwork,
        "extended": ExtendedNetwork,
        "krum": KrumNetwork,
    }[variant.name]
    resources = initialize_run(
        args,
        date_time,
        log_files_folder_path,
        "snapshots",
        dev,
        entry_file,
        network_class,
        variant,
        command_line=argv,
    )
    run_rounds(
        args,
        dev,
        resources.devices_in_network,
        resources.devices_list,
        resources.latest_round_num,
        resources.log_files_folder_path,
        resources.network_snapshot_save_path,
        resources.rewards,
        resources.workers_needed,
        resources.miners_needed,
        resources.validators_needed,
        resources.mining_consensus,
        resources.conn,
        resources.conn_cursor,
        variant,
    )
    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{resources.log_files_folder_path}/time.txt", "a") as file:
        file.write("\nend time: " + end_time)
    resources.conn.close()
