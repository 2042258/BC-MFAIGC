"""Argument definitions and profile-specific compatibility checks."""

import argparse


def build_parser(variant):
    """Build the original argument surface for a supported execution profile."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Block_FedAvg_Simulation",
    )
    parser.add_argument(
        "-g", "--gpu", type=str, default="0", help="gpu id to use(e.g. 0,1,2,3)"
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1, help="print verbose debug log"
    )
    parser.add_argument(
        "-sn",
        "--save_network_snapshots",
        type=int,
        default=0,
        help="only save network_snapshots if this is set to 1; will create a folder with date in the snapshots folder",
    )
    parser.add_argument(
        "-dtx",
        "--destroy_tx_in_block",
        type=int,
        default=0,
        help="currently transactions stored in the blocks are occupying GPU ram and have not figured out a way to move them to CPU ram or harddisk, so turn it on to save GPU ram in order for PoS to run 100+ rounds. NOT GOOD if there needs to perform chain resyncing.",
    )
    parser.add_argument(
        "-rp",
        "--resume_path",
        type=str,
        default=None,
        help="resume from the path of saved network_snapshots; only provide the date",
    )
    parser.add_argument(
        "-sf",
        "--save_freq",
        type=int,
        default=5,
        help="save frequency of the network_snapshot",
    )
    parser.add_argument(
        "-sm",
        "--save_most_recent",
        type=int,
        default=2,
        help="in case of saving space, keep only the recent specified number of snapshops; 0 means keep all",
    )
    parser.add_argument(
        "--save_global_checkpoints",
        type=int,
        default=1,
        help="save the global DDPM state after each configured round",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=1,
        help="save a global checkpoint every N communication rounds",
    )
    parser.add_argument(
        "-B", "--batchsize", type=int, default=16, help="local train batch size"
    )
    parser.add_argument(
        "-mn", "--model_name", type=str, default="ddpm", help="the model to train"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate, use value from origin paper as default",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for data allocation, roles and training",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="optional unique suffix for the log directory (matrix runs)",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=1,
        help="fixed seed for deterministic validation noise",
    )
    parser.add_argument(
        "--protocol_attack",
        choices=["none", "validator_tamper", "block_tamper", "replay"],
        default="none",
        help="controlled protocol-level attack for the blockchain supplement",
    )
    parser.add_argument(
        "--protocol_attack_start_round",
        type=int,
        default=2,
        help="first communication round that may receive a controlled protocol attack",
    )
    parser.add_argument(
        "--protocol_attack_every",
        type=int,
        default=1,
        help="inject a controlled protocol attack every N eligible rounds",
    )
    if variant.attack_dispatch:
        parser.add_argument(
            "--attack_type",
            choices=["gaussian_weight_noise", "sign_flip"],
            default="gaussian_weight_noise",
            help="model-poisoning attack applied by malicious workers",
        )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="Adam",
        help="optimizer to be used, by default implementing Adam",
    )
    parser.add_argument(
        "-iid", "--IID", type=int, default=0, help="the way to allocate data to devices"
    )
    parser.add_argument(
        "-max_ncomm",
        "--max_num_comm",
        type=int,
        default=100,
        help="maximum number of communication rounds, may terminate early if converges",
    )
    parser.add_argument(
        "-nd",
        "--num_devices",
        type=int,
        default=10,
        help="numer of the devices in  the simulation network",
    )
    parser.add_argument(
        "-st",
        "--shard_test_data",
        type=int,
        default=0,
        help="it is easy to see the global models are consistent across devices when the test dataset is NOT sharded",
    )
    parser.add_argument(
        "--test_batchsize",
        type=int,
        default=256,
        help="batch size for test-set evaluation; larger values reduce validation time",
    )
    parser.add_argument(
        "-nm",
        "--num_malicious",
        type=int,
        default=1,
        help="number of malicious nodes in the network. malicious node's data sets will be introduced Gaussian noise",
    )
    parser.add_argument(
        "-nv",
        "--noise_variance",
        type=float,
        default=2.0,
        help="Gaussian weight-noise scale; the resulting variance is approximately this value squared",
    )
    parser.add_argument(
        "-le",
        "--default_local_epochs",
        type=int,
        default=5,
        help="local train epoch. Train local model by this same num of epochs for each worker, if -mt is not specified",
    )
    parser.add_argument(
        "-ur",
        "--unit_reward",
        type=int,
        default=1,
        help="unit reward for providing data, verification of signature, validation and so forth",
    )
    parser.add_argument(
        "-ko",
        "--knock_out_rounds",
        type=int,
        default=6,
        help="a worker or validator device is kicked out of the device's peer list(put in black list) if it's identified as malicious for this number of rounds",
    )
    parser.add_argument(
        "-lo",
        "--lazy_worker_knock_out_rounds",
        type=int,
        default=10,
        help="a worker device is kicked out of the device's peer list(put in black list) if it does not provide updates for this number of rounds, due to too slow or just lazy to do updates and only accept the model udpates.(do not care lazy validator or miner as they will just not receive rewards)",
    )
    parser.add_argument(
        "-pow",
        "--pow_difficulty",
        type=int,
        default=0,
        help="if set to 0, meaning miners are using PoS",
    )
    parser.add_argument(
        "--blockchain_on",
        type=int,
        default=1,
        help="1: full blockchain flow; 0: direct aggregation without consensus, mining or block propagation while keeping the same validator filter",
    )
    parser.add_argument(
        "--defense_on",
        type=int,
        default=1,
        help="1: reject worker updates above the validator threshold; 0: accept all updates (plain FedAvg baseline)",
    )
    parser.add_argument(
        "--aggregation_method",
        choices=["hard_filter", "trust_weighted"]
        + (["multikrum"] if variant.multikrum else [])
        + (["krum"] if variant.single_krum else []),
        default="hard_filter",
        help="aggregation rule after validator decisions",
    )
    if variant.single_krum:
        parser.add_argument(
            "--multikrum_f",
            "--krum_f",
            dest="multikrum_f",
            type=int,
            default=1,
            help="assumed Byzantine-client count for Krum or Multi-Krum",
        )
    elif variant.multikrum:
        parser.add_argument(
            "--multikrum_f",
            type=int,
            default=1,
            help="assumed Byzantine-client count for the Multi-Krum baseline",
        )
    parser.add_argument(
        "--trust_center",
        type=float,
        default=0.0,
        help="center used to normalize validator loss differences",
    )
    parser.add_argument(
        "--trust_scale",
        type=float,
        default=1.0,
        help="scale used to normalize validator loss differences",
    )
    parser.add_argument(
        "--trust_soft_threshold",
        type=float,
        default=0.0,
        help="soft threshold in normalized-score space for trust weighting",
    )
    parser.add_argument(
        "--trust_beta",
        type=float,
        default=1.0,
        help="slope of the trust-weighting sigmoid",
    )
    parser.add_argument(
        "--trust_min_weight",
        type=float,
        default=0.05,
        help="lower bound for an accepted update trust weight",
    )
    parser.add_argument(
        "--trust_hard_reject_score",
        type=float,
        default=5.0,
        help="normalized-score upper bound for soft aggregation eligibility",
    )
    parser.add_argument(
        "-mt",
        "--miner_acception_wait_time",
        type=float,
        default=0.0,
        help="default time window for miners to accept transactions, in seconds. 0 means no time limit, and each device will just perform same amount(-le) of epochs per round like in FedAvg paper",
    )
    parser.add_argument(
        "-ml",
        "--miner_accepted_transactions_size_limit",
        type=float,
        default=0.0,
        help="no further transactions will be accepted by miner after this limit. 0 means no size limit. either this or -mt has to be specified, or both. This param determines the final block_size",
    )
    parser.add_argument(
        "-mp",
        "--miner_pos_propagated_block_wait_time",
        type=float,
        default=float("inf"),
        help="this wait time is counted from the beginning of the comm round, used to simulate forking events in PoS",
    )
    parser.add_argument(
        "-vh",
        "--validator_threshold",
        type=float,
        default=0.1,
        help="a threshold value of accuracy difference to determine malicious worker",
    )
    parser.add_argument(
        "-md",
        "--malicious_updates_discount",
        type=float,
        default=0.0,
        help="do not entirely drop the voted negative worker transaction because that risks the same worker dropping the entire transactions and repeat its accuracy again and again and will be kicked out. Apply a discount factor instead to the false negative worker's updates are by some rate applied so it won't repeat",
    )
    parser.add_argument(
        "-mv",
        "--malicious_validator_on",
        type=int,
        default=0,
        help="let malicious validator flip voting result",
    )
    parser.add_argument(
        "-ns",
        "--network_stability",
        type=float,
        default=1.0,
        help="the odds a device is online",
    )
    parser.add_argument(
        "-els",
        "--even_link_speed_strength",
        type=int,
        default=1,
        help="This variable is used to simulate transmission delay. Default value 1 means every device is assigned to the same link speed strength -dts bytes/sec. If set to 0, link speed strength is randomly initiated between 0 and 1, meaning a device will transmit  -els*-dts bytes/sec - during experiment, one transaction is around 35k bytes.",
    )
    parser.add_argument(
        "-dts",
        "--base_data_transmission_speed",
        type=float,
        default=70000.0,
        help="volume of data can be transmitted per second when -els == 1. set this variable to determine transmission speed (bandwidth), which further determines the transmission delay - during experiment, one transaction is around 35k bytes.",
    )
    parser.add_argument(
        "-ecp",
        "--even_computation_power",
        type=int,
        default=1,
        help="This variable is used to simulate strength of hardware equipment. The calculation time will be shrunk down by this value. Default value 1 means evenly assign computation power to 1. If set to 0, power is randomly initiated as an int between 0 and 4, both included.",
    )
    parser.add_argument(
        "-ha",
        "--hard_assign",
        type=str,
        default="*,*,*",
        help='hard assign number of roles in the network, order by worker, validator and miner. e.g. 12,5,3 assign 12 workers, 5 validators and 3 miners. "*,*,*" means completely random role-assigning in each communication round ',
    )
    parser.add_argument(
        "-aio",
        "--all_in_one",
        type=int,
        default=1,
        help="let all nodes be aware of each other in the network while registering",
    )
    parser.add_argument(
        "-cs",
        "--check_signature",
        type=int,
        default=1,
        help="if set to 0, all signatures are assumed to be verified to save execution time",
    )
    return parser


def validate_arguments(parser, args, variant):
    if (
        variant.multikrum
        and args["aggregation_method"] in ("multikrum", "krum")
        and (args["blockchain_on"] or args["defense_on"])
    ):
        if variant.single_krum:
            parser.error(
                "Krum and Multi-Krum require the no-blockchain, no-validator-filter baseline"
            )
        parser.error(
            "--aggregation_method multikrum is only valid for the no-blockchain, no-validator-filter baseline"
        )
