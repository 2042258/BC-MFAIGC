"""Protocol event and model/network persistence, preserving original schemas."""

import os
import json
import pickle
from pathlib import Path
import torch
from bcmfaigc.blockchain.serialization import stable_digest


def write_protocol_event(log_files_folder_path, event):
    """Append one controlled protocol-attack event without changing normal logs."""
    event_path = Path(log_files_folder_path) / "protocol_attack_events.jsonl"
    with event_path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def write_global_checkpoint(args, devices_list, log_files_folder_path, comm_round):
    if args["save_global_checkpoints"] and (
        comm_round == 1 or comm_round % args["checkpoint_frequency"] == 0
    ):
        checkpoint_dir = Path(log_files_folder_path) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        canonical_devices = sorted(devices_list, key=lambda device: device.return_idx())
        canonical_device = next(
            (device for device in canonical_devices if device.is_online()),
            canonical_devices[0],
        )
        state_digests = {
            device.return_idx(): stable_digest(device.global_parameters).hex()
            for device in canonical_devices
        }
        global_versions = {
            device.return_idx(): device.global_version for device in canonical_devices
        }
        checkpoint_state = {
            key: value.detach().cpu().clone()
            for key, value in canonical_device.global_parameters.items()
        }
        torch.save(
            {
                "round": comm_round,
                "model_name": args["model_name"],
                "timesteps": 1000,
                "state_dict": checkpoint_state,
                "checkpoint_source_device": canonical_device.return_idx(),
                "global_state_digests": state_digests,
                "global_versions": global_versions,
                "global_state_consistent": len(set(state_digests.values())) == 1,
                "seed": args["seed"],
                "validation_seed": args["validation_seed"],
            },
            checkpoint_dir / f"global_round_{comm_round:04d}.pt",
        )


def save_network_snapshot(
    args, devices_in_network, network_snapshot_save_path, comm_round
):
    if args["save_network_snapshots"] and (
        comm_round == 1 or comm_round % args["save_freq"] == 0
    ):
        if args["save_most_recent"]:
            paths = sorted(
                Path(network_snapshot_save_path).iterdir(), key=os.path.getmtime
            )
            if len(paths) > args["save_most_recent"]:
                for _ in range(len(paths) - args["save_most_recent"]):
                    open(paths[_], "w").close()
                    os.remove(paths[_])
        snapshot_file_path = f"{network_snapshot_save_path}/snapshot_r_{comm_round}"
        print(f"Saving network snapshot to {snapshot_file_path}")
        pickle.dump(devices_in_network, open(snapshot_file_path, "wb"))
