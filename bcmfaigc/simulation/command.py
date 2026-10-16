"""One training run from explicit JSON arguments, without an experiment manager."""

import argparse
import json
import os
from pathlib import Path

from bcmfaigc.variants import get_variant
from .runner import run

PROJECT = Path(__file__).resolve().parents[2]
PROFILE_DIRECTORIES = {
    "classic": "comparison",
    "extended": "byzantine_ratio",
    "krum": "krum",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one federated DDPM simulation from a config or original CLI arguments.",
        epilog="Other arguments are forwarded to the simulation (for example --seed 17 -max_ncomm 100). Install this source checkout with pip install -e .",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON with profile and arguments (a list of CLI tokens)",
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_DIRECTORIES), default=None)
    parser.add_argument(
        "--run-id",
        help="Select exactly one run from an existing matrix manifest; never launches the whole matrix",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Data/log workspace; requires data/cifar-10-batches-py. Defaults to the source experiment directory.",
    )
    options, overrides = parser.parse_known_args(argv)
    config = json.loads(options.config.read_text()) if options.config else {}
    profile = options.profile or config.get("profile", "extended")
    if profile not in PROFILE_DIRECTORIES:
        parser.error("Unknown simulation profile")
    if config.get("execution_enabled") is False:
        parser.error("This configuration explicitly disables execution")
    if "runs" in config:
        matches = [item for item in config["runs"] if item.get("id") == options.run_id]
        if len(matches) != 1:
            parser.error("Matrix manifests require one unique --run-id")
        arguments = (
            config.get("common", [])
            + ["--run_tag", matches[0]["id"]]
            + matches[0].get("args", [])
        )
    else:
        arguments = config.get("arguments", [])
        if config.get("run_id") and "--run_tag" not in arguments:
            arguments = [*arguments, "--run_tag", config["run_id"]]
    if not isinstance(arguments, list) or not all(
        isinstance(value, str) for value in arguments
    ):
        parser.error("Config arguments must be a list of strings")
    experiment = PROJECT / "experiments" / PROFILE_DIRECTORIES[profile]
    entry_file = experiment / "BFL/main.py"
    if not entry_file.is_file():
        parser.error(
            "Source compatibility entrypoints missing; use the documented editable source installation"
        )
    workspace = options.workspace.resolve() if options.workspace else experiment
    working_directory = workspace / "BFL"
    working_directory.mkdir(parents=True, exist_ok=True)
    (working_directory / "snapshots").mkdir(exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(working_directory)
        run(get_variant(profile), arguments + overrides, entry_file=entry_file)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
