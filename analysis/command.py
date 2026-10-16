"""Explicit analysis command selection with a caller-owned experiment root."""

import argparse
import importlib
from pathlib import Path
import sys

COMMANDS = (
    "collect_runs",
    "collect_comparison_runs",
    "collect_central",
    "analyze_matrix",
    "analyze_threshold",
    "choose_sigma",
    "aggregate_calibration_scores",
    "build_reports",
    "export_latex_tables",
    "plot_revision_figures",
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect or analyze existing experiment outputs. Never starts training."
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Experiment workspace containing logs and report inputs",
    )
    options, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    module = importlib.import_module("analysis." + options.command)
    previous = sys.argv
    try:
        sys.argv = [options.command, *forwarded]
        module.main(project_root=options.project_root.resolve())
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
