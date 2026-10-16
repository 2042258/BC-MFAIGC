"""Compatibility entrypoint for shared analysis with experiment-local paths."""
from pathlib import Path
from analysis.build_reports import *
from analysis import build_reports as _implementation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def figure_data(*args, **kwargs):
    kwargs.setdefault("project_root", PROJECT_ROOT)
    return _implementation.figure_data(*args, **kwargs)

def main(*args, **kwargs):
    kwargs.setdefault("project_root", PROJECT_ROOT)
    return _implementation.main(*args, **kwargs)

def timing_table(*args, **kwargs):
    kwargs.setdefault("project_root", PROJECT_ROOT)
    return _implementation.timing_table(*args, **kwargs)

if __name__ == "__main__":
    main()
