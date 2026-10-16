"""Compatibility entrypoint for shared analysis with experiment-local paths."""
from pathlib import Path
from analysis.plot_revision_figures import *
from analysis import plot_revision_figures as _implementation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main(*args, **kwargs):
    kwargs.setdefault("project_root", PROJECT_ROOT)
    return _implementation.main(*args, **kwargs)

if __name__ == "__main__":
    main()
