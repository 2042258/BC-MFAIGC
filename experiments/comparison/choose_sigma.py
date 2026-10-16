"""Compatibility entrypoint; analysis implementation is shared."""
from pathlib import Path
from analysis.choose_sigma import *
from analysis import choose_sigma as _implementation

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    return _implementation.main(project_root=PROJECT_ROOT)

if __name__ == "__main__":
    main()
