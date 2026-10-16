"""Compatibility entrypoint; analysis implementation is shared."""
from pathlib import Path
from analysis.collect_central import *
from analysis import collect_central as _implementation

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    return _implementation.main(project_root=PROJECT_ROOT)

if __name__ == "__main__":
    main()
