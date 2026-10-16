"""Legacy experiment entrypoint backed by the shared simulation implementation."""
from bcmfaigc.simulation.runner import run
from bcmfaigc.variants import KRUM

if __name__ == "__main__":
    run(KRUM, entry_file=__file__)
