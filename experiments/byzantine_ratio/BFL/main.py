"""Legacy experiment entrypoint backed by the shared simulation implementation."""
from bcmfaigc.simulation.runner import run
from bcmfaigc.variants import EXTENDED

if __name__ == "__main__":
    run(EXTENDED, entry_file=__file__)
