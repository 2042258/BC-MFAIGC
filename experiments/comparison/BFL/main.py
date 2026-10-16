"""Legacy experiment entrypoint backed by the shared simulation implementation."""
from bcmfaigc.simulation.runner import run
from bcmfaigc.variants import CLASSIC

if __name__ == "__main__":
    run(CLASSIC, entry_file=__file__)
