"""Execution differences retained from the original experiment implementations.

These values describe compatibility policy, not manuscript sections or a second
simulation engine. They never choose seeds, thresholds or training parameters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationVariant:
    name: str
    attack_dispatch: bool
    multikrum: bool
    single_krum: bool
    create_validation_log_parent: bool
    rejection_event_logs: bool
    selection_count_log: bool


CLASSIC = SimulationVariant("classic", False, False, False, True, False, False)
EXTENDED = SimulationVariant("extended", True, True, False, False, True, False)
KRUM = SimulationVariant("krum", True, True, True, False, True, True)


def get_variant(name):
    return {"classic": CLASSIC, "extended": EXTENDED, "krum": KRUM}[name]
