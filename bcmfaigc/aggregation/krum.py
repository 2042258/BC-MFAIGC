"""Krum distance scores and selection, without node/ledger/reporting state."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class KrumSelection:
    usable_updates: list
    configured_f: int
    effective_f: int
    scores: list
    selected_positions: list[int]
    selected_ids: set
    selected_weight: float
    fallback_reason: str | None


def select_krum_updates(local_updates, configured_f=1, method="multikrum"):
    usable_updates = [
        (worker_idx, params)
        for worker_idx, params in local_updates
        if params is not None
    ]
    if not usable_updates:
        return
    num_updates = len(usable_updates)
    configured_f = max(int(configured_f), 0)
    if method == "krum" and num_updates < 2 * configured_f + 3:
        raise ValueError("Krum requires n >= 2f + 3; refusing a silent fallback")
    effective_f = min(configured_f, max(0, (num_updates - 3) // 2))
    neighbor_count = num_updates - effective_f - 2
    fallback_reason = None
    selected_positions = list(range(num_updates))
    scores = [None] * num_updates
    if num_updates < 3 or neighbor_count < 1:
        fallback_reason = "insufficient_updates_for_multikrum"
    else:
        ordered_keys = sorted(usable_updates[0][1].keys())
        vectors = []
        for _, params in usable_updates:
            parts = [
                params[key].detach().reshape(-1).float().cpu()
                for key in ordered_keys
                if torch.is_floating_point(params[key])
            ]
            vectors.append(torch.cat(parts) if parts else torch.zeros(1))
        vector_matrix = torch.stack(vectors)
        distance_matrix = torch.cdist(vector_matrix, vector_matrix, p=2).pow(2)
        for position in range(num_updates):
            distances = [
                float(distance_matrix[position, other])
                for other in range(num_updates)
                if other != position
            ]
            scores[position] = sum(sorted(distances)[:neighbor_count])
        selected_positions = sorted(
            range(num_updates),
            key=lambda position: (scores[position], usable_updates[position][0]),
        )[: 1 if method == "krum" else max(1, num_updates - effective_f - 2)]
    selected_ids = {usable_updates[position][0] for position in selected_positions}
    selected_weight = 1.0 / len(selected_positions)
    return KrumSelection(
        usable_updates,
        configured_f,
        effective_f,
        scores,
        selected_positions,
        selected_ids,
        selected_weight,
        fallback_reason,
    )
