"""CPU-only checks of the exact Device method used by the training runs."""
import ast
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
from bcmfaigc.federation.compat import KrumDevice
from bcmfaigc.variants import KRUM

apply = KrumDevice.apply_multikrum
configured = SimpleNamespace(variant=KRUM)
KrumDevice.set_aggregation_config(configured, aggregation_method='krum', multikrum_f=1)
assert configured.aggregation_method == 'krum' and configured.multikrum_f == 1


def execute(updates, f=1, mode='krum'):
    received = []
    dummy = SimpleNamespace(
        multikrum_f=f, aggregation_method=mode, variant=KRUM,
        devices_dict={k: SimpleNamespace(return_is_malicious=lambda: False) for k, _ in updates},
        return_idx=lambda: 'observer', is_online=lambda: True,
        global_update=lambda records, **kwargs: received.extend(records) or True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        apply(dummy, updates, tmp, 1)
        records = [json.loads(x) for x in (Path(tmp) / 'aggregation_decisions.jsonl').read_text().splitlines()]
    return received, records


def reference(updates, f):
    vectors = [(k, torch.cat([v[name].double().flatten() for name in sorted(v)
                             if torch.is_floating_point(v[name])])) for k, v in updates]
    count = len(vectors) - f - 2
    scores = {}
    for key, vector in vectors:
        distances = [float(((vector - other) ** 2).sum())
                     for other_key, other in vectors if other_key != key]
        scores[key] = sum(sorted(distances)[:count])
    return min(scores, key=lambda k: (scores[k], k)), scores


checks = 0
for seed in range(100):
    generator = torch.Generator().manual_seed(seed)
    for n, f in [(3, 0), (5, 1), (7, 2)]:
        updates = [(f'w{i}', {'b': torch.randn(2, generator=generator),
                              'w': torch.randn(7, generator=generator),
                              'integer_buffer': torch.tensor(1)}) for i in range(n)]
        expected, scores = reference(updates, f)
        received, records = execute(updates, f)
        assert len(received) == 1 and received[0][0] == expected and received[0][2] == 1.0
        assert sum(r['accepted_for_aggregation'] for r in records) == 1
        assert all(r['selected_count'] == 1 and r['krum_fallback_reason'] is None for r in records)
        assert all(math.isclose(r['krum_score'], scores[r['worker_id']], rel_tol=2e-5, abs_tol=2e-5)
                   for r in records)
        assert execute(list(reversed(updates)), f)[0][0][0] == expected
        checks += 1

updates = [(f'w{i}', {'w': torch.zeros(3)}) for i in reversed(range(5))]
assert execute(updates)[0][0][0] == 'w0'
assert len(execute(updates, mode='multikrum')[0]) == 2
try:
    execute(updates[:4])
except ValueError:
    pass
else:
    raise AssertionError('Krum silently accepted n < 2f+3')
print(f'PASS: {checks} independent-reference fixtures, permutation checks, ties, single selection, Multi-Krum preservation, and invalid-n rejection.')
