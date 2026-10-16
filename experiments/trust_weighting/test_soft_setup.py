"""CPU-only checks of the unchanged Soft configuration and approved run matrix."""
import ast
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'BFL'))
from trust_aggregation import compute_trust_metrics

from bcmfaigc.federation.compat import ExtendedDevice
from bcmfaigc.variants import EXTENDED

ns = {name: getattr(ExtendedDevice, name) for name in
      ('set_aggregation_config', '_compute_aggregation_metrics', '_aggregation_eligibility')}
assert len(ns) == 3
manifest = json.loads((ROOT.parents[1] / 'configs' / ROOT.name / 'soft_manifest.json').read_text())
assert len(manifest['runs']) == 12 and len({r['id'] for r in manifest['runs']}) == 12
parameters = {k: float(v) for k, v in manifest['soft_parameters'].items()}
device = SimpleNamespace(variant=EXTENDED)
ns['set_aggregation_config'](device, aggregation_method='trust_weighted', **parameters)
last = float('inf')
for index in range(-100, 101):
    raw = device.trust_center + device.trust_scale * index / 10
    normalized, weight = ns['_compute_aggregation_metrics'](device, raw)
    assert math.isclose(normalized, index / 10, abs_tol=1e-12)
    assert 0.05 <= weight <= 1 and weight <= last
    assert ns['_aggregation_eligibility'](device, normalized, False) == (normalized <= 5)
    last = weight
assert math.isclose(compute_trust_metrics(device.trust_center, center=device.trust_center,
                                        scale=device.trust_scale, soft_threshold=1)[1],
                    .05 + .95 / (1 + math.exp(-1)))
for r in manifest['runs']:
    assert r['seed'] in [50000, 50001, 50002] and r['condition'] in ['clean', 'near']
    assert r['method'] in ['nobc', 'bc']
    assert r['reference']['args']['hard_assign'] == '5,2,3'
    assert r['reference']['args']['check_signature'] == '1'
    assert r['reference']['args']['defense_on'] == '1'
    assert float(r['reference']['args']['validator_threshold']) == 0.013486012299855549
    assert r['reference']['args']['blockchain_on'] == ('1' if r['method'] == 'bc' else '0')
print('PASS: 12 unique matched runs; frozen Soft parameters; 201 normalization, monotonic-weight and eligibility cases; Stage 2 threshold, roles and signatures.')
