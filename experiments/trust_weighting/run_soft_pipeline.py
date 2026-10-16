"""Execute the 12 authorized Stage 2 Soft runs without modifying project code."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
BFL = ROOT / 'BFL'
sys.path.insert(0, str(BFL))
from run_matrix import summarize_run
from collect_runs import summarize as collect_final

MANIFEST = json.loads((ROOT.parents[1] / 'configs' / ROOT.name / 'soft_manifest.json').read_text())
PYTHON = sys.executable
LOCK = threading.Lock()
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def state(status, **details):
    path = ROOT / 'status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'status': status, 'updated_at': time.time(), **details}, indent=2) + '\n')
    temporary.replace(path)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def arguments(path):
    return dict(x[2:].split(' ', 1) for x in path.read_text().splitlines()
                if x.startswith('--') and ' ' in x)


def role_map(path):
    return {(r['round'], r['device_id']): (r['role'], r['malicious']) for r in rows(path)}


def append(name, record):
    with LOCK:
        with (ROOT / 'results' / name).open('a') as output:
            output.write(json.dumps(record, sort_keys=True) + '\n')


def train(spec):
    sources = [BFL / name for name in ('main.py', 'Device.py', 'DatasetLoad.py', 'Block.py', 'stable_hash.py', 'trust_aggregation.py')]
    sources.append(ROOT / 'models/Models.py')
    current = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    expected = spec['reference']['code_fingerprint']
    differences = {name for name in set(current) | set(expected) if current.get(name) != expected.get(name)}
    if differences != set():
        raise RuntimeError('Historical exact-source audit is incompatible with this reorganized source. Use bcmfaigc-train for fresh runs; historical assertions were not relaxed.')
    tag = spec['id']
    if list((ROOT / 'logs/blogs').glob('*_' + tag)):
        raise RuntimeError(f'Refusing duplicate execution: {tag}')
    command = [PYTHON, str(BFL / 'main.py'), *MANIFEST['common'], '--run_tag', tag, *spec['args']]
    start = time.monotonic()
    print('TRAIN_START', tag, flush=True)
    with (ROOT / 'results' / (tag + '.stdout.log')).open('w') as output:
        subprocess.run(command, cwd=BFL, env=ENV, stdout=output,
                       stderr=subprocess.STDOUT, timeout=7200, check=True)
    matches = list((ROOT / 'logs/blogs').glob('*_' + tag))
    assert len(matches) == 1
    directory = matches[0]
    summary = summarize_run(directory, spec, time.monotonic() - start)
    summary['log_dir'] = str(directory.relative_to(ROOT))
    assert summary['rounds'] == 100
    assert summary['checkpoints'][-1].endswith('global_round_0100.pt')
    actual = arguments(directory / 'args_used.txt')
    expected = dict(spec['reference']['args'])
    expected.update(MANIFEST['soft_parameters'])
    expected.update(aggregation_method='trust_weighted', run_tag=tag)
    assert actual == expected, {k: (actual.get(k), expected.get(k)) for k in set(actual) | set(expected)
                                if actual.get(k) != expected.get(k)}
    reference_dir = Path(spec['reference']['log_dir'])
    assert role_map(directory / 'round_metadata.jsonl') == role_map(reference_dir / 'round_metadata.jsonl')
    fingerprint = json.loads((directory / 'code_fingerprint.json').read_text())
    assert fingerprint == spec['reference']['code_fingerprint']
    decisions = rows(directory / 'aggregation_decisions.jsonl')
    assert decisions and all(r.get('aggregation_method') == 'trust_weighted' for r in decisions)
    for r in decisions:
        if 'aggregation_weight' in r:
            assert 0.0 <= r['aggregation_weight'] <= 1.0
    collected = collect_final(directory, tag, spec['tag'])
    assert collected['rounds'] == 100 and collected['validation_vote_records'] > 0
    total = collected['malicious_worker_round_total']
    rejected = collected['malicious_worker_round_filter_rejected']
    assert (total > 0) == (spec['condition'] == 'near')
    summary['malicious_final_decisions'] = total
    summary['malicious_final_rejected'] = rejected
    summary['final_detection_percent'] = 100 * rejected / total if total else None
    summary['observer_disagreements'] = collected['observer_disagreements']
    summary['runtime_qa_passed'] = True
    append('training.jsonl', summary)
    append('collected_final.jsonl', collected)
    print('TRAIN_DONE', tag, round(summary['wall_seconds'], 1), flush=True)
    return summary


def evaluate(spec, record):
    directory = ROOT / record['log_dir']
    checkpoint = ROOT / record['checkpoints'][-1]
    sample_dir = directory / 'generated'
    fid_path = sample_dir / 'fid.json'
    print('FID_START', spec['id'], flush=True)
    with (ROOT / 'results' / (spec['id'] + '.fid.log')).open('w') as output:
        subprocess.run([PYTHON, str(BFL / 'evaluate_ddpm.py'), str(checkpoint),
                        '--output-dir', str(sample_dir), '--num-samples', '2000',
                        '--batch-size', '64', '--device', 'cuda', '--seed', str(spec['sample_seed'])],
                       cwd=BFL, env=ENV, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=7200)
        subprocess.run([PYTHON, str(BFL / 'evaluate_fid.py'), str(sample_dir / 'samples.pt'),
                        '--output', str(fid_path), '--max-real-samples', '10000',
                        '--batch-size', '64', '--device', 'cuda'],
                       cwd=BFL, env=ENV, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=3600)
    document = json.loads(fid_path.read_text())
    assert document['generated_samples'] == 2000 and document['real_samples'] == 10000
    assert 0 <= document['fid'] < float('inf')
    meta = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'num_samples': 2000, 'max_real_samples': 10000, 'batch_size': 64, 'sample_seed': spec['sample_seed']}
    (sample_dir / 'evaluation_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    result = {'id': spec['id'], 'tag': spec['tag'], 'seed': spec['seed'], 'fid': document['fid'],
              'num_samples': 2000, 'real_samples': 10000, 'sample_seed': spec['sample_seed'],
              'checkpoint': str(checkpoint.relative_to(ROOT)), 'reference_id': spec['reference']['id'],
              'final_detection_percent': record['final_detection_percent']}
    append('fid.jsonl', result)
    print('FID_DONE', spec['id'], document['fid'], flush=True)
    return result


def main():
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'logs/blogs').mkdir(parents=True, exist_ok=True)
    if (ROOT / 'results/training.jsonl').exists() or (ROOT / 'results/fid.jsonl').exists():
        raise RuntimeError('Results already exist; no automatic rerun is permitted')
    runs = MANIFEST['runs']
    assert len(runs) == 12
    records = []
    for start in range(0, 12, 2):
        batch = runs[start:start + 2]
        state('training', current_batch=[r['id'] for r in batch], completed_training=len(records), completed_fid=0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            records.extend(pool.map(train, batch))
    byid = {r['id']: r for r in records}
    output = []
    for spec in runs:
        state('evaluating_fid', current=spec['id'], completed_training=12, completed_fid=len(output))
        output.append(evaluate(spec, byid[spec['id']]))
    summary = {}
    for condition in ['clean', 'near']:
        for method in ['nobc', 'bc']:
            tag = f'soft2_{condition}_{method}'
            group = [r for r in output if r['tag'] == tag]
            values = [r['fid'] for r in group]
            assert len(values) == 3
            summary[tag] = {'n': 3, 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
                            'final_detection_percent': statistics.mean(r['final_detection_percent'] for r in group)
                            if condition == 'near' else None}
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    state('completed', completed_training=12, completed_fid=12, summary=summary)
    print('COMPLETED', json.dumps(summary), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed', error=repr(error))
        raise
