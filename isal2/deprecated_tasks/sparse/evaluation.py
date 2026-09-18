"""Frozen scene manifests, metrics and explicit phase admission rules."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from .terrain_cfg import ACQUIRE, ROBUST, REVIEW, WIDTHS
from .config import ROBUST_STEPS


def manifest(unlocked=None, command_stage='C0', samples=32):
    if samples < 32:
        raise ValueError('At least 32 fixed scenes are required per group')
    groups = {(kind, level, None) for kind in ('single_beam', 'radial_beams') for level in (0, 2, 6, 9)}
    groups |= {('legacy_star', level, level/9) for level in (0, 3, 6, 9)}
    groups |= {(kind, 0, None) for kind in ACQUIRE if kind not in ('single_beam', 'radial_beams')}
    groups |= {(kind, min(level, 9), None) for kind, level in (unlocked or {}).items() if kind in ROBUST and kind not in REVIEW}
    scenes = []
    for kind, level, difficulty in sorted(groups, key=lambda x: (x[0], x[1])):
        for i in range(samples):
            # Fixed low-discrepancy offsets sweep the scan grid edge phase.
            heading = math.pi/2 if command_stage == 'C2' and i % 2 else 0.
            scenes.append(dict(id=f'{kind}:{level}:{i}', kind=kind, level=level,
                legacy_difficulty=difficulty, route=i % 24 if kind in ('radial_beams', 'legacy_star') else 0,
                offset=[-.045+.09*(i % 8)/7, -.045+.09*(i//8 % 4)/3],
                yaw_offset=math.radians((-5 if command_stage == 'C0' else -15) * (1-2*(i % 4)/3)),
                heading_offset=heading, speed=.35+.30*(i % 8)/7, seed=73000+i))
    return scenes


def wilson(k, n):
    if not n:
        return [0., 1.]
    z = 1.95996398454
    p, denom = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [center-half, center+half]


def summarize(records):
    groups = []
    for kind, level in sorted({(r['kind'], r['level']) for r in records}):
        subset = [r for r in records if (r['kind'], r['level']) == (kind, level)]
        n = len(subset)
        row = dict(kind=kind, level=level, n=n)
        if 'geometry' in subset[0]:
            row['geometry'] = subset[0]['geometry']
        for key in ('success', 'fall', 'bypass', 'stuck', 'timeout'):
            k = sum(bool(r[key]) for r in subset)
            row[key+'_rate'] = k/n
            row[key+'_ci95'] = wilson(k, n)
        successes = [r['seconds'] for r in subset if r['success']]
        row['success_seconds'] = float(np.mean(successes)) if successes else None
        for key in ('progress', 'tracking_error'):
            row[key] = float(np.mean([r[key] for r in subset]))
        groups.append(row)
    return groups


def write_report(output, records, scenes, checkpoint, signature, condition='stage'):
    from isal2.modified_rsl.runners.checkpoint import model_digest
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(version=1, iteration=checkpoint['iter'], model_digest=model_digest(checkpoint['model_state_dict']),
        signature=signature, condition=condition,
        manifest_digest=hashlib.sha256(json.dumps(scenes, sort_keys=True).encode()).hexdigest(),
        groups=summarize(records))
    (output/'scenes.json').write_text(json.dumps(scenes, indent=2), encoding='utf-8')
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output/'groups.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(report['groups'][0]))
        writer.writeheader()
        writer.writerows(report['groups'])
    return report


def _passes(report, targets, baseline):
    indexed = {(g['kind'], g['level']): g for g in report['groups']}
    for key in targets:
        g = indexed.get(key)
        if g is None or g['n'] < 32 or g['success_rate'] < .85 or g['fall_rate'] >= .10:
            return False
    flat = indexed.get(('flat', 0))
    return bool(flat and flat['n'] >= 32 and flat['fall_rate'] < .10 and
                flat['tracking_error'] <= 1.1*max(baseline, 1.e-6))


def check_advance(checkpoint, target, report, target_terrain=None):
    from isal2.modified_rsl.runners.checkpoint import model_digest
    source = checkpoint.get('sparse_state')
    if source is None:
        raise ValueError('Phase migration requires a sparse checkpoint')
    old = source['signature']
    if report['model_digest'] != model_digest(checkpoint['model_state_dict']) or report['iteration'] != checkpoint['iter']:
        raise ValueError('Validation report does not match the source model')
    if report['condition'] != 'stage' or report['signature'] != old:
        raise ValueError('Validation must use the source stage settings')
    history = source['curriculum']['validation']
    reports = {r['iteration']: r for r in history}
    reports[report['iteration']] = report
    latest = sorted(reports.values(), key=lambda r: r['iteration'])
    if len(latest) < 2:
        raise ValueError('Two distinct successful evaluations are required')
    baseline = source['curriculum'].get('flat_baseline')
    if baseline is None:
        raise ValueError('Missing flat baseline')
    rescue = source.get('terrain') == 'sparse_rescue'
    target_terrain = target_terrain or source.get('terrain', 'sparse')
    command_change = old['command_stage'] != target['command_stage']
    if rescue:
        if target_terrain != 'sparse' or target['phase'] != 'acquire' or target['command_stage'] != 'C0':
            raise ValueError('Rescue advances only to acquire/C0 width curriculum')
    elif target_terrain != source.get('terrain', 'sparse'):
        raise ValueError('Terrain changes require the explicit rescue-to-width transition')
    elif command_change:
        if old['phase'] != target['phase'] or old['robust_step'] != target['robust_step'] or \
                ('C0', 'C1', 'C2').index(target['command_stage']) != ('C0', 'C1', 'C2').index(old['command_stage'])+1:
            raise ValueError('Advance exactly one command stage without changing perturbations')
    elif old['phase'] == 'acquire':
        if target['phase'] != 'robust' or target['robust_step'] != 'clean':
            raise ValueError('Acquire must advance to robust/clean')
    elif target['phase'] != 'robust' or ROBUST_STEPS.index(target['robust_step']) != ROBUST_STEPS.index(old['robust_step'])+1:
        raise ValueError('Advance exactly one robust step')
    target_level = old['target_level']
    legacy_level = max(0, min(9, round((.4-WIDTHS[target_level])/.15*9)))
    targets = [('single_beam', 2)] if rescue else [('single_beam', target_level), ('radial_beams', target_level), ('legacy_star', legacy_level)]
    if not all(_passes(r, targets, baseline) for r in latest[-2:]):
        raise ValueError('Target and flat regression gates have not passed twice')
    entry = source['curriculum'].get('entry_report') or latest[-2]
    previous = {(g['kind'], g['level']): g for g in entry['groups']}
    if any(g['success_rate'] < previous[(g['kind'], g['level'])]['success_rate']-.10
           for g in latest[-1]['groups'] if (g['kind'], g['level']) in targets):
        raise ValueError('Pass rate regressed more than 10 percentage points; hold this stage with 40% replay')


def validation_callback(task, log_dir, device):
    """Evaluation lives in a separate process and never touches training physics/RNG."""
    def evaluate(runner):
        raw = runner.env.unwrapped
        if raw.sparse_phase_updates % raw.cfg.sparse.eval_interval and raw.sparse_curriculum.last_evaluation >= 0:
            return
        if raw.sparse_curriculum.last_evaluation >= runner.current_learning_iteration:
            return
        folder = Path(log_dir)/'evaluation'/f'update_{runner.current_learning_iteration}'
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder/'policy.pt'
        runner.save(str(checkpoint))
        args = [sys.executable, str(Path(__file__).resolve().parents[2]/'scripts/evaluate_sparse.py'),
                '--task', task, '--checkpoint', str(checkpoint), '--output', str(folder), '--headless', '--device', device]
        with (folder/'evaluation.log').open('w', encoding='utf-8') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((folder/'stage/summary.json').read_text(encoding='utf-8'))
        raw.sparse_curriculum.validate(report)
    return evaluate
