"""Read-only finite-sample reporting for the fixed H033 crossfit output.

No confidence interval treats overlapping crossfit scores as independent.
The admission file must record the coordinator/user-approved screening gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS = ('crossfit', 'fixed_keep', 'fixed_replace', 'intercept_only', 'frequency_rule')


def average(values):
    return math.fsum(values) / len(values)


def summarize(packet, admission):
    if packet['format'] != 'h033_crossfit_v1':
        raise ValueError('unexpected frozen diagnostic output format')
    if admission.get('state') != 'APPROVED_BEFORE_RUN':
        raise ValueError('a pre-run approval record is required')
    rows = packet['rows']
    seeds = [r['seed'] for r in rows]
    if len(rows) != 16 or sorted(seeds) != sorted(admission['seeds']):
        raise ValueError('all sixteen frozen seeds are required')
    if len(set(seeds)) != 16:
        raise ValueError('duplicate seed')
    folds = {m['fold']: m for m in packet['folds']}
    if set(folds) != {0, 1, 2, 3}:
        raise ValueError('all four frozen folds are required')
    for f, model in folds.items():
        test = [r['seed'] for r in rows if r['fold'] == f]
        if len(test) != 4 or set(test) != set(model['test_seeds']):
            raise ValueError('held-out fold coverage mismatch')
        if set(model['train_seeds']) != set(seeds)-set(test):
            raise ValueError('training/held-out overlap or omission')
    for row in rows:
        decisions = row['decisions']
        expected = {
            'crossfit': 'replace' if row['predicted_delta'] > 0 else 'keep',
            'fixed_keep': 'keep', 'fixed_replace': 'replace',
            'intercept_only': 'replace' if folds[row['fold']]['intercept'] > 0 else 'keep',
            'frequency_rule': 'replace' if row['features']['recent_frequency_contrast'] > 0 else 'keep',
        }
        if decisions != expected:
            raise ValueError('a rule differs from the frozen decision contract')
        for domain in ('values_primary', 'values_balanced'):
            values = row[domain]
            if not all(math.isfinite(values[m]) for m in METHODS):
                raise ValueError('non-finite policy value')
            for m in METHODS:
                actual = values['fixed_replace' if decisions[m] == 'replace' else 'fixed_keep']
                if not math.isclose(values[m], actual, abs_tol=1e-12, rel_tol=0):
                    raise ValueError('selected policy value disagrees with its action')
        delta = row['values_primary']['fixed_replace']-row['values_primary']['fixed_keep']
        if not math.isclose(delta, row['target_delta'], abs_tol=1e-12, rel_tol=0):
            raise ValueError('paired target differs from the frozen arm values')

    policies = {}
    for method in METHODS:
        policies[method] = {
            'replace_count': sum(r['decisions'][method] == 'replace' for r in rows),
            'seed_count': 16,
            'mean_selected_primary': average([r['values_primary'][method] for r in rows]),
            'mean_primary_gain_vs_keep': average([
                r['values_primary'][method]-r['values_primary']['fixed_keep'] for r in rows]),
            'mean_selected_balanced': average([r['values_balanced'][method] for r in rows]),
            'mean_balanced_gain_vs_keep': average([
                r['values_balanced'][method]-r['values_balanced']['fixed_keep'] for r in rows]),
        }
    contrasts = {}
    for control in METHODS[1:]:
        differences = {r['seed']: r['values_primary']['crossfit']-r['values_primary'][control]
                       for r in rows}
        fold_means = {str(f): average([differences[r['seed']] for r in rows if r['fold'] == f])
                      for f in range(4)}
        point = average(list(differences.values()))
        contrasts[control] = {
            'mean': point, 'per_seed': differences, 'fold_means': fold_means,
            'fold_range': [min(fold_means.values()), max(fold_means.values())],
            'positive_seeds': sum(v > 0 for v in differences.values()),
            'zero_seeds': sum(v == 0 for v in differences.values()),
            'negative_seeds': sum(v < 0 for v in differences.values()),
            'minimum_gain_pass': point >= admission['minimum_gain_reward_per_step'],
            'all_four_fold_means_positive': all(v > 0 for v in fold_means.values()),
        }
    passed = all(c['minimum_gain_pass'] and c['all_four_fold_means_positive']
                 for c in contrasts.values())
    return {
        'format': 'h033_experiment_report_v1', 'scope': 'exploratory_E161_reuse',
        'policy_values': policies, 'crossfit_minus_controls': contrasts,
        'screening_gate_met': passed,
        'screening_interpretation': ('RETURN_TO_COORDINATOR_FOR_UNSEEN_SEED_CONFIRMATION_DESIGN'
                                      if passed else 'MINIMAL_DIAGNOSTIC_DOES_NOT_PASS_SCREEN'),
        'automatic_followup_authorized': False,
        'uncertainty': 'All seed differences and four fold means/ranges are descriptive. '
                       'No independent-score t interval or significance claim; training folds overlap.',
        'environment_interactions': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.crossfit.read_bytes()
    admission_raw = args.admission.read_bytes()
    packet = json.loads(raw)
    report = summarize(packet, json.loads(admission_raw))
    report['provenance'] = {
        'crossfit': str(args.crossfit.resolve()), 'crossfit_sha256': hashlib.sha256(raw).hexdigest(),
        'admission': str(args.admission.resolve()),
        'admission_sha256': hashlib.sha256(admission_raw).hexdigest(),
        'reporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    fields = ['seed', 'fold', 'predicted_delta', 'target_delta']
    fields += [f'{m}_decision' for m in METHODS]
    fields += [f'crossfit_minus_{m}' for m in METHODS[1:]]
    with (args.output/'per_seed.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in packet['rows']:
            row = {k: r[k] for k in fields[:4]}
            row.update({f'{m}_decision': r['decisions'][m] for m in METHODS})
            row.update({f'crossfit_minus_{m}': r['values_primary']['crossfit']-r['values_primary'][m]
                        for m in METHODS[1:]})
            writer.writerow(row)
    lines = ['# H033 offline selection diagnostic', '',
             'Exploratory reuse of E161; no new environment interactions.', '',
             '| Rule | Replace / 16 | Mean primary value | Gain vs keep | Balanced gain vs keep |',
             '| --- | ---: | ---: | ---: | ---: |']
    for m, v in report['policy_values'].items():
        lines.append(f"| {m} | {v['replace_count']} | {v['mean_selected_primary']:.9f} | "
                     f"{v['mean_primary_gain_vs_keep']:+.9f} | {v['mean_balanced_gain_vs_keep']:+.9f} |")
    lines += ['', '| Crossfit minus control | Mean | Four held-out fold means | Minimum-gain / stability gate |',
              '| --- | ---: | --- | --- |']
    for m, v in report['crossfit_minus_controls'].items():
        fold_values = ', '.join(f'{x:+.9f}' for x in v['fold_means'].values())
        lines.append(f"| {m} | {v['mean']:+.9f} | {fold_values} | "
                     f"{v['minimum_gain_pass']} / {v['all_four_fold_means_positive']} |")
    lines += ['', report['uncertainty'], '', f"Screening gate met: {report['screening_gate_met']}.",
              'No automatic follow-up. Research integration remains with the coordinator.', '']
    (args.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
