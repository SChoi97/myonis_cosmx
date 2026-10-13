#!/usr/bin/env python3
"""Plot saved nucleus QC scores and rethreshold them without loading any images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qc-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Default: QC_DIR/score_threshold_analysis')
    parser.add_argument('--aggregation', choices=['max', 'mean'], default='max')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.0])
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in args.thresholds):
        parser.error('Thresholds must be finite values in [0,1]')

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = args.qc_dir / 'nucleus_summary.csv'
    metadata = json.loads((args.qc_dir / 'statistics_metadata.json').read_text())
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or metadata['filter_mode'] != 'registration_based':
        raise ValueError('Requires a nonempty registration-based nucleus QC run')
    score_column = f'{args.aggregation}_change_score'
    eligible = [row for row in rows if row['decision'] in {'kept', 'filtered_change'}]
    if any(row['is_edge'].lower() == 'true' for row in eligible):
        raise ValueError('An edge nucleus was marked evaluable in the source table')
    scores = np.sort(np.asarray([float(row[score_column]) for row in eligible]))
    if not len(scores) or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Evaluable nuclei must have finite scores in [0,1]')
    total, evaluated = len(rows), len(scores)
    edges = sum(row['is_edge'].lower() == 'true' for row in rows)
    other_unknown = total - evaluated - edges
    current = float(metadata['change_threshold'])

    def retained_at(threshold):
        # Existing QC retains score <= threshold. Eligibility is independent of threshold.
        return int(np.searchsorted(scores, threshold, side='right'))

    if args.aggregation == metadata['score_aggregation']:
        original_kept = sum(row['decision'] == 'kept' for row in rows)
        if retained_at(current) != original_kept:
            raise ValueError('Saved scores do not reproduce the original retained count')
    out = args.output_dir or args.qc_dir / 'score_threshold_analysis'
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for threshold in sorted(set([*args.thresholds, current])):
        kept = retained_at(threshold)
        table.append(dict(threshold=threshold, retained_nuclei=kept,
                          retained_pct_all=100*kept/total, retained_pct_evaluable=100*kept/evaluated,
                          excluded_total=total-kept, excluded_pct_all=100*(total-kept)/total,
                          failed_score=evaluated-kept, excluded_edge=edges,
                          excluded_other_unevaluable=other_unknown))
    with (out / f'{args.aggregation}_threshold_counts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    hist, curve = axes
    hist.hist(scores, bins=np.linspace(0, 1, 51), color='#476c91', edgecolor='white', linewidth=.4)
    hist.set(title=f'Score distribution: {evaluated:,} evaluable nuclei',
             xlabel=f'Nucleus {args.aggregation}-map SSIM change score (lower is better)', ylabel='Nuclei', xlim=(0, 1))
    xs = np.unique(np.concatenate(([0.0], scores, [1.0])))
    counts = np.searchsorted(scores, xs, side='right')
    curve.step(xs, 100*counts/total, where='post', color='#476c91', linewidth=2.1, label='Retained / all nuclei')
    curve.step(xs, 100*counts/evaluated, where='post', color='#777777', linewidth=1.5,
               linestyle=':', label='Retained / evaluable nuclei')
    curve.set(title='Retention at each threshold', xlabel='Retain nuclei with score ≤ threshold',
              ylabel='Retained nuclei (%)', xlim=(0, 1), ylim=(0, 103))
    for threshold, color in [(0.30, '#b86a23'), (current, '#27816d')]:
        count = retained_at(threshold)
        label = f'{threshold:.2f}: {count:,} retained ({100*count/total:.2f}% of all)'
        hist.axvline(threshold, color=color, linestyle='--', linewidth=1.7, label=label)
        curve.axvline(threshold, color=color, linestyle='--', linewidth=1.2, alpha=.8)
        curve.scatter([threshold], [100*count/total], color=color, s=35, zorder=4)
    hist.legend(fontsize=9, frameon=False, loc='upper right')
    curve.legend(fontsize=10, frameon=False, loc='lower right')
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle(f'{metadata["score_metric"].upper()} nuclear QC · {args.aggregation} across rounds · '
                 f'{metadata["completed_patches"]:,} patches · {total:,} reference nuclei', fontsize=14)
    fig.text(.5, .025, f'Fixed exclusions at every threshold: {edges:,} edge + {other_unknown:,} other unevaluable nuclei. '
             'Only evaluable nuclei contribute scores.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .93))
    figure = out / f'{args.aggregation}_score_distribution_and_retention.png'
    fig.savefig(figure, dpi=160, facecolor='white')
    plt.close(fig)
    summary = dict(source=str(source.resolve()), score_column=score_column,
                   score_definition=f'Area mean within the nucleus of the pixelwise {args.aggregation} change across selected rounds',
                   threshold_rule='<=', total_nuclei=total, evaluable_nuclei=evaluated,
                   fixed_exclusions=dict(edge=edges, other_unevaluable=other_unknown),
                   comparison_rounds=metadata.get('comparison_rounds'),
                   excluded_filter_rounds=metadata.get('excluded_filter_rounds'),
                   original_threshold=current, original_aggregation=metadata['score_aggregation'],
                   score_quantiles={str(p): float(np.quantile(scores, p)) for p in [.05, .25, .5, .75, .95]},
                   thresholds=table)
    (out / f'{args.aggregation}_threshold_analysis.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'All nuclei: {total:,}; evaluable: {evaluated:,}; fixed exclusions: {total-evaluated:,}')
    for row in table:
        print(f"Threshold {row['threshold']:.2f}: retained {row['retained_nuclei']:,} "
              f"({row['retained_pct_all']:.2f}% of all); excluded {row['excluded_total']:,} "
              f"({row['excluded_pct_all']:.2f}%)")
    print(f'Figure: {figure}')


if __name__ == '__main__':
    main()
