"""Summarize continuation runs; optionally render learning and comparison plots."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(root):
    rows, histories = [], {}
    for path in sorted(root.glob('lag_*/seed_*/status.json')):
        status = json.loads(path.read_text())
        row = {key: status[key] for key in ('lag', 'seed', 'status', 'total_steps',
                                            'best_validation_return')}
        row['qualified'] = (status['status'] == 'target_reached_on_validation'
                            and status.get('holdout_training_return', 0) >= status['target_return'])
        histories[(row['lag'], row['seed'])] = status
        if 'holdout_training_return' in status:
            for name in ('training', 'original'):
                report_path = path.parent / f'holdout_{name}_environment.json'
                if name == 'training':
                    audit = path.parent / 'audit_training_environment.json'
                    if audit.exists():
                        digest = hashlib.sha256((path.parent / 'policy.bin').read_bytes()).hexdigest()
                        if json.loads(audit.read_text()).get('checkpoint_sha256') == digest:
                            report_path = audit
                report = json.loads(report_path.read_text())
                row[f'{name}_report'] = report_path.name
                row[f'{name}_return'] = report['mean_return_including_capped']
                row[f'{name}_win_rate'] = report['win_rate']
                row[f'{name}_episodes'] = len(report['episodes'])
                row[f'{name}_capped_episodes'] = report['capped_episodes']
                row[f'{name}_max_decision_steps'] = report.get('max_steps', 30000)
                for metric in ('action_switch_rate', 'direction_reversal_rate',
                               'noop_fraction', 'paddle_distance_pixels'):
                    values = [episode[metric] for episode in report['episodes'] if metric in episode]
                    if values:
                        row[f'{name}_{metric}'] = statistics.mean(values)
        rows.append(row)
    if not rows:
        raise ValueError(f'No continuation status files found in {root}')
    aggregates = []
    for lag in sorted({row['lag'] for row in rows}):
        group = [row for row in rows if row['lag'] == lag and 'original_return' in row]
        aggregate = {'lag': lag, 'evaluated_policies': len(group),
                     'policies_passing_target': sum(row['qualified'] for row in group)}
        for key in ('training_return', 'original_return', 'training_win_rate',
                    'original_win_rate', 'original_action_switch_rate',
                    'original_direction_reversal_rate', 'original_noop_fraction'):
            values = [row[key] for row in group if key in row]
            if values:
                aggregate[key] = statistics.mean(values)
                aggregate[key + '_seed_standard_error'] = (
                    statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None)
        aggregates.append(aggregate)
    (root / 'summary.json').write_text(json.dumps({'runs': rows, 'lag_averages': aggregates}, indent=2) + '\n')
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
        writer.writeheader()
        writer.writerows(rows)
    return rows, aggregates, histories


def plots(root, rows, aggregates, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lags = sorted({row['lag'] for row in rows})
    seeds = sorted({row['seed'] for row in rows})
    colors = dict(zip(seeds, ['#31688e', '#35b779', '#c69524', '#913a87']))
    phases_path = root / 'warm_start_phases.json'
    phases = json.loads(phases_path.read_text()) if phases_path.exists() else []
    fig, axes = plt.subplots(math.ceil(len(lags) / 2), 2, figsize=(10, 3.5 * math.ceil(len(lags) / 2)),
                             sharex=True, sharey=True, squeeze=False)
    for lag, ax in zip(lags, axes.flat):
        for seed in seeds:
            s = histories.get((lag, seed))
            if not s:
                continue
            history = s['history']
            xs = [(s['source_steps'] + h['additional_steps']) / 1e6 for h in history]
            ys = [h['validation_return'] for h in history]
            color = colors.get(seed, '#777777')
            ax.plot(xs, ys, color=color, linewidth=1.6, marker='.', markersize=3, label=f'Seed {seed}')
            for phase in phases:
                if phase['lag'] == lag and phase['seed'] == seed:
                    following = [h for h in history if h['additional_steps'] > phase['resumed_steps']]
                    if following:
                        h = following[0]
                        ax.scatter((s['source_steps'] + h['additional_steps']) / 1e6,
                                   h['validation_return'], marker='^', color=color, s=65, zorder=5)
        ax.axhline(850, color='#777777', linewidth=.8, linestyle='--')
        ax.set_title(f'Lag {lag} microsteps ({lag / 60 * 1000:.0f} ms)')
        ax.set_ylim(0, 885)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    for ax in list(axes.flat)[len(lags):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Steps in this run (millions)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean validation return')
    fig.suptitle('Lagged Breakout: validation at the training lag, volatility = 6', fontsize=13)
    fig.text(.5, .025, 'Dashed line: 850/864 target. Triangles: first check after adopting a same-seed pretrained policy.\n'
             'Run counters include discarded updates; borrowed pretraining is recorded separately in warm_start_phases.json.',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .065, 1, .96])
    fig.savefig(root / 'learning_curves.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = {lag: i for i, lag in enumerate(lags)}
    for field, label, color, offset in [('training_return', 'Training settings', '#278466', -.1),
                                        ('original_return', 'Original game', '#31688e', .1)]:
        group = [a for a in aggregates if field in a]
        axes[0].errorbar([positions[a['lag']] + offset for a in group], [a[field] for a in group],
                         yerr=[a[field + '_seed_standard_error'] or 0 for a in group],
                         fmt='o-', capsize=4, color=color, label=label)
        for row in rows:
            if field in row:
                jitter = .045 * (seeds.index(row['seed']) - (len(seeds) - 1) / 2)
                axes[0].scatter(positions[row['lag']] + offset + jitter, row[field],
                                color=color, s=16, alpha=.45)
    axes[0].set_ylabel('Mean earned return')
    axes[0].set_ylim(0, 885)
    axes[0].legend(frameon=False, fontsize=9)
    field = 'original_action_switch_rate'
    group = [a for a in aggregates if field in a]
    axes[1].errorbar([positions[a['lag']] for a in group], [a[field] for a in group],
                     yerr=[a[field + '_seed_standard_error'] or 0 for a in group],
                     fmt='o-', capsize=4, color='#31688e')
    for row in rows:
        if field in row:
            jitter = .05 * (seeds.index(row['seed']) - (len(seeds) - 1) / 2)
            axes[1].scatter(positions[row['lag']] + jitter, row[field], color=colors.get(row['seed']),
                            s=22, alpha=.8)
    axes[1].set_ylabel('Command changes / decision')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Behavior on the original game', fontsize=11)
    for ax in axes:
        ax.set_xticks(range(len(lags)), [str(lag) for lag in lags])
        ax.set_xlabel('Training lag (microsteps)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Trained policies: own environment and common original game', fontsize=13)
    capped = sum(row.get('original_capped_episodes', 0) for row in rows)
    episode_count = sum(row.get('original_episodes', 0) for row in rows)
    completion_note = (f'Original game: all {episode_count} episodes completed.' if capped == 0 else
                       f'Original game: {capped}/{episode_count} episodes capped; scores and behavior include them.')
    fig.text(.5, .02, 'Small points: individual training seeds. Error bars: standard error across training seeds.\n'
             f'{completion_note}\n'
             'Adaptive budgets and same-seed pretraining make this an exploratory comparison.',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .12, 1, .96])
    fig.savefig(root / 'policy_comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', action='store_true', help='Requires matplotlib')
    args = parser.parse_args()
    root = args.output.resolve()
    rows, aggregates, histories = summarize(root)
    if args.plots:
        plots(root, rows, aggregates, histories)
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
