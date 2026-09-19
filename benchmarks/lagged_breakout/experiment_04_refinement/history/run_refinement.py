"""Run the recorded four-policy refinement and compare on an unused audit cohort."""

import hashlib
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continue_lagged_breakout import atomic_json, evaluation
from scripts.summarize_lagged_breakout import summarize

child = None
stop_requested = False


def stop(signum, frame):
    global stop_requested
    stop_requested = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


def run_command(command, log):
    global child
    if stop_requested:
        raise InterruptedError('Refinement stopped by the user')
    with log.open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        code = child.wait()
    child = None
    if stop_requested:
        raise InterruptedError('Refinement stopped by the user')
    if code:
        raise RuntimeError(f'Command exited {code}; see {log}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config, checkpoint, report, plan, lag):
    checkpoint_hash = digest(checkpoint)
    if report.exists():
        saved = json.loads(report.read_text())
        if (saved.get('checkpoint_sha256') == checkpoint_hash
                and saved['seed'] == plan['audit_seed']
                and len(saved['episodes']) == plan['audit_episodes']):
            return saved
    result = evaluation(config, checkpoint, plan['audit_episodes'], plan['audit_seed'],
                        lag, config['env']['volatility'])
    result['checkpoint_sha256'] = checkpoint_hash
    atomic_json(report, result)
    return result


def write_time_comparison(plan):
    rows = []
    for run in plan['runs']:
        lag, seed = run['lag'], run['seed']
        parent = ROOT / run['parent']
        current = OUTPUT / f'lag_{lag}/seed_{seed}'
        config = json.loads((current / 'config.json').read_text())
        seconds_per_decision = config['env']['frameskip'] / 60.0
        reports = {
            'training_settings': (OUTPUT / 'comparison' / f'lag_{lag}_seed_{seed}_before.json',
                                  current / 'audit_training_environment.json'),
            'original_game': (parent / 'holdout_original_environment.json',
                              current / 'holdout_original_environment.json'),
        }
        for environment, (before_path, after_path) in reports.items():
            row = {'lag': lag, 'seed': seed, 'environment': environment}
            winners = {}
            for label, path in [('before', before_path), ('after', after_path)]:
                episodes = json.loads(path.read_text())['episodes']
                winners[label] = {e['environment_seed']: e['decision_steps'] * seconds_per_decision
                                  for e in episodes if e['completed'] and e['return'] == 864}
                times = list(winners[label].values())
                row[f'{label}_wins'] = len(times)
                row[f'{label}_episodes'] = len(episodes)
                row[f'{label}_mean_win_seconds'] = statistics.mean(times) if times else None
                row[f'{label}_median_win_seconds'] = statistics.median(times) if times else None
            common = winners['before'].keys() & winners['after'].keys()
            changes = [winners['after'][s] - winners['before'][s] for s in sorted(common)]
            row['paired_wins'] = len(common)
            row['paired_mean_change_seconds'] = statistics.mean(changes) if changes else None
            row['paired_standard_error_seconds'] = (statistics.stdev(changes) / len(changes) ** .5
                                                     if len(changes) > 1 else None)
            rows.append(row)
    atomic_json(OUTPUT / 'comparison' / 'completion_times.json',
                {'units': 'simulated seconds',
                 'definition': 'Time to return 864 among completed winning episodes; compare win counts alongside times.',
                 'paired_definition': 'After minus before on environment seeds where both policies won; negative means faster.',
                 'runs': rows})
    return rows


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(1)
    plan = json.loads((OUTPUT / 'refinement_plan.json').read_text())
    completed = []

    def progress(stage, **fields):
        value = {'stage': stage, 'completed_training_lags': completed,
                 'updated_at': time.time(), **fields}
        atomic_json(OUTPUT / 'progress.json', value)
        print(json.dumps(value), flush=True)

    try:
        for run in plan['runs']:
            parent = ROOT / run['parent']
            assert digest(parent / 'policy.bin') == run['parent_policy_sha256']
            assert digest(parent / 'training_state.pt') == run['parent_training_state_sha256']
            current = OUTPUT / f"lag_{run['lag']}/seed_{run['seed']}"
            status = json.loads((current / 'status.json').read_text())
            if status['additional_steps'] < run['total_step_budget']:
                progress('training', lag=run['lag'], budget=run['total_step_budget'])
                run_command(run['command'], OUTPUT / 'history/training_logs' / f"lag_{run['lag']}_seed_42.log")
                status = json.loads((current / 'status.json').read_text())
                if status['additional_steps'] < run['total_step_budget']:
                    raise RuntimeError(f"Training stopped before budget: {current}")
            completed.append(run['lag'])
            progress('training_finished', lag=run['lag'], steps=status['additional_steps'])

        comparisons = []
        for run in plan['runs']:
            if stop_requested:
                raise InterruptedError('Refinement stopped by the user')
            lag, seed = run['lag'], run['seed']
            parent = ROOT / run['parent']
            current = OUTPUT / f'lag_{lag}/seed_{seed}'
            progress('paired_audit', lag=lag)
            before = audit(json.loads((parent / 'config.json').read_text()), parent / 'policy.bin',
                           OUTPUT / 'comparison' / f'lag_{lag}_seed_{seed}_before.json', plan, lag)
            after = audit(json.loads((current / 'config.json').read_text()), current / 'policy.bin',
                          current / 'audit_training_environment.json', plan, lag)
            # Same native vector layout, seed and episode count on both sides.
            deltas = [b['return'] - a['return'] for a, b in zip(before['episodes'], after['episodes'])]
            error = statistics.stdev(deltas) / len(deltas) ** .5
            comparisons.append({'lag': lag, 'seed': seed,
                                'before_return': before['mean_return_including_capped'],
                                'after_return': after['mean_return_including_capped'],
                                'paired_return_change': statistics.mean(deltas),
                                'paired_standard_error': error,
                                'before_win_rate': before['win_rate'], 'after_win_rate': after['win_rate'],
                                'before_capped': before['capped_episodes'], 'after_capped': after['capped_episodes'],
                                'before_sha256': before['checkpoint_sha256'], 'after_sha256': after['checkpoint_sha256']})
            atomic_json(OUTPUT / 'comparison' / 'paired_audit.json',
                        {'episodes_per_policy': plan['audit_episodes'], 'seed': plan['audit_seed'],
                         'selection': 'Training-setting validation only; audit and original-game results do not select checkpoints.',
                         'runs': comparisons})

        progress('original_comparison')
        command = [str(ROOT / '.venv/bin/python'), 'scripts/compare_lagged_breakout.py',
                   '--output', str(OUTPUT), '--evaluate-only', '--eval-episodes', str(plan['original_episodes']),
                   '--eval-seed', str(plan['original_seed']), '--max-eval-steps', str(plan['original_max_steps']),
                   '--trace-steps', '600']
        run_command(command, OUTPUT / 'history/training_logs/original_comparison.log')
        evaluation_dir = sorted((OUTPUT / 'evaluations').glob('evaluation_*'))[-1]
        result = json.loads((evaluation_dir / 'results.json').read_text())
        for row in result['results']:
            current = OUTPUT / f"lag_{row['train_lag']}/seed_{row['train_seed']}"
            row.update(lag=0, volatility=0.0, seed=plan['original_seed'], max_steps=plan['original_max_steps'],
                       win_rate=sum(e['completed'] and e['return'] == 864 for e in row['episodes']) / len(row['episodes']),
                       mean_return_including_capped=statistics.mean(e['return'] for e in row['episodes']))
            atomic_json(current / 'holdout_original_environment.json', row)
            status = json.loads((current / 'status.json').read_text())
            status['holdout_original_return'] = row['mean_return_including_capped']
            atomic_json(current / 'status.json', status)
            for comparison in comparisons:
                if comparison['lag'] == row['train_lag']:
                    parent = ROOT / plan['parent_experiment'] / f"lag_{row['train_lag']}/seed_42"
                    original_before = json.loads((parent / 'holdout_original_environment.json').read_text())
                    comparison['original_before_return'] = original_before['mean_return_including_capped']
                    comparison['original_after_return'] = row['mean_return_including_capped']
        atomic_json(OUTPUT / 'comparison' / 'paired_audit.json',
                    {'episodes_per_policy': plan['audit_episodes'], 'seed': plan['audit_seed'],
                     'selection': 'Training-setting validation only; audit and original-game results do not select checkpoints.',
                     'runs': comparisons})
        atomic_json(OUTPUT / 'final_evaluation.json',
                    {'original_comparison': str(evaluation_dir.relative_to(OUTPUT)),
                     'training_audit_episodes_per_policy': plan['audit_episodes'],
                     'training_audit_seed': plan['audit_seed'],
                     'original_episodes_per_policy': plan['original_episodes'],
                     'original_seed': plan['original_seed'],
                     'original_max_decision_steps': plan['original_max_steps']})
        summarize(OUTPUT)
        write_time_comparison(plan)
        progress('complete', comparison='comparison/paired_audit.json')
    except BaseException as exc:
        progress('paused' if stop_requested else 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
