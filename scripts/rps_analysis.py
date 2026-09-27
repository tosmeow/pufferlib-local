"""RPS probability summaries, adaptation timing, and CPU/GPU parity checks."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import pathlib
import statistics


def summarize(root, output):
    groups = {}
    conditional = {}
    match_returns = collections.defaultdict(lambda: collections.defaultdict(float))


    def accumulate(table, key, row):
        p = [float(row[f"p_{move}"]) for move in ("rock", "paper", "scissors")]
        g = table.setdefault(key, dict(n=0, sums=[0.0]*3, squares=[0.0]*3,
            minimum=[1.0]*3, maximum=[0.0]*3, actions=[0]*3, reward=0.0,
            entropy=0.0, mean_tv=0.0, max_tv=0.0, oracle_edge=0.0))
        g["n"] += 1
        for i in range(3):
            g["sums"][i] += p[i]
            g["squares"][i] += p[i]*p[i]
            g["minimum"][i] = min(g["minimum"][i], p[i])
            g["maximum"][i] = max(g["maximum"][i], p[i])
        g["actions"][int(row["action"])] += 1
        g["reward"] += int(row["reward"])
        g["entropy"] += -sum(x*math.log(x) for x in p if x)
        tv = sum(abs(x - 1/3) for x in p)/2
        g["mean_tv"] += tv
        g["max_tv"] = max(g["max_tv"], tv)
        g["oracle_edge"] += max(p[2]-p[1], p[0]-p[2], p[1]-p[0])


    with (root / "trajectories.csv").open() as f:
        for row in csv.DictReader(f):
            scenario = row["scenario"]
            accumulate(groups, scenario, row)
            if scenario.startswith("bot_"):
                for rounds in (1, 5, 10, 50):
                    if int(row["step"]) < rounds:
                        accumulate(groups, f"{scenario}:first_{rounds}", row)
                if int(row["step"]) >= 50:
                    accumulate(groups, scenario + ":after_50", row)
            # Exclude first 100 rounds for a separate warm-memory view.
            if int(row["step"]) >= 100:
                accumulate(groups, scenario + ":warm", row)
            key = f'{scenario}:{row["previous_self"]},{row["previous_opponent"]}'
            accumulate(conditional, key, row)
            match_returns[scenario][int(row["match"])] += int(row["reward"]) / 1000


    def finish(g):
        n = g["n"]
        means = [x/n for x in g["sums"]]
        return dict(n=n, mean_probabilities=means,
            probability_std=[math.sqrt(max(0, g["squares"][i]/n-means[i]**2)) for i in range(3)],
            minimum_probabilities=g["minimum"], maximum_probabilities=g["maximum"],
            sampled_frequencies=[x/n for x in g["actions"]], mean_reward=g["reward"]/n,
            mean_entropy_nats=g["entropy"]/n, mean_tv_from_uniform=g["mean_tv"]/n,
            max_tv_from_uniform=g["max_tv"], informed_one_step_opponent_edge=g["oracle_edge"]/n)


    histories = collections.defaultdict(list)
    same_observation = collections.defaultdict(list)
    zero_memory = []
    with (root / "probes.csv").open() as f:
        for row in csv.DictReader(f):
            p = [float(row[f"p_{move}"]) for move in ("rock", "paper", "scissors")]
            if row["scenario"] == "zero_memory":
                zero_memory.append(dict(previous_self=int(row["previous_self"]),
                    previous_opponent=int(row["previous_opponent"]), probabilities=p))
            elif int(row["previous_self"]) >= 0:
                key = (row["scenario"], row["match"], row["step"])
                histories[key].append(p)
                same_observation[(row["previous_self"], row["previous_opponent"])].append(p)

    # Hold hidden state fixed while changing only the last-move observation.
    obs_ranges = [max(max(p[a] for p in ps)-min(p[a] for p in ps) for a in range(3))
                  for ps in histories.values()]
    # Hold observation fixed while changing hidden state accumulated on actual paths.
    history_ranges = {','.join(key): [max(p[a] for p in ps)-min(p[a] for p in ps)
                        for a in range(3)] for key, ps in same_observation.items()}
    summary = dict(groups={k: finish(v) for k, v in groups.items()},
        conditional={k: finish(v) for k, v in conditional.items()},
        zero_memory=zero_memory,
        observation_effect=dict(histories=len(obs_ranges), mean_max_probability_range=statistics.mean(obs_ranges),
            max_probability_range=max(obs_ranges)),
        history_effect_ranges=history_ranges,
        match_reward_ci95={k: [statistics.mean(v.values()) - 1.96*statistics.stdev(v.values())/math.sqrt(len(v)),
            statistics.mean(v.values()) + 1.96*statistics.stdev(v.values())/math.sqrt(len(v))]
            for k, v in match_returns.items()})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != "conditional"}, indent=2))


def adaptation(source, expected_hash, root):
    with source.open('rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
        actual_hash = digest.hexdigest()
    assert actual_hash == expected_hash, 'Trajectory checksum mismatch'

    matches = collections.defaultdict(list)
    with source.open() as f:
        for row in csv.DictReader(f):
            scenario = row['scenario']
            if scenario not in ('bot_uniform', 'bot_counter', 'bot_rock'):
                continue
            turn = int(row['step']) + 1
            p = [float(row['p_' + a]) for a in ('rock', 'paper', 'scissors')]
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in p)
            assert abs(sum(p) - 1) < 1e-8
            if scenario == 'bot_uniform':
                target = [1/3] * 3  # Uniformity diagnostic, not optimality.
            elif turn == 1:
                matches[scenario, int(row['match'])].append(False)
                continue  # All bots open uniformly: no uniquely optimal action.
            else:
                best = 1 if scenario == 'bot_rock' else (int(row['previous_self']) + 2) % 3
                target = [float(a == best) for a in range(3)]
            close = max(abs(x - y) for x, y in zip(p, target)) <= 0.01
            flags = matches[scenario, int(row['match'])]
            assert len(flags) == turn - 1
            flags.append(close)

    rows = []
    for (scenario, match), flags in sorted(matches.items()):
        assert len(flags) == 1000
        start = next((i + 1 for i in range(998) if all(flags[i:i+3])), None)
        confirm = None if start is None else start + 2
        last_bad = max((i for i, good in enumerate(flags) if not good), default=-1)
        sustained = last_bad + 2 if 1000 - (last_bad + 1) >= 3 else None
        rows.append(dict(scenario=scenario, match=match, first_streak_start=start,
            confirmation_round=confirm, sustained_start=sustained,
            failures_after_confirmation=None if confirm is None else sum(not x for x in flags[confirm:])))

    def distribution(values):
        values = [v for v in values if v is not None]
        return dict(count=len(values), minimum=min(values) if values else None,
            median=statistics.median(values) if values else None,
            maximum=max(values) if values else None,
            histogram=dict(sorted(collections.Counter(values).items())))

    summary = {}
    for scenario in ('bot_uniform', 'bot_counter', 'bot_rock'):
        subset = [r for r in rows if r['scenario'] == scenario]
        assert len(subset) == 64
        summary[scenario] = dict(matches=64,
            first_streak_start=distribution(r['first_streak_start'] for r in subset),
            confirmation_round=distribution(r['confirmation_round'] for r in subset),
            sustained_start=distribution(r['sustained_start'] for r in subset),
            matches_with_later_failure=sum((r['failures_after_confirmation'] or 0) > 0 for r in subset),
            fraction_close_by_round={t: sum(matches[scenario, r['match']][t-1] for r in subset)/64
                for t in range(1, 21)})

    result = dict(source=str(source), source_sha256=actual_hash, tolerance=0.01,
        consecutive_rounds=3, round_numbering='one-based; opening excluded for deterministic bots',
        uniform_metric='closeness to uniform only; all policies are optimal against uniform',
        summary=summary)
    (root/'adaptation.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root/'matches.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


def parity(root):
    def read(name):
        with (root/name).open() as f:
            return {(r['agent'],r['step']): [float(r['p_'+a]) for a in ['rock','paper','scissors']] for r in csv.DictReader(f)}
    cpu,gpu=read('cpu.csv'),read('gpu.csv')
    assert cpu.keys()==gpu.keys() and len(cpu)==10000
    error=max(abs(x-y) for k in cpu for x,y in zip(cpu[k],gpu[k]))
    result=dict(rows=len(cpu),max_absolute_probability_error=error,tolerance=1e-5,passed=error<1e-5,
        gpu_first_move=gpu[('0','0')],gpu_second_move=gpu[('0','1')])
    (root/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert result['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    summary = commands.add_parser('summary', help='Summarize trajectory/probe CSVs')
    summary.add_argument('directory', type=pathlib.Path)
    summary.add_argument('--output', type=pathlib.Path)
    adapt = commands.add_parser('adaptation', help='Check three-round probability tolerance')
    adapt.add_argument('source', type=pathlib.Path)
    adapt.add_argument('sha256')
    adapt.add_argument('--output', type=pathlib.Path,
                       default=os.environ.get('CLUSTER_RESULTS_DIR'))
    check = commands.add_parser('parity', help='Compare cpu.csv and gpu.csv')
    check.add_argument('directory', type=pathlib.Path)
    args = parser.parse_args()
    if args.command == 'summary':
        output = args.output or args.directory
        output.mkdir(parents=True, exist_ok=True)
        summarize(args.directory, output)
    elif args.command == 'adaptation':
        if args.output is None:
            parser.error('adaptation needs --output or CLUSTER_RESULTS_DIR')
        args.output.mkdir(parents=True, exist_ok=True)
        adaptation(args.source, args.sha256, args.output)
    else:
        parity(args.directory)


if __name__ == '__main__':
    main()
