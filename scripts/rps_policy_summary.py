"""Summarize exact softmax probabilities, not just sampled action counts."""
import collections
import csv
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
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
(root / "summary.json").write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != "conditional"}, indent=2))
