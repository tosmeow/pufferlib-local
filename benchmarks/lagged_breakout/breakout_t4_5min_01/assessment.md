# Five-minute T4 training assessment

The policies learned substantially, but none reliably solves Breakout yet. The lag-2 checkpoint is strongest in these evaluations; lag 8 remains weak. The maximum episode score is 864.

| Training lag | Before: common game | After: common game | After: training conditions | Wins in training conditions |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.72 | 375.38 | 355.69 | 0/128 |
| 2 | 6.62 | 626.59 | 529.21 | 3/128 |
| 4 | 4.34 | 233.44 | 317.88 | 0/128 |
| 8 | 6.59 | 27.38 | 44.00 | 0/128 |

The common game uses lag 0, volatility 0, greedy actions and 32 episodes per checkpoint (environment seeds 100000–100031). Both parent and new policies were evaluated on this Mac with the same configuration, so the parent numbers differ slightly from the earlier Linux evaluation. None won a common-game episode.

Training-condition evaluation uses each policy’s own lag, volatility 6, greedy actions and 128 episodes (environment seeds 200000–200127). Lag 2 won 3/128 episodes (2.34%); the other policies won none. All 640 evaluation episodes of the new checkpoints terminated before the 120000-decision cap. Scores are episode-return means, not training-log rewards.

The standard errors of the new common-game means are 6.48, 27.99, 14.94 and 3.73, respectively. For the training-condition means they are 5.22, 13.09, 4.55 and 2.17. This assesses one training seed per lag; it does not establish an optimal lag across independently trained models.

## Training execution

| Lag | New decisions | Training seconds | Mean SPS | Rollout/update cycles | Optimizer steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 266,338,304 | 300.31 | 886,886 | 127 | 2,794 |
| 2 | 266,338,304 | 300.75 | 885,575 | 127 | 2,794 |
| 4 | 270,532,608 | 302.17 | 895,293 | 129 | 2,838 |
| 8 | 239,075,328 | 300.23 | 796,314 | 114 | 2,508 |

All four jobs completed and all downloaded policies contain finite weights. The native/Python source hashes in their manifests match the current checkout. Training scores were still increasing at the cutoff, especially for lags 0, 2 and 4; the logs show no clear plateau yet.

The large rollout/minibatches were selected for throughput. Relative to 4096 agents and minibatch 16384, this configuration uses eight times as many agents and eight times the minibatch size; it performs the same 22 optimizer steps per rollout but eight times fewer optimizer steps per million game decisions. This changes learning dynamics. High SPS alone is not evidence of fastest improvement in score. A next experiment should compare score gained per minute when continuing these weights with the current batch versus a smaller one; this assessment does not establish which will be better.

## Raw evaluations

- [Common-game results](evaluations/evaluation_1788708115339128000/results.json)
- [Training-condition results](evaluations/training_conditions_1788708161596131000/results.json)
- [Parent reevaluation](../modal_20260906_141333_119624d8/evaluations/evaluation_1788708116533668000/results.json)
