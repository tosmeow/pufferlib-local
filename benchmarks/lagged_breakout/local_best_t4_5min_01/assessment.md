# Five-minute T4 continuation from the best local checkpoints

All four seed-42 policies completed five minutes on separate T4 GPUs. The run preserves their normalized velocity encoder and uses 4096 agents, horizon 64 and minibatch 16384. Learning rate decays from 0.0001 to 0.00001. Optimizer momentum starts fresh because the source bundle contains the selected best policy weights only.

Performance at the training conditions remains close to the starting policies. This run does not show a uniform improvement. Lag 8 loses performance on the common unlagged game; the original selected checkpoints remain available.

## Each policy at its own training lag, volatility 6

| Lag | Before mean | After mean | Before wins | After wins |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 862.08 | 864.00 | 254/256 | 256/256 |
| 2 | 864.00 | 864.00 | 256/256 | 256/256 |
| 4 | 862.11 | 859.80 | 254/256 | 249/256 |
| 8 | 854.89 | 854.48 | 229/256 | 238/256 |

Greedy actions; 256 matched test episodes per checkpoint, starting at environment seed 600000. All episodes terminated before the 120000-decision cap. Maximum score is 864. Small mean changes should be interpreted in light of episode variability.

[Before results](../local_best_seed42_20260906_153321/evaluations/continuation_own_1788708901857523000/results.json) · [After results](evaluations/continuation_own_1788709252239806000/results.json)

## Common game: lag 0, volatility 0

| Lag | Before mean | After mean | Before wins | After wins |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 864.00 | 864.00 | 64/64 | 64/64 |
| 2 | 835.70 | 843.20 | 38/64 | 42/64 |
| 4 | 564.45 | 564.77 | 3/64 | 1/64 |
| 8 | 125.28 | 93.91 | 0/64 | 0/64 |

Greedy actions; 64 matched test episodes per checkpoint, starting at environment seed 100000. All episodes terminated before the 120000-decision cap. Maximum score is 864. Small mean changes should be interpreted in light of episode variability.

[Before results](../local_best_seed42_20260906_153321/evaluations/continuation_common_1788708901857347000/results.json) · [After results](evaluations/continuation_common_1788709252239805000/results.json)

## Training execution

| Lag | Seconds | Additional steps | Average SPS |
| ---: | ---: | ---: | ---: |
| 0 | 300.388 | 180,355,072 | 600,406 |
| 2 | 300.034 | 179,568,640 | 598,494 |
| 4 | 300.298 | 182,190,080 | 606,697 |
| 8 | 300.441 | 184,811,520 | 615,134 |

Verification: four distinct GPU UUIDs, overlapping execution, completed status in every manifest, finite updated policy weights, matching weights in optimizer checkpoints, saved CUDA RNG state, and consistent step counts. Final checkpoints and logs are downloaded alongside this report.

[Source checkpoint selection and hashes](../local_best_seed42_20260906_153321/selection.json) · [Modal run](https://modal.com/apps/tosmeow/main/ap-iRJ7D6lMAHpwuzUIB1cbrH)
