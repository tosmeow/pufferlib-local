# Lag-8 transfer diagnostic

The main evidence points to increased dependence on the delayed-action environment. This is a diagnosis of the selected before/after checkpoint pair, not an isolated causal test of optimizer, hardware, or feature dependence.

Each cell evaluates the same lag-8-trained policy on 128 new environment seeds (700000–700127), using greedy actions. The only evaluation conditions varied are action lag and volatility. All episodes finished before the 120000-decision cap.

| Evaluation lag | Volatility | Before score | After score | Mean change | Paired standard error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 6 | 861.42 | 862.34 | +0.92 | 1.50 |
| 8 | 0 | 864.00 | 863.17 | -0.83 | 0.51 |
| 0 | 6 | 174.52 | 124.00 | -50.52 | 8.40 |
| 0 | 0 | 130.86 | 95.78 | -35.08 | 4.60 |

Keeping lag 8 preserves near-ceiling performance with either volatility. Removing the delay produces poor scores with either volatility, and additional training makes both lag-0 results worse. The degradation therefore replicates on new seeds and is associated with removing action delay, rather than removing volatility alone.

During continuation, the policy trained only at lag 8 and volatility 6. Its observations explicitly include lag, volatility and the pending action queue (ocean/lagged_breakout/lagged_breakout.h). Setting lag to zero changes both physical action timing and those observation features. This diagnostic does not separate sensitivity to the features from dependence on the delayed dynamics.

A plausible mechanism is that continued optimization changes a controller already tuned to compensate for delayed actions, while providing no training feedback about immediate-action behavior. Training can keep changing a near-ceiling policy: its objective uses discounted/clipped rewards and value/entropy terms, and this continuation still performed 15510 optimizer steps. The test measures undiscounted episode score.

The original checkpoint was selected by performance at its own lag; the continuation retained final weights without selecting on common-game validation. Neither procedure protects transfer performance. The fresh optimizer, Linux/CUDA training and changed schedule were not independently controlled, so their contribution cannot be assigned from this pair of checkpoints.

A useful next experiment for measuring transfer is periodic evaluation on a separate unlagged validation seed set during fixed-lag training, retaining snapshots and reserving untouched seeds for the final test. Mixing lag-0 episodes into training would instead change the training distribution and answer a different experimental question.

[Raw diagnostic results](evaluations/lag8_transfer_diagnostic_1788710475746375000/results.json)

General background: [PPO](https://arxiv.org/abs/1707.06347) optimizes a surrogate objective from environment interactions; [Quantifying Generalization in Reinforcement Learning](https://proceedings.mlr.press/v97/cobbe19a.html) documents that strong training-environment performance need not imply strong generalization. These papers provide context, not proof of the specific mechanism in this run.
